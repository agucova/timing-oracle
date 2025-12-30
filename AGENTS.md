# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timing-oracle` is a Rust library for detecting timing side channels in cryptographic code. It uses statistical methodology to compare timing distributions between fixed and random inputs, outputting leak probability, effect sizes, and exploitability assessments.

## Build Commands

```bash
cargo build                    # Build the library
cargo build --features parallel # Build with parallel/rayon support
cargo test                     # Run all tests (many are ignored pending implementation)
cargo test -- --ignored        # Run ignored tests (requires full implementation)
cargo test <test_name>         # Run a specific test
cargo check                    # Type-check without building
cargo run --example simple     # Run the simple example
cargo run --example compare    # Run the compare example
```

## Architecture

### Core Pipeline

The timing oracle follows a multi-layer analysis pipeline:
1. **Preflight checks** (`src/preflight/`) - Validates measurement setup before analysis
2. **Measurement** (`src/measurement/`) - High-resolution cycle counting with interleaved sampling
3. **Outlier filtering** - Symmetric percentile-based trimming
4. **Quantile computation** - Decile differences between fixed/random classes
5. **CI Gate** (`src/analysis/ci_gate.rs`) - Fast frequentist screening with bounded false positive rate
6. **Bayesian inference** (`src/analysis/bayes.rs`) - Posterior probability of timing leak
7. **Effect decomposition** (`src/analysis/effect.rs`) - Separates uniform shift from tail effects

### Module Structure

- `TimingOracle` (`src/oracle.rs`) - Builder-pattern entry point
- `Config` (`src/config.rs`) - All tunable parameters (samples, alpha, thresholds)
- `TestResult` (`src/result.rs`) - Output struct with leak_probability, effect, ci_gate, exploitability
- `statistics/` - Quantile computation, bootstrap resampling, covariance estimation
- `output/` - Terminal and JSON formatters

### Key Types

- `Class::Fixed` / `Class::Random` - Input class identifiers
- `Matrix9`, `Vector9` - nalgebra types for 9 decile quantile differences
- `Exploitability` - Negligible (<100ns), PossibleLAN, LikelyLAN, PossibleRemote (>20μs)

### Test Organization

**Core Validation Tests:**
- `tests/known_leaky.rs` - Tests that MUST detect timing leaks (early-exit comparison, branches) [2 tests]
- `tests/known_safe.rs` - Tests that MUST NOT false-positive (XOR, constant-time comparison) [2 tests]
- `tests/calibration.rs` - Statistical validation (CI gate FPR, Bayesian calibration) [2 tests, 100 trials each]

**Comprehensive Integration Tests:**

All integration tests use **DudeCT's two-class pattern**:
- **Class 0**: All-zero data (0x00 repeated)
- **Class 1**: Random data

This pattern tests for data-dependent timing rather than specific value comparisons.

- `tests/crypto_attacks.rs` - Real-world crypto timing attacks (AES S-box, modular exponentiation, cache effects, effect patterns, exploitability thresholds) [20 tests total]
  - 4 cache-based tests (AES, cache lines, memory patterns)
  - 2 modular exponentiation tests (square-and-multiply, bit patterns)
  - 3 table lookup tests (L1/L2/L3 cache)
  - 3 effect pattern validation tests (UniformShift, TailEffect, Mixed)
  - 4 exploitability threshold tests (Negligible, PossibleLAN, LikelyLAN, PossibleRemote)
  - 4 tests marked `#[ignore]` for thorough validation
- `tests/async_timing.rs` - Async/await and concurrent task timing [9 tests total]
  - 2 baseline tests (executor overhead, block_on symmetry)
  - 3 leak detection tests (conditional await, early exit, sleep duration)
  - 2 concurrent task tests (crosstalk, spawn count)
  - 2 optional thorough tests (runtime comparison, flag effectiveness)
- `tests/aes_timing.rs` - AES-128 encryption timing tests inspired by DudeCT [7 tests total]
  - Block encryption with zeros vs random plaintexts
  - Different keys with fixed plaintext
  - Multiple blocks cumulative timing
  - Key initialization timing
  - Hamming weight independence (0x00 vs 0xFF)
  - Byte pattern independence (sequential vs reverse)
- `tests/ecc_timing.rs` - Curve25519/X25519 elliptic curve timing tests [8 tests total]
  - Scalar multiplication with zeros vs random scalars
  - Different basepoints timing
  - Multiple operations cumulative timing
  - Scalar clamping timing
  - Hamming weight independence (0x00 vs 0xFF)
  - Byte pattern independence (sequential vs reverse)
  - Full ECDH key exchange timing

**Test Execution:**
```bash
# Fast suite (non-ignored tests)
cargo test --test crypto_attacks  # ~3-5 minutes
cargo test --test async_timing    # ~20-40 seconds
cargo test --test aes_timing      # ~2-3 minutes
cargo test --test ecc_timing      # ~2-3 minutes

# Full suite (includes ignored tests)
cargo test --test crypto_attacks -- --ignored  # ~15-20 minutes
cargo test --test async_timing -- --ignored    # ~5-10 minutes

# All integration tests
cargo test --test known_leaky --test known_safe --test calibration --test crypto_attacks --test async_timing --test aes_timing --test ecc_timing
```

See `TESTING.md` for detailed documentation of each test category.

## API Usage Pattern

```rust
// Simple API
let result = timing_oracle::test(
    || my_function(&fixed_input),
    || my_function(&random_input()),
);

// Builder API with configuration
let result = TimingOracle::new()
    .samples(100_000)
    .ci_alpha(0.01)
    .effect_prior_ns(10.0)
    .test(fixed_closure, random_closure);
```

## Performance Optimization

### Configuration Presets

Choose the right preset for your use case to balance speed and accuracy:

```rust
// Default - Most accurate, slowest (~5-10 seconds per test)
// 100k samples, 100 CI bootstrap, 50 covariance bootstrap
TimingOracle::new()

// Balanced - Recommended for production (recommended, ~1-2 seconds per test)
// 20k samples, 100 CI bootstrap, 50 covariance bootstrap
TimingOracle::balanced()

// Quick - Fast iteration during development (~0.2-0.5 seconds per test)
// 5k samples, 50 CI bootstrap, 50 covariance bootstrap
TimingOracle::quick()

// Calibration - For running many trials (100+) (~0.1-0.2 seconds per test)
// 2k samples, 30 CI bootstrap, 20 covariance bootstrap
TimingOracle::calibration()
```

### Parallel Processing

Enable the `parallel` feature for 4-8x speedup on multi-core systems:

```toml
[dependencies]
timing-oracle = { version = "0.1", features = ["parallel"] }
```

```bash
# Build with parallel support
cargo build --release --features parallel
cargo test --release --features parallel
```

The parallel feature uses rayon to parallelize bootstrap iterations across CPU cores.

### Timer Reuse

When running many trials (e.g., calibration tests), reuse a single Timer to avoid repeated calibration overhead (~100ms per Timer::new()):

```rust
use timing_oracle::{TimingOracle, Timer};

// Calibrate once
let timer = Timer::new();

// Reuse across 100 trials (saves ~10 seconds)
for _ in 0..100 {
    let result = TimingOracle::calibration()
        .with_timer(timer.clone())
        .test(|| fixed_op(), || random_op());
}
```

### Performance Tips

1. **Start with `.balanced()`** for most use cases - 5x faster than default with minimal accuracy loss
2. **Use `.calibration()` for test suites** that run 100+ trials
3. **Enable `parallel` feature** for maximum performance on multi-core systems
4. **Reuse timers** in loops to avoid calibration overhead
5. **Use `.quick()` during development** for rapid iteration

### Performance Comparison

With parallel feature enabled on an 8-core machine:

| Preset | Samples | Runtime | Speedup vs Default |
|--------|---------|---------|-------------------|
| Default | 100k | ~5s | 1x |
| Balanced | 20k | ~1s | 5x |
| Quick | 5k | ~0.3s | 17x |
| Calibration | 2k | ~0.15s | 33x |

## Feature Flags

- `default` - No extra features
- `parallel` - Enables rayon for parallel bootstrap computation
