# timing-oracle

**Detect timing side channels in Rust code with statistically rigorous methods.**

`timing-oracle` measures whether your code's execution time depends on secret data. It compares timing between a "fixed" input (e.g., a test vector) and random inputs, then uses Bayesian inference to compute the probability of a timing leak and estimate its magnitude. It can be used to evaluate cryptographic implementations, constant-time algorithms, or any code where timing leaks are a concern, and can be easily integrated into CI pipelines.

Traditional timing leak detection tools often rely on t-tests or other mean-based statistics, which provide hard-to-interpret results and usually require a lot of fiddling to prevent test flakiness. `timing-oracle` improves upon these methods by using quantile-based statistics and a two-layer Bayesian analysis, which simply reports the probability of a leak in the tested code, mediums by which a leak might be exploitable (is it exploitable over the internet? just locally?) and provides bounded guarantees on false positive rates (so it's easy to use in CI).

It's inspired by [DudeCT](https://appsec.guide/docs/crypto/constant_time_tool/dudect/), [DudeCT-bencher](github.com/rozbb/dudect-bencher) and the paper [
With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors](https://www.usenix.org/conference/usenixsecurity24/presentation/dunsche), though it uses a novel statistical approach that aims to be more accurate and interpretable.

## Quick Start

```shell
$ cargo add timing-oracle
```

```rust
use timing_oracle::{test, helpers::InputPair};

let secret = [0u8; 32];

// Pre-generate inputs BEFORE measurement
let inputs = InputPair::new(
    [0u8; 32],                    // Fixed: all zeros
    || rand::random::<[u8; 32]>() // Random: generated per sample
);

let result = test(
    || compare(&secret, inputs.fixed()),
    || compare(&secret, inputs.random()),
);

if result.leak_probability > 0.9 {
    panic!("Timing leak detected: {:.0}%", result.leak_probability * 100.0);
}
```

See [examples/](examples/) for more usage patterns.

## How It Works

### Two-Layer Analysis

```
                    Timing Samples
                          |
           +--------------+--------------+
           v                             v
   +---------------+           +------------------+
   |  Layer 1:     |           |  Layer 2:        |
   |  CI Gate      |           |  Bayesian        |
   +---------------+           +------------------+
   | Bootstrap     |           | Conjugate Bayes  |
   | thresholds    |           | inference        |
   |               |           |                  |
   | Bounded FPR   |           | Probability +    |
   | Pass/Fail     |           | Effect size      |
   +---------------+           +------------------+
```

**Layer 1 (CI Gate)** answers: "Should this block my build?"
- Uses RTLF-style bootstrap thresholds for bounded false positive rate
- Guarantees FPR <= alpha (default 1%)

**Layer 2 (Bayesian)** answers: "What's the probability and magnitude?"
- Computes Bayes factor between no-leak and leak hypotheses
- Decomposes effect into uniform shift and tail components
- Provides 95% credible intervals

### Quantile-Based Statistics

Rather than comparing means (which miss distributional differences), we compare nine deciles:

```
Fixed class:   [q10, q20, q30, q40, q50, q60, q70, q80, q90]
Random class:  [q10, q20, q30, q40, q50, q60, q70, q80, q90]
Difference:    delta in R^9
```

This captures:
- **Uniform shifts**: Different code path (affects all quantiles equally)
- **Tail effects**: Cache misses (affects upper quantiles more)

### Statistical Guarantees

- **Bounded false positive rate**: CI gate guarantees FPR <= alpha
- **Interpretable probabilities**: Bayesian posterior, not p-values
- **Effect decomposition**: Separates uniform shift from tail effects
- **Exploitability assessment**: Based on Crosby et al. (2009) thresholds

For full methodology details, see [docs/guide.md](docs/guide.md).

## Basic Usage

### Builder Pattern

```rust
use timing_oracle::TimingOracle;

let result = TimingOracle::new()
    .samples(100_000)           // Samples per class
    .ci_alpha(0.01)             // CI false positive rate
    .test(fixed_fn, random_fn);

// Key result fields
println!("Leak probability: {:.0}%", result.leak_probability * 100.0);
println!("CI gate: {}", if result.ci_gate.passed { "PASS" } else { "FAIL" });
println!("Quality: {:?}", result.quality);
```

### Presets

```rust
TimingOracle::new()        // Default: 100k samples (~5s)
TimingOracle::balanced()   // Production CI: 20k samples (~1s)
TimingOracle::quick()      // Development: 5k samples (~0.3s)
```

### Exploitability Thresholds

| Effect Size | Assessment | Implications |
|------------|------------|--------------|
| < 100 ns | `Negligible` | Requires impractical measurement count |
| 100-500 ns | `PossibleLAN` | Exploitable on LAN with ~100k queries |
| 500 ns - 20 us | `LikelyLAN` | Likely exploitable on LAN |
| > 20 us | `PossibleRemote` | Possibly exploitable over internet |

For complete API documentation, see [docs/api-reference.md](docs/api-reference.md).

## Common Mistakes

### Don't Generate Random Data Inside Closures

The most common mistake is calling RNG functions inside the measured closures:

```rust
// WRONG - RNG overhead measured in random closure only
let result = test(
    || encrypt(&KEY, &[0u8; 32]),
    || encrypt(&KEY, &rand::random()),  // Measures RNG + encrypt!
);

// CORRECT - Pre-generate inputs with InputPair
let inputs = InputPair::new([0u8; 32], || rand::random());
let result = test(
    || encrypt(&KEY, inputs.fixed()),
    || encrypt(&KEY, inputs.random()),
);
```

Both closures must execute **identical code paths** - only the data differs.

### Side-Effects to Avoid

Inside measured closures, never:
- Call `rand::random()` or any RNG functions
- Allocate with `vec![]`, `Vec::new()`, `String::new()`, etc.
- Perform I/O (file reads, network calls)
- Use `println!()` or other logging

For more patterns and advanced examples, see [docs/guide.md](docs/guide.md).

## CI Integration

```rust
#[test]
fn test_constant_time_compare() {
    let inputs = InputPair::new([0u8; 32], || rand::random());

    let result = TimingOracle::balanced()
        .test(
            || my_compare(inputs.fixed()),
            || my_compare(inputs.random()),
        );

    assert!(result.ci_gate.passed, "Timing leak detected");
    assert!(result.leak_probability < 0.5, "Elevated leak probability");
}
```

### Handling Unreliable Measurements

Some tests may be unreliable on certain platforms (e.g., Apple Silicon with coarse timers):

```rust
use timing_oracle::{Outcome, skip_if_unreliable};

#[test]
fn test_cache_timing() {
    let result = TimingOracle::new().test(fixed, random);
    let outcome = Outcome::Completed(result);

    // Skip if measurement is unreliable (prints warning, returns early)
    let result = skip_if_unreliable!(outcome, "test_cache_timing");

    assert!(result.leak_probability > 0.5);
}
```

For detailed reliability handling and troubleshooting, see [docs/troubleshooting.md](docs/troubleshooting.md).

## Build & Test

```bash
cargo build                     # Build with all features
cargo test                      # Run all tests
cargo test --test known_leaky   # Leak detection tests
cargo test --test known_safe    # False-positive tests
cargo test --test aes_timing    # AES-128 timing tests
cargo test --test ecc_timing    # Curve25519 timing tests
cargo bench --bench comparison  # Compare with DudeCT (~10 min)
```

For test organization details, see [TESTING.md](TESTING.md).

## Documentation

| Document | Description |
|----------|-------------|
| [docs/guide.md](docs/guide.md) | Conceptual overview, methodology, advanced examples |
| [docs/api-reference.md](docs/api-reference.md) | Complete API documentation |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Debugging, reliability handling, performance tips |
| [docs/architecture.md](docs/architecture.md) | Internal architecture and implementation details |
| [examples/README.md](examples/README.md) | Example catalog with suggested reading order |
| [TESTING.md](TESTING.md) | Test organization and running instructions |

## Feature Flags

- **`parallel`** (default) - Rayon-based parallel bootstrap (4-8x speedup)
- **`kperf`** (default, macOS ARM64) - PMU-based cycle counting support
- **`perf`** (default, Linux) - perf_event-based cycle counting support

```toml
# Minimal build without optional features
timing-oracle = { version = "0.1", default-features = false }
```

## References

1. Reparaz, O., Balasch, J., & Verbauwhede, I. (2016). "Dude, is my code constant time?" DATE.
2. Dunsche, M., et al. (2024). "With Great Power Come Great Side Channels." USENIX Security.
3. Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009). "Opportunities and limits of remote timing attacks." ACM TISSEC.

## License

MIT OR Apache-2.0
