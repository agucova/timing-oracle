# Testing Guide

This document describes the testing strategy for `timing-oracle` and how to run the test suite.

## Test Organization

### Unit Tests (`cargo test --lib`)

Fast tests that validate individual components:
- **Statistical functions**: Quantile computation, bootstrap, covariance
- **Analysis layers**: CI gate, Bayesian inference, effect decomposition
- **Measurement**: Timer calibration, sample collection, outlier filtering
- **Preflight checks**: Sanity, generator cost, autocorrelation, system

**Runtime:** ~3 seconds

### Integration Tests

#### Known Leaky (`tests/known_leaky.rs`)

Tests that **must detect** timing leaks:

```rust
cargo test --test known_leaky
```

**Tests:**
- `detects_early_exit_comparison` - Early-exit byte-by-byte comparison
- `detects_branch_timing` - Branch-based timing (if x == 0)

**Expected:**
- `leak_probability > 0.9`
- `ci_gate.passed == false`

**Runtime:** ~2-3 minutes each (100k samples)

#### Known Safe (`tests/known_safe.rs`)

Tests that **must not false-positive** on constant-time code:

```rust
cargo test --test known_safe
```

**Tests:**
- `no_false_positive_xor_compare` - Constant-time XOR-based comparison
- `no_false_positive_xor` - Simple XOR operation

**Expected:**
- `leak_probability < 0.5`
- `ci_gate.passed == true`

**Runtime:** ~2-3 minutes each (100k samples)

#### Calibration (`tests/calibration.rs`)

Statistical validation tests:

```rust
cargo test --test calibration
```

**Tests:**
- `ci_gate_fpr_calibration` - Verify false positive rate ≤ 2α (100 trials)
- `bayesian_calibration` - Verify posterior doesn't over-concentrate (100 trials)

**Expected:**
- CI gate: rejection rate ≤ 2% with α=0.01
- Bayesian: <10% of null trials have probability > 0.9

**Runtime:** ~5-10 minutes (many trials with quick mode)

## Running Tests

### Quick Iteration

```bash
# Run only unit tests (fast)
cargo test --lib

# Run a specific integration test
cargo test --test known_leaky detects_early_exit_comparison

# Run with nextest (faster parallel execution)
cargo nextest run
```

### Full Test Suite

```bash
# Run everything (unit + integration)
cargo test

# Run with single thread (more reliable timing)
cargo test -- --test-threads=1

# Run with nextest in timing profile (single-threaded)
cargo nextest run --profile timing
```

### Continuous Integration

For CI pipelines, use the full suite with single-threaded execution:

```bash
cargo test -- --test-threads=1
```

## Test Configuration

### Timer Reuse Optimization

The calibration tests use `Timer::new()` once and reuse it across trials to avoid ~50ms calibration overhead per test:

```rust
let timer = Timer::new();  // Calibrate once

for _ in 0..100 {
    let result = TimingOracle::quick()
        .with_timer(timer.clone())  // Reuse
        .test(|| noise(), || noise());
}
```

### Quick Mode

Integration tests use `TimingOracle::new()` with 100k samples for thorough validation. During development, use `TimingOracle::quick()` (5k samples) for faster iteration.

## Interpreting Test Failures

### Known Leaky Tests Fail to Detect

**Possible causes:**
1. High noise environment (check `result.quality`)
2. Insufficient sample size
3. Timer calibration issue

**Debug:**
```bash
# Run with verbose output
RUST_LOG=debug cargo test --test known_leaky -- --nocapture

# Check system configuration
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor  # Linux
```

### Known Safe Tests False-Positive

**Possible causes:**
1. Environmental interference
2. Unstable measurement conditions
3. Multiple testing effects

**Debug:**
1. Check preflight warnings in output
2. Run sanity check manually
3. Reduce alpha: `ci_alpha(0.001)`

### Calibration Tests Fail

**CI Gate FPR > 2α:**
- Indicates block bootstrap doesn't preserve null distribution
- May need adjustment to block size or resampling strategy

**Bayesian over-concentration:**
- Too many high probabilities on null data
- Check prior settings and sample splitting

## Platform-Specific Notes

### macOS / Apple Silicon

- Uses ARM `cntvct_el0` timer (~24 MHz)
- Lower "cycles per nanosecond" than x86 (0.024 vs 3.0)
- Results are still valid - conversion is handled automatically

### Linux

- Ensure CPU governor is set to "performance"
- Disable turbo boost for consistent results
- Consider CPU isolation for very precise measurements

### Windows

- Uses x86 `rdtsc` timer
- May have higher noise due to timer precision
- Run with administrator privileges for best results

## Benchmarks

Microbenchmarks for performance testing:

```bash
cargo bench
```

These test individual components (quantile computation, bootstrap, etc.) with small sample sizes to track performance regressions.

## Adding New Tests

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_feature() {
        // Arrange
        let input = ...;

        // Act
        let result = my_function(input);

        // Assert
        assert!(result.is_valid());
    }
}
```

### Integration Test Template

```rust
// tests/my_test.rs
use timing_oracle::TimingOracle;

#[test]
fn test_my_timing_property() {
    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || fixed_operation(),
            || random_operation(),
        );

    assert!(/* your condition */,
            "Description with value: {}", result.leak_probability);
}
```

## Test Coverage

Run with coverage report:

```bash
# Install cargo-llvm-cov
cargo install cargo-llvm-cov

# Generate coverage
cargo llvm-cov --html

# View in browser
open target/llvm-cov/html/index.html
```

## Continuous Integration Setup

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable

      - name: Run unit tests
        run: cargo test --lib

      - name: Run integration tests
        run: cargo test --test known_leaky --test known_safe -- --test-threads=1

      - name: Run calibration tests
        run: cargo test --test calibration -- --test-threads=1
        # Optional: allow calibration tests to fail (they're statistical)
        continue-on-error: true
```

## Troubleshooting Test Issues

See [README.md § Troubleshooting](README.md#troubleshooting) for detailed debugging steps for common test failure scenarios.
