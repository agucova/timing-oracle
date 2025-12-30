//! Tests that must not false-positive on constant-time code.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA
//! should differ. If fixed uses a constant while random does Vec indexing,
//! the indexing overhead creates a false timing difference.

use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

/// Test that XOR-based comparison is not flagged.
///
/// Both closures use identical code paths - only the data differs.
#[test]
fn no_false_positive_xor_compare() {
    let secret = [0xABu8; 32];

    // Pre-generate inputs using InputPair
    const SAMPLES: usize = 100_000;
    let inputs = InputPair::with_samples(SAMPLES, [0xABu8; 32], rand_bytes);

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || constant_time_compare(&secret, inputs.fixed()),
            || constant_time_compare(&secret, inputs.random()),
        );

    assert!(
        result.leak_probability < 0.5,
        "Should not detect leak in constant-time code, got {}",
        result.leak_probability
    );
    assert!(
        result.ci_gate.passed,
        "CI gate should pass for constant-time code"
    );
}

/// Test that simple XOR operation is not flagged.
///
/// Both closures use identical code paths - only the data differs.
#[test]
fn no_false_positive_xor() {
    // Pre-generate inputs using InputPair - pairs of arrays
    const SAMPLES: usize = 100_000;
    let inputs = InputPair::from_fn_with_samples(
        SAMPLES,
        || ([0u8; 32], [0u8; 32]),
        || (rand_bytes(), rand_bytes()),
    );

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                let (a, b) = inputs.fixed();
                xor_bytes(a, b)
            },
            || {
                let (a, b) = inputs.random();
                xor_bytes(a, b)
            },
        );

    assert!(
        result.ci_gate.passed,
        "CI gate should pass for XOR operation"
    );
}

fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len().min(b.len()) {
        acc |= a[i] ^ b[i];
    }
    acc == 0 && a.len() == b.len()
}

fn xor_bytes(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
