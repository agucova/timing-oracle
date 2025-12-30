//! Tests that must not false-positive on constant-time code.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA
//! should differ. If fixed uses a constant while random does Vec indexing,
//! the indexing overhead creates a false timing difference.
//!
//! NOTE: Simple XOR operations can leak timing information due to CPU
//! microarchitectural effects (store buffer optimizations, cache behavior).
//! XORing zeros vs random data shows ~40ns timing difference on most platforms.

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

fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    let mut acc = 0u8;
    for i in 0..a.len().min(b.len()) {
        acc |= a[i] ^ b[i];
    }
    acc == 0 && a.len() == b.len()
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
