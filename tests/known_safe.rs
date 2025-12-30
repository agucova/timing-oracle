//! Tests that must not false-positive on constant-time code.
//!
//! IMPORTANT: Both closures must execute IDENTICAL code paths - only the DATA
//! should differ. If fixed uses a constant while random does Vec indexing,
//! the indexing overhead creates a false timing difference.

use std::cell::Cell;
use timing_oracle::TimingOracle;

/// Test that XOR-based comparison is not flagged.
///
/// Both closures use identical code paths (Vec indexing + Cell ops).
/// Only the underlying data differs.
#[test]
#[ignore = "requires full implementation"]
fn no_false_positive_xor_compare() {
    let secret = [0xABu8; 32];

    // Pre-generate inputs for BOTH classes - same code path, different data
    let fixed_inputs: Vec<[u8; 32]> = vec![[0xABu8; 32]; 100_000]; // All match secret
    let random_inputs: Vec<[u8; 32]> = (0..100_000).map(|_| rand_bytes()).collect();

    // Shared index counter - both closures do identical Cell operations
    let idx = Cell::new(0usize);

    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                constant_time_compare(&secret, &fixed_inputs[i % fixed_inputs.len()])
            },
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                constant_time_compare(&secret, &random_inputs[i % random_inputs.len()])
            },
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
/// Both closures use identical code paths (Vec indexing + Cell ops).
#[test]
#[ignore = "requires full implementation"]
fn no_false_positive_xor() {
    // Pre-generate inputs for BOTH classes
    let fixed_a: Vec<[u8; 32]> = vec![[0u8; 32]; 100_000];
    let fixed_b: Vec<[u8; 32]> = vec![[0u8; 32]; 100_000];
    let random_a: Vec<[u8; 32]> = (0..100_000).map(|_| rand_bytes()).collect();
    let random_b: Vec<[u8; 32]> = (0..100_000).map(|_| rand_bytes()).collect();

    let idx = Cell::new(0usize);

    let result = TimingOracle::new()
        .samples(100_000)
        .test(
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                xor_bytes(&fixed_a[i % fixed_a.len()], &fixed_b[i % fixed_b.len()])
            },
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                xor_bytes(&random_a[i % random_a.len()], &random_b[i % random_b.len()])
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
