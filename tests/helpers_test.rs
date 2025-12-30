//! Tests for timing_oracle::helpers utilities

use timing_oracle::helpers::{InputPair, byte_arrays_32, byte_vecs};
use timing_oracle::TimingOracle;

#[test]
fn test_input_pair_basic() {
    let inputs = InputPair::new(42u64, || rand::random());

    // Should cycle through
    for _ in 0..100 {
        assert_eq!(*inputs.fixed(), 42);
        let _ = inputs.random(); // Random values vary
    }
}

#[test]
fn test_input_pair_indexing() {
    let inputs = InputPair::with_samples(10, 0u32, || 1u32);

    // Both should advance the same index
    assert_eq!(*inputs.fixed(), 0);
    assert_eq!(*inputs.random(), 1);
    assert_eq!(*inputs.fixed(), 0);
}

#[test]
fn test_byte_arrays_helper() {
    let inputs = byte_arrays_32();

    // Fixed should be all zeros
    assert_eq!(*inputs.fixed(), [0u8; 32]);

    // Random should vary
    let r1 = *inputs.random();
    let r2 = *inputs.random();
    assert_ne!(r1, r2); // Very unlikely to be equal
}

#[test]
fn test_byte_vecs_helper() {
    let inputs = byte_vecs(128);

    // Fixed should be all zeros
    assert_eq!(*inputs.fixed(), vec![0u8; 128]);

    // Random should vary and have correct length
    let r = inputs.random();
    assert_eq!(r.len(), 128);
}

#[test]
fn test_no_false_positive_with_helpers() {
    // Use helpers for constant-time XOR - should not false positive
    let inputs_a = byte_arrays_32();
    let inputs_b = byte_arrays_32();

    let result = TimingOracle::new()
        .samples(50_000)
        .test(
            || xor_bytes(inputs_a.fixed(), inputs_b.fixed()),
            || xor_bytes(inputs_a.random(), inputs_b.random()),
        );

    assert!(result.ci_gate.passed, "Should not detect leak with helpers");
    assert!(result.leak_probability < 0.5);
}

#[test]
fn test_from_fn_generators() {
    // Test from_fn with separate generators for both sides
    let counter = std::sync::atomic::AtomicU32::new(0);
    let inputs = InputPair::from_fn_with_samples(
        100,
        || 0u32,  // Fixed always returns 0
        || counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1000,  // Random returns 1000, 1001, 1002, ...
    );

    // Fixed always 0
    for _ in 0..10 {
        assert_eq!(*inputs.fixed(), 0);
    }

    // Random values are sequential from 1000
    inputs.reset();
    assert!(*inputs.random() >= 1000);
}

#[test]
fn test_reset_functionality() {
    let inputs = InputPair::with_samples(5, 1u64, || 2u64);

    // Advance the index
    for _ in 0..10 {
        let _ = inputs.fixed();
    }

    // Reset should go back to start
    inputs.reset();
    assert_eq!(*inputs.fixed(), 1);
    assert_eq!(*inputs.random(), 2);
}

fn xor_bytes(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for i in 0..32 {
        result[i] = a[i] ^ b[i];
    }
    result
}
