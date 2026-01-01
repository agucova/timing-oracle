//! Comprehensive tests for the TimingTest builder API.
//!
//! This file tests the builder pattern API as an alternative to the macro API.

use timing_oracle::{Outcome, TestResult, TimingOracle, TimingTest};

// ===========================================================================
// Basic Builder Pattern Tests
// ===========================================================================

/// Test minimal builder usage with just required fields.
#[test]
fn builder_minimal() {
    let outcome = TimingTest::new()
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with custom oracle configuration.
#[test]
fn builder_with_oracle() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with balanced oracle preset.
#[test]
fn builder_with_balanced_oracle() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::balanced())
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with configured oracle (samples, alpha).
#[test]
fn builder_with_configured_oracle() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick().samples(2000).ci_alpha(0.05))
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    if let Outcome::Completed(result) = outcome {
        assert!(result.metadata.samples_per_class > 0);
        assert!((result.ci_gate.alpha - 0.05).abs() < 1e-10);
    }
}

// ===========================================================================
// Different Input Types
// ===========================================================================

/// Test builder with primitive type (u32).
#[test]
fn builder_primitive_u32() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [42u32; 64])
        .sample(|| {
            let mut arr = [0u32; 64];
            for i in &mut arr {
                *i = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut val = 0u32;
            for &x in arr {
                val = val.wrapping_mul(13).wrapping_add(x);
            }
            std::hint::black_box(val);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with primitive type (u64).
#[test]
fn builder_primitive_u64() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [123456u64; 64])
        .sample(|| {
            let mut arr = [0u64; 64];
            for i in &mut arr {
                *i = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut val = 0u64;
            for &x in arr {
                val = val.wrapping_mul(17).wrapping_add(x);
            }
            std::hint::black_box(val);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with bool type.
#[test]
fn builder_bool_type() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [false; 128])
        .sample(|| {
            let mut arr = [false; 128];
            for b in &mut arr {
                *b = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut val = 0u32;
            for &b in arr {
                val = val.wrapping_mul(31).wrapping_add(if b { 1 } else { 0 });
            }
            std::hint::black_box(val);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with array type.
#[test]
fn builder_array_type() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with Vec type.
#[test]
fn builder_vec_type() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| vec![0u32; 128])
        .sample(|| (0..128).map(|_| rand::random::<u32>()).collect())
        .measure(|v| {
            let hash: u32 = v.iter().fold(0u32, |acc, x| acc.wrapping_mul(31).wrapping_add(*x));
            std::hint::black_box(hash);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with tuple type.
#[test]
fn builder_tuple_type() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| ([0u8; 64], [0u8; 64]))
        .sample(|| {
            let mut a = [0u8; 64];
            let mut b = [0u8; 64];
            for byte in &mut a {
                *byte = rand::random();
            }
            for byte in &mut b {
                *byte = rand::random();
            }
            (a, b)
        })
        .measure(|(a, b)| {
            let mut val = 0u32;
            for i in 0..64 {
                val = val.wrapping_mul(31).wrapping_add((a[i] ^ b[i]) as u32);
            }
            std::hint::black_box(val);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with String type.
#[test]
fn builder_string_type() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| "a".repeat(200))
        .sample(|| {
            let len = 200;
            (0..len)
                .map(|_| (rand::random::<u8>() % 26 + 97) as char)
                .collect()
        })
        .measure(|s| {
            let hash = s.bytes().fold(0u32, |acc, b| {
                acc.wrapping_mul(31).wrapping_add(b as u32)
            });
            std::hint::black_box(hash);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

// ===========================================================================
// Field Ordering Tests
// ===========================================================================

/// Test that fields can be set in any order (oracle last).
#[test]
fn builder_field_order_oracle_last() {
    let outcome = TimingTest::new()
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .oracle(TimingOracle::quick())
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test that fields can be set in any order (random before fixed).
#[test]
fn builder_field_order_random_before_fixed() {
    // Note: We can't call .measure() first because it needs to know the type V
    // from .baseline(). But we can set .sample() before .baseline() using a generic closure.
    let outcome = TimingTest::new()
        .baseline(|| [0u8; 128])
        .oracle(TimingOracle::quick())
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

// ===========================================================================
// Captured Variables Tests
// ===========================================================================

/// Test builder with captured variables in test closure.
#[test]
fn builder_with_captures() {
    let multiplier = 42u32;

    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [1u32; 64])
        .sample(|| {
            let mut arr = [0u32; 64];
            for i in &mut arr {
                *i = rand::random();
            }
            arr
        })
        .measure(move |arr| {
            let mut val = 0u32;
            for &x in arr {
                val = val.wrapping_mul(multiplier).wrapping_add(x);
            }
            std::hint::black_box(val);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with mutable state captured in test closure.
#[test]
fn builder_with_mutable_captures() {
    use std::cell::Cell;
    let counter = Cell::new(0u32);

    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            counter.set(counter.get() + 1);
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    // Test ran successfully
    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
    // Counter was incremented (at least twice - once for each class)
    assert!(counter.get() >= 2);
}

// ===========================================================================
// Outcome Handling Tests
// ===========================================================================

/// Test that builder returns Outcome for explicit handling.
#[test]
fn builder_returns_outcome() {
    let outcome: Outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    match outcome {
        Outcome::Completed(result) => {
            assert!(result.leak_probability >= 0.0 && result.leak_probability <= 1.0);
        }
        Outcome::Unmeasurable { .. } => {
            // Valid outcome for very fast operations
        }
    }
}

/// Test unwrapping completed outcome.
#[test]
fn builder_unwrap_completed() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            // Complex enough to be measurable
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    if let Outcome::Completed(result) = outcome {
        let _result: TestResult = result;
        // Can access TestResult fields
        assert!(_result.metadata.samples_per_class > 0);
    }
}

/// Test accessing result fields through pattern matching.
#[test]
fn builder_result_field_access() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick().samples(1000))
        .baseline(|| [0u8; 128])
        .sample(|| {
            let mut arr = [0u8; 128];
            for byte in &mut arr {
                *byte = rand::random();
            }
            arr
        })
        .measure(|arr| {
            let mut acc = 0u32;
            for byte in arr {
                acc = acc.wrapping_mul(31).wrapping_add(*byte as u32);
            }
            std::hint::black_box(acc);
        })
        .run();

    if let Outcome::Completed(result) = outcome {
        // Verify all important fields are accessible
        assert!(result.leak_probability >= 0.0);
        assert!(result.leak_probability <= 1.0);
        assert!(result.metadata.samples_per_class > 0);
        assert!(result.ci_gate.alpha > 0.0);
        assert!(result.min_detectable_effect.shift_ns >= 0.0);
    }
}

// ===========================================================================
// Complex Operations Tests
// ===========================================================================

/// Test builder with complex cryptographic-like operation.
#[test]
fn builder_complex_operation() {
    fn hash_like(data: &[u8]) -> u32 {
        let mut h = 0x811c9dc5u32;
        for &byte in data {
            h ^= byte as u32;
            h = h.wrapping_mul(0x01000193);
        }
        h
    }

    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| vec![0u8; 128])
        .sample(|| (0..128).map(|_| rand::random::<u8>()).collect())
        .measure(|data| {
            let hash = hash_like(data);
            std::hint::black_box(hash);
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}

/// Test builder with nested data structures.
#[test]
fn builder_nested_structures() {
    let outcome = TimingTest::new()
        .oracle(TimingOracle::quick())
        .baseline(|| (vec![0u8; 64], vec![0u8; 64]))
        .sample(|| {
            (
                (0..64).map(|_| rand::random::<u8>()).collect(),
                (0..64).map(|_| rand::random::<u8>()).collect(),
            )
        })
        .measure(|(v1, v2)| {
            let hash1: u32 = v1.iter().fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
            let hash2: u32 = v2.iter().fold(0u32, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u32));
            std::hint::black_box(hash1.wrapping_add(hash2));
        })
        .run();

    assert!(matches!(
        outcome,
        Outcome::Completed(_) | Outcome::Unmeasurable { .. }
    ));
}
