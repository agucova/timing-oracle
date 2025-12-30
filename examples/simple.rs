//! Simple example demonstrating basic timing-oracle usage.
//!
//! This demonstrates the CORRECT way to test operations:
//! - Pre-generate all inputs before measurement
//! - Both closures execute identical code paths
//! - Only the input data differs

use timing_oracle::{test, helpers::InputPair, TimingOracle};

fn main() {
    println!("timing-oracle simple example\n");

    // Example: Testing a potentially leaky comparison
    let secret = [0u8; 32];

    // ✅ CORRECT: Pre-generate inputs outside closures
    // Both fixed and random generators are called BEFORE measurement
    let inputs = InputPair::new(
        [0u8; 32],  // Fixed: all zeros (same as secret)
        || {
            let mut arr = [0u8; 32];
            for i in 0..32 {
                arr[i] = rand::random();
            }
            arr
        },
    );

    // Simple API with default config
    let result = test(
        || compare_bytes(&secret, inputs.fixed()),
        || compare_bytes(&secret, inputs.random()),
    );

    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
    println!("CI gate passed: {}", result.ci_gate.passed);
    println!("Quality: {:?}", result.quality);

    // Builder API with custom config
    let result = TimingOracle::new()
        .samples(10_000) // Fewer samples for quick demo
        .ci_alpha(0.01)
        .test(
            || compare_bytes(&secret, inputs.fixed()),
            || compare_bytes(&secret, inputs.random()),
        );

    println!("\nWith custom config:");
    println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
}

/// Non-constant-time comparison (intentionally leaky for demo).
fn compare_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false; // Early exit - timing leak!
        }
    }
    true
}
