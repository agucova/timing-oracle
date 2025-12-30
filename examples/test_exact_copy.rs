//! Tests that must not false-positive on constant-time code.
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

fn main() {
    test_no_false_positive_xor_compare();
}

fn test_no_false_positive_xor_compare() {
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

    println!("Leak probability: {}", result.leak_probability);
    println!("CI gate passed: {}", result.ci_gate.passed);
    
    if result.leak_probability > 0.5 {
        println!("ERROR: False positive! leak_probability={}", result.leak_probability);
    } else {
        println!("OK: No false positive detected");
    }
    if !result.ci_gate.passed {
        println!("ERROR: CI gate failed!");
    } else {
        println!("OK: CI gate passed");
    }
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
