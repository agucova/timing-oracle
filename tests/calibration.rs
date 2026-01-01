//! Calibration tests to verify statistical properties.
//!
//! These tests use `TimingOracle::quick()` with a cached timer for fast execution.
//!
//! IMPORTANT: With the new API, inputs are pre-generated before measurement,
//! ensuring identical code paths for fixed and random inputs.

use timing_oracle::helpers::InputPair;
use timing_oracle::{Outcome, TimingOracle};

const SAMPLES: usize = 5_000;

/// Verify CI gate false positive rate is bounded.
///
/// Run many trials on pure noise data and check rejection rate <= 2*alpha.
#[test]
fn ci_gate_fpr_calibration() {
    const TRIALS: usize = 100;
    const ALPHA: f64 = 0.01;

    eprintln!("\n[ci_gate_fpr] Starting {} trials (alpha={})", TRIALS, ALPHA);

    let mut rejections = 0;

    for trial in 0..TRIALS {
        // Pre-generate inputs using InputPair
        // Pure noise: both classes are random data
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .samples(SAMPLES)
            .ci_alpha(ALPHA)
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if !result.ci_gate.passed {
                rejections += 1;
            }

            let rate = rejections as f64 / (trial + 1) as f64;
            eprintln!(
                "[ci_gate_fpr] Trial {}/{}: {} rejections (rate={:.1}%)",
                trial + 1,
                TRIALS,
                rejections,
                rate * 100.0
            );
        }
    }

    let rejection_rate = rejections as f64 / TRIALS as f64;
    eprintln!(
        "[ci_gate_fpr] Complete: {} rejections, rate={:.1}% (limit={:.1}%)",
        rejections,
        rejection_rate * 100.0,
        2.0 * ALPHA * 100.0
    );
    assert!(
        rejection_rate <= 2.0 * ALPHA,
        "FPR {} exceeds 2*alpha={}",
        rejection_rate,
        2.0 * ALPHA
    );
}

/// Verify Bayesian layer doesn't over-concentrate on high probabilities for null data.
#[test]
fn bayesian_calibration() {
    const TRIALS: usize = 100;

    eprintln!("\n[bayesian] Starting {} trials", TRIALS);

    let mut high_prob_count = 0;

    for trial in 0..TRIALS {
        // Pre-generate inputs using InputPair
        let inputs = InputPair::new(rand_bytes, rand_bytes);

        let outcome = TimingOracle::quick()
            .test(inputs, |data| {
                std::hint::black_box(data);
            });

        if let Outcome::Completed(result) = outcome {
            if result.leak_probability > 0.9 {
                high_prob_count += 1;
            }

            let rate = high_prob_count as f64 / (trial + 1) as f64;
            eprintln!(
                "[bayesian] Trial {}/{}: {} high-prob (rate={:.1}%)",
                trial + 1,
                TRIALS,
                high_prob_count,
                rate * 100.0
            );
        }
    }

    let high_prob_rate = high_prob_count as f64 / TRIALS as f64;
    eprintln!(
        "[bayesian] Complete: {} high-prob, rate={:.1}% (limit=10%)",
        high_prob_count,
        high_prob_rate * 100.0
    );
    assert!(
        high_prob_rate < 0.1,
        "Too many high probabilities on null data: {}",
        high_prob_rate
    );
}

fn rand_bytes() -> [u8; 32] {
    let mut arr = [0u8; 32];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
