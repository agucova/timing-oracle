//! Fixed-vs-Fixed sanity check.
//!
//! This check splits the fixed samples in half and runs the analysis
//! between the two halves. If a "leak" is detected between identical
//! input classes, it indicates a broken measurement harness or
//! environmental interference.

use serde::{Deserialize, Serialize};

use crate::analysis::{compute_bayes_factor, estimate_mde, run_ci_gate, CiGateInput};
use crate::statistics::{bootstrap_covariance_matrix, compute_deciles};
use crate::types::Vector9;

/// Warning from the sanity check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SanityWarning {
    /// Fixed-vs-Fixed comparison detected a spurious "leak".
    ///
    /// This is a critical warning indicating the measurement harness
    /// is producing unreliable results.
    BrokenHarness {
        /// Leak probability from Fixed-vs-Fixed comparison.
        leak_probability: f64,
    },

    /// Insufficient samples to perform sanity check.
    InsufficientSamples {
        /// Number of samples available.
        available: usize,
        /// Minimum required for the check.
        required: usize,
    },
}

impl SanityWarning {
    /// Check if this warning indicates a critical issue.
    pub fn is_critical(&self) -> bool {
        matches!(self, SanityWarning::BrokenHarness { .. })
    }

    /// Get a human-readable description of the warning.
    pub fn description(&self) -> String {
        match self {
            SanityWarning::BrokenHarness { leak_probability } => {
                format!(
                    "CRITICAL: Fixed-vs-Fixed comparison detected spurious 'leak' \
                     (probability: {:.1}%). This indicates a broken measurement harness \
                     or significant environmental interference. Results are unreliable.",
                    leak_probability * 100.0
                )
            }
            SanityWarning::InsufficientSamples { available, required } => {
                format!(
                    "Insufficient samples for sanity check: {} available, {} required. \
                     Skipping Fixed-vs-Fixed validation.",
                    available, required
                )
            }
        }
    }
}

/// Minimum samples required to perform sanity check.
const MIN_SAMPLES_FOR_SANITY: usize = 1000;

/// Threshold for leak probability to trigger warning.
const LEAK_THRESHOLD: f64 = 0.5;

/// Default alpha for sanity check.
const SANITY_ALPHA: f64 = 0.01;

/// Default bootstrap iterations for sanity check.
const SANITY_BOOTSTRAP: usize = 1000;

/// Perform Fixed-vs-Fixed sanity check.
///
/// Splits the fixed samples in half and runs analysis between the halves.
/// If a leak is detected, returns a warning indicating a broken harness.
///
/// # Arguments
///
/// * `fixed_samples` - All timing samples from the fixed input class
///
/// # Returns
///
/// `Some(SanityWarning)` if an issue is detected, `None` otherwise.
pub fn sanity_check(fixed_samples: &[f64], timer_resolution_ns: f64) -> Option<SanityWarning> {
    // Check if we have enough samples
    if fixed_samples.len() < MIN_SAMPLES_FOR_SANITY {
        return Some(SanityWarning::InsufficientSamples {
            available: fixed_samples.len(),
            required: MIN_SAMPLES_FOR_SANITY,
        });
    }

    // Split samples in half
    let mid = fixed_samples.len() / 2;
    let first_half = &fixed_samples[..mid];
    let second_half = &fixed_samples[mid..];

    let leak_probability = compute_full_leak_check(first_half, second_half, timer_resolution_ns);

    if leak_probability > LEAK_THRESHOLD {
        Some(SanityWarning::BrokenHarness { leak_probability })
    } else {
        None
    }
}

/// Full leak check using the core quantile-based analysis pipeline.
fn compute_full_leak_check(first: &[f64], second: &[f64], timer_resolution_ns: f64) -> f64 {
    if first.is_empty() || second.is_empty() {
        return 0.0;
    }

    let q_first = compute_deciles(first);
    let q_second = compute_deciles(second);
    let delta: Vector9 = q_first - q_second;

    let cov_first = bootstrap_covariance_matrix(first, SANITY_BOOTSTRAP, 123);
    let cov_second = bootstrap_covariance_matrix(second, SANITY_BOOTSTRAP, 456);
    let sigma0 = cov_first.matrix + cov_second.matrix;

    let mde = estimate_mde(&sigma0, 200, (1e6, 1e6));
    let min_effect = 10.0;
    let prior_sigmas = (
        (2.0 * mde.shift_ns).max(min_effect),
        (2.0 * mde.tail_ns).max(min_effect),
    );

    let ci_gate_input = CiGateInput {
        observed_diff: delta,
        fixed_samples: first,
        random_samples: second,
        alpha: SANITY_ALPHA,
        bootstrap_iterations: SANITY_BOOTSTRAP,
        seed: Some(999),
        timer_resolution_ns,
    };
    let ci_gate = run_ci_gate(&ci_gate_input);

    let bayes = compute_bayes_factor(&delta, &sigma0, prior_sigmas, 0.5);
    if !ci_gate.passed {
        bayes.posterior_probability.max(LEAK_THRESHOLD)
    } else {
        bayes.posterior_probability
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insufficient_samples() {
        let samples = vec![1.0; 100];
        let result = sanity_check(&samples, 1.0);
        assert!(matches!(
            result,
            Some(SanityWarning::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_identical_samples_pass() {
        let samples: Vec<f64> = (0..2000).map(|i| 100.0 + (i % 10) as f64).collect();
        let result = sanity_check(&samples, 1.0);

        match result {
            None => {}
            Some(SanityWarning::InsufficientSamples { .. }) => {}
            Some(SanityWarning::BrokenHarness { leak_probability }) => {
                assert!(
                    leak_probability < LEAK_THRESHOLD,
                    "Identical samples should not trigger broken harness warning"
                );
            }
        }
    }

    #[test]
    fn test_warning_description() {
        let warning = SanityWarning::BrokenHarness {
            leak_probability: 0.95,
        };
        let desc = warning.description();
        assert!(desc.contains("CRITICAL"));
        assert!(desc.contains("95.0%"));
    }
}
