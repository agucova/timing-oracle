//! Layer 1: RTLF-style CI gate with bounded false positive rate.
//!
//! This implements a frequentist screening layer that provides:
//! - Pass/fail decision suitable for CI pipelines
//! - Guaranteed false positive rate via Bonferroni correction
//! - Block-bootstrap-based threshold computation within each class
//!
//! The approach follows the methodology from:
//! Dunsche et al. (RTLF) for constant-time validation.

use rand::SeedableRng;

use crate::result::CiGate;
use crate::statistics::{block_bootstrap_resample, compute_block_size, compute_deciles};
use crate::types::Vector9;

/// Input data for CI gate analysis.
#[derive(Debug, Clone)]
pub struct CiGateInput<'a> {
    /// Observed quantile differences (fixed - random) for 9 deciles.
    pub observed_diff: Vector9,
    /// Fixed-class samples (nanoseconds).
    pub fixed_samples: &'a [f64],
    /// Random-class samples (nanoseconds).
    pub random_samples: &'a [f64],
    /// Significance level (e.g., 0.05 for 5% false positive rate).
    pub alpha: f64,
    /// Number of bootstrap iterations for thresholds.
    pub bootstrap_iterations: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
}

/// Run the CI gate analysis.
///
/// This implements RTLF-style testing with:
/// 1. Bootstrap within each class separately to estimate null distribution
/// 2. Compute per-class thresholds at corrected alpha level
/// 3. Take maximum threshold across classes for each quantile
/// 4. Apply Bonferroni correction (alpha/9) for multiple testing
pub fn run_ci_gate(input: &CiGateInput<'_>) -> CiGate {
    // Bonferroni correction for 9 simultaneous tests
    let corrected_alpha = input.alpha / 9.0;

    // Compute thresholds via block bootstrap within each class
    let thresholds = compute_bootstrap_thresholds(
        input.fixed_samples,
        input.random_samples,
        corrected_alpha,
        input.bootstrap_iterations,
        input.seed,
    );

    // Convert observed differences to array
    let observed: [f64; 9] = input.observed_diff.as_slice().try_into().unwrap_or([0.0; 9]);

    // Convert thresholds to array
    let thresholds_arr: [f64; 9] = thresholds.as_slice().try_into().unwrap_or([0.0; 9]);

    // Check if any observed difference exceeds its threshold
    let passed = observed
        .iter()
        .zip(thresholds_arr.iter())
        .all(|(obs, thresh)| obs.abs() <= *thresh);

    CiGate {
        alpha: input.alpha,
        passed,
        thresholds: thresholds_arr,
        observed,
    }
}

fn compute_bootstrap_thresholds(
    fixed_samples: &[f64],
    random_samples: &[f64],
    alpha: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> Vector9 {
    let mut rng = match seed {
        Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
        None => rand::rngs::StdRng::seed_from_u64(42),
    };

    let n = fixed_samples.len().min(random_samples.len());
    let block_size = compute_block_size(n);

    let mut fixed_diffs: Vec<Vector9> = Vec::with_capacity(n_bootstrap);
    let mut random_diffs: Vec<Vector9> = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let f1 = block_bootstrap_resample(fixed_samples, block_size, &mut rng);
        let f2 = block_bootstrap_resample(fixed_samples, block_size, &mut rng);
        fixed_diffs.push(compute_deciles(&f1) - compute_deciles(&f2));

        let r1 = block_bootstrap_resample(random_samples, block_size, &mut rng);
        let r2 = block_bootstrap_resample(random_samples, block_size, &mut rng);
        random_diffs.push(compute_deciles(&r1) - compute_deciles(&r2));
    }

    let threshold_quantile = 1.0 - alpha;
    let idx = ((n_bootstrap as f64) * threshold_quantile).ceil() as usize;
    let idx = idx.saturating_sub(1).min(n_bootstrap.saturating_sub(1));

    let mut thresholds = Vector9::zeros();
    for i in 0..9 {
        let mut abs_fixed: Vec<f64> = fixed_diffs.iter().map(|s| s[i].abs()).collect();
        abs_fixed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut abs_random: Vec<f64> = random_diffs.iter().map(|s| s[i].abs()).collect();
        abs_random.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let thresh_fixed = abs_fixed[idx];
        let thresh_random = abs_random[idx];
        thresholds[i] = thresh_fixed.max(thresh_random);
    }

    thresholds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_gate_passes_on_identical_samples() {
        let fixed: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let random = fixed.clone();
        let observed = compute_deciles(&fixed) - compute_deciles(&random);

        let input = CiGateInput {
            observed_diff: observed,
            fixed_samples: &fixed,
            random_samples: &random,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
        };

        let result = run_ci_gate(&input);
        assert!(result.passed, "CI gate should pass when no difference observed");
    }

    #[test]
    fn test_ci_gate_fails_on_large_difference() {
        let fixed: Vec<f64> = (0..2000).map(|x| x as f64 + 1000.0).collect();
        let random: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let observed = compute_deciles(&fixed) - compute_deciles(&random);

        let input = CiGateInput {
            observed_diff: observed,
            fixed_samples: &fixed,
            random_samples: &random,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
        };

        let result = run_ci_gate(&input);
        assert!(!result.passed, "CI gate should fail on large difference");
    }
}
