//! Layer 1: RTLF-style CI gate with controlled false positive rate.
//!
//! This implements a frequentist screening layer that provides:
//! - Pass/fail decision suitable for CI pipelines
//! - Controlled false positive rate via max-bootstrap thresholds
//! - Block-bootstrap-based threshold computation within each class
//!
//! The approach follows the methodology from:
//! Dunsche et al. (RTLF) for constant-time validation.
//!
//! ## Max-Bootstrap Method
//!
//! We use the maximum absolute quantile difference as our test statistic:
//! M = max_p |Δ_p|
//!
//! This is more powerful than per-quantile thresholds with Bonferroni/Šidák
//! correction because it naturally captures the correlation structure of the
//! quantile differences through bootstrap resampling.

use rand::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::result::CiGate;
use crate::statistics::{compute_deciles_inplace, counter_rng_seed};
use crate::types::Vector9;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Input data for CI gate analysis.
#[derive(Debug, Clone)]
pub struct CiGateInput<'a> {
    /// Observed quantile differences (baseline - sample) for 9 deciles.
    pub observed_diff: Vector9,
    /// Baseline-class samples (nanoseconds).
    pub baseline_samples: &'a [f64],
    /// Sample-class samples (nanoseconds).
    pub sample_samples: &'a [f64],
    /// Significance level (e.g., 0.05 for 5% false positive rate).
    pub alpha: f64,
    /// Number of bootstrap iterations for thresholds.
    pub bootstrap_iterations: usize,
    /// Optional seed for reproducibility.
    pub seed: Option<u64>,
    /// Timer resolution in nanoseconds (for quantization to integer ticks).
    pub timer_resolution_ns: f64,
    /// Minimum effect of concern in nanoseconds (threshold floor for practical significance).
    pub min_effect_of_concern: f64,
}

/// Run the CI gate analysis.
///
/// This implements RTLF-style testing with max-bootstrap:
/// 1. Pool all samples and bootstrap to estimate null distribution of max|Δ_p|
/// 2. Compute single threshold at (1-α) quantile of max statistic distribution
/// 3. Compare observed max|Δ_p| against threshold
///
/// # Max-Bootstrap Method
///
/// We test using M = max_p |Δ_p| rather than testing each quantile separately.
/// This captures the correlation structure naturally and provides more power
/// than Bonferroni/Šidák corrections.
///
/// # Timer Quantization Handling
///
/// Since timing measurements are discrete (quantized to timer ticks), this
/// implementation uses integer tick comparisons rather than floating-point
/// nanosecond comparisons:
///
/// - Threshold is floored to ≥1 timer tick to prevent false positives from
///   quantization noise. When bootstrap finds zero variance (all tied values),
///   a zero threshold would reject any non-zero difference, even if that
///   difference is below timer resolution and thus unmeasurable.
///
/// - Both observed max and threshold are rounded to integer ticks before
///   comparison to eliminate floating-point precision artifacts.
///
/// This makes the test more conservative (slightly higher threshold) but
/// ensures it respects the fundamental constraint that differences below
/// one timer tick are indistinguishable from quantization noise.
pub fn run_ci_gate(input: &CiGateInput<'_>) -> CiGate {
    // Compute threshold via max-bootstrap (single threshold, not per-quantile)
    let threshold = compute_max_bootstrap_threshold(
        input.baseline_samples,
        input.sample_samples,
        input.alpha,
        input.bootstrap_iterations,
        input.seed,
    );

    // Convert observed differences to array
    let observed: [f64; 9] = input.observed_diff.as_slice().try_into().unwrap_or([0.0; 9]);

    // Compute observed max statistic: M = max_p |Δ_p|
    let max_observed = observed.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);

    // Floor threshold to at least 1 timer tick and min_effect_of_concern
    // This ensures: (1) we don't reject on sub-tick noise, and (2) we don't
    // reject on effects below the practically significant threshold
    let timer_res = input.timer_resolution_ns;
    let threshold = threshold.max(timer_res).max(input.min_effect_of_concern);

    // Convert to integer ticks for comparison
    // This respects the discrete nature of timing measurements: differences
    // smaller than one timer tick are indistinguishable from quantization noise,
    // so we work in the natural units (ticks) rather than derived units (ns).
    let obs_ticks = (max_observed / timer_res).round() as u64;
    let thresh_ticks = (threshold / timer_res).round() as u64;

    // Integer comparison eliminates floating-point precision artifacts
    // and aligns with the physical reality of discrete timer readings
    let passed = obs_ticks <= thresh_ticks;

    CiGate {
        alpha: input.alpha,
        passed,
        threshold,
        max_observed,
        observed,
    }
}

/// Compute the bootstrap threshold for the max statistic M = max_p |Δ_p| (spec §2.4).
///
/// Under the null hypothesis (no leak), we pool ALL samples from both classes and
/// resample to estimate the distribution of M. This simulates "what if there's no
/// difference?" by treating both classes as coming from the same distribution.
///
/// Algorithm:
/// 1. Pool all samples (both classes together)
/// 2. For each bootstrap iteration:
///    - Resample n values with replacement for "pseudo-fixed"
///    - Resample n values with replacement for "pseudo-random"
///    - Compute quantile difference vector Δ*
///    - Record M* = max|Δ*|
/// 3. Return (1-α) quantile of M* distribution
fn compute_max_bootstrap_threshold(
    baseline_samples: &[f64],
    sample_samples: &[f64],
    alpha: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> f64 {
    let base_seed = seed.unwrap_or(42);
    let n = baseline_samples.len().min(sample_samples.len());

    // Pool all samples under the null hypothesis (no difference between classes)
    let pooled: Vec<f64> = baseline_samples
        .iter()
        .chain(sample_samples.iter())
        .copied()
        .collect();

    // Parallel bootstrap: compute max|Δ| for each bootstrap iteration
    #[cfg(feature = "parallel")]
    let max_stats: Vec<f64> = crate::thread_pool::install(|| {
        let mut out = vec![0.0_f64; n_bootstrap];

        out.par_iter_mut()
            .enumerate()
            .map_init(
                || {
                    // Per-thread initialization: RNG and scratch buffers
                    (
                        Xoshiro256PlusPlus::seed_from_u64(base_seed),
                        vec![0.0; n],  // pseudo-fixed
                        vec![0.0; n],  // pseudo-random
                    )
                },
                |(rng, pseudo_fixed, pseudo_random), (i, out)| {
                    // Use counter-based RNG for deterministic, well-distributed seeding
                    *rng = Xoshiro256PlusPlus::seed_from_u64(counter_rng_seed(base_seed, i as u64));

                    // Resample from POOLED distribution to create pseudo-classes
                    // Simple random sampling (with replacement) from the pooled null distribution
                    for val in pseudo_fixed.iter_mut() {
                        *val = *pooled.choose(rng).unwrap();
                    }
                    for val in pseudo_random.iter_mut() {
                        *val = *pooled.choose(rng).unwrap();
                    }

                    // Compute quantile difference between pseudo-classes
                    let delta = compute_deciles_inplace(pseudo_fixed) - compute_deciles_inplace(pseudo_random);

                    // Record max|Δ| for this iteration
                    *out = delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
                },
            )
            .count(); // Force execution

        out
    });

    #[cfg(not(feature = "parallel"))]
    let max_stats: Vec<f64> = {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed);
        let mut stats = Vec::with_capacity(n_bootstrap);

        // Reusable buffers
        let mut pseudo_fixed = vec![0.0; n];
        let mut pseudo_random = vec![0.0; n];

        for _ in 0..n_bootstrap {
            // Resample from POOLED distribution to create pseudo-classes
            for val in pseudo_fixed.iter_mut() {
                *val = *pooled.choose(&mut rng).unwrap();
            }
            for val in pseudo_random.iter_mut() {
                *val = *pooled.choose(&mut rng).unwrap();
            }

            // Compute quantile difference between pseudo-classes
            let delta = compute_deciles_inplace(&mut pseudo_fixed) - compute_deciles_inplace(&mut pseudo_random);

            // Record max|Δ|
            stats.push(delta.iter().map(|x| x.abs()).fold(0.0_f64, f64::max));
        }

        stats
    };

    // Sort and find the (1-α) quantile
    let mut sorted_stats = max_stats;
    sorted_stats.sort_by(|a, b| a.total_cmp(b));

    let threshold_quantile = 1.0 - alpha;
    let idx = ((n_bootstrap as f64) * threshold_quantile).ceil() as usize;
    let idx = idx.saturating_sub(1).min(n_bootstrap.saturating_sub(1));

    sorted_stats[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::compute_deciles;

    #[test]
    fn test_ci_gate_passes_on_identical_samples() {
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let sample = baseline.clone();
        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0, // 1ns resolution for test
            min_effect_of_concern: 0.0,
        };

        let result = run_ci_gate(&input);
        assert!(result.passed, "CI gate should pass when no difference observed");
    }

    #[test]
    fn test_ci_gate_fails_on_large_difference() {
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64 + 1000.0).collect();
        let sample: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0, // 1ns resolution for test
            min_effect_of_concern: 0.0,
        };

        let result = run_ci_gate(&input);
        assert!(!result.passed, "CI gate should fail on large difference");
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_ci_gate_max_bootstrap_determinism() {
        // Test that parallel implementation produces identical results
        // across multiple runs with the same seed
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let sample: Vec<f64> = (0..2000).map(|x| x as f64 + 0.5).collect();

        let result1 = compute_max_bootstrap_threshold(&baseline, &sample, 0.01, 1000, Some(42));
        let result2 = compute_max_bootstrap_threshold(&baseline, &sample, 0.01, 1000, Some(42));
        let result3 = compute_max_bootstrap_threshold(&baseline, &sample, 0.01, 1000, Some(42));

        // Verify threshold is identical across runs
        assert!(
            (result1 - result2).abs() < 1e-12,
            "Run 1 vs 2 differ: {} vs {}",
            result1,
            result2
        );
        assert!(
            (result2 - result3).abs() < 1e-12,
            "Run 2 vs 3 differ: {} vs {}",
            result2,
            result3
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_ci_gate_max_bootstrap_edge_cases() {
        // Test edge cases: n_bootstrap < typical chunk size, exact multiples, remainders
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();

        // n_bootstrap < chunk_size (should still work)
        let r1 = compute_max_bootstrap_threshold(&data, &data, 0.01, 5, Some(42));
        assert!(r1 >= 0.0, "Threshold should be non-negative");

        // n_bootstrap = chunk_size (one chunk per core)
        let r2 = compute_max_bootstrap_threshold(&data, &data, 0.01, 50, Some(42));
        assert!(r2 >= 0.0, "Threshold should be non-negative");

        // n_bootstrap % chunk_size != 0 (last chunk is partial)
        let r3 = compute_max_bootstrap_threshold(&data, &data, 0.01, 150, Some(42));
        assert!(r3 >= 0.0, "Threshold should be non-negative");

        // All should be deterministic
        let r3_repeat = compute_max_bootstrap_threshold(&data, &data, 0.01, 150, Some(42));
        assert!(
            (r3 - r3_repeat).abs() < 1e-12,
            "Edge case should be deterministic: {} vs {}",
            r3,
            r3_repeat
        );
    }

    #[test]
    fn test_ci_gate_new_fields() {
        // Test that the new CiGate struct has correct fields
        let baseline: Vec<f64> = (0..2000).map(|x| x as f64).collect();
        let sample = baseline.clone();
        let observed = compute_deciles(&baseline) - compute_deciles(&sample);

        let input = CiGateInput {
            observed_diff: observed,
            baseline_samples: &baseline,
            sample_samples: &sample,
            alpha: 0.05,
            bootstrap_iterations: 200,
            seed: Some(123),
            timer_resolution_ns: 1.0,
            min_effect_of_concern: 0.0,
        };

        let result = run_ci_gate(&input);

        // Check new fields exist and are sensible
        assert!(result.threshold >= 0.0, "threshold should be non-negative");
        assert!(result.max_observed >= 0.0, "max_observed should be non-negative");
        assert_eq!(result.observed.len(), 9, "observed should have 9 elements");
    }
}
