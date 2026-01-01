//! Layer 2: Bayesian inference for timing leak detection.
//!
//! This module computes the posterior probability of a timing leak using:
//! - Sample splitting (calibration/inference handled by the caller)
//! - Closed-form Bayes factor via multivariate normal log-pdf ratio
//! - Cholesky decomposition for numerical stability

use nalgebra::Cholesky;

use crate::constants::{B_TAIL, LOG_2PI, ONES};
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector9};

/// Result from Bayesian analysis.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Log Bayes factor: log(P(data|H1) / P(data|H0))
    pub log_bayes_factor: f64,
    /// Posterior probability of leak: P(H1|data)
    pub posterior_probability: f64,
    /// Whether the posterior probability was clamped due to numerical limits.
    pub is_clamped: bool,
    /// Null covariance used for inference.
    pub sigma0: Matrix9,
    /// Leak covariance used for inference.
    pub sigma1: Matrix9,
}

/// Compute the Bayes factor for timing leak detection.
///
/// Bayes factor compares:
/// - H0: delta ~ N(0, Sigma0)
/// - H1: delta ~ N(0, Sigma1), where Sigma1 = Sigma0 + X * Lambda0 * X^T
///
/// # Arguments
///
/// * `observed_diff` - Observed quantile differences (fixed - random)
/// * `sigma0` - Covariance of delta under H0 (from calibration)
/// * `prior_sigmas` - Prior standard deviations (shift, tail) in ns
/// * `prior_no_leak` - Prior probability of no leak
pub fn compute_bayes_factor(
    observed_diff: &Vector9,
    sigma0: &Matrix9,
    prior_sigmas: (f64, f64),
    prior_no_leak: f64,
) -> BayesResult {
    let design = build_design_matrix();
    let lambda0 = prior_covariance(prior_sigmas);
    let sigma1 = sigma0 + design * lambda0 * design.transpose();

    // Compute log BF with proper fallback handling (spec: return log BF = 0 if Cholesky fails)
    let log_bf = match (
        mvn_log_pdf_zero(observed_diff, &sigma1),
        mvn_log_pdf_zero(observed_diff, sigma0),
    ) {
        (Some(log_pdf1), Some(log_pdf0)) => log_pdf1 - log_pdf0,
        _ => {
            // If either Cholesky decomposition failed, return neutral evidence (log BF = 0)
            // Per spec: "return log BF = 0 (BF = 1, 'no evidence either way')"
            0.0
        }
    };

    let (posterior, is_clamped) = compute_posterior_probability(log_bf, prior_no_leak);

    BayesResult {
        log_bayes_factor: log_bf,
        posterior_probability: posterior,
        is_clamped,
        sigma0: *sigma0,
        sigma1,
    }
}

/// Convert log Bayes factor to posterior probability.
///
/// P(H1|data) = BF * prior_odds / (1 + BF * prior_odds)
///
/// Returns (probability, is_clamped) where is_clamped indicates if the result
/// hit numerical stability limits.
pub fn compute_posterior_probability(log_bf: f64, prior_no_leak: f64) -> (f64, bool) {
    let prior_no_leak = prior_no_leak.clamp(1e-12, 1.0 - 1e-12);
    let prior_odds = (1.0 - prior_no_leak) / prior_no_leak;

    let log_posterior_odds = log_bf + prior_odds.ln();

    if log_posterior_odds > 700.0 {
        (0.9999, true)
    } else if log_posterior_odds < -700.0 {
        (0.0001, true)
    } else {
        (1.0 / (1.0 + (-log_posterior_odds).exp()), false)
    }
}

pub(crate) fn build_design_matrix() -> Matrix9x2 {
    let mut x = Matrix9x2::zeros();
    for i in 0..9 {
        x[(i, 0)] = ONES[i];
        x[(i, 1)] = B_TAIL[i];
    }
    x
}

fn prior_covariance(prior_sigmas: (f64, f64)) -> Matrix2 {
    let (sigma_mu, sigma_tau) = prior_sigmas;
    let mut lambda0 = Matrix2::zeros();
    lambda0[(0, 0)] = sigma_mu.max(1e-12).powi(2);
    lambda0[(1, 1)] = sigma_tau.max(1e-12).powi(2);
    lambda0
}

/// Compute log pdf of MVN(0, sigma) at point x.
///
/// Returns None if Cholesky decomposition fails even after jitter.
/// Per spec: caller should return log BF = 0 (neutral evidence) on failure.
fn mvn_log_pdf_zero(x: &Vector9, sigma: &Matrix9) -> Option<f64> {
    let chol = match Cholesky::new(*sigma) {
        Some(c) => c,
        None => {
            let regularized = add_jitter(*sigma);
            match Cholesky::new(regularized) {
                Some(c) => c,
                None => return None,
            }
        }
    };

    // Solve L * z = x
    let z = chol.l().solve_lower_triangular(x).unwrap_or(*x);
    let mahal_sq = z.dot(&z);
    let log_det = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

    Some(-0.5 * (9.0 * LOG_2PI + log_det + mahal_sq))
}

/// Add diagonal jitter for numerical stability (spec §2.6).
///
/// Formula: ε = 10⁻¹⁰ + (tr(Σ)/9) × 10⁻⁸
///
/// The base jitter (10⁻¹⁰) handles near-zero variance cases.
/// The trace-scaled term adapts to the matrix's magnitude.
fn add_jitter(mut sigma: Matrix9) -> Matrix9 {
    let trace: f64 = (0..9).map(|i| sigma[(i, i)]).sum();
    let base_jitter = 1e-10;
    let adaptive_jitter = (trace / 9.0) * 1e-8;
    let jitter = base_jitter + adaptive_jitter;
    for i in 0..9 {
        sigma[(i, i)] += jitter;
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_probability_bounds() {
        let (prob_high, clamped_high) = compute_posterior_probability(100.0, 0.5);
        assert!(prob_high > 0.999);
        assert!(!clamped_high); // 100.0 is below the 700.0 threshold

        let (prob_low, clamped_low) = compute_posterior_probability(-100.0, 0.5);
        assert!(prob_low < 0.001);
        assert!(!clamped_low);

        let (prob_equal, clamped_equal) = compute_posterior_probability(0.0, 0.5);
        assert!((prob_equal - 0.5).abs() < 0.001);
        assert!(!clamped_equal);
    }

    #[test]
    fn test_posterior_probability_clamping() {
        // Test clamping at upper threshold
        let (prob_clamped_high, clamped_high) = compute_posterior_probability(800.0, 0.5);
        assert_eq!(prob_clamped_high, 0.9999);
        assert!(clamped_high);

        // Test clamping at lower threshold
        let (prob_clamped_low, clamped_low) = compute_posterior_probability(-800.0, 0.5);
        assert_eq!(prob_clamped_low, 0.0001);
        assert!(clamped_low);
    }

    #[test]
    fn test_mvn_log_pdf_at_zero() {
        let mean = Vector9::zeros();
        let cov = Matrix9::identity();
        let log_pdf_at_mean = mvn_log_pdf_zero(&mean, &cov).expect("Cholesky should succeed");
        let expected = -0.5 * 9.0 * LOG_2PI;
        assert!((log_pdf_at_mean - expected).abs() < 0.001);
    }

    #[test]
    fn test_bayes_factor_cholesky_fallback() {
        // Test that when Cholesky would fail on a pathological matrix,
        // the Bayes factor falls back to 0.0 (neutral evidence)
        use crate::types::Vector9;

        let observed_diff = Vector9::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

        // Create a pathological covariance with huge condition number
        // that might cause numerical issues (though jitter often fixes it)
        let mut pathological = Matrix9::identity();
        for i in 0..9 {
            pathological[(i, i)] = if i == 0 { 1e20 } else { 1e-20 };
        }

        // Even if Cholesky succeeds (due to jitter), verify the code path works
        // The important part is that the API handles Option<f64> correctly
        let result = compute_bayes_factor(&observed_diff, &pathological, (10.0, 10.0), 0.75);

        // Result should be valid (either computed or fallback to 0.0)
        assert!(result.log_bayes_factor.is_finite());
        assert!(result.posterior_probability >= 0.0 && result.posterior_probability <= 1.0);
    }
}
