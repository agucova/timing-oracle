//! Layer 2: Bayesian inference for timing leak detection.
//!
//! This module computes the posterior probability of a timing leak using:
//! - Sample splitting (calibration/inference handled by the caller)
//! - Closed-form Bayes factor via multivariate normal log-pdf ratio
//! - Cholesky decomposition for numerical stability

use nalgebra::Cholesky;

use crate::constants::{B_TAIL, LOG_2PI, ONES};
use crate::types::{Matrix2, Matrix9, Matrix9x2, Vector2, Vector9};

/// Result from Bayesian analysis.
#[derive(Debug, Clone)]
pub struct BayesResult {
    /// Log Bayes factor: log(P(data|H1) / P(data|H0))
    pub log_bayes_factor: f64,
    /// Posterior probability of leak: P(H1|data)
    pub posterior_probability: f64,
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

    let log_bf = mvn_log_pdf_zero(observed_diff, &sigma1)
        - mvn_log_pdf_zero(observed_diff, sigma0);

    let posterior = compute_posterior_probability(log_bf, prior_no_leak);

    BayesResult {
        log_bayes_factor: log_bf,
        posterior_probability: posterior,
        sigma0: *sigma0,
        sigma1,
    }
}

/// Convert log Bayes factor to posterior probability.
///
/// P(H1|data) = BF * prior_odds / (1 + BF * prior_odds)
pub fn compute_posterior_probability(log_bf: f64, prior_no_leak: f64) -> f64 {
    let prior_no_leak = prior_no_leak.clamp(1e-12, 1.0 - 1e-12);
    let prior_odds = (1.0 - prior_no_leak) / prior_no_leak;

    let log_posterior_odds = log_bf + prior_odds.ln();

    if log_posterior_odds > 700.0 {
        1.0
    } else if log_posterior_odds < -700.0 {
        0.0
    } else {
        1.0 / (1.0 + (-log_posterior_odds).exp())
    }
}

fn build_design_matrix() -> Matrix9x2 {
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

fn mvn_log_pdf_zero(x: &Vector9, sigma: &Matrix9) -> f64 {
    let chol = match Cholesky::new(*sigma) {
        Some(c) => c,
        None => {
            let regularized = add_jitter(*sigma);
            match Cholesky::new(regularized) {
                Some(c) => c,
                None => return 0.0,
            }
        }
    };

    // Solve L * z = x
    let z = chol.l().solve_lower_triangular(x).unwrap_or(*x);
    let mahal_sq = z.dot(&z);
    let log_det = 2.0 * chol.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();

    -0.5 * (9.0 * LOG_2PI + log_det + mahal_sq)
}

fn add_jitter(mut sigma: Matrix9) -> Matrix9 {
    let max_diag = (0..9).map(|i| sigma[(i, i)].abs()).fold(0.0, f64::max);
    let jitter = 1e-9 * max_diag.max(1.0);
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
        let prob_high = compute_posterior_probability(100.0, 0.5);
        assert!(prob_high > 0.999);

        let prob_low = compute_posterior_probability(-100.0, 0.5);
        assert!(prob_low < 0.001);

        let prob_equal = compute_posterior_probability(0.0, 0.5);
        assert!((prob_equal - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_mvn_log_pdf_at_zero() {
        let mean = Vector9::zeros();
        let cov = Matrix9::identity();
        let log_pdf_at_mean = mvn_log_pdf_zero(&mean, &cov);
        let expected = -0.5 * 9.0 * LOG_2PI;
        assert!((log_pdf_at_mean - expected).abs() < 0.001);
    }
}
