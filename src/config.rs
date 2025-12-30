//! Configuration for timing analysis.

/// Configuration options for `TimingOracle`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Samples per class (default: 100,000).
    pub samples: usize,

    /// Warmup iterations before measurement (default: 1,000).
    pub warmup: usize,

    /// False positive rate for CI gate (default: 0.01).
    pub ci_alpha: f64,

    /// Minimum effect size we care about in nanoseconds (default: 10.0).
    ///
    /// Effects smaller than this won't trigger high posterior probabilities
    /// even if statistically detectable. This encodes practical relevance.
    pub min_effect_of_concern_ns: f64,

    /// Optional hard effect threshold in nanoseconds for reporting/panic.
    pub effect_threshold_ns: Option<f64>,

    /// Bootstrap iterations for CI thresholds (default: 10,000).
    pub ci_bootstrap_iterations: usize,

    /// Bootstrap iterations for covariance estimation (default: 2,000).
    pub cov_bootstrap_iterations: usize,

    /// Percentile for outlier filtering (default: 0.999).
    /// Set to 1.0 to disable filtering.
    pub outlier_percentile: f64,

    /// Prior probability of no leak (default: 0.75).
    pub prior_no_leak: f64,

    /// Fraction of samples held out for calibration/preflight (default: 0.3).
    pub calibration_fraction: f32,

    /// Optional guardrail for max duration in milliseconds.
    pub max_duration_ms: Option<u64>,

    /// Optional deterministic seed for measurement randomness.
    pub measurement_seed: Option<u64>,

    /// Iterations per timing sample (default: Auto).
    ///
    /// When set to `Auto`, the library detects timer resolution and
    /// automatically batches iterations when needed for coarse timers.
    /// Set to a specific value to override auto-detection.
    pub iterations_per_sample: IterationsPerSample,
}

/// Configuration for iterations per timing sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationsPerSample {
    /// Automatically detect based on timer resolution.
    ///
    /// On ARM64 with coarse timers (~40ns on Apple Silicon, Ampere Altra),
    /// this will batch multiple iterations per sample for reliable timing.
    /// On x86 or ARMv8.6+ (~1ns resolution), this typically uses 1 iteration.
    Auto,

    /// Use exactly N iterations per sample.
    ///
    /// The measured time will be divided by N to get per-iteration timing.
    Fixed(usize),
}

impl Default for Config {
    fn default() -> Self {
        Self {
            samples: 100_000,
            warmup: 1_000,
            ci_alpha: 0.01,
            min_effect_of_concern_ns: 10.0,
            effect_threshold_ns: None,
            ci_bootstrap_iterations: 10_000,  // Distribution-free FPR guarantee requires sufficient iterations
            cov_bootstrap_iterations: 2_000,  // Accurate covariance estimation for MVN test
            outlier_percentile: 0.999,
            prior_no_leak: 0.75,
            calibration_fraction: 0.3,
            max_duration_ms: None,
            measurement_seed: None,
            iterations_per_sample: IterationsPerSample::Auto,
        }
    }
}

impl Default for IterationsPerSample {
    fn default() -> Self {
        Self::Auto
    }
}

impl IterationsPerSample {
    /// Resolve the iterations count for a given timer.
    ///
    /// For `Auto`, uses the timer's resolution to suggest iterations.
    /// For `Fixed(n)`, returns `n`.
    pub fn resolve(&self, timer: &crate::measurement::Timer) -> usize {
        match self {
            Self::Auto => {
                // Target 10ns effective resolution for statistical reliability
                timer.suggested_iterations(10.0)
            }
            Self::Fixed(n) => *n,
        }
    }
}
