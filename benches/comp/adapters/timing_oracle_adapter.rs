//! Adapter for timing-oracle library.

use super::{DetectionResult, Detector, RawData};
use std::any::Any;
use std::time::Instant;
use timing_oracle::TimingOracle;

/// Adapter for timing-oracle
pub struct TimingOracleDetector {
    /// Use balanced preset for faster comparison runs
    use_balanced: bool,
}

impl TimingOracleDetector {
    pub fn new() -> Self {
        Self {
            use_balanced: true,
        }
    }

    pub fn with_balanced(mut self, balanced: bool) -> Self {
        self.use_balanced = balanced;
        self
    }
}

impl Default for TimingOracleDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector for TimingOracleDetector {
    fn name(&self) -> &str {
        "timing-oracle"
    }

    fn detect(&self, fixed: &dyn Fn(), random: &dyn Fn(), samples: usize) -> DetectionResult {
        let start = Instant::now();

        let oracle = if self.use_balanced {
            TimingOracle::balanced()
        } else {
            TimingOracle::new()
        };

        let result = oracle.samples(samples).test(fixed, random);

        let duration = start.elapsed();

        DetectionResult {
            detected_leak: result.leak_probability > 0.5,
            confidence_metric: result.leak_probability,
            samples_used: result.metadata.samples_per_class,
            duration,
            raw_data: Some(RawData::TimingOracle {
                leak_probability: result.leak_probability,
                ci_gate_passed: result.ci_gate.passed,
            }),
        }
    }

    fn default_threshold(&self) -> f64 {
        0.5
    }

    fn exceeds_threshold(&self, confidence_metric: f64, threshold: f64) -> bool {
        confidence_metric > threshold
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
