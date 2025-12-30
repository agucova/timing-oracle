//! CI-focused builder for running timing checks inside cargo test/nextest.

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{Config, TimingOracle};
use crate::result::TestResult;

/// Preset modes for CI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    /// Faster checks for per-commit CI.
    Smoke,
    /// Deeper checks for nightly/PR.
    Full,
}

/// Pass/fail policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FailCriterion {
    /// Use CI gate (bounded FPR). Default.
    CiGate,
    /// Fail if posterior leak probability crosses threshold.
    Probability(f64),
    /// Fail if either CI gate trips or posterior crosses threshold.
    Either {
        /// Probability threshold for Bayesian leak detection.
        probability: f64,
    },
}

/// CI-facing builder for timing tests.
pub struct CiTestBuilder {
    oracle: TimingOracle,
    mode: Mode,
    report_path: Option<PathBuf>,
    seed: Option<u64>,
    fail_on: FailCriterion,
    async_workload: bool,
}

/// Result of a CI run with context for reporting.
#[derive(Debug)]
pub struct CiRunOutcome {
    /// Full timing result.
    pub result: TestResult,
    /// Report location if written.
    pub report_path: Option<PathBuf>,
    /// Mode that was applied (Smoke/Full).
    pub mode: Mode,
    /// RNG seed, if provided.
    pub seed: Option<u64>,
    /// Config snapshot used for the run.
    pub config: Config,
    /// Fail criterion in effect.
    pub fail_on: FailCriterion,
    /// Whether the workload was marked async (higher noise expectations).
    pub async_workload: bool,
}

/// Failure reasons from CI helper.
#[derive(Debug)]
pub enum CiFailure {
    /// Leak detected according to fail criterion.
    LeakDetected {
        /// Full run context when leak was detected.
        outcome: Box<CiRunOutcome>,
    },
    /// Failed to write report to disk.
    ReportIo {
        /// Path we attempted to write.
        path: PathBuf,
        /// Underlying IO error.
        source: io::Error,
    },
    /// Generic measurement/config error.
    Measurement(String),
}

impl CiTestBuilder {
    /// Create a CI-oriented builder with smoke defaults.
    pub fn new() -> Self {
        let mut builder = Self {
            oracle: TimingOracle::new(),
            mode: Mode::Smoke,
            report_path: None,
            seed: None,
            fail_on: FailCriterion::CiGate,
            async_workload: false,
        };
        builder.apply_mode(Mode::Smoke);
        builder
    }

    /// Attach to TimingOracle via associated function.
    pub(crate) fn from_oracle(oracle: TimingOracle) -> Self {
        let mut builder = Self {
            oracle,
            mode: Mode::Smoke,
            report_path: None,
            seed: None,
            fail_on: FailCriterion::CiGate,
            async_workload: false,
        };
        builder.apply_mode(Mode::Smoke);
        builder
    }

    /// Merge configuration from environment variables.
    pub fn from_env(mut self) -> Self {
        if let Some(mode) = parse_mode_env("TO_MODE") {
            self = self.mode(mode);
        }
        if let Some(samples) = parse_usize_env("TO_SAMPLES") {
            self = self.samples(samples);
        }
        if let Some(alpha) = parse_f64_env("TO_ALPHA") {
            self = self.alpha(alpha);
        }
        if let Some(prior) = parse_f64_env("TO_EFFECT_PRIOR_NS") {
            self = self.effect_prior_ns(prior);
        }
        if let Some(threshold) = parse_f64_env("TO_EFFECT_THRESHOLD_NS") {
            self = self.effect_threshold_ns(threshold);
        }
        if let Some(frac) = parse_f32_env("TO_CALIBRATION_FRAC") {
            self = self.calibration_fraction(frac);
        }
        if let Some(path) = parse_path_env("TO_REPORT") {
            self = self.report_path(path);
        }
        if let Some(seed) = parse_u64_env("TO_SEED") {
            self = self.seed(seed);
        }
        if let Some(ms) = parse_u64_env("TO_MAX_DURATION_MS") {
            self = self.max_duration_ms(ms);
        }
        if let Some(fail) = parse_fail_on_env("TO_FAIL_ON") {
            self = self.fail_on(fail);
        }
        if let Ok(val) = env::var("TO_ASYNC_WORKLOAD") {
            if val == "1" || val.eq_ignore_ascii_case("true") {
                self = self.async_workload(true);
            }
        }
        self
    }

    /// Override preset mode.
    pub fn mode(mut self, mode: Mode) -> Self {
        self.apply_mode(mode);
        self
    }

    /// Override sample count.
    pub fn samples(mut self, n: usize) -> Self {
        self.oracle = self.oracle.samples(n);
        self
    }

    /// Override alpha.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.oracle = self.oracle.ci_alpha(alpha);
        self
    }

    /// Override effect prior.
    pub fn effect_prior_ns(mut self, ns: f64) -> Self {
        self.oracle = self.oracle.effect_prior_ns(ns);
        self
    }

    /// Optional hard effect threshold.
    pub fn effect_threshold_ns(mut self, ns: f64) -> Self {
        self.oracle = self.oracle.effect_threshold_ns(ns);
        self
    }

    /// Override calibration fraction.
    pub fn calibration_fraction(mut self, frac: f32) -> Self {
        self.oracle = self.oracle.calibration_fraction(frac);
        self
    }

    /// Override max duration.
    pub fn max_duration_ms(mut self, ms: u64) -> Self {
        self.oracle = self.oracle.max_duration_ms(ms);
        self
    }

    /// Set report path.
    pub fn report_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.report_path = Some(path.into());
        self
    }

    /// Set deterministic seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self.oracle = self.oracle.seed(seed);
        self
    }

    /// Configure pass/fail criterion.
    pub fn fail_on(mut self, criterion: FailCriterion) -> Self {
        self.fail_on = criterion;
        self
    }

    /// Mark workload as async, inflating priors/thresholds to account for noise.
    pub fn async_workload(mut self, yes: bool) -> Self {
        self.async_workload = yes;
        if yes {
            // Nudge priors/thresholds upward for expected jitter.
            self.oracle = self.oracle.effect_prior_ns(30.0);
            if self.oracle.config().effect_threshold_ns.is_none() {
                self.oracle = self.oracle.effect_threshold_ns(30.0);
            }
        }
        self
    }

    /// Run the timing test and return result.
    pub fn run<F, R, T>(self, fixed: F, random: R) -> Result<TestResult, CiFailure>
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        self.run_with_context(fixed, random).map(|outcome| outcome.result)
    }

    /// Run and panic on failure, printing a concise summary and optionally writing a report.
    pub fn unwrap_or_report<F, R, T>(self, fixed: F, random: R) -> TestResult
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        match self.run_with_context(fixed, random) {
            Ok(outcome) => {
                print_summary(&outcome, /*failed=*/ false);
                outcome.result
            }
            Err(CiFailure::LeakDetected { outcome }) => {
                print_summary(&outcome, /*failed=*/ true);
                panic!(
                    "timing leak detected (fail_on={:?}, p={:.2}, seed={})",
                    outcome.fail_on,
                    outcome.result.leak_probability,
                    outcome.seed
                        .map(|s| format!("0x{s:08x}"))
                        .unwrap_or_else(|| "none".to_string())
                );
            }
            Err(CiFailure::ReportIo { path, source }) => {
                panic!("timing-oracle: failed to write report to {}: {}", path.display(), source);
            }
            Err(CiFailure::Measurement(msg)) => {
                panic!("timing-oracle: measurement failed: {}", msg);
            }
        }
    }

    fn run_with_context<F, R, T>(
        self,
        fixed: F,
        random: R,
    ) -> Result<CiRunOutcome, CiFailure>
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        let CiTestBuilder {
            oracle,
            mode,
            report_path,
            seed,
            fail_on,
            async_workload,
        } = self;

        if let Some(seed) = seed {
            // TODO: plumb seed into measurement RNG when available.
            let _ = seed;
        }

        let report_path = compute_report_path(report_path.as_ref());
        let config_snapshot = oracle.config().clone();

        let result = oracle.test(fixed, random);
        let outcome = CiRunOutcome {
            report_path,
            result,
            mode,
            seed,
            config: config_snapshot,
            fail_on,
            async_workload,
        };

        write_report(&outcome)?;

        if should_fail(fail_on, &outcome.result) {
            return Err(CiFailure::LeakDetected { outcome: Box::new(outcome) });
        }

        Ok(outcome)
    }

    fn apply_mode(&mut self, mode: Mode) {
        self.mode = mode;
        match mode {
            Mode::Smoke => {
                let oracle = std::mem::take(&mut self.oracle);
                self.oracle = oracle
                    .samples(20_000)
                    .ci_alpha(0.01)
                    .effect_prior_ns(20.0);
            }
            Mode::Full => {
                let oracle = std::mem::take(&mut self.oracle);
                self.oracle = oracle
                    .samples(100_000)
                    .ci_alpha(0.01)
                    .effect_prior_ns(10.0);
            }
        }
    }

}

impl Default for CiTestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn compute_report_path(explicit: Option<&PathBuf>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.clone());
    }

    let dump_requested = env::args().any(|arg| arg == "--timing-oracle-dump");
    if !dump_requested {
        return None;
    }

    let mut path = env::temp_dir();
    path.push("timing-oracle");
    let _ = fs::create_dir_all(&path);
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    path.push(format!("report-{}.json", unique));
    Some(path)
}

fn print_summary(outcome: &CiRunOutcome, failed: bool) {
    let seed_str = outcome
        .seed
        .map(|s| format!("0x{s:08x}"))
        .unwrap_or_else(|| "none".to_string());

    eprintln!(
        "timing-oracle: seed={} mode={:?} samples={} alpha={:.4} split={:.0}/{:.0} async={}",
        seed_str,
        outcome.mode,
        outcome.config.samples,
        outcome.config.ci_alpha,
        outcome.config.calibration_fraction * 100.0,
        (1.0 - outcome.config.calibration_fraction as f64) * 100.0,
        outcome.async_workload,
    );

    eprintln!(
        "timing-oracle: ci_gate={} leak_probability={:.2} effect_threshold_ns={:?} min_effect_of_concern_ns={:.2}",
        if outcome.result.ci_gate.passed { "passed" } else { "tripped" },
        outcome.result.leak_probability,
        outcome.config.effect_threshold_ns,
        outcome.config.min_effect_of_concern_ns,
    );

    if let Some(path) = &outcome.report_path {
        eprintln!("timing-oracle: report saved to {}", path.display());
    }

    if !failed {
        return;
    }

    eprintln!(
        "timing-oracle: fail_on={:?} exploitability={:?}",
        outcome.fail_on, outcome.result.exploitability
    );
}

fn should_fail(criterion: FailCriterion, result: &TestResult) -> bool {
    match criterion {
        FailCriterion::CiGate => !result.ci_gate.passed,
        FailCriterion::Probability(thresh) => result.leak_probability >= thresh,
        FailCriterion::Either { probability } => {
            !result.ci_gate.passed || result.leak_probability >= probability
        }
    }
}

#[derive(Serialize)]
struct CiReport<'a> {
    mode: Mode,
    seed: Option<u64>,
    fail_on: FailCriterion,
    samples: usize,
    ci_alpha: f64,
    min_effect_of_concern_ns: f64,
    effect_threshold_ns: Option<f64>,
    calibration_fraction: f32,
    max_duration_ms: Option<u64>,
    outlier_percentile: f64,
    prior_no_leak: f64,
    async_workload: bool,
    result: &'a TestResult,
}

fn write_report(outcome: &CiRunOutcome) -> Result<(), CiFailure> {
    let Some(path) = &outcome.report_path else {
        return Ok(());
    };

    let report = CiReport {
        mode: outcome.mode,
        seed: outcome.seed,
        fail_on: outcome.fail_on,
        samples: outcome.config.samples,
        ci_alpha: outcome.config.ci_alpha,
        min_effect_of_concern_ns: outcome.config.min_effect_of_concern_ns,
        effect_threshold_ns: outcome.config.effect_threshold_ns,
        calibration_fraction: outcome.config.calibration_fraction,
        max_duration_ms: outcome.config.max_duration_ms,
        outlier_percentile: outcome.config.outlier_percentile,
        prior_no_leak: outcome.config.prior_no_leak,
        async_workload: outcome.async_workload,
        result: &outcome.result,
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| CiFailure::ReportIo { path: path.clone(), source: err })?;
    }
    let file = fs::File::create(path)
        .map_err(|err| CiFailure::ReportIo { path: path.clone(), source: err })?;
    serde_json::to_writer_pretty(file, &report)
        .map_err(|err| CiFailure::ReportIo { path: path.clone(), source: io::Error::other(err) })
}

fn parse_mode_env(key: &str) -> Option<Mode> {
    match env::var(key).ok()?.to_ascii_lowercase().as_str() {
        "smoke" => Some(Mode::Smoke),
        "full" => Some(Mode::Full),
        _ => None,
    }
}

fn parse_usize_env(key: &str) -> Option<usize> {
    env::var(key).ok()?.parse().ok()
}

fn parse_u64_env(key: &str) -> Option<u64> {
    env::var(key).ok()?.parse().ok()
}

fn parse_f64_env(key: &str) -> Option<f64> {
    env::var(key).ok()?.parse().ok()
}

fn parse_f32_env(key: &str) -> Option<f32> {
    env::var(key).ok()?.parse().ok()
}

fn parse_path_env(key: &str) -> Option<PathBuf> {
    env::var(key).ok().map(PathBuf::from)
}

fn parse_fail_on_env(key: &str) -> Option<FailCriterion> {
    let raw = env::var(key).ok()?;
    let lower = raw.to_ascii_lowercase();
    if lower == "ci_gate" || lower == "cigate" {
        return Some(FailCriterion::CiGate);
    }
    if lower.starts_with("prob") {
        if let Some(val) = lower.split(':').nth(1) {
            if let Ok(p) = val.parse::<f64>() {
                return Some(FailCriterion::Probability(p));
            }
        }
    }
    if lower.starts_with("either") {
        if let Some(val) = lower.split(':').nth(1) {
            if let Ok(p) = val.parse::<f64>() {
                return Some(FailCriterion::Either { probability: p });
            }
        }
    }
    None
}
