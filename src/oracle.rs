//! Main `TimingOracle` entry point and builder.

use std::time::Instant;

#[allow(unused_imports)]
use rand::Rng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::analysis::{compute_bayes_factor, decompose_effect, estimate_mde, run_ci_gate, CiGateInput};
use crate::ci::CiTestBuilder;
use crate::config::Config;
use crate::measurement::{filter_outliers, Collector, Timer};
use crate::preflight::run_all_checks;
use crate::result::{
    CiGate, Effect, Exploitability, MeasurementQuality, Metadata, MinDetectableEffect, TestResult,
};
use crate::statistics::{bootstrap_covariance_matrix, compute_deciles};
use crate::types::Vector9;

/// Main entry point for timing analysis.
///
/// Use the builder pattern to configure and run timing tests.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::TimingOracle;
///
/// let result = TimingOracle::new()
///     .samples(50_000)
///     .ci_alpha(0.001)
///     .test(
///         || my_function(&fixed_input),
///         || my_function(&random_input()),
///     );
/// ```
#[derive(Debug, Clone)]
pub struct TimingOracle {
    config: Config,
    /// Optional pre-calibrated timer to avoid repeated calibration overhead.
    timer: Option<Timer>,
}

impl Default for TimingOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingOracle {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: Config::default(),
            timer: None,
        }
    }

    /// Create with fast configuration for testing/calibration.
    ///
    /// Uses reduced sample counts, warmup, and bootstrap iterations
    /// for faster execution while maintaining statistical validity.
    ///
    /// Settings:
    /// - 5,000 samples (vs 100,000 default)
    /// - 50 warmup iterations (vs 1,000 default)
    /// - 50 covariance bootstrap iterations (vs 200 default)
    /// - 50 CI bootstrap iterations (vs 500 default)
    pub fn quick() -> Self {
        Self {
            config: Config {
                samples: 5_000,
                warmup: 50,
                cov_bootstrap_iterations: 50,
                ci_bootstrap_iterations: 50,
                ..Config::default()
            },
            timer: None,
        }
    }

    /// Create a CI-focused builder with ergonomic defaults.
    pub fn ci_test() -> CiTestBuilder {
        CiTestBuilder::from_oracle(Self::new())
    }

    /// Use a pre-calibrated timer to avoid calibration overhead.
    ///
    /// Timer calibration (`cycles_per_ns()`) takes ~50ms. When running
    /// many trials (e.g., in calibration tests), reusing a single timer
    /// can significantly speed up execution.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use timing_oracle::{TimingOracle, Timer};
    ///
    /// // Calibrate once
    /// let timer = Timer::new();
    ///
    /// // Reuse across many trials
    /// for _ in 0..100 {
    ///     let result = TimingOracle::quick()
    ///         .with_timer(timer.clone())
    ///         .test(|| fixed_op(), || random_op());
    /// }
    /// ```
    pub fn with_timer(mut self, timer: Timer) -> Self {
        self.timer = Some(timer);
        self
    }

    /// Set samples per class.
    pub fn samples(mut self, n: usize) -> Self {
        self.config.samples = n;
        self
    }

    /// Set warmup iterations.
    pub fn warmup(mut self, n: usize) -> Self {
        self.config.warmup = n;
        self
    }

    /// Set CI false positive rate.
    pub fn ci_alpha(mut self, alpha: f64) -> Self {
        self.config.ci_alpha = alpha;
        self
    }

    /// Set minimum effect of concern in nanoseconds.
    pub fn min_effect_of_concern(mut self, ns: f64) -> Self {
        self.config.min_effect_of_concern_ns = ns;
        self
    }

    /// Set minimum effect of concern in nanoseconds.
    ///
    /// Alias for [`min_effect_of_concern`] to preserve compatibility.
    pub fn effect_prior_ns(mut self, ns: f64) -> Self {
        self.config.min_effect_of_concern_ns = ns;
        self
    }

    /// Optional hard effect threshold in nanoseconds for reporting/panic.
    pub fn effect_threshold_ns(mut self, ns: f64) -> Self {
        self.config.effect_threshold_ns = Some(ns);
        self
    }

    /// Set bootstrap iterations for CI thresholds.
    pub fn ci_bootstrap_iterations(mut self, n: usize) -> Self {
        self.config.ci_bootstrap_iterations = n;
        self
    }

    /// Set bootstrap iterations for covariance estimation.
    pub fn cov_bootstrap_iterations(mut self, n: usize) -> Self {
        self.config.cov_bootstrap_iterations = n;
        self
    }

    /// Set outlier filtering percentile.
    pub fn outlier_percentile(mut self, p: f64) -> Self {
        self.config.outlier_percentile = p;
        self
    }

    /// Set prior probability of no leak.
    pub fn prior_no_leak(mut self, p: f64) -> Self {
        self.config.prior_no_leak = p;
        self
    }

    /// Set calibration fraction for sample splitting (0.0-1.0).
    pub fn calibration_fraction(mut self, frac: f32) -> Self {
        self.config.calibration_fraction = frac;
        self
    }

    /// Set maximum duration guardrail (milliseconds).
    pub fn max_duration_ms(mut self, ms: u64) -> Self {
        self.config.max_duration_ms = Some(ms);
        self
    }

    /// Set deterministic measurement seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.measurement_seed = Some(seed);
        self
    }

    /// Get the current configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run test with simple closures.
    ///
    /// # Arguments
    ///
    /// * `fixed` - Closure that executes the operation with a fixed input
    /// * `random` - Closure that executes the operation with random inputs
    ///
    /// # Returns
    ///
    /// A `TestResult` containing the analysis results.
    pub fn test<F, R, T>(self, mut fixed: F, mut random: R) -> TestResult
    where
        F: FnMut() -> T,
        R: FnMut() -> T,
    {
        let start_time = Instant::now();

        // Step 1: Create timer (reuse if provided) and collector
        let timer = self.timer.clone().unwrap_or_else(Timer::new);
        let collector = Collector::with_timer(timer.clone(), self.config.warmup);

        // Step 2: Collect timing samples with randomized interleaving
        let samples = collector.collect(self.config.samples, &mut fixed, &mut random);
        let mut fixed_cycles = Vec::with_capacity(self.config.samples);
        let mut random_cycles = Vec::with_capacity(self.config.samples);
        let mut interleaved_cycles = Vec::with_capacity(self.config.samples * 2);

        for sample in samples {
            interleaved_cycles.push(sample.cycles);
            match sample.class {
                crate::types::Class::Fixed => fixed_cycles.push(sample.cycles),
                crate::types::Class::Random => random_cycles.push(sample.cycles),
            }
        }

        // Run the analysis pipeline
        self.run_pipeline(
            fixed_cycles,
            random_cycles,
            interleaved_cycles,
            None,
            None,
            &timer,
            start_time,
        )
    }

    /// Run test with setup and state.
    ///
    /// This variant allows more control over test setup and input generation.
    ///
    /// # Type Parameters
    ///
    /// * `S` - State type created by setup
    /// * `I` - Input type produced by generators
    /// * `F` - Fixed input generator
    /// * `R` - Random input generator
    /// * `E` - Executor that runs the operation under test
    ///
    /// # Arguments
    ///
    /// * `setup` - Creates initial state (called once)
    /// * `fixed_input` - Generates the fixed input
    /// * `random_input` - Generates random inputs
    /// * `execute` - Runs the operation under test with given input
    pub fn test_with_state<S, F, R, I, E>(
        self,
        setup: impl FnOnce() -> S,
        mut fixed_input: F,
        mut random_input: R,
        mut execute: E,
    ) -> TestResult
    where
        F: FnMut(&mut S) -> I,
        R: FnMut(&mut S, &mut rand::rngs::ThreadRng) -> I,
        E: FnMut(&mut S, I),
        I: Clone,
    {
        let mut rng = rand::rng();
        self.test_with_state_rng(
            setup,
            &mut rng,
            &mut fixed_input,
            &mut random_input,
            &mut execute,
        )
    }

    /// Run test with setup and state, using a caller-provided RNG.
    pub fn test_with_state_rng<S, F, R, I, E, RNG>(
        self,
        setup: impl FnOnce() -> S,
        rng: &mut RNG,
        fixed_input: &mut F,
        random_input: &mut R,
        execute: &mut E,
    ) -> TestResult
    where
        RNG: rand::Rng + ?Sized,
        F: FnMut(&mut S) -> I,
        R: FnMut(&mut S, &mut RNG) -> I,
        E: FnMut(&mut S, I),
        I: Clone,
    {
        let start_time = Instant::now();

        // Create state
        let mut state = setup();

        // Create timer (reuse if provided)
        let timer = self.timer.clone().unwrap_or_else(Timer::new);

        // Pre-generate all inputs to avoid borrow conflicts
        let fixed_gen_start = Instant::now();
        let fixed_inputs: Vec<I> = (0..self.config.samples)
            .map(|_| fixed_input(&mut state))
            .collect();
        let fixed_gen_time_ns = if self.config.samples > 0 {
            fixed_gen_start.elapsed().as_nanos() as f64 / self.config.samples as f64
        } else {
            0.0
        };

        let random_gen_start = Instant::now();
        let random_inputs: Vec<I> = (0..self.config.samples)
            .map(|_| random_input(&mut state, rng))
            .collect();
        let random_gen_time_ns = if self.config.samples > 0 {
            random_gen_start.elapsed().as_nanos() as f64 / self.config.samples as f64
        } else {
            0.0
        };

        // Run warmup
        for _ in 0..self.config.warmup {
            if let Some(input) = fixed_inputs.first() {
                execute(&mut state, input.clone());
            }
            if let Some(input) = random_inputs.first() {
                execute(&mut state, input.clone());
            }
        }

        // Create randomized schedule
        let mut schedule: Vec<(crate::types::Class, usize)> =
            Vec::with_capacity(self.config.samples * 2);
        for i in 0..self.config.samples {
            schedule.push((crate::types::Class::Fixed, i));
            schedule.push((crate::types::Class::Random, i));
        }
        schedule.shuffle(rng);

        // Collect timing samples
        let mut fixed_cycles = Vec::with_capacity(self.config.samples);
        let mut random_cycles = Vec::with_capacity(self.config.samples);
        let mut interleaved_cycles = Vec::with_capacity(self.config.samples * 2);

        for (class, idx) in schedule {
            match class {
                crate::types::Class::Fixed => {
                    let cycles = timer.measure_cycles(|| {
                        execute(&mut state, fixed_inputs[idx].clone());
                    });
                    fixed_cycles.push(cycles);
                    interleaved_cycles.push(cycles);
                }
                crate::types::Class::Random => {
                    let cycles = timer.measure_cycles(|| {
                        execute(&mut state, random_inputs[idx].clone());
                    });
                    random_cycles.push(cycles);
                    interleaved_cycles.push(cycles);
                }
            }
        }

        // Run the analysis pipeline
        self.run_pipeline(
            fixed_cycles,
            random_cycles,
            interleaved_cycles,
            Some(fixed_gen_time_ns),
            Some(random_gen_time_ns),
            &timer,
            start_time,
        )
    }

    /// Run the full analysis pipeline on collected samples.
    fn run_pipeline(
        &self,
        fixed_cycles: Vec<u64>,
        random_cycles: Vec<u64>,
        interleaved_cycles: Vec<u64>,
        fixed_gen_time_ns: Option<f64>,
        random_gen_time_ns: Option<f64>,
        timer: &Timer,
        start_time: Instant,
    ) -> TestResult {
        // Step 3: Outlier filtering (pooled symmetric)
        let (filtered_fixed, filtered_random, outlier_stats) =
            filter_outliers(&fixed_cycles, &random_cycles, self.config.outlier_percentile);

        // Convert cycles to nanoseconds
        let fixed_ns: Vec<f64> = filtered_fixed
            .iter()
            .map(|&c| timer.cycles_to_ns(c))
            .collect();
        let random_ns: Vec<f64> = filtered_random
            .iter()
            .map(|&c| timer.cycles_to_ns(c))
            .collect();

        // Step 4: Compute full-sample quantile differences for CI gate
        let q_fixed_full = compute_deciles(&fixed_ns);
        let q_random_full = compute_deciles(&random_ns);
        let delta_full: Vector9 = q_fixed_full - q_random_full;

        // Step 5: Sample splitting (calibration/inference)
        let n = fixed_ns.len().min(random_ns.len());
        let n_calib = ((n as f64) * self.config.calibration_fraction as f64).round() as usize;
        let n_calib = n_calib.max(10).min(n.saturating_sub(10)); // Ensure at least 10 samples each

        let mut rng = if let Some(seed) = self.config.measurement_seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_rng(&mut rand::rng())
        };

        let mut fixed_shuffled = fixed_ns.clone();
        let mut random_shuffled = random_ns.clone();
        fixed_shuffled.shuffle(&mut rng);
        random_shuffled.shuffle(&mut rng);

        let (calib_fixed, infer_fixed) = fixed_shuffled.split_at(n_calib);
        let (calib_random, infer_random) = random_shuffled.split_at(n_calib);

        // Run preflight checks
        let interleaved_ns: Vec<f64> = interleaved_cycles
            .iter()
            .map(|&c| timer.cycles_to_ns(c))
            .collect();
        let _preflight = run_all_checks(
            &fixed_ns,
            &random_ns,
            &interleaved_ns,
            fixed_gen_time_ns,
            random_gen_time_ns,
        );

        // CALIBRATION PHASE
        // Step 6: Estimate per-class covariances from calibration data
        let bootstrap_iters = self.config.cov_bootstrap_iterations.min(500);

        let cov_fixed = bootstrap_covariance_matrix(calib_fixed, bootstrap_iters, 42);
        let cov_random = bootstrap_covariance_matrix(calib_random, bootstrap_iters, 43);

        // Step 7: Estimate MDE from calibration covariance
        let pooled_cov = cov_fixed.matrix + cov_random.matrix;
        let mde_estimate = estimate_mde(
            &pooled_cov,
            100, // Reduced simulations for speed
            (1e6, 1e6), // Weak prior for near-MLE estimates
        );

        // Prior scales from calibration (spec: max(2*MDE, min_effect_of_concern))
        let min_effect = self.config.min_effect_of_concern_ns;
        let prior_sigmas = (
            (2.0 * mde_estimate.shift_ns).max(min_effect),
            (2.0 * mde_estimate.tail_ns).max(min_effect),
        );

        // INFERENCE PHASE
        // Step 8: Compute quantile difference vector from inference data
        let q_fixed = compute_deciles(infer_fixed);
        let q_random = compute_deciles(infer_random);
        let delta_infer: Vector9 = q_fixed - q_random;

        // Use calibration covariance for inference per spec
        let cov_estimate = pooled_cov;

        // Step 9: Run CI Gate (Layer 1)
        let ci_gate_input = CiGateInput {
            observed_diff: delta_full,
            fixed_samples: &fixed_ns,
            random_samples: &random_ns,
            alpha: self.config.ci_alpha,
            bootstrap_iterations: self.config.ci_bootstrap_iterations,
            seed: self.config.measurement_seed,
        };
        let ci_gate = run_ci_gate(&ci_gate_input);

        // Step 10: Compute Bayes factor and posterior probability (Layer 2)
        let bayes_result =
            compute_bayes_factor(&delta_infer, &cov_estimate, prior_sigmas, self.config.prior_no_leak);
        let leak_probability = bayes_result.posterior_probability;

        // Step 11: Effect decomposition (if leak detected)
        let effect = if leak_probability > 0.5 || !ci_gate.passed {
            let decomp = decompose_effect(
                &delta_infer,
                &cov_estimate,
                prior_sigmas,
            );

            Some(Effect {
                shift_ns: decomp.posterior_mean[0],
                tail_ns: decomp.posterior_mean[1],
                credible_interval_ns: decomp.effect_magnitude_ci,
                pattern: decomp.pattern,
            })
        } else {
            None
        };

        // Step 12: Exploitability assessment
        let effect_magnitude = effect
            .as_ref()
            .map(|e| (e.shift_ns.powi(2) + e.tail_ns.powi(2)).sqrt())
            .unwrap_or(0.0);
        let exploitability = Exploitability::from_effect_ns(effect_magnitude);

        // Step 13: Measurement quality assessment
        let quality = MeasurementQuality::from_mde_ns(mde_estimate.shift_ns);

        // Timer identification
        let timer_name = if cfg!(target_arch = "x86_64") {
            "rdtsc"
        } else if cfg!(target_arch = "aarch64") {
            "cntvct_el0"
        } else {
            "Instant"
        };

        let runtime_secs = start_time.elapsed().as_secs_f64();

        // Step 14: Assemble and return TestResult
        TestResult {
            leak_probability,
            effect,
            exploitability,
            min_detectable_effect: MinDetectableEffect {
                shift_ns: mde_estimate.shift_ns,
                tail_ns: mde_estimate.tail_ns,
            },
            ci_gate: CiGate {
                alpha: ci_gate.alpha,
                passed: ci_gate.passed,
                thresholds: ci_gate.thresholds,
                observed: ci_gate.observed,
            },
            quality,
            outlier_fraction: outlier_stats.outlier_fraction,
            metadata: Metadata {
            samples_per_class: infer_fixed.len().min(infer_random.len()),
                cycles_per_ns: timer.cycles_per_ns(),
                timer: timer_name.to_string(),
                runtime_secs,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_default_config() {
        let oracle = TimingOracle::new();
        assert_eq!(oracle.config().samples, 100_000);
        assert_eq!(oracle.config().warmup, 1_000);
        assert_eq!(oracle.config().ci_alpha, 0.01);
    }

    #[test]
    fn test_oracle_builder() {
        let oracle = TimingOracle::new()
            .samples(50_000)
            .warmup(500)
            .ci_alpha(0.05)
            .min_effect_of_concern(5.0)
            .prior_no_leak(0.9);

        assert_eq!(oracle.config().samples, 50_000);
        assert_eq!(oracle.config().warmup, 500);
        assert_eq!(oracle.config().ci_alpha, 0.05);
        assert_eq!(oracle.config().min_effect_of_concern_ns, 5.0);
        assert_eq!(oracle.config().prior_no_leak, 0.9);
    }

    #[test]
    fn test_oracle_quick() {
        let oracle = TimingOracle::quick();
        assert_eq!(oracle.config().samples, 5_000);
        assert_eq!(oracle.config().warmup, 50);
        assert_eq!(oracle.config().cov_bootstrap_iterations, 50);
        assert_eq!(oracle.config().ci_bootstrap_iterations, 50);
    }

    #[test]
    fn test_oracle_with_timer() {
        let timer = Timer::with_cycles_per_ns(3.0);
        let oracle = TimingOracle::quick().with_timer(timer);
        assert!(oracle.timer.is_some());
        assert_eq!(oracle.timer.unwrap().cycles_per_ns(), 3.0);
    }
}
