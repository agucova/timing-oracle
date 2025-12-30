//! # timing-oracle
//!
//! Detect timing side channels in cryptographic code.
//!
//! This crate provides statistical methodology for detecting timing variations
//! between two input classes (fixed vs random), outputting:
//! - Probability of timing leak (0.0-1.0)
//! - Effect size estimates in nanoseconds
//! - CI gate pass/fail with bounded false positive rate
//! - Exploitability assessment
//!
//! ## ⚠️ Common Pitfall: Side-Effects in Closures
//!
//! The closures you provide to `test()` must execute **identical code paths**.
//! Only the input *data* should differ - not the operations performed.
//!
//! ```ignore
//! // ❌ WRONG - Random closure has extra RNG/allocation overhead
//! test(
//!     || my_op(&[0u8; 32]),
//!     || my_op(&rand::random()),  // RNG called during measurement!
//! );
//!
//! // ✅ CORRECT - Pre-generate inputs, both closures identical
//! use timing_oracle::helpers::InputPair;
//! let inputs = InputPair::new([0u8; 32], || rand::random());
//! test(
//!     || my_op(&inputs.fixed()),
//!     || my_op(&inputs.random()),
//! );
//! ```
//!
//! See the `helpers` module for utilities that make this pattern easier.
//!
//! ## Quick Start
//!
//! ```ignore
//! use timing_oracle::{TimingOracle, test};
//!
//! // Simple API
//! let result = test(
//!     || my_function(&fixed_input),
//!     || my_function(&random_input()),
//! );
//!
//! println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

// Core modules
mod config;
mod constants;
pub mod ci;
mod oracle;
mod result;
mod types;

// Functional modules
pub mod analysis;
pub mod measurement;
pub mod output;
pub mod preflight;
pub mod statistics;
pub mod helpers;

// Re-exports for public API
pub use config::Config;
pub use constants::{B_TAIL, DECILES, LOG_2PI, ONES};
pub use measurement::Timer;
pub use oracle::TimingOracle;
pub use result::{
    CiGate, Effect, EffectPattern, Exploitability, MeasurementQuality, Metadata,
    MinDetectableEffect, TestResult,
};
pub use ci::{CiFailure, CiRunOutcome, CiTestBuilder, FailCriterion, Mode};
pub use types::{Class, Matrix9, Matrix9x2, Vector9};

// Re-export helpers for convenience
pub use helpers::InputPair;

/// Convenience function for simple timing tests with default configuration.
///
/// This runs a timing analysis comparing a fixed input operation against
/// a random input operation.
///
/// # ⚠️ Critical Requirement
///
/// Both closures must execute **identical operations** - only the input data should differ.
/// Never call RNG functions, allocate memory, or perform I/O inside the closures.
///
/// ```ignore
/// // ❌ WRONG
/// test(|| op(&FIXED), || op(&rand::random()));  // RNG overhead!
///
/// // ✅ CORRECT
/// use timing_oracle::helpers::InputPair;
/// let inputs = InputPair::new(FIXED, || rand::random());
/// test(|| op(&inputs.fixed()), || op(&inputs.random()));
/// ```
///
/// See `helpers::InputPair` for utilities to pre-generate inputs correctly.
///
/// # Arguments
///
/// * `fixed` - Closure that executes with a fixed (potentially vulnerable) input
/// * `random` - Closure that executes with random inputs
///
/// # Returns
///
/// A `TestResult` containing leak probability, effect estimates, and CI gate result.
pub fn test<F, R, T>(fixed: F, random: R) -> TestResult
where
    F: FnMut() -> T,
    R: FnMut() -> T,
{
    TimingOracle::new().test(fixed, random)
}

/// Declarative timing test macro.
///
/// Generates a #[test] that fails if leak_probability > 0.9.
#[cfg(feature = "macros")]
#[macro_export]
macro_rules! timing_test {
    ($name:ident {
        setup: $setup:block,
        fixed: $fixed:expr,
        random: $random:expr,
        test: $test:expr $(,)?
    }) => {
        #[test]
        fn $name() {
            let setup_fn = || $setup;
            let result = $crate::TimingOracle::new()
                .test_with_state(setup_fn, $fixed, $random, $test);

            assert!(
                result.leak_probability <= 0.9,
                "timing leak detected: probability {:.2}",
                result.leak_probability
            );
        }
    };
    ($name:ident {
        fixed: $fixed:expr,
        random: $random:expr,
        test: $test:expr $(,)?
    }) => {
        #[test]
        fn $name() {
            let _ = $test;
            let result = $crate::TimingOracle::new().test($fixed, $random);

            assert!(
                result.leak_probability <= 0.9,
                "timing leak detected: probability {:.2}",
                result.leak_probability
            );
        }
    };
}
