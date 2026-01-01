//! # timing-oracle
//!
//! Detect timing side channels in cryptographic code.
//!
//! This crate provides statistical methodology for detecting timing variations
//! between two input classes (baseline vs sample), outputting:
//! - Probability of timing leak (0.0-1.0)
//! - Effect size estimates in nanoseconds
//! - CI gate pass/fail with bounded false positive rate
//! - Exploitability assessment
//!
//! ## Common Pitfall: Side-Effects in Closures
//!
//! The closures you provide to `test()` must execute **identical code paths**.
//! Only the input *data* should differ - not the operations performed.
//!
//! ```ignore
//! // WRONG - Sample closure has extra RNG/allocation overhead
//! test(
//!     || my_op(&[0u8; 32]),
//!     || my_op(&rand::random()),  // RNG called during measurement!
//! );
//!
//! // CORRECT - Pre-generate inputs, both closures identical
//! use timing_oracle::helpers::InputPair;
//! let inputs = InputPair::new(|| [0u8; 32], || rand::random());
//! test(inputs, |data| {
//!     my_op(data);
//! });
//! ```
//!
//! See the `helpers` module for utilities that make this pattern easier.
//!
//! ## Quick Start
//!
//! ```ignore
//! use timing_oracle::{test, helpers::InputPair};
//!
//! // Simple API with InputPair
//! let inputs = InputPair::new(|| [0u8; 32], || rand::random());
//! let outcome = test(inputs, |data| {
//!     my_function(data);
//! });
//!
//! match outcome {
//!     timing_oracle::Outcome::Completed(result) => {
//!         println!("Leak probability: {:.1}%", result.leak_probability * 100.0);
//!     }
//!     timing_oracle::Outcome::Unmeasurable { recommendation, .. } => {
//!         println!("Skipping: {}", recommendation);
//!     }
//! }
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
mod builder;
mod thread_pool;

// Functional modules
pub mod analysis;
pub mod measurement;
pub mod output;
pub mod preflight;
pub mod statistics;
pub mod helpers;

// Re-exports for public API
pub use config::{Config, IterationsPerSample};
pub use constants::{B_TAIL, DECILES, LOG_2PI, ONES};
pub use measurement::{BoxedTimer, Timer, TimerSpec};
pub use oracle::TimingOracle;
pub use result::{
    BatchingInfo, CiGate, Diagnostics, Effect, EffectPattern, Exploitability, MeasurementQuality,
    Metadata, MinDetectableEffect, Outcome, TestResult, UnmeasurableInfo, UnreliablePolicy,
};
pub use ci::{CiFailure, CiRunOutcome, CiTestBuilder, FailCriterion, Mode};
pub use types::{Class, Matrix9, Matrix9x2, TimingSample, Vector9};

// Re-export helpers for convenience
pub use helpers::InputPair;

// Re-export builder for the builder API
pub use builder::TimingTest;

/// Convenience function for simple timing tests with default configuration.
///
/// This runs a timing analysis comparing a baseline input operation against
/// a sample input operation.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::{test, helpers::InputPair, Outcome};
///
/// let inputs = InputPair::new(|| [0u8; 32], || rand::random());
/// let outcome = test(inputs, |data| {
///     my_crypto_function(data);
/// });
///
/// match outcome {
///     Outcome::Completed(result) => {
///         assert!(result.leak_probability < 0.5, "Timing leak detected!");
///     }
///     Outcome::Unmeasurable { .. } => {
///         // Test skipped due to noisy environment
///     }
/// }
/// ```
///
/// # Arguments
///
/// * `inputs` - An `InputPair` containing the baseline and sample closures
/// * `operation` - Closure that performs the operation to time (receives `&T`)
///
/// # Returns
///
/// An `Outcome` which is either `Completed(TestResult)` or `Unmeasurable`.
pub fn test<T, F1, F2, F>(inputs: InputPair<T, F1, F2>, operation: F) -> Outcome
where
    T: Clone + std::hash::Hash,
    F1: FnMut() -> T,
    F2: FnMut() -> T,
    F: FnMut(&T),
{
    TimingOracle::new().test(inputs, operation)
}

// ============================================================================
// Reliability Handling Macros
// ============================================================================

/// Skip test if measurement is unreliable (fail-open).
///
/// Prints `[SKIPPED]` message and returns early if unreliable.
/// Returns `TestResult` if reliable.
///
/// # Example
/// ```ignore
/// use timing_oracle::{TimingOracle, InputPair, skip_if_unreliable};
///
/// #[test]
/// fn test_aes() {
///     let inputs = InputPair::new(|| [0u8; 16], || rand::random());
///     let outcome = TimingOracle::new().test(inputs, |data| encrypt(data));
///     let result = skip_if_unreliable!(outcome, "test_aes");
///     assert!(result.leak_probability < 0.1);
/// }
/// ```
#[macro_export]
macro_rules! skip_if_unreliable {
    ($outcome:expr, $name:expr) => {
        match $outcome.handle_unreliable($name, $crate::UnreliablePolicy::FailOpen) {
            Some(result) => result,
            None => return,
        }
    };
}

/// Require measurement to be reliable (fail-closed).
///
/// Panics if unreliable. Returns `TestResult` if reliable.
///
/// # Example
/// ```ignore
/// use timing_oracle::{TimingOracle, InputPair, require_reliable};
///
/// #[test]
/// fn test_aes_critical() {
///     let inputs = InputPair::new(|| [0u8; 16], || rand::random());
///     let outcome = TimingOracle::new().test(inputs, |data| encrypt(data));
///     let result = require_reliable!(outcome, "test_aes_critical");
///     assert!(result.leak_probability < 0.1);
/// }
/// ```
#[macro_export]
macro_rules! require_reliable {
    ($outcome:expr, $name:expr) => {
        match $outcome.handle_unreliable($name, $crate::UnreliablePolicy::FailClosed) {
            Some(result) => result,
            None => unreachable!(),
        }
    };
}

// Re-export the timing_test! and timing_test_checked! proc macros when the macros feature is enabled
#[cfg(feature = "macros")]
pub use timing_oracle_macros::{timing_test, timing_test_checked};
