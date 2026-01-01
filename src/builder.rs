//! Builder API for timing tests.
//!
//! The [`TimingTest`] builder provides an explicit, method-chaining API for
//! configuring and running timing tests. It's an alternative to the `timing_test!`
//! macro for users who prefer explicit code over macros.
//!
//! # Example
//!
//! ```ignore
//! use timing_oracle::{TimingTest, TimingOracle, Outcome};
//!
//! let result = TimingTest::new()
//!     .oracle(TimingOracle::balanced())
//!     .baseline(|| [0u8; 32])
//!     .sample(|| rand::random::<[u8; 32]>())
//!     .measure(|input| secret.ct_eq(&input))
//!     .run();
//! ```
//!
//! # Required Fields
//!
//! The following fields must be set before calling `.run()`:
//! - `baseline`: Closure that generates the baseline input (typically constant)
//! - `sample`: Closure that generates varied sample inputs
//! - `measure`: The test body that receives inputs and performs the operation
//!
//! # Optional Fields
//!
//! - `oracle`: Custom [`TimingOracle`] configuration (defaults to `TimingOracle::new()`)

use std::hash::Hash;
use std::hint::black_box;
use std::marker::PhantomData;

use crate::helpers::InputPair;
use crate::{Outcome, TimingOracle};

/// Builder for timing tests.
///
/// Use the builder pattern to configure and run a timing test:
///
/// ```ignore
/// let result = TimingTest::new()
///     .baseline(|| [0u8; 32])
///     .sample(|| rand::random::<[u8; 32]>())
///     .measure(|input| encrypt(&input))
///     .run();
/// ```
///
/// See the [module documentation](self) for details.
pub struct TimingTest<V, F1, F2, E> {
    oracle: Option<TimingOracle>,
    baseline_gen: Option<F1>,
    sample_gen: Option<F2>,
    measure_body: Option<E>,
    _phantom: PhantomData<V>,
}

impl TimingTest<(), (), (), ()> {
    /// Create a new timing test builder.
    ///
    /// All required fields (`baseline`, `sample`, `measure`) must be set before
    /// calling `.run()`.
    pub fn new() -> Self {
        TimingTest {
            oracle: None,
            baseline_gen: None,
            sample_gen: None,
            measure_body: None,
            _phantom: PhantomData,
        }
    }
}

impl Default for TimingTest<(), (), (), ()> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, F1, F2, E> TimingTest<V, F1, F2, E> {
    /// Set the [`TimingOracle`] configuration.
    ///
    /// This is optional. If not set, defaults to `TimingOracle::new()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// TimingTest::new()
    ///     .oracle(TimingOracle::balanced())  // Use balanced preset
    ///     .baseline(|| [0u8; 32])
    ///     // ...
    /// ```
    pub fn oracle(mut self, oracle: TimingOracle) -> Self {
        self.oracle = Some(oracle);
        self
    }
}

impl<F2, E> TimingTest<(), (), F2, E> {
    /// Set the baseline input generator.
    ///
    /// This closure is called to generate baseline inputs. Typically returns
    /// a constant value, but it's a closure for API symmetry with `sample()`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// TimingTest::new()
    ///     .baseline(|| [0u8; 32])  // Baseline: all zeros
    ///     // ...
    /// ```
    pub fn baseline<V, F1: FnMut() -> V>(self, gen: F1) -> TimingTest<V, F1, F2, E> {
        TimingTest {
            oracle: self.oracle,
            baseline_gen: Some(gen),
            sample_gen: self.sample_gen,
            measure_body: self.measure_body,
            _phantom: PhantomData,
        }
    }
}

impl<V, F1, E> TimingTest<V, F1, (), E> {
    /// Set the sample input generator.
    ///
    /// This closure is called once per sample to generate a fresh value.
    /// It must return the same type as the baseline generator.
    ///
    /// # Example
    ///
    /// ```ignore
    /// TimingTest::new()
    ///     .baseline(|| [0u8; 32])
    ///     .sample(|| rand::random::<[u8; 32]>())  // Generate varied bytes
    ///     // ...
    /// ```
    pub fn sample<NewF2: FnMut() -> V>(self, gen: NewF2) -> TimingTest<V, F1, NewF2, E> {
        TimingTest {
            oracle: self.oracle,
            baseline_gen: self.baseline_gen,
            sample_gen: Some(gen),
            measure_body: self.measure_body,
            _phantom: PhantomData,
        }
    }
}

impl<V, F1, F2> TimingTest<V, F1, F2, ()> {
    /// Set the measurement body.
    ///
    /// This closure receives a reference to the input (either baseline or sample)
    /// and performs the operation to be timed. It's called for each measurement.
    ///
    /// **Note:** The closure receives `&V` (reference) not `V` (owned) because
    /// inputs are pre-generated for accurate timing measurement.
    ///
    /// # Example
    ///
    /// ```ignore
    /// TimingTest::new()
    ///     .baseline(|| [0u8; 32])
    ///     .sample(|| rand::random::<[u8; 32]>())
    ///     .measure(|input| {
    ///         // `input` is &[u8; 32]
    ///         secret.ct_eq(input)
    ///     })
    ///     // ...
    /// ```
    pub fn measure<NewE: FnMut(&V)>(self, body: NewE) -> TimingTest<V, F1, F2, NewE> {
        TimingTest {
            oracle: self.oracle,
            baseline_gen: self.baseline_gen,
            sample_gen: self.sample_gen,
            measure_body: Some(body),
            _phantom: PhantomData,
        }
    }
}

impl<V, F1, F2, E> TimingTest<V, F1, F2, E>
where
    V: Clone + Hash,
    F1: FnMut() -> V,
    F2: FnMut() -> V,
    E: FnMut(&V),
{
    /// Run the timing test and return the result.
    ///
    /// # Panics
    ///
    /// Panics if any required field (`baseline`, `sample`, `measure`) is not set.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = TimingTest::new()
    ///     .baseline(|| [0u8; 32])
    ///     .sample(|| rand::random::<[u8; 32]>())
    ///     .measure(|input| encrypt(input))
    ///     .run();
    ///
    /// match result {
    ///     Outcome::Completed(r) => println!("Leak probability: {}", r.leak_probability),
    ///     Outcome::Unmeasurable { recommendation, .. } => println!("{}", recommendation),
    /// }
    /// ```
    pub fn run(self) -> Outcome {
        let baseline_gen = self
            .baseline_gen
            .expect("TimingTest: missing required field `baseline`. Call `.baseline(|| generator)` before `.run()`");

        let sample_gen = self
            .sample_gen
            .expect("TimingTest: missing required field `sample`. Call `.sample(|| generator)` before `.run()`");

        let mut measure_body = self
            .measure_body
            .expect("TimingTest: missing required field `measure`. Call `.measure(|input| body)` before `.run()`");

        let oracle = self.oracle.unwrap_or_else(TimingOracle::new);

        // Create InputPair for pre-generation
        let inputs = InputPair::new(baseline_gen, sample_gen);

        // Run the test - oracle handles pre-generation internally
        oracle.test(inputs, |input| {
            black_box(measure_body(black_box(input)));
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        // Use quick config to speed up test
        let result = TimingTest::new()
            .oracle(TimingOracle::quick())
            .baseline(|| 42u64)
            .sample(|| rand::random::<u64>())
            .measure(|x| {
                std::hint::black_box(*x);
            })
            .run();

        // Just verify the builder runs without error
        assert!(matches!(result, Outcome::Completed(_) | Outcome::Unmeasurable { .. }));
    }

    #[test]
    fn test_builder_with_oracle() {
        let result = TimingTest::new()
            .oracle(TimingOracle::quick())
            .baseline(|| [0u8; 32])
            .sample(|| rand::random::<[u8; 32]>())
            .measure(|input| {
                std::hint::black_box(input);
            })
            .run();

        assert!(matches!(result, Outcome::Completed(_) | Outcome::Unmeasurable { .. }));
    }

    // Note: Panic tests for missing fields are not possible because
    // the type system enforces that all required fields are set before
    // calling .run(). For example:
    //
    //   TimingTest::new()           // TimingTest<(), (), (), ()>
    //       .baseline(|| 42u64)     // TimingTest<u64, impl FnMut() -> u64, (), ()>
    //       .sample(|| 42u64)       // TimingTest<u64, ..., impl FnMut() -> u64, ()>
    //       .measure(|x| { ... })   // TimingTest<u64, ..., ..., impl FnMut(&u64)>
    //       .run()                  // Only compiles with all fields set
    //
    // If you try to skip a field, the types won't match and you'll get
    // a compile-time error rather than a runtime panic.
}
