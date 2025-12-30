//! Utilities for pre-generating test inputs correctly.
//!
//! The most common mistake when using timing-oracle is calling RNG functions
//! or allocating memory inside the measured closures. This creates timing
//! overhead that drowns out the actual signal.
//!
//! These helpers make it easy to pre-generate inputs with identical code paths.
//!
//! # Example
//!
//! ```ignore
//! use timing_oracle::{test, helpers::InputPair};
//!
//! // Pre-generate inputs - RNG called BEFORE measurement
//! let inputs = InputPair::new(
//!     [0u8; 32],                    // Fixed value
//!     || rand::random::<[u8; 32]>() // Generator called per sample
//! );
//!
//! let result = test(
//!     || my_function(&inputs.fixed()),   // Identical code path
//!     || my_function(&inputs.random()),  // Identical code path
//! );
//! ```

use std::cell::Cell;
use std::rc::Rc;

/// Manages pre-generated fixed and random inputs for timing tests.
///
/// Automatically handles indexing and ensures both closures execute
/// identical code paths (only the data differs).
///
/// # Type Parameters
///
/// - `T`: The input type (e.g., `[u8; 32]`, `Vec<u8>`, etc.)
///
/// # Example
///
/// ```ignore
/// use timing_oracle::helpers::InputPair;
///
/// // Bytes
/// let bytes = InputPair::new([0u8; 32], || rand::random());
///
/// // Vectors
/// let vecs = InputPair::from_fn(
///     || vec![0u8; 1024],
///     || (0..1024).map(|_| rand::random()).collect()
/// );
///
/// // Use in test
/// test(
///     || encrypt(&bytes.fixed()),
///     || encrypt(&bytes.random()),
/// );
/// ```
pub struct InputPair<T> {
    fixed: Vec<T>,
    random: Vec<T>,
    fixed_index: Rc<Cell<usize>>,
    random_index: Rc<Cell<usize>>,
}

impl<T: Clone> InputPair<T> {
    /// Create with a fixed value and a generator for random values.
    ///
    /// The fixed value is cloned `samples` times (default 100k).
    /// The random generator is called `samples` times to pre-generate all inputs.
    ///
    /// # Arguments
    ///
    /// * `fixed_value` - The fixed input (cloned for all samples)
    /// * `random_gen` - Function that generates one random input
    ///
    /// # Example
    ///
    /// ```ignore
    /// let inputs = InputPair::new(
    ///     [0u8; 32],
    ///     || rand::random::<[u8; 32]>()
    /// );
    /// ```
    pub fn new<F>(fixed_value: T, random_gen: F) -> Self
    where
        F: Fn() -> T,
    {
        Self::with_samples(100_000, fixed_value, random_gen)
    }

    /// Create with explicit sample count.
    ///
    /// Use this to match your `TimingOracle` configuration:
    ///
    /// ```ignore
    /// let inputs = InputPair::with_samples(
    ///     50_000,  // Match .samples(50_000)
    ///     [0u8; 32],
    ///     || rand::random()
    /// );
    /// ```
    pub fn with_samples<F>(samples: usize, fixed_value: T, random_gen: F) -> Self
    where
        F: Fn() -> T,
    {
        let fixed = vec![fixed_value; samples];
        let random = (0..samples).map(|_| random_gen()).collect();

        Self {
            fixed,
            random,
            fixed_index: Rc::new(Cell::new(0)),
            random_index: Rc::new(Cell::new(0)),
        }
    }

    /// Create from separate generators for both fixed and random.
    ///
    /// Use when the fixed input also needs generation logic:
    ///
    /// ```ignore
    /// let inputs = InputPair::from_fn(
    ///     || vec![0u8; 1024],      // Fixed generator
    ///     || rand::random_vec(1024) // Random generator
    /// );
    /// ```
    pub fn from_fn<FF, FR>(fixed_gen: FF, random_gen: FR) -> Self
    where
        FF: Fn() -> T,
        FR: Fn() -> T,
    {
        Self::from_fn_with_samples(100_000, fixed_gen, random_gen)
    }

    /// Create from generators with explicit sample count.
    pub fn from_fn_with_samples<FF, FR>(
        samples: usize,
        fixed_gen: FF,
        random_gen: FR,
    ) -> Self
    where
        FF: Fn() -> T,
        FR: Fn() -> T,
    {
        let fixed = (0..samples).map(|_| fixed_gen()).collect();
        let random = (0..samples).map(|_| random_gen()).collect();

        Self {
            fixed,
            random,
            fixed_index: Rc::new(Cell::new(0)),
            random_index: Rc::new(Cell::new(0)),
        }
    }

    /// Get the next fixed input.
    ///
    /// Intended for single-threaded use; relies on `Cell` for interior mutability.
    /// Wraps around when reaching the end.
    /// Uses a separate index from `random()` so alternating calls work correctly.
    #[inline]
    pub fn fixed(&self) -> &T {
        let i = self.fixed_index.get();
        self.fixed_index.set(i.wrapping_add(1));
        &self.fixed[i % self.fixed.len()]
    }

    /// Get the next random input.
    ///
    /// Intended for single-threaded use; relies on `Cell` for interior mutability.
    /// Wraps around when reaching the end.
    /// Uses a separate index from `fixed()` so alternating calls work correctly.
    #[inline]
    pub fn random(&self) -> &T {
        let i = self.random_index.get();
        self.random_index.set(i.wrapping_add(1));
        &self.random[i % self.random.len()]
    }

    /// Reset both index counters to 0.
    ///
    /// Rarely needed - mainly for testing the helper itself.
    pub fn reset(&self) {
        self.fixed_index.set(0);
        self.random_index.set(0);
    }
}

// Convenience constructors for common types

/// Helper for byte arrays.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::helpers;
///
/// let inputs = helpers::byte_arrays_32();
/// // Equivalent to InputPair::new([0u8; 32], || rand::random())
/// ```
pub fn byte_arrays_32() -> InputPair<[u8; 32]> {
    InputPair::new([0u8; 32], rand::random)
}

/// Helper for byte vectors of specific length.
///
/// # Example
///
/// ```ignore
/// use timing_oracle::helpers;
///
/// let inputs = helpers::byte_vecs(1024);
/// test(
///     || encrypt(&inputs.fixed()),
///     || encrypt(&inputs.random()),
/// );
/// ```
pub fn byte_vecs(len: usize) -> InputPair<Vec<u8>> {
    InputPair::from_fn(
        || vec![0u8; len],
        || (0..len).map(|_| rand::random()).collect(),
    )
}
