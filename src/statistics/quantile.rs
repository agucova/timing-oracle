//! Quantile computation using O(n) selection algorithms.
//!
//! This module provides efficient quantile computation using Rust's
//! `slice.select_nth_unstable()` which uses introselect for O(n) average time.

use crate::constants::DECILES;
use crate::types::Vector9;

/// Compute a single quantile from a mutable slice.
///
/// Uses `select_nth_unstable()` for O(n) expected time complexity.
/// The slice is partially reordered as a side effect.
///
/// # Arguments
///
/// * `data` - Mutable slice of measurements (will be partially reordered)
/// * `p` - Quantile probability in [0, 1]
///
/// # Returns
///
/// The quantile value at probability `p`.
///
/// # Panics
///
/// Panics if `data` is empty or if `p` is outside [0, 1].
pub fn compute_quantile(data: &mut [f64], p: f64) -> f64 {
    assert!(!data.is_empty(), "Cannot compute quantile of empty slice");
    assert!(
        (0.0..=1.0).contains(&p),
        "Quantile probability must be in [0, 1]"
    );

    let n = data.len();

    // Handle edge cases
    if n == 1 {
        return data[0];
    }

    // Compute the index for the quantile
    // Using the "R-7" quantile definition (linear interpolation)
    let h = (n - 1) as f64 * p;
    let h_floor = h.floor() as usize;
    let h_frac = h - h.floor();

    if h_floor >= n - 1 {
        // At or beyond the last element
        let (_, &mut max, _) = data.select_nth_unstable_by(n - 1, |a, b| a.total_cmp(b));
        return max;
    }

    // Get the lower value using select_nth_unstable
    let (_, &mut lower, upper) = data.select_nth_unstable_by(h_floor, |a, b| a.total_cmp(b));

    if h_frac == 0.0 {
        return lower;
    }

    // Find the minimum of the upper partition for interpolation
    let upper_min = upper
        .iter()
        .copied()
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(lower);

    // Linear interpolation between lower and upper
    lower + h_frac * (upper_min - lower)
}

/// Compute all 9 deciles [0.1, 0.2, ..., 0.9] from timing measurements.
///
/// Returns a Vector9 containing the quantile values at each decile.
/// The input slice is cloned and sorted once for efficiency.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements
///
/// # Returns
///
/// A `Vector9` with decile values at positions 0-8 corresponding to
/// quantiles 0.1-0.9.
///
/// # Panics
///
/// Panics if `data` is empty.
pub fn compute_deciles(data: &[f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Sort once, then compute all quantiles from sorted data - O(n log n) total
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    compute_deciles_sorted(&sorted)
}

/// Compute all 9 deciles by sorting and indexing.
///
/// Uses unstable sort (O(n log n)) followed by direct indexing for deciles.
/// This is semantically identical to `compute_deciles()` but uses unstable
/// sort which is faster when stability is not needed.
///
/// Uses the same R-7 quantile definition as `compute_deciles_sorted` for
/// exact result matching, including linear interpolation.
///
/// # Arguments
///
/// * `data` - Slice of timing measurements
///
/// # Returns
///
/// A `Vector9` with decile values at positions 0-8 corresponding to
/// quantiles 0.1-0.9.
///
/// # Panics
///
/// Panics if `data` is empty.
///
/// # Note
///
/// Despite the name, this function uses O(n log n) sorting, not O(n) selection.
/// The name is historical; true multi-quantile selection was not significantly
/// faster in practice due to overhead.
pub fn compute_deciles_fast(data: &[f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Use unstable sort which is faster than stable sort (don't need stability)
    let mut working = data.to_vec();
    working.sort_unstable_by(|a, b| a.total_cmp(b));

    compute_deciles_sorted(&working)
}

/// Compute deciles with a reusable buffer to avoid allocations.
///
/// This is an optimization for hot loops where `compute_deciles_fast` is called
/// repeatedly. By reusing the same buffer, we avoid repeated allocations.
///
/// # Arguments
///
/// * `data` - Input data slice
/// * `buffer` - Reusable working buffer (will be resized as needed)
///
/// # Returns
///
/// A `Vector9` with decile values.
pub fn compute_deciles_with_buffer(data: &[f64], buffer: &mut Vec<f64>) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Reuse buffer, avoiding allocation if it's already large enough
    buffer.clear();
    buffer.extend_from_slice(data);
    buffer.sort_unstable_by(|a, b| a.total_cmp(b));

    compute_deciles_sorted(buffer)
}

/// Compute deciles by sorting a mutable slice in-place.
///
/// This is the most efficient version for hot loops where you already have
/// the data in a mutable buffer and don't need to preserve the unsorted order.
/// It sorts the buffer in-place (no allocations) and reads deciles directly.
///
/// # Arguments
///
/// * `data` - Mutable slice that will be sorted in-place
///
/// # Returns
///
/// A `Vector9` with decile values.
///
/// # Note
///
/// After this call, `data` will be sorted in ascending order.
pub fn compute_deciles_inplace(data: &mut [f64]) -> Vector9 {
    assert!(!data.is_empty(), "Cannot compute deciles of empty slice");

    // Sort in-place (no allocation)
    data.sort_unstable_by(|a, b| a.total_cmp(b));

    // Read deciles from sorted data
    compute_deciles_sorted(data)
}

/// Compute all 9 deciles from pre-sorted timing measurements.
///
/// This is an optimization when you need to compute deciles multiple times
/// on the same data - sort once and reuse.
///
/// # Arguments
///
/// * `sorted` - Slice of timing measurements that MUST be sorted in ascending order
///
/// # Returns
///
/// A `Vector9` with decile values at positions 0-8 corresponding to
/// quantiles 0.1-0.9.
///
/// # Panics
///
/// Panics if `sorted` is empty.
///
/// # Safety
///
/// The caller must ensure the data is sorted. No verification is performed.
pub fn compute_deciles_sorted(sorted: &[f64]) -> Vector9 {
    assert!(!sorted.is_empty(), "Cannot compute deciles of empty slice");

    let n = sorted.len();
    let mut result = Vector9::zeros();

    for (i, &p) in DECILES.iter().enumerate() {
        // Compute index using R-7 quantile definition (linear interpolation)
        let h = (n - 1) as f64 * p;
        let h_floor = h.floor() as usize;
        let h_frac = h - h.floor();

        if h_floor >= n - 1 {
            result[i] = sorted[n - 1];
        } else if h_frac == 0.0 {
            result[i] = sorted[h_floor];
        } else {
            // Linear interpolation
            result[i] = sorted[h_floor] + h_frac * (sorted[h_floor + 1] - sorted[h_floor]);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_quantile_median() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = compute_quantile(&mut data, 0.5);
        assert!((median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_quantile_extremes() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let min = compute_quantile(&mut data.clone(), 0.0);
        let max = compute_quantile(&mut data, 1.0);
        assert!((min - 1.0).abs() < 1e-10);
        assert!((max - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_deciles_sorted() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let deciles = compute_deciles(&data);

        // Check that deciles are monotonically increasing
        for i in 1..9 {
            assert!(deciles[i] >= deciles[i - 1]);
        }
    }

    #[test]
    fn test_compute_deciles_fast_matches_sort() {
        // Test on sequential data
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-10,
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }
    }

    #[test]
    fn test_compute_deciles_fast_random_data() {
        // Test on random-ish data
        let data: Vec<f64> = vec![
            3.7, 1.2, 9.5, 2.1, 7.3, 4.8, 6.2, 8.9, 1.5, 5.4, 2.7, 9.1, 3.3, 6.8, 4.5, 7.9, 2.4,
            8.3, 5.7, 1.9,
        ];
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-10,
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }

        // Verify monotonicity
        for i in 1..9 {
            assert!(deciles_fast[i] >= deciles_fast[i - 1]);
        }
    }

    #[test]
    fn test_compute_deciles_fast_small_data() {
        // Test edge case: small dataset
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-10,
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }
    }

    #[test]
    fn test_compute_deciles_fast_large_data() {
        // Test on larger dataset (typical bootstrap size)
        let data: Vec<f64> = (0..20000).map(|x| (x as f64 * 1.234) % 1000.0).collect();
        let deciles_sort = compute_deciles(&data);
        let deciles_fast = compute_deciles_fast(&data);

        for i in 0..9 {
            let diff = (deciles_sort[i] - deciles_fast[i]).abs();
            assert!(
                diff < 1e-8, // Slightly relaxed tolerance for large data
                "Decile {} differs: sort={}, fast={}, diff={}",
                i,
                deciles_sort[i],
                deciles_fast[i],
                diff
            );
        }
    }

    #[test]
    #[should_panic(expected = "Cannot compute quantile of empty slice")]
    fn test_empty_slice_panics() {
        let mut data: Vec<f64> = vec![];
        compute_quantile(&mut data, 0.5);
    }

    #[test]
    #[should_panic(expected = "Cannot compute deciles of empty slice")]
    fn test_compute_deciles_fast_empty_panics() {
        let data: Vec<f64> = vec![];
        compute_deciles_fast(&data);
    }
}
