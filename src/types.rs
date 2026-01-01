//! Type aliases and common types.

use nalgebra::{SMatrix, SVector};

/// 9x9 covariance matrix for quantile differences.
pub type Matrix9 = SMatrix<f64, 9, 9>;

/// 9-dimensional vector for quantile differences.
pub type Vector9 = SVector<f64, 9>;

/// 9x2 design matrix [ones | b_tail] for effect decomposition.
pub type Matrix9x2 = SMatrix<f64, 9, 2>;

/// 2x2 matrix for effect covariance.
pub type Matrix2 = SMatrix<f64, 2, 2>;

/// 2-dimensional vector for effect parameters (shift, tail).
pub type Vector2 = SVector<f64, 2>;

/// Input class identifier for timing measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Baseline input (typically constant) that establishes the reference timing.
    Baseline,
    /// Sample input (typically varied) for comparison against baseline.
    Sample,
}

/// A timing sample with its class label, preserving measurement order.
///
/// Used for joint resampling in covariance estimation, which preserves
/// temporal pairing between baseline and sample measurements.
#[derive(Debug, Clone, Copy)]
pub struct TimingSample {
    /// Timing value in nanoseconds.
    pub time_ns: f64,
    /// Which class this sample belongs to.
    pub class: Class,
}
