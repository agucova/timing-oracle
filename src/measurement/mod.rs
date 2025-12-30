//! Measurement infrastructure for timing analysis.
//!
//! This module provides:
//! - High-resolution cycle counting with platform-specific implementations
//! - Sample collection with randomized interleaved design
//! - Symmetric outlier filtering for robust analysis
//!
//! # Timer Selection
//!
//! By default, timing uses platform timers:
//! - **x86_64**: `rdtsc` instruction (~1ns resolution)
//! - **aarch64**: `cntvct_el0` virtual timer (~41ns on Apple Silicon)
//!
//! On Apple Silicon, the library automatically batches iterations to compensate
//! for the coarse timer resolution. For cycle-accurate timing, enable the
//! `kperf` feature and run with sudo:
//!
//! ```toml
//! [dependencies]
//! timing-oracle = { version = "0.1", features = ["kperf"] }
//! ```
//!
//! ```bash
//! sudo cargo test
//! ```

mod collector;
mod outlier;
mod timer;

#[cfg(feature = "kperf")]
pub mod kperf;

pub use collector::{Collector, Sample};
pub use outlier::{filter_outliers, OutlierStats};
pub use timer::{black_box, cycles_per_ns, rdtsc, Timer};
