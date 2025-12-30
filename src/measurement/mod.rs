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
//! - **aarch64**: `cntvct_el0` virtual timer (resolution varies by SoC)
//!
//! ARM64 timer resolution depends on the SoC's counter frequency:
//! - ARMv8.6+ (Graviton4): ~1ns (1 GHz mandated by spec)
//! - Apple Silicon: ~42ns (24 MHz)
//! - Ampere Altra: ~40ns (25 MHz)
//! - Raspberry Pi 4: ~18ns (54 MHz)
//!
//! On ARM64 with coarse timers, the library automatically batches iterations
//! to compensate for timer resolution. On macOS, for cycle-accurate timing,
//! enable the `kperf` feature and run with sudo:
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

#[cfg(feature = "perf")]
pub mod perf;

pub use collector::{Collector, Sample};
pub use outlier::{filter_outliers, OutlierStats};
pub use timer::{black_box, cycles_per_ns, rdtsc, Timer};
