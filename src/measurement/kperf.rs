//! PMU-based cycle counting for Apple Silicon using kperf.
//!
//! This module provides cycle-accurate timing on Apple Silicon by accessing
//! hardware performance counters through Apple's private kperf framework.
//!
//! # Requirements
//!
//! - macOS on Apple Silicon (M1/M2/M3)
//! - **Must run with sudo/root privileges**
//! - Enable with `--features kperf`
//!
//! # Usage
//!
//! ```rust,ignore
//! use timing_oracle::measurement::kperf::PmuTimer;
//!
//! // Must run as root!
//! let timer = PmuTimer::new().expect("Failed to init PMU (run with sudo)");
//! let cycles = timer.measure_cycles(|| {
//!     // code to measure
//! });
//! ```
//!
//! # How it works
//!
//! Apple Silicon CPUs have performance monitoring counters (PMCs) that count
//! actual CPU cycles. These are accessed through the undocumented kperf framework.
//! Unlike the virtual timer (cntvct_el0) which runs at 24 MHz, PMCs run at CPU
//! frequency (~3 GHz), providing ~100x better resolution.

use std::sync::atomic::{compiler_fence, Ordering};

/// Error type for PMU initialization failures.
#[derive(Debug, Clone)]
pub enum PmuError {
    /// Not running on Apple Silicon
    UnsupportedPlatform,
    /// kperf framework not available
    FrameworkNotFound,
    /// Permission denied (need sudo)
    PermissionDenied,
    /// Counter configuration failed
    ConfigurationFailed(String),
}

impl std::fmt::Display for PmuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PmuError::UnsupportedPlatform => write!(f, "PMU timing requires Apple Silicon"),
            PmuError::FrameworkNotFound => write!(f, "kperf framework not found"),
            PmuError::PermissionDenied => write!(f, "Permission denied - run with sudo"),
            PmuError::ConfigurationFailed(msg) => write!(f, "PMU configuration failed: {}", msg),
        }
    }
}

impl std::error::Error for PmuError {}

/// PMU-based timer for cycle-accurate measurement on Apple Silicon.
///
/// This timer uses hardware performance counters to measure actual CPU cycles,
/// providing much better resolution than the virtual timer.
///
/// # Requirements
///
/// - Must run with sudo/root privileges
/// - Only works on Apple Silicon (M1/M2/M3)
#[derive(Debug)]
pub struct PmuTimer {
    /// Estimated cycles per nanosecond (CPU frequency in GHz)
    cycles_per_ns: f64,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod apple_silicon {
    use super::*;

    // kperf constants (reverse-engineered from Apple's private framework)
    const KPC_CLASS_FIXED: u32 = 0;
    const KPC_CLASS_CONFIGURABLE: u32 = 1;
    const KPC_CLASS_FIXED_MASK: u32 = 1 << KPC_CLASS_FIXED;
    const KPC_CLASS_CONFIGURABLE_MASK: u32 = 1 << KPC_CLASS_CONFIGURABLE;

    // ARM PMU event: CPU cycles
    #[allow(dead_code)]
    const ARMV8_PMCR_E: u64 = 1 << 0; // Enable (for reference)
    const CPMU_CORE_CYCLE: u64 = 0x02; // Core cycles event

    // Thread counter buffer size
    const KPC_MAX_COUNTERS: usize = 32;

    // Dynamic linking to kperf framework
    type KpcForceAllCtrsSet = unsafe extern "C" fn(i32) -> i32;
    type KpcSetCounting = unsafe extern "C" fn(u32) -> i32;
    type KpcSetThreadCounting = unsafe extern "C" fn(u32) -> i32;
    type KpcSetConfig = unsafe extern "C" fn(u32, *const u64) -> i32;
    type KpcGetThreadCounters = unsafe extern "C" fn(u32, u32, *mut u64) -> i32;
    type KpcGetCounterCount = unsafe extern "C" fn(u32) -> u32;
    type KpcGetConfigCount = unsafe extern "C" fn(u32) -> u32;

    struct KperfFunctions {
        force_all_ctrs_set: KpcForceAllCtrsSet,
        set_counting: KpcSetCounting,
        set_thread_counting: KpcSetThreadCounting,
        set_config: KpcSetConfig,
        get_thread_counters: KpcGetThreadCounters,
        get_counter_count: KpcGetCounterCount,
        get_config_count: KpcGetConfigCount,
    }

    use std::sync::OnceLock;
    static KPERF: OnceLock<Option<KperfFunctions>> = OnceLock::new();

    fn load_kperf() -> Result<&'static KperfFunctions, PmuError> {
        let kperf = KPERF.get_or_init(|| {
            unsafe {
                let lib = libc::dlopen(
                    b"/System/Library/PrivateFrameworks/kperf.framework/kperf\0".as_ptr() as *const i8,
                    libc::RTLD_NOW,
                );
                if lib.is_null() {
                    return None;
                }

                macro_rules! load_fn {
                    ($name:expr) => {{
                        let sym = libc::dlsym(lib, concat!($name, "\0").as_ptr() as *const i8);
                        if sym.is_null() {
                            return None;
                        }
                        std::mem::transmute(sym)
                    }};
                }

                Some(KperfFunctions {
                    force_all_ctrs_set: load_fn!("kpc_force_all_ctrs_set"),
                    set_counting: load_fn!("kpc_set_counting"),
                    set_thread_counting: load_fn!("kpc_set_thread_counting"),
                    set_config: load_fn!("kpc_set_config"),
                    get_thread_counters: load_fn!("kpc_get_thread_counters"),
                    get_counter_count: load_fn!("kpc_get_counter_count"),
                    get_config_count: load_fn!("kpc_get_config_count"),
                })
            }
        });

        kperf.as_ref().ok_or(PmuError::FrameworkNotFound)
    }

    impl PmuTimer {
        /// Initialize PMU counters for cycle counting.
        ///
        /// # Errors
        ///
        /// Returns an error if:
        /// - Not running on Apple Silicon
        /// - kperf framework not available
        /// - Not running with sudo/root privileges
        pub fn new() -> Result<Self, PmuError> {
            let kperf = load_kperf()?;

            unsafe {
                // Force all counters to be available (requires root)
                if (kperf.force_all_ctrs_set)(1) != 0 {
                    return Err(PmuError::PermissionDenied);
                }

                let classes = KPC_CLASS_FIXED_MASK | KPC_CLASS_CONFIGURABLE_MASK;

                // Get counter counts
                let n_configs = (kperf.get_config_count)(classes);
                if n_configs == 0 {
                    return Err(PmuError::ConfigurationFailed("No configurable counters".into()));
                }

                // Configure counters for cycle counting
                let mut config = vec![0u64; n_configs as usize];
                // First configurable counter: count cycles
                if !config.is_empty() {
                    config[0] = CPMU_CORE_CYCLE | (1 << 16); // Enable counting
                }

                if (kperf.set_config)(classes, config.as_ptr()) != 0 {
                    return Err(PmuError::ConfigurationFailed("kpc_set_config failed".into()));
                }

                // Enable counting
                if (kperf.set_counting)(classes) != 0 {
                    return Err(PmuError::ConfigurationFailed("kpc_set_counting failed".into()));
                }

                // Enable thread counting
                if (kperf.set_thread_counting)(classes) != 0 {
                    return Err(PmuError::ConfigurationFailed("kpc_set_thread_counting failed".into()));
                }
            }

            // Calibrate cycles per nanosecond
            let cycles_per_ns = Self::calibrate();

            Ok(Self { cycles_per_ns })
        }

        /// Read the current cycle count from PMU.
        #[inline]
        pub fn read_cycles(&self) -> u64 {
            let kperf = match load_kperf() {
                Ok(k) => k,
                Err(_) => return 0,
            };

            let mut counters = [0u64; KPC_MAX_COUNTERS];
            unsafe {
                let classes = KPC_CLASS_FIXED_MASK | KPC_CLASS_CONFIGURABLE_MASK;
                let n_counters = (kperf.get_counter_count)(classes);
                (kperf.get_thread_counters)(0, n_counters, counters.as_mut_ptr());
            }

            // First counter should be cycles
            counters[0]
        }

        fn calibrate() -> f64 {
            use std::time::Instant;

            let timer = match Self::new_uncalibrated() {
                Ok(t) => t,
                Err(_) => return 3.0, // Fallback
            };

            let mut ratios = Vec::with_capacity(10);
            for _ in 0..10 {
                let start_cycles = timer.read_cycles();
                let start_time = Instant::now();

                std::thread::sleep(std::time::Duration::from_millis(1));

                let end_cycles = timer.read_cycles();
                let elapsed_nanos = start_time.elapsed().as_nanos() as u64;

                if elapsed_nanos > 0 {
                    let cycles = end_cycles.saturating_sub(start_cycles);
                    ratios.push(cycles as f64 / elapsed_nanos as f64);
                }
            }

            if ratios.is_empty() {
                return 3.0;
            }

            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ratios[ratios.len() / 2]
        }

        fn new_uncalibrated() -> Result<Self, PmuError> {
            Ok(Self { cycles_per_ns: 1.0 })
        }

        /// Measure execution time in cycles.
        #[inline]
        pub fn measure_cycles<F, T>(&self, f: F) -> u64
        where
            F: FnOnce() -> T,
        {
            compiler_fence(Ordering::SeqCst);
            let start = self.read_cycles();
            std::hint::black_box(f());
            let end = self.read_cycles();
            compiler_fence(Ordering::SeqCst);
            end.saturating_sub(start)
        }

        /// Convert cycles to nanoseconds.
        #[inline]
        pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
            cycles as f64 / self.cycles_per_ns
        }

        /// Get the calibrated cycles per nanosecond.
        pub fn cycles_per_ns(&self) -> f64 {
            self.cycles_per_ns
        }

        /// Get the timer resolution in nanoseconds (~0.3ns for 3GHz CPU).
        pub fn resolution_ns(&self) -> f64 {
            1.0 / self.cycles_per_ns
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl PmuTimer {
    /// PMU timer is only available on Apple Silicon.
    pub fn new() -> Result<Self, PmuError> {
        Err(PmuError::UnsupportedPlatform)
    }

    #[inline]
    pub fn read_cycles(&self) -> u64 {
        0
    }

    #[inline]
    pub fn measure_cycles<F, T>(&self, _f: F) -> u64
    where
        F: FnOnce() -> T,
    {
        0
    }

    #[inline]
    pub fn cycles_to_ns(&self, cycles: u64) -> f64 {
        cycles as f64
    }

    pub fn cycles_per_ns(&self) -> f64 {
        1.0
    }

    pub fn resolution_ns(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn test_pmu_timer_requires_root() {
        // This test documents the expected behavior
        match PmuTimer::new() {
            Ok(_) => {
                // Running as root - timer should work
                eprintln!("PMU timer initialized (running as root)");
            }
            Err(PmuError::PermissionDenied) => {
                // Expected when not running as root
                eprintln!("PMU timer requires root (expected)");
            }
            Err(e) => {
                eprintln!("PMU timer error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    fn test_pmu_unsupported_platform() {
        assert!(matches!(PmuTimer::new(), Err(PmuError::UnsupportedPlatform)));
    }
}
