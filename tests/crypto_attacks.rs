//! Crypto attack integration tests - validates detection of realistic timing vulnerabilities.
//!
//! Tests cover:
//! - Cache-based attacks (AES S-box, cache lines, memory access patterns)
//! - Algorithmic attacks (modular exponentiation, bit patterns)
//! - Table lookup timing (L1/L2/L3 cache)
//! - Effect pattern validation (uniform shift, tail effect, mixed)
//! - Exploitability classification (Crosby et al. thresholds)

use num_bigint::BigUint;
use num_traits::{One, Zero};
use timing_oracle::helpers::InputPair;
use timing_oracle::TimingOracle;

// ============================================================================
// Helper Functions Module
// ============================================================================

mod helpers {
    use super::*;

    /// Naive modular exponentiation - intentionally leaky!
    ///
    /// Uses square-and-multiply algorithm where timing depends on
    /// Hamming weight of the exponent (number of 1 bits).
    pub fn modpow_naive(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
        let mut result = BigUint::one();
        let mut base = base.clone();
        let mut exp = exp.clone();

        while exp > BigUint::zero() {
            if &exp & BigUint::one() == BigUint::one() {
                result = (result * &base) % modulus;
            }
            base = (&base * &base) % modulus;
            exp >>= 1;
        }

        result
    }

    /// Generate a simplified AES S-box or permutation table
    pub fn generate_sbox() -> [u8; 256] {
        let mut sbox = [0u8; 256];
        for i in 0..256 {
            // Simple permutation based on bit reversal and XOR
            let mut x = i as u8;
            x = (x & 0xF0) >> 4 | (x & 0x0F) << 4;
            x = (x & 0xCC) >> 2 | (x & 0x33) << 2;
            x = (x & 0xAA) >> 1 | (x & 0x55) << 1;
            sbox[i] = x ^ 0x63;
        }
        sbox
    }

    /// Platform-specific busy-wait for controlled timing delays
    #[cfg(target_arch = "x86_64")]
    pub fn busy_wait_cycles(cycles: u64) {
        let start = unsafe { core::arch::x86_64::_rdtsc() };
        while unsafe { core::arch::x86_64::_rdtsc() } - start < cycles {
            std::hint::spin_loop();
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn busy_wait_cycles(cycles: u64) {
        let start = unsafe {
            let cnt: u64;
            core::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt);
            cnt
        };
        loop {
            let now = unsafe {
                let cnt: u64;
                core::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt);
                cnt
            };
            if now - start >= cycles {
                break;
            }
            std::hint::spin_loop();
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn busy_wait_cycles(_cycles: u64) {
        // Fallback: just use a fixed delay
        for _ in 0..100 {
            std::hint::black_box(std::hint::spin_loop());
        }
    }

    /// Cache line size aligned buffer for cache timing tests
    #[repr(align(64))]
    pub struct CacheAligned<T> {
        pub data: T,
    }

    impl<T> CacheAligned<T> {
        pub fn new(data: T) -> Self {
            Self { data }
        }
    }
}

// ============================================================================
// Category 1: Cache-Based Attacks
// ============================================================================

/// 1.1 AES S-box Timing (Fast) - Should detect cache-based timing
#[test]
fn aes_sbox_timing_fast() {
    let sbox = helpers::generate_sbox();
    let secret_key = 0xABu8;

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let indices = InputPair::with_samples(SAMPLES, secret_key, rand::random::<u8>);

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                // Fixed: always use secret_key
                let val = std::hint::black_box(*indices.fixed());
                std::hint::black_box(sbox[val as usize])
            },
            || {
                // Random: use random byte
                let val = std::hint::black_box(*indices.random());
                std::hint::black_box(sbox[val as usize])
            },
        );

    eprintln!("\n[aes_sbox_timing_fast]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // On many systems, S-box lookups show cache effects
    // We expect moderate leak probability or TailEffect pattern
    let has_tail_effect = result.effect.as_ref()
        .map(|e| matches!(e.pattern, timing_oracle::EffectPattern::TailEffect))
        .unwrap_or(false);

    assert!(
        result.leak_probability > 0.3 || has_tail_effect,
        "Expected to detect some cache timing effect (got leak_probability={})",
        result.leak_probability
    );
}

/// 1.2 AES S-box Timing (Thorough) - High confidence detection
#[test]
#[ignore = "slow test - run with --ignored"]
fn aes_sbox_timing_thorough() {
    let sbox = helpers::generate_sbox();
    let secret_key = 0xABu8;

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 100_000;
    let indices = InputPair::with_samples(SAMPLES, secret_key, rand::random::<u8>);

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .ci_alpha(0.01)
        .test(
            || {
                let val = std::hint::black_box(*indices.fixed());
                std::hint::black_box(sbox[val as usize])
            },
            || {
                let val = std::hint::black_box(*indices.random());
                std::hint::black_box(sbox[val as usize])
            },
        );

    eprintln!("\n[aes_sbox_timing_thorough]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // With 100k samples, cache effects should be more pronounced
    assert!(
        result.leak_probability > 0.5 || !result.ci_gate.passed,
        "Expected high confidence leak detection (got leak_probability={}, ci_gate.passed={})",
        result.leak_probability,
        result.ci_gate.passed
    );
}

/// 1.3 Cache Line Boundary Effects
#[test]
fn cache_line_boundary_effects() {
    // Create a large buffer with cache-line aligned sections
    let buffer = vec![0u8; 4096];
    let secret_offset_same_line = 0usize;
    let secret_offset_diff_line = 64usize; // Different cache line

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let indices = InputPair::from_fn_with_samples(
        SAMPLES,
        || secret_offset_same_line,
        || secret_offset_diff_line + (rand::random::<u32>() as usize % 4) * 64,
    );

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                // Access same cache line repeatedly
                let idx = std::hint::black_box(*indices.fixed());
                std::hint::black_box(buffer[idx])
            },
            || {
                // Access different cache lines
                let idx = std::hint::black_box(*indices.random());
                std::hint::black_box(buffer[idx % buffer.len()])
            },
        );

    eprintln!("\n[cache_line_boundary_effects]");
    eprintln!("{}", timing_oracle::output::format_result(&result));
}

/// 1.4 Memory Access Pattern Leak
#[test]
fn memory_access_pattern_leak() {
    use std::cell::Cell;

    let data = vec![rand::random::<u64>(); 1024];
    let secret_pattern = [0usize, 64, 128, 192]; // Sequential in large strides

    // Pre-generate access indices using InputPair
    const SAMPLES: usize = 8_000;
    let data_len = data.len();
    let pattern_idx = Cell::new(0usize);
    let indices = InputPair::from_fn_with_samples(
        SAMPLES,
        || {
            // Cycle through secret_pattern
            let i = pattern_idx.get();
            let access_idx = secret_pattern[i % secret_pattern.len()];
            pattern_idx.set((i + 1) % secret_pattern.len());
            access_idx
        },
        || rand::random::<u32>() as usize % data_len,
    );

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                // Sequential access pattern (predictable for CPU)
                let access_idx = *indices.fixed();
                std::hint::black_box(data[access_idx])
            },
            || {
                // Random access pattern (less predictable)
                let access_idx = std::hint::black_box(*indices.random());
                std::hint::black_box(data[access_idx])
            },
        );

    eprintln!("\n[memory_access_pattern_leak]");
    eprintln!("{}", timing_oracle::output::format_result(&result));
}

// ============================================================================
// Category 2: Modular Exponentiation
// ============================================================================

/// 2.1 Square-and-Multiply Timing (Fast) - Should detect algorithmic timing
#[test]
fn modexp_square_and_multiply_timing() {
    let base = BigUint::from(5u32);
    let modulus = BigUint::from(1000000007u64); // Large prime

    // High Hamming weight exponent (many 1 bits = more multiplies)
    let exp_high_hamming = BigUint::from(0xFFFFu32); // 16 ones
    // Low Hamming weight exponent (few 1 bits = fewer multiplies)
    let exp_low_hamming = BigUint::from(0x8001u32); // 2 ones

    let result = TimingOracle::new()
        .samples(8_000)
        .test(
            || std::hint::black_box(helpers::modpow_naive(&base, &exp_high_hamming, &modulus)),
            || std::hint::black_box(helpers::modpow_naive(&base, &exp_low_hamming, &modulus)),
        );

    // Print full formatted output
    eprintln!("\n{}", timing_oracle::output::format_result(&result));

    // Square-and-multiply creates a uniform shift (more iterations = more time)
    assert!(
        result.leak_probability > 0.8,
        "Expected to detect modexp timing leak (got {})",
        result.leak_probability
    );

    // Should show UniformShift or Mixed pattern (BigInt ops may have variance)
    if let Some(ref effect) = result.effect {
        assert!(
            matches!(effect.pattern, timing_oracle::EffectPattern::UniformShift | timing_oracle::EffectPattern::Mixed),
            "Expected UniformShift or Mixed pattern for algorithmic timing (got {:?})",
            effect.pattern
        );

        // Should have significant shift
        assert!(
            effect.shift_ns > 50.0,
            "Expected significant shift_ns (got {:.1})",
            effect.shift_ns
        );
    } else {
        panic!("Expected effect data for high leak probability test");
    }
}

/// 2.2 Exponent Bit Pattern Timing
#[test]
fn modexp_bit_pattern_timing() {
    let base = BigUint::from(7u32);
    let modulus = BigUint::from(1000000007u64);

    // Create random exponents with controlled Hamming weights
    let exp_many_ones = BigUint::from(0xAAAAAAAAu32); // 50% ones
    let exp_few_ones = BigUint::from(0x80000001u32); // ~6% ones

    let result = TimingOracle::new()
        .samples(6_000)
        .test(
            || std::hint::black_box(helpers::modpow_naive(&base, &exp_many_ones, &modulus)),
            || std::hint::black_box(helpers::modpow_naive(&base, &exp_few_ones, &modulus)),
        );

    eprintln!("\n[modexp_bit_pattern_timing]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        result.leak_probability > 0.8,
        "Expected high leak probability for bit pattern timing (got {})",
        result.leak_probability
    );
}

// ============================================================================
// Category 3: Table Lookup Timing
// ============================================================================

/// 3.1 Small Table (L1 Cache) - Should show minimal timing
///
/// Uses DudeCT's two-class pattern: all-zero index vs random index
#[test]
fn table_lookup_small_l1() {
    let table = [rand::random::<u64>(); 4]; // 32 bytes, fits in L1

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let table_len = table.len();
    let indices = InputPair::from_fn_with_samples(
        SAMPLES,
        || 0usize,
        || rand::random::<u32>() as usize % table_len,
    );

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                // DudeCT Class 0: All zeros
                let idx = std::hint::black_box(*indices.fixed());
                std::hint::black_box(table[idx])
            },
            || {
                // DudeCT Class 1: Random
                let idx = std::hint::black_box(*indices.random());
                std::hint::black_box(table[idx])
            },
        );

    eprintln!("\n[table_lookup_small_l1]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // L1-resident small table: may detect timing difference but should be Negligible
    assert!(
        matches!(result.exploitability, timing_oracle::Exploitability::Negligible),
        "Expected Negligible exploitability for L1-resident table (got {:?})",
        result.exploitability
    );
}

/// 3.2 Medium Table (L2/L3 Cache) - May show cache effects
#[test]
fn table_lookup_medium_l2() {
    let table = vec![rand::random::<u64>(); 32]; // 256 bytes (AES S-box size)

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let table_len = table.len();
    let indices = InputPair::from_fn_with_samples(
        SAMPLES,
        || 0usize,
        || rand::random::<u32>() as usize % table_len,
    );

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                let idx = std::hint::black_box(*indices.fixed());
                std::hint::black_box(table[idx])
            },
            || {
                let idx = std::hint::black_box(*indices.random());
                std::hint::black_box(table[idx])
            },
        );

    eprintln!("\n[table_lookup_medium_l2]");
    eprintln!("{}", timing_oracle::output::format_result(&result));
}

/// 3.3 Large Table (Cache Thrashing) - Should show cache effects
#[test]
fn table_lookup_large_cache_thrash() {
    let table = vec![rand::random::<u64>(); 512]; // 4KB

    // Pre-generate indices using InputPair
    const SAMPLES: usize = 10_000;
    let table_len = table.len();
    let indices = InputPair::from_fn_with_samples(
        SAMPLES,
        || 0usize,
        || rand::random::<u32>() as usize % table_len,
    );

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                let idx = std::hint::black_box(*indices.fixed());
                std::hint::black_box(table[idx])
            },
            || {
                let idx = std::hint::black_box(*indices.random());
                std::hint::black_box(table[idx])
            },
        );

    eprintln!("\n[table_lookup_large_cache_thrash]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    // Large table should show cache timing effects
    let has_tail_effect = result.effect.as_ref()
        .map(|e| matches!(e.pattern, timing_oracle::EffectPattern::TailEffect))
        .unwrap_or(false);

    assert!(
        result.leak_probability > 0.4 || has_tail_effect,
        "Expected cache effects for large table (got leak_probability={})",
        result.leak_probability
    );
}

// ============================================================================
// Category 4: Effect Pattern Validation
// ============================================================================

/// 4.1 Pure Uniform Shift - Validates UniformShift classification
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn effect_pattern_pure_uniform_shift() {
    // Use a larger delay to ensure uniform shift dominates any measurement noise
    const DELAY_CYCLES: u64 = 500;

    let result = TimingOracle::new()
        .samples(10_000)
        .test(
            || {
                // No delay
                std::hint::black_box(42)
            },
            || {
                // Constant delay
                helpers::busy_wait_cycles(DELAY_CYCLES);
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[effect_pattern_pure_uniform_shift]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    if let Some(ref effect) = result.effect {
        // Should classify as UniformShift (or Mixed with dominant shift)
        // On real hardware, constant delays may have small jitter that creates a minor tail
        assert!(
            matches!(
                effect.pattern,
                timing_oracle::EffectPattern::UniformShift | timing_oracle::EffectPattern::Mixed
            ),
            "Expected UniformShift or Mixed pattern (got {:?})",
            effect.pattern
        );

        // Should have significant shift component
        // Use abs() since sign depends on which class is slower
        assert!(
            effect.shift_ns.abs() > 50.0,
            "Expected significant shift component (got {:.1}ns)",
            effect.shift_ns
        );

        // Shift should dominate tail (at least 5x larger)
        assert!(
            effect.shift_ns.abs() > effect.tail_ns.abs() * 5.0,
            "Expected shift to dominate tail (got shift={:.1}ns, tail={:.1}ns)",
            effect.shift_ns,
            effect.tail_ns
        );
    } else {
        panic!("Expected effect data for uniform shift test");
    }
}

/// 4.2 Pure Tail Effect - Validates TailEffect classification
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn effect_pattern_pure_tail() {
    use std::cell::Cell;

    // Use larger values to ensure tail effect is measurable
    const EXPENSIVE_CYCLES: u64 = 2000;
    const TAIL_PROBABILITY: f64 = 0.15;
    const SAMPLES: usize = 10_000;

    // Pre-generate spike decisions - both closures use identical code paths
    let spike_decisions: Vec<bool> = (0..SAMPLES)
        .map(|_| rand::random::<f64>() < TAIL_PROBABILITY)
        .collect();
    let idx = Cell::new(0usize);

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                // No expensive operation regardless of decision
                let _ = spike_decisions[i % SAMPLES];
                std::hint::black_box(42)
            },
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                // Apply spike based on pre-generated decision
                if spike_decisions[i % SAMPLES] {
                    helpers::busy_wait_cycles(EXPENSIVE_CYCLES);
                }
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[effect_pattern_pure_tail]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    if let Some(ref effect) = result.effect {
        // Should classify as TailEffect (or Mixed with dominant tail)
        // On real hardware, probabilistic delays may have some shift component
        assert!(
            matches!(
                effect.pattern,
                timing_oracle::EffectPattern::TailEffect | timing_oracle::EffectPattern::Mixed
            ),
            "Expected TailEffect or Mixed pattern (got {:?})",
            effect.pattern
        );

        // Should have significant tail component
        assert!(
            effect.tail_ns.abs() > 20.0,
            "Expected significant tail component (got {:.1}ns)",
            effect.tail_ns
        );

        // Tail should dominate shift (at least 2x larger for probabilistic delays)
        assert!(
            effect.tail_ns.abs() > effect.shift_ns.abs() * 2.0,
            "Expected |tail_ns| > 2*|shift_ns| (got tail={:.1}ns, shift={:.1}ns)",
            effect.tail_ns,
            effect.shift_ns
        );
    } else {
        panic!("Expected effect data for tail effect test");
    }
}

/// 4.3 Mixed Pattern - Validates Mixed classification
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn effect_pattern_mixed() {
    use std::cell::Cell;

    const BASE_DELAY: u64 = 30;
    const SPIKE_DELAY: u64 = 150;
    const SPIKE_PROBABILITY: f64 = 0.15;
    const SAMPLES: usize = 10_000;

    // Pre-generate spike decisions - both closures use identical code paths
    let spike_decisions: Vec<bool> = (0..SAMPLES)
        .map(|_| rand::random::<f64>() < SPIKE_PROBABILITY)
        .collect();
    let idx = Cell::new(0usize);

    let result = TimingOracle::new()
        .samples(SAMPLES)
        .test(
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                // No delays, but access spike_decisions for identical code path
                let _ = spike_decisions[i % SAMPLES];
                std::hint::black_box(42)
            },
            || {
                let i = idx.get();
                idx.set(i.wrapping_add(1));
                // Base delay (uniform shift)
                helpers::busy_wait_cycles(BASE_DELAY);

                // Plus occasional spike (tail effect)
                if spike_decisions[i % SAMPLES] {
                    helpers::busy_wait_cycles(SPIKE_DELAY);
                }
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[effect_pattern_mixed]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    if let Some(ref effect) = result.effect {
        // Should classify as Mixed
        assert!(
            matches!(effect.pattern, timing_oracle::EffectPattern::Mixed),
            "Expected Mixed pattern (got {:?})",
            effect.pattern
        );

        // Both components should be significant (use abs for sign-independence)
        assert!(
            effect.shift_ns.abs() > 3.0 && effect.tail_ns.abs() > 3.0,
            "Expected both |shift| and |tail| > 3ns (got shift={:.1}ns, tail={:.1}ns)",
            effect.shift_ns,
            effect.tail_ns
        );
    } else {
        panic!("Expected effect data for mixed pattern test");
    }
}

// ============================================================================
// Category 5: Exploitability Thresholds
// ============================================================================

/// 5.1 Negligible (<100ns) - Should classify as Negligible
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_negligible() {
    // Small delay (~10-30ns depending on platform)
    const SMALL_DELAY: u64 = 15;

    let result = TimingOracle::new()
        .samples(10_000)
        .test(
            || std::hint::black_box(42),
            || {
                helpers::busy_wait_cycles(SMALL_DELAY);
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[exploitability_negligible]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        matches!(result.exploitability, timing_oracle::Exploitability::Negligible),
        "Expected Negligible exploitability (got {:?})",
        result.exploitability
    );
}

/// 5.2 PossibleLAN (100-500ns) - Should classify appropriately
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_possible_lan() {
    // Medium delay (~200-300ns)
    const MEDIUM_DELAY: u64 = 500; // cycles

    let result = TimingOracle::new()
        .samples(10_000)
        .test(
            || std::hint::black_box(42),
            || {
                helpers::busy_wait_cycles(MEDIUM_DELAY);
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[exploitability_possible_lan]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        matches!(
            result.exploitability,
            timing_oracle::Exploitability::PossibleLAN | timing_oracle::Exploitability::LikelyLAN
        ),
        "Expected PossibleLAN or LikelyLAN exploitability (got {:?})",
        result.exploitability
    );
}

/// 5.3 LikelyLAN (500ns - 20μs) - Should classify appropriately
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_likely_lan() {
    // Large delay (~2μs)
    const LARGE_DELAY: u64 = 5000; // cycles

    let result = TimingOracle::new()
        .samples(10_000)
        .test(
            || std::hint::black_box(42),
            || {
                helpers::busy_wait_cycles(LARGE_DELAY);
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[exploitability_likely_lan]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        matches!(result.exploitability, timing_oracle::Exploitability::LikelyLAN),
        "Expected LikelyLAN exploitability (got {:?})",
        result.exploitability
    );
}

/// 5.4 PossibleRemote (>20μs) - Should classify appropriately
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn exploitability_possible_remote() {
    // Very large delay (~50μs)
    const HUGE_DELAY: u64 = 150_000; // cycles

    let result = TimingOracle::new()
        .samples(10_000)
        .test(
            || std::hint::black_box(42),
            || {
                helpers::busy_wait_cycles(HUGE_DELAY);
                std::hint::black_box(42)
            },
        );

    eprintln!("\n[exploitability_possible_remote]");
    eprintln!("{}", timing_oracle::output::format_result(&result));

    assert!(
        matches!(result.exploitability, timing_oracle::Exploitability::PossibleRemote),
        "Expected PossibleRemote exploitability (got {:?})",
        result.exploitability
    );
}
