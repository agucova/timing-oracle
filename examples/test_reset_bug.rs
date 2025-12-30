//! Reproduction of the kperf-rs reset() bug in calibration scenarios.
//! Run with: cargo build --features kperf --example test_reset_bug && sudo ./target/debug/examples/test_reset_bug

use kperf_rs::{event::Event, PerfCounterBuilder};
use std::time::Instant;

fn calibrate_with_reset(counter: &mut kperf_rs::PerfCounter) -> f64 {
    let mut ratios = Vec::new();

    for i in 0..5 {
        // Reset before each measurement (THE BUG!)
        counter.reset().expect("Reset failed");

        let start_time = Instant::now();

        // Busy loop ~1ms
        let mut dummy: u64 = 1;
        loop {
            dummy = dummy.wrapping_mul(6364136223846793005).wrapping_add(1);
            std::hint::black_box(dummy);
            if dummy & 0xFFFF == 0 && start_time.elapsed().as_micros() >= 1000 {
                break;
            }
        }

        let cycles = counter.read().expect("Read failed");
        let elapsed_ns = start_time.elapsed().as_nanos() as u64;

        if elapsed_ns > 0 && cycles > 0 {
            let ratio = cycles as f64 / elapsed_ns as f64;
            println!("  Iteration {}: {} cycles / {} ns = {:.4} cycles/ns",
                i + 1, cycles, elapsed_ns, ratio);
            ratios.push(ratio);
        }
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ratios.get(ratios.len() / 2).copied().unwrap_or(0.0)
}

fn calibrate_with_delta(counter: &mut kperf_rs::PerfCounter) -> f64 {
    let mut ratios = Vec::new();
    let mut prev_cycles = counter.read().expect("Initial read failed");

    for i in 0..5 {
        let start_time = Instant::now();

        // Busy loop ~1ms
        let mut dummy: u64 = 1;
        loop {
            dummy = dummy.wrapping_mul(6364136223846793005).wrapping_add(1);
            std::hint::black_box(dummy);
            if dummy & 0xFFFF == 0 && start_time.elapsed().as_micros() >= 1000 {
                break;
            }
        }

        let current_cycles = counter.read().expect("Read failed");
        let delta_cycles = current_cycles.saturating_sub(prev_cycles);
        prev_cycles = current_cycles;
        let elapsed_ns = start_time.elapsed().as_nanos() as u64;

        if elapsed_ns > 0 && delta_cycles > 0 {
            let ratio = delta_cycles as f64 / elapsed_ns as f64;
            println!("  Iteration {}: {} cycles / {} ns = {:.4} cycles/ns",
                i + 1, delta_cycles, elapsed_ns, ratio);
            ratios.push(ratio);
        }
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ratios.get(ratios.len() / 2).copied().unwrap_or(0.0)
}

fn main() {
    kperf_rs::check_kpc_permission().expect("Need sudo");

    let mut counter = PerfCounterBuilder::new()
        .track_event(Event::Cycles)
        .build_counter()
        .expect("Failed to build counter");

    counter.start().expect("Failed to start");

    println!("=== Calibration WITH reset() (buggy) ===\n");
    let buggy_ratio = calibrate_with_reset(&mut counter);
    println!("\nResult: {:.4} cycles/ns\n", buggy_ratio);

    // Need to restart after reset broke things
    counter.start().expect("Restart failed");

    println!("=== Calibration with delta tracking (workaround) ===\n");
    let fixed_ratio = calibrate_with_delta(&mut counter);
    println!("\nResult: {:.4} cycles/ns\n", fixed_ratio);

    println!("=== Summary ===");
    println!("With reset():      {:.4} cycles/ns", buggy_ratio);
    println!("With delta track:  {:.4} cycles/ns", fixed_ratio);

    if buggy_ratio < 1.0 && fixed_ratio > 2.0 {
        println!("\nBUG CONFIRMED: reset() causes incorrect calibration!");
        println!("Expected ~3-4 cycles/ns on Apple Silicon, got {:.4}", buggy_ratio);
    } else if buggy_ratio < fixed_ratio / 2.0 {
        println!("\nBUG LIKELY: reset() calibration significantly lower");
    } else {
        println!("\nResults similar - bug may not manifest in this run");
    }
}
