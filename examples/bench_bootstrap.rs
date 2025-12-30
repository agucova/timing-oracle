use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;
use timing_oracle::statistics::{block_bootstrap_resample, block_bootstrap_resample_into, compute_block_size};

fn main() {
    println!("=== Bootstrap Resampling Micro-Benchmark ===\n");

    // Test on typical bootstrap size
    let n = 20_000;
    let data: Vec<f64> = (0..n).map(|x| (x as f64 * 1.234) % 1000.0).collect();
    let block_size = compute_block_size(n);

    println!("Testing with n = {} samples, block_size = {}\n", n, block_size);

    // Warmup
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut buffer = vec![0.0; n];
    for _ in 0..100 {
        let _ = block_bootstrap_resample(&data, block_size, &mut rng);
        block_bootstrap_resample_into(&data, block_size, &mut rng, &mut buffer);
    }

    // Benchmark allocating version
    let iterations = 10_000;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = block_bootstrap_resample(&data, block_size, &mut rng);
    }
    let alloc_time = start.elapsed();
    let alloc_avg_us = alloc_time.as_micros() as f64 / iterations as f64;

    // Benchmark _into version
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
    let mut buffer = vec![0.0; n];
    let start = Instant::now();
    for _ in 0..iterations {
        block_bootstrap_resample_into(&data, block_size, &mut rng, &mut buffer);
    }
    let into_time = start.elapsed();
    let into_avg_us = into_time.as_micros() as f64 / iterations as f64;

    let speedup = alloc_avg_us / into_avg_us;

    println!("Allocating version:  {:.2} µs/iteration", alloc_avg_us);
    println!("Into version:        {:.2} µs/iteration", into_avg_us);
    println!(
        "Speedup:             {:.2}× {}",
        speedup,
        if speedup > 1.0 { "✓ FASTER" } else { "✗ SLOWER" }
    );

    // Verify correctness
    println!("\n=== Correctness Verification ===");
    let mut rng1 = Xoshiro256PlusPlus::seed_from_u64(123);
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(123);

    let alloc_result = block_bootstrap_resample(&data, block_size, &mut rng1);
    let mut into_result = vec![0.0; n];
    block_bootstrap_resample_into(&data, block_size, &mut rng2, &mut into_result);

    let mut matches = 0;
    for i in 0..n {
        if (alloc_result[i] - into_result[i]).abs() < 1e-10 {
            matches += 1;
        }
    }

    println!("Matching values: {}/{} ({:.1}%)", matches, n, 100.0 * matches as f64 / n as f64);

    if matches == n {
        println!("✓ Results are identical");
    } else {
        println!("✗ Results differ (expected for same RNG state)");
    }
}
