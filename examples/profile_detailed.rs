//! Detailed profiling with stage-level timing instrumentation.
//!
//! This manually instruments the oracle to measure time spent in each stage.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use timing_oracle::helpers::InputPair;
use timing_oracle::measurement::{Collector, Timer};
use timing_oracle::statistics::{
    block_bootstrap_resample, bootstrap_covariance_matrix, compute_block_size, compute_deciles,
};
use timing_oracle::TimingOracle;

fn early_exit_compare(a: &[u8], b: &[u8]) -> bool {
    for i in 0..a.len().min(b.len()) {
        if a[i] != b[i] {
            return false;
        }
    }
    a.len() == b.len()
}

fn main() {
    println!("=== Detailed Stage-Level Profiling ===\n");

    let secret = [0u8; 512];
    let config = TimingOracle::balanced().config().clone();
    let timer = Timer::new();
    let collector = Collector::with_timer(timer.clone(), config.warmup);
    let inputs = InputPair::with_samples(config.samples, [0u8; 512], rand_bytes_512);

    println!("Config: {} samples per class", config.samples);
    println!("        {} CI bootstrap iterations", config.ci_bootstrap_iterations);
    println!("        {} covariance bootstrap iterations\n", config.cov_bootstrap_iterations);

    // Stage 1: Sample Collection
    println!("Stage 1: Sample Collection");
    let start = Instant::now();
    let (fixed_cycles, random_cycles, batching) = collector.collect_separated(
        config.samples,
        || early_exit_compare(&secret, inputs.fixed()),
        || early_exit_compare(&secret, inputs.random()),
    );
    let stage1_time = start.elapsed();
    println!("  Time: {:.3}s ({:.1}%)", stage1_time.as_secs_f64(), 100.0);
    println!(
        "  Batching: enabled={}, K={}, ticks_per_batch={:.1}",
        batching.enabled, batching.k, batching.ticks_per_batch
    );

    // Stage 2: Quantile Computation (first time - on raw samples)
    println!("\nStage 2: Initial Quantile Computation");
    let start = Instant::now();
    let fixed_data: Vec<f64> = fixed_cycles.iter().map(|&x| x as f64).collect();
    let random_data: Vec<f64> = random_cycles.iter().map(|&x| x as f64).collect();

    let _fixed_deciles = compute_deciles(&fixed_data);
    let _random_deciles = compute_deciles(&random_data);
    let stage2_time = start.elapsed();
    let stage2_pct = stage2_time.as_secs_f64() / stage1_time.as_secs_f64() * 100.0;
    println!("  Time: {:.3}s ({:.1}%)", stage2_time.as_secs_f64(), stage2_pct);

    // Stage 3: CI Gate Bootstrap (many quantile computations)
    println!("\nStage 3: CI Gate Bootstrap ({} iterations)", config.ci_bootstrap_iterations);
    let start = Instant::now();
    let block_size = compute_block_size(fixed_data.len());

    for i in 0..config.ci_bootstrap_iterations {
        let mut rng = StdRng::seed_from_u64(42u64.wrapping_add(i as u64));
        let resampled = block_bootstrap_resample(&fixed_data, block_size, &mut rng);
        let _ = compute_deciles(&resampled);
    }

    let stage3_time = start.elapsed();
    let stage3_pct = stage3_time.as_secs_f64() / stage1_time.as_secs_f64() * 100.0;
    println!("  Time: {:.3}s ({:.1}%)", stage3_time.as_secs_f64(), stage3_pct);
    println!("  Time per bootstrap: {:.2}ms", stage3_time.as_millis() as f64 / config.ci_bootstrap_iterations as f64);
    println!("  Time per compute_deciles call: {:.2}μs",
             stage3_time.as_micros() as f64 / config.ci_bootstrap_iterations as f64);

    // Stage 4: Covariance Bootstrap
    println!("\nStage 4: Covariance Bootstrap ({} iterations)", config.cov_bootstrap_iterations);
    let start = Instant::now();

    let cov = bootstrap_covariance_matrix(&fixed_data, config.cov_bootstrap_iterations, 42);

    let stage4_time = start.elapsed();
    let stage4_pct = stage4_time.as_secs_f64() / stage1_time.as_secs_f64() * 100.0;
    println!("  Time: {:.3}s ({:.1}%)", stage4_time.as_secs_f64(), stage4_pct);
    println!(
        "  Time per bootstrap: {:.2}ms",
        stage4_time.as_millis() as f64 / config.cov_bootstrap_iterations as f64
    );
    println!("  Jitter added: {:.2e}", cov.jitter_added);

    // Summary
    let total_bootstrap_time = stage3_time + stage4_time;
    let total_quantile_time = stage2_time + stage3_time;

    println!("\n=== Summary ===");
    println!("Total bootstrap time: {:.3}s ({:.1}%)",
             total_bootstrap_time.as_secs_f64(),
             total_bootstrap_time.as_secs_f64() / stage1_time.as_secs_f64() * 100.0);
    println!("Total quantile computation time: {:.3}s ({:.1}%)",
             total_quantile_time.as_secs_f64(),
             total_quantile_time.as_secs_f64() / stage1_time.as_secs_f64() * 100.0);

    println!("\nBottlenecks:");
    println!("1. CI Gate Bootstrap: {:.3}s ({} × compute_deciles)",
             stage3_time.as_secs_f64(), config.ci_bootstrap_iterations);
    println!(
        "2. Covariance Bootstrap: {:.3}s ({} × compute_deciles)",
        stage4_time.as_secs_f64(),
        config.cov_bootstrap_iterations
    );

    let total_decile_calls = config.ci_bootstrap_iterations + config.cov_bootstrap_iterations + 2;
    println!("\nTotal compute_deciles calls: {}", total_decile_calls);
    println!("Average time per call: {:.2}μs",
             total_quantile_time.as_micros() as f64 / total_decile_calls as f64);
}

fn rand_bytes_512() -> [u8; 512] {
    let mut arr = [0u8; 512];
    for byte in &mut arr {
        *byte = rand::random();
    }
    arr
}
