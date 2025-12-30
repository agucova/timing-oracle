use std::time::Instant;
use timing_oracle::TimingOracle;

fn main() {
    println!("=== Performance Baseline (With Correct Bootstrap Counts) ===\n");

    // Measurable operation (should be >5 ticks on ARM)
    let operation = || {
        let mut x = 0u64;
        for i in 0..1000 {
            x = x.wrapping_add(std::hint::black_box(i));
        }
        std::hint::black_box(x)
    };

    let leak_operation = || {
        let mut x = 0u64;
        for i in 0..1100 {
            x = x.wrapping_add(std::hint::black_box(i));
        }
        std::hint::black_box(x)
    };

    // Test 1: Quick preset (5k samples, 50 CI bootstrap, 50 cov bootstrap)
    println!("Testing Quick preset (5k samples, 50 CI bootstrap, 50 cov bootstrap)...");
    let start = Instant::now();
    let result = TimingOracle::quick().test(operation, leak_operation);
    let quick_time = start.elapsed();
    println!("  Time: {:>8.3}s", quick_time.as_secs_f64());
    println!("  Leak prob: {:.3}", result.leak_probability);
    println!();

    // Test 2: Balanced preset (20k samples, 100 CI bootstrap, 50 cov bootstrap)
    println!("Testing Balanced preset (20k samples, 100 CI bootstrap, 50 cov bootstrap)...");
    let start = Instant::now();
    let result = TimingOracle::balanced().test(operation, leak_operation);
    let balanced_time = start.elapsed();
    println!("  Time: {:>8.3}s", balanced_time.as_secs_f64());
    println!("  Leak prob: {:.3}", result.leak_probability);
    println!();

    // Test 3: Default preset (100k samples, 10k CI bootstrap, 2k cov bootstrap)
    println!("Testing Default preset (100k samples, 10k CI bootstrap, 2k cov bootstrap)...");
    let start = Instant::now();
    let result = TimingOracle::new().test(operation, leak_operation);
    let default_time = start.elapsed();
    println!("  Time: {:>8.3}s", default_time.as_secs_f64());
    println!("  Leak prob: {:.3}", result.leak_probability);
    println!();

    println!("=== Summary ===");
    println!("Quick:    {:>8.3}s", quick_time.as_secs_f64());
    println!("Balanced: {:>8.3}s", balanced_time.as_secs_f64());
    println!("Default:  {:>8.3}s", default_time.as_secs_f64());
    println!("\nNote: Default uses CORRECT bootstrap counts (10k/2k)");
    println!("      Presets use OLD bootstrap counts (need updating per plan)");
}
