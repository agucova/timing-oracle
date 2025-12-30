use std::time::Instant;
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
    let timer = timing_oracle::Timer::new();

    // Run a representative leaky test
    let secret = [0u8; 512];

    println!("Starting profiling run with balanced preset...");
    let start = Instant::now();

    let result = TimingOracle::balanced()
        .with_timer(timer)
        .test(
            || {
                let input = [0u8; 512];
                std::hint::black_box(early_exit_compare(&secret, &input));
            },
            || {
                let mut input = [0u8; 512];
                for i in 0..512 {
                    input[i] = rand::random();
                }
                std::hint::black_box(early_exit_compare(&secret, &input));
            },
        );

    let total_time = start.elapsed();

    println!("\nTotal execution time: {:.2}s", total_time.as_secs_f64());
    println!("Leak probability: {:.3}", result.leak_probability);
}
