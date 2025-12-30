//! Comparison benchmark suite entry point.
//!
//! Run with:
//! ```bash
//! cargo bench --bench comparison
//! ```
//!
//! Or with custom configuration:
//! ```bash
//! cargo bench --bench comparison -- --samples 50000 --trials 20
//! ```

mod comp;

use comp::{run_comparison_benchmark, BenchmarkConfig};

fn main() {
    // Parse command-line arguments (simple version)
    let args: Vec<String> = std::env::args().collect();

    let mut config = BenchmarkConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--samples" => {
                if i + 1 < args.len() {
                    config.samples = args[i + 1].parse().unwrap_or(20_000);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--trials" => {
                if i + 1 < args.len() {
                    config.detection_trials = args[i + 1].parse().unwrap_or(10);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--no-roc" => {
                config.run_roc = false;
                i += 1;
            }
            "--json" => {
                config.export_json = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    config.json_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    i += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    run_comparison_benchmark(config);
}
