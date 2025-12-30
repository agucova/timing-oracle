//! Reporting and output formatting for comparison results.

use super::metrics::{DetectionRateResult, RocCurveResult, SampleEfficiencyResult};
use serde::{Deserialize, Serialize};

/// Complete comparison results for a single test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseComparison {
    pub test_case_name: String,
    pub expected_leaky: bool,
    pub results: Vec<ToolResult>,
}

/// Results from a single tool on a test case
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_name: String,
    pub detection_rate: f64,
    pub avg_confidence: f64,
    pub avg_time_secs: f64,
    pub min_samples_to_detect: Option<usize>,
}

/// Complete benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub test_cases: Vec<TestCaseComparison>,
    pub roc_curves: Vec<RocCurveResult>,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            test_cases: Vec::new(),
            roc_curves: Vec::new(),
        }
    }

    /// Add test case comparison results
    pub fn add_test_case(&mut self, comparison: TestCaseComparison) {
        self.test_cases.push(comparison);
    }

    /// Add ROC curve results
    pub fn add_roc_curve(&mut self, roc: RocCurveResult) {
        self.roc_curves.push(roc);
    }

    /// Export results as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Print formatted terminal report
    pub fn print_terminal_report(&self) {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│                        DETECTION COMPARISON RESULTS                          │");
        println!("└─────────────────────────────────────────────────────────────────────────────┘");
        println!();

        // Print results for each test case
        for tc in &self.test_cases {
            self.print_test_case(tc);
            println!();
        }

        // Print ROC summary
        if !self.roc_curves.is_empty() {
            self.print_roc_summary();
        }
    }

    fn print_test_case(&self, tc: &TestCaseComparison) {
        let leak_status = if tc.expected_leaky {
            "KNOWN LEAKY"
        } else {
            "KNOWN SAFE"
        };

        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Test Case: {} ({})", tc.test_case_name, leak_status);
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ {:15} │ {:14} │ {:10} │ {:11} │ {:15} │",
            "Tool", "Detection Rate", "Avg Time", "Min Samples", "Confidence"
        );
        println!(
            "│ {:─<15} │ {:─<14} │ {:─<10} │ {:─<11} │ {:─<15} │",
            "", "", "", "", ""
        );

        for result in &tc.results {
            let detection_pct = format!("{:.1}%", result.detection_rate * 100.0);
            let avg_time = format!("{:.2}s", result.avg_time_secs);
            let min_samples = result
                .min_samples_to_detect
                .map(|s| format!("{}", s))
                .unwrap_or_else(|| "N/A".to_string());
            let confidence = format!("{:.3}", result.avg_confidence);

            println!(
                "│ {:15} │ {:>14} │ {:>10} │ {:>11} │ {:>15} │",
                result.tool_name, detection_pct, avg_time, min_samples, confidence
            );
        }

        println!("└─────────────────────────────────────────────────────────────────────────────┘");
    }

    fn print_roc_summary(&self) {
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ ROC CURVE SUMMARY                                                            │");
        println!("├─────────────────────────────────────────────────────────────────────────────┤");
        println!(
            "│ {:15} │ {:13} │ {:45} │",
            "Tool", "AUC", "Optimal Threshold"
        );
        println!(
            "│ {:─<15} │ {:─<13} │ {:─<45} │",
            "", "", ""
        );

        for roc in &self.roc_curves {
            let auc_str = format!("{:.3}", roc.auc);

            // Find optimal threshold (closest to top-left corner)
            let optimal = find_optimal_threshold(roc);
            let opt_str = format!(
                "{:.2} (TPR={:.2}, FPR={:.2})",
                optimal.threshold, optimal.tpr, optimal.fpr
            );

            println!(
                "│ {:15} │ {:>13} │ {:45} │",
                roc.detector_name, auc_str, opt_str
            );
        }

        println!("└─────────────────────────────────────────────────────────────────────────────┘");
    }
}

impl Default for BenchmarkResults {
    fn default() -> Self {
        Self::new()
    }
}

struct OptimalPoint {
    threshold: f64,
    tpr: f64,
    fpr: f64,
}

fn find_optimal_threshold(roc: &RocCurveResult) -> OptimalPoint {
    let mut best_idx = 0;
    let mut best_distance = f64::MAX;

    for (idx, &(fpr, tpr)) in roc.roc_points.iter().enumerate() {
        // Distance to top-left corner (0, 1)
        let distance = ((1.0 - tpr).powi(2) + fpr.powi(2)).sqrt();
        if distance < best_distance {
            best_distance = distance;
            best_idx = idx;
        }
    }

    let (fpr, tpr) = roc.roc_points[best_idx];
    let threshold = roc.thresholds[best_idx];

    OptimalPoint {
        threshold,
        tpr,
        fpr,
    }
}

/// Helper to convert DetectionRateResult and SampleEfficiencyResult to ToolResult
pub fn create_tool_result(
    tool_name: &str,
    detection: &DetectionRateResult,
    efficiency: &SampleEfficiencyResult,
) -> ToolResult {
    ToolResult {
        tool_name: tool_name.to_string(),
        detection_rate: detection.detection_rate,
        avg_confidence: detection.avg_confidence,
        avg_time_secs: detection.avg_duration.as_secs_f64(),
        min_samples_to_detect: efficiency.min_samples_to_detect,
    }
}
