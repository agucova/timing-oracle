//! Metrics collection for comparing timing analysis tools.

use super::adapters::dudect_adapter::DudectDetector;
use super::adapters::Detector;
use super::test_cases::TestCase;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Results of detection rate measurement
#[derive(Debug, Clone)]
pub struct DetectionRateResult {
    pub detections: usize,
    pub total_trials: usize,
    pub detection_rate: f64,
    pub avg_confidence: f64,
    pub avg_duration: Duration,
}

/// Results of sample efficiency measurement
#[derive(Debug, Clone)]
pub struct SampleEfficiencyResult {
    pub min_samples_to_detect: Option<usize>,
    pub tested_sample_sizes: Vec<usize>,
    pub detection_results: Vec<bool>,
}

/// Measure detection rate (true positive rate) on a test case
pub fn measure_detection_rate(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
    samples: usize,
    trials: usize,
) -> DetectionRateResult {
    let mut detections = 0;
    let mut total_confidence = 0.0;
    let mut total_duration = Duration::ZERO;

    for trial in 0..trials {
        eprintln!(
            "[{}] Detection rate trial {}/{} for {}",
            detector.name(),
            trial + 1,
            trials,
            test_case.name()
        );

        // For DudectDetector, prepare the test case before calling detect()
        if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
            dudect.prepare_test_case(test_case);
        }

        let fixed_op = test_case.fixed_operation();
        let random_op = test_case.random_operation();

        let result = detector.detect(
            &|| fixed_op(),
            &|| random_op(),
            samples,
        );

        if result.detected_leak {
            detections += 1;
        }
        total_confidence += result.confidence_metric;
        total_duration += result.duration;
    }

    DetectionRateResult {
        detections,
        total_trials: trials,
        detection_rate: detections as f64 / trials as f64,
        avg_confidence: total_confidence / trials as f64,
        avg_duration: total_duration / trials as u32,
    }
}

/// Measure false positive rate on a known-safe test case
pub fn measure_false_positive_rate(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
    samples: usize,
    trials: usize,
) -> DetectionRateResult {
    // Same as detection rate, but on safe cases
    measure_detection_rate(detector, test_case, samples, trials)
}

/// Measure sample efficiency - find minimum samples needed to detect leak
pub fn measure_sample_efficiency(
    detector: &dyn Detector,
    test_case: &dyn TestCase,
) -> SampleEfficiencyResult {
    let sample_sizes = vec![1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000];
    let mut detection_results = Vec::new();
    let mut min_samples = None;

    for &samples in &sample_sizes {
        eprintln!(
            "[{}] Sample efficiency test with {} samples for {}",
            detector.name(),
            samples,
            test_case.name()
        );

        // For DudectDetector, prepare the test case before calling detect()
        if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
            dudect.prepare_test_case(test_case);
        }

        let fixed_op = test_case.fixed_operation();
        let random_op = test_case.random_operation();

        let result = detector.detect(
            &|| fixed_op(),
            &|| random_op(),
            samples,
        );

        let detected = result.detected_leak;
        detection_results.push(detected);

        if detected && min_samples.is_none() {
            min_samples = Some(samples);
        }
    }

    SampleEfficiencyResult {
        min_samples_to_detect: min_samples,
        tested_sample_sizes: sample_sizes,
        detection_results,
    }
}

/// Results of ROC curve analysis for a single detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocCurveResult {
    /// Tool name
    pub detector_name: String,
    /// Threshold values tested
    pub thresholds: Vec<f64>,
    /// (FPR, TPR) pairs for each threshold
    pub roc_points: Vec<(f64, f64)>,
    /// Area under the curve
    pub auc: f64,
}

/// Generate ROC curve by varying detection threshold
pub fn generate_roc_curve(
    detector: &dyn Detector,
    leaky_cases: &[&dyn TestCase],
    safe_cases: &[&dyn TestCase],
    thresholds: &[f64],
    samples: usize,
    trials_per_case: usize,
) -> RocCurveResult {
    let mut roc_points = Vec::new();

    for &threshold in thresholds {
        eprintln!(
            "[{}] ROC analysis at threshold {}",
            detector.name(),
            threshold
        );

        // Measure TPR on leaky cases
        let mut true_positives = 0;
        let mut total_positives = 0;

        for test_case in leaky_cases {
            for _ in 0..trials_per_case {
                // For DudectDetector, prepare the test case before calling detect()
                if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
                    dudect.prepare_test_case(*test_case);
                }

                let fixed_op = test_case.fixed_operation();
                let random_op = test_case.random_operation();

                let result = detector.detect(
                    &|| fixed_op(),
                    &|| random_op(),
                    samples,
                );

                total_positives += 1;
                if detector.exceeds_threshold(result.confidence_metric, threshold) {
                    true_positives += 1;
                }
            }
        }

        let tpr = if total_positives > 0 {
            true_positives as f64 / total_positives as f64
        } else {
            0.0
        };

        // Measure FPR on safe cases
        let mut false_positives = 0;
        let mut total_negatives = 0;

        for test_case in safe_cases {
            for _ in 0..trials_per_case {
                // For DudectDetector, prepare the test case before calling detect()
                if let Some(dudect) = detector.as_any().downcast_ref::<DudectDetector>() {
                    dudect.prepare_test_case(*test_case);
                }

                let fixed_op = test_case.fixed_operation();
                let random_op = test_case.random_operation();

                let result = detector.detect(
                    &|| fixed_op(),
                    &|| random_op(),
                    samples,
                );

                total_negatives += 1;
                if detector.exceeds_threshold(result.confidence_metric, threshold) {
                    false_positives += 1;
                }
            }
        }

        let fpr = if total_negatives > 0 {
            false_positives as f64 / total_negatives as f64
        } else {
            0.0
        };

        roc_points.push((fpr, tpr));
    }

    // Calculate AUC using trapezoidal rule
    let auc = calculate_auc(&roc_points);

    RocCurveResult {
        detector_name: detector.name().to_string(),
        thresholds: thresholds.to_vec(),
        roc_points,
        auc,
    }
}

/// Calculate area under ROC curve using trapezoidal rule
fn calculate_auc(roc_points: &[(f64, f64)]) -> f64 {
    if roc_points.len() < 2 {
        return 0.0;
    }

    // Sort by FPR (x-axis)
    let mut sorted_points = roc_points.to_vec();
    sorted_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut auc = 0.0;
    for i in 1..sorted_points.len() {
        let (x0, y0) = sorted_points[i - 1];
        let (x1, y1) = sorted_points[i];
        // Trapezoidal area
        auc += (x1 - x0) * (y0 + y1) / 2.0;
    }

    auc
}
