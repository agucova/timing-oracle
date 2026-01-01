use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use timing_oracle::helpers::InputPair;
use timing_oracle::{CiFailure, CiTestBuilder, FailCriterion, Mode};

struct EnvGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var(key).ok();
        env::set_var(key, value);
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(val) = &self.original {
            env::set_var(self.key, val);
        } else {
            let _ = env::remove_var(self.key);
        }
    }
}

fn temp_report_path(name: &str) -> PathBuf {
    let mut path = env::temp_dir();
    path.push(format!(
        "timing-oracle-test-{}-{}.json",
        name,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros()
    ));
    path
}

// Helper for simulating timing difference via iteration count
fn do_iterations(count: &u32) {
    let mut x = 0u64;
    for i in 0..*count {
        x = x.wrapping_add(std::hint::black_box(i as u64));
    }
    std::hint::black_box(x);
}

#[test]
fn env_merge_and_fail_on_probability() {
    let _g1 = EnvGuard::set("TO_MODE", "full");
    let _g2 = EnvGuard::set("TO_SAMPLES", "25");
    let _g3 = EnvGuard::set("TO_ALPHA", "0.02");
    let _g4 = EnvGuard::set("TO_EFFECT_PRIOR_NS", "12.0");
    let _g5 = EnvGuard::set("TO_EFFECT_THRESHOLD_NS", "15.0");
    let _g6 = EnvGuard::set("TO_CALIBRATION_FRAC", "0.25");
    let _g7 = EnvGuard::set("TO_MAX_DURATION_MS", "5000");
    let _g8 = EnvGuard::set("TO_SEED", "12345");
    let _g9 = EnvGuard::set("TO_FAIL_ON", "prob:0.0");
    let report_path = temp_report_path("env-merge");
    let _g10 = EnvGuard::set("TO_REPORT", report_path.to_str().unwrap());

    let builder = CiTestBuilder::new().from_env();
    // Fixed: 1000 iterations, Random: 1100 iterations (creates timing diff)
    let inputs = InputPair::new(|| 1000u32, || 1100u32);

    match builder.run(inputs, do_iterations) {
        Err(CiFailure::LeakDetected { outcome }) => {
            assert_eq!(outcome.mode, Mode::Full);
            assert_eq!(outcome.config.samples, 25);
            assert!((outcome.config.ci_alpha - 0.02).abs() < 1e-12);
            assert!((outcome.config.min_effect_of_concern_ns - 12.0).abs() < 1e-12);
            assert_eq!(outcome.config.effect_threshold_ns, Some(15.0));
            assert!((outcome.config.calibration_fraction - 0.25).abs() < 1e-6);
            assert_eq!(outcome.config.max_duration_ms, Some(5000));
            assert_eq!(outcome.config.measurement_seed, Some(12345));
            assert_eq!(
                outcome.fail_on,
                FailCriterion::Probability(0.0),
                "env fail_on should map to probability threshold"
            );
            assert_eq!(outcome.seed, Some(12345));
            assert_eq!(outcome.report_path.as_ref(), Some(&report_path));
            assert!(report_path.exists(), "report should be written");
        }
        other => panic!(
            "expected leak detected due to prob=0.0 fail_on, got {:?}",
            other
        ),
    }

    // Clean up the report file if it was created.
    let _ = fs::remove_file(&report_path);
}

#[test]
fn async_flag_inflates_priors_and_thresholds() {
    let builder = CiTestBuilder::new()
        .async_workload(true)
        .samples(40)
        .fail_on(FailCriterion::Probability(0.0));

    let inputs = InputPair::new(|| 1000u32, || 1100u32);

    match builder.run(inputs, do_iterations) {
        Err(CiFailure::LeakDetected { outcome }) => {
            assert!(
                outcome.async_workload,
                "async flag should propagate to outcome"
            );
            assert!(
                outcome.config.min_effect_of_concern_ns >= 30.0,
                "async mode should inflate effect prior"
            );
            assert_eq!(
                outcome.config.effect_threshold_ns,
                Some(30.0),
                "async mode sets default effect threshold when absent"
            );
        }
        other => panic!(
            "expected leak detected due to prob=0.0 fail_on, got {:?}",
            other
        ),
    }
}

#[test]
fn report_contains_metadata() {
    let report_path = temp_report_path("metadata");
    let builder = CiTestBuilder::new()
        .mode(Mode::Smoke)
        .samples(40)
        .fail_on(FailCriterion::Probability(0.0))
        .report_path(&report_path);

    let inputs = InputPair::new(|| 1000u32, || 1100u32);

    match builder.run(inputs, do_iterations) {
        Err(CiFailure::LeakDetected { outcome: _ }) => {
            assert!(report_path.exists(), "report should be written");
            let contents = fs::read_to_string(&report_path).expect("read report");
            let v: serde_json::Value =
                serde_json::from_str(&contents).expect("report should be valid JSON");
            assert_eq!(v["mode"], serde_json::json!("Smoke"));
            assert_eq!(
                v["fail_on"]["Probability"],
                serde_json::json!(0.0),
                "fail_on enum should be present"
            );
            assert!(
                v.get("result").is_some(),
                "result payload should be present in report"
            );
            assert!(
                v.get("samples").is_some(),
                "config fields should be present in report"
            );
        }
        other => panic!(
            "expected leak detected due to prob=0.0 fail_on, got {:?}",
            other
        ),
    }

    let _ = fs::remove_file(&report_path);
}
