//! Terminal output formatting with colors and box drawing.

use colored::Colorize;

use crate::result::{EffectPattern, Exploitability, MeasurementQuality, TestResult};

/// Format a TestResult for human-readable terminal output.
///
/// Uses ANSI colors and a spec-aligned layout for clear presentation.
pub fn format_result(result: &TestResult) -> String {
    let mut output = String::new();
    let sep = "\u{2500}".repeat(62);

    output.push_str("timing-oracle\n");
    output.push_str(&sep);
    output.push('\n');
    output.push('\n');

    output.push_str(&format!(
        "  Samples: {} per class\n",
        result.metadata.samples_per_class
    ));
    output.push_str(&format!("  Quality: {}\n", format_quality(result.quality)));
    output.push_str(&format!(
        "  Min detectable effect: {:.1} ns (shift), {:.1} ns (tail)\n",
        result.min_detectable_effect.shift_ns, result.min_detectable_effect.tail_ns
    ));
    output.push('\n');

    if result.ci_gate.passed {
        output.push_str(&format!("  {}\n\n", "\u{2713} No timing leak detected".green().bold()));
    } else {
        output.push_str(&format!("  {}\n\n", "\u{26A0} Timing leak detected".yellow().bold()));
    }

    let prob_pct = result.leak_probability * 100.0;
    output.push_str(&format!(
        "    Probability of leak: {:.0}%\n",
        prob_pct
    ));

    if let Some(ref effect) = result.effect {
        output.push_str(&format!(
            "    Effect: {:.1} ns {} (95% CI: {:.1}–{:.1} ns)\n",
            (effect.shift_ns.powi(2) + effect.tail_ns.powi(2)).sqrt(),
            format_pattern(effect.pattern),
            effect.credible_interval_ns.0,
            effect.credible_interval_ns.1
        ));
    }

    output.push('\n');
    output.push_str("    Exploitability (heuristic):\n");
    let (lan, internet) = exploitability_lines(result.exploitability);
    output.push_str(&format!("      Local network:  {}\n", lan));
    output.push_str(&format!("      Internet:       {}\n", internet));
    output.push('\n');

    output.push_str(&sep);
    output.push('\n');

    output.push_str(
        "Note: Exploitability is a heuristic estimate based on effect magnitude.\n",
    );

    output
}

/// Format MeasurementQuality for display.
fn format_quality(quality: MeasurementQuality) -> String {
    match quality {
        MeasurementQuality::Excellent => "Excellent".green().to_string(),
        MeasurementQuality::Good => "Good".green().to_string(),
        MeasurementQuality::Poor => "Poor".yellow().to_string(),
        MeasurementQuality::TooNoisy => "Too Noisy".red().to_string(),
    }
}

/// Format EffectPattern for display.
fn format_pattern(pattern: EffectPattern) -> &'static str {
    match pattern {
        EffectPattern::UniformShift => "UniformShift",
        EffectPattern::TailEffect => "TailEffect",
        EffectPattern::Mixed => "Mixed",
    }
}

fn exploitability_lines(exploit: Exploitability) -> (String, String) {
    match exploit {
        Exploitability::Negligible => (
            "Negligible".green().to_string(),
            "Unlikely".green().to_string(),
        ),
        Exploitability::PossibleLAN => (
            "Possible (~10\u{2075} queries)".yellow().to_string(),
            "Unlikely".green().to_string(),
        ),
        Exploitability::LikelyLAN => (
            "Likely (~10\u{2074} queries)".red().to_string(),
            "Unlikely".yellow().to_string(),
        ),
        Exploitability::PossibleRemote => (
            "Likely".red().to_string(),
            "Possible".red().bold().to_string(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{CiGate, Effect, Metadata, MinDetectableEffect};

    fn make_test_result(passed: bool, leak_probability: f64) -> TestResult {
        TestResult {
            leak_probability,
            effect: if leak_probability > 0.5 {
                Some(Effect {
                    shift_ns: 150.0,
                    tail_ns: 25.0,
                    credible_interval_ns: (100.0, 200.0),
                    pattern: EffectPattern::UniformShift,
                })
            } else {
                None
            },
            exploitability: Exploitability::PossibleLAN,
            min_detectable_effect: MinDetectableEffect {
                shift_ns: 10.0,
                tail_ns: 15.0,
            },
            ci_gate: CiGate {
                alpha: 0.001,
                passed,
                thresholds: [0.0; 9],
                observed: [0.0; 9],
            },
            quality: MeasurementQuality::Good,
            outlier_fraction: 0.02,
            metadata: Metadata {
                samples_per_class: 10000,
                cycles_per_ns: 3.0,
                timer: "rdtsc".to_string(),
                timer_resolution_ns: 0.33,
                iterations_per_sample: 1,
                runtime_secs: 1.5,
            },
        }
    }

    #[test]
    fn test_format_passing_result() {
        let result = make_test_result(true, 0.1);
        let output = format_result(&result);
        assert!(output.contains("timing-oracle"));
        assert!(output.contains("Probability of leak: 10%"));
    }

    #[test]
    fn test_format_failing_result() {
        let result = make_test_result(false, 0.95);
        let output = format_result(&result);
        assert!(output.contains("Timing leak detected"));
        assert!(output.contains("Probability of leak: 95%"));
        assert!(output.contains("Effect:"));
    }
}
