# `timing-oracle`: Detecting Timing Side Channels in Rust

## Implementation Specification v1.6

This document specifies `timing-oracle`, a Rust crate for detecting timing side channels. It provides the statistical methodology, API design, and architectural decisions needed for implementation.

**For the implementer:** This spec is intentionally concise—focus on correctness of the statistical methodology and the API contract. Standard Rust patterns (error handling, module organization, etc.) are left to your judgment.

---

## Table of Contents

1. [Overview](#overview)
2. [Public API](#public-api)
3. [Architecture](#architecture)
4. [Statistical Methodology](#statistical-methodology)
5. [Implementation Details](#implementation-details)
6. [Module Structure](#module-structure)
7. [Dependencies](#dependencies)
8. [Testing Strategy](#testing-strategy)
9. [References](#references)

---

## 1. Overview

### What It Does

`timing-oracle` detects timing side channels by comparing execution times between two input classes:
- **Fixed**: A specific input that might trigger timing variations (e.g., all zeros, valid padding)
- **Random**: Randomly sampled inputs

It outputs:
- **Probability of leak** (0.0–1.0): Bayesian posterior probability
- **Effect size**: Estimated timing difference in nanoseconds
- **CI gate**: Pass/fail with bounded false positive rate
- **Exploitability**: Heuristic assessment based on effect magnitude

### Design Goals

1. **Interpretable**: Output probabilities, not t-statistics
2. **Reliable CI integration**: Bounded false positive rate via RTLF-style thresholds
3. **Fast**: < 1 second for 100k samples
4. **Robust**: Handles noisy environments, outliers, correlated quantiles

### Key Innovation

Two-layer architecture:
- **Layer 1 (CI Gate)**: RTLF-style bounded FPR for pass/fail decisions
- **Layer 2 (Interpretation)**: Closed-form Bayesian inference for probabilities and effect sizes

These use different null constructions because they serve different purposes.

---

## 2. Public API

### Core Types

```rust
/// Main entry point for timing analysis
pub struct TimingOracle {
    config: Config,
}

/// Configuration options
pub struct Config {
    /// Samples per class (default: 100_000)
    pub samples: usize,
    
    /// Warmup iterations before measurement (default: 1_000)
    pub warmup: usize,
    
    /// False positive rate for CI gate (default: 0.01)
    pub ci_alpha: f64,
    
    /// Minimum effect size we care about in nanoseconds (default: 10.0)
    /// Effects smaller than this won't trigger high posterior probabilities
    /// even if statistically detectable. This encodes practical relevance.
    pub min_effect_of_concern_ns: f64,
    
    /// Bootstrap iterations for CI thresholds (default: 10_000)
    pub ci_bootstrap_iterations: usize,
    
    /// Bootstrap iterations for covariance estimation (default: 2_000)
    pub cov_bootstrap_iterations: usize,
    
    /// Percentile for outlier filtering (default: 0.999)
    /// Set to 1.0 to disable filtering
    pub outlier_percentile: f64,
    
    /// Prior probability of no leak (default: 0.75)
    pub prior_no_leak: f64,
    
    /// Maximum batch size for adaptive batching (default: 1000)
    /// Maximum batch size for adaptive batching (default: 20)
    /// Set to 1 to disable batching entirely
    pub max_batch_size: u32,
    
    /// Target ticks per batch for adaptive K selection (default: 50.0)
    pub target_ticks_per_batch: f64,
    
    /// Minimum ticks per call to consider measurable (default: 5.0)
    /// Below this, operations are too fast to measure reliably
    pub min_ticks_single_call: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            samples: 100_000,
            warmup: 1_000,
            ci_alpha: 0.01,
            min_effect_of_concern_ns: 10.0,
            ci_bootstrap_iterations: 10_000,
            cov_bootstrap_iterations: 2_000,
            outlier_percentile: 0.999,
            prior_no_leak: 0.75,
            max_batch_size: 20,
            target_ticks_per_batch: 50.0,
            min_ticks_single_call: 5.0,
        }
    }
}

/// Analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Probability of timing leak given data and model (0.0 to 1.0)
    pub leak_probability: f64,
    
    /// Effect size estimate (present if leak_probability > 0.5)
    pub effect: Option<Effect>,
    
    /// Exploitability assessment (heuristic)
    pub exploitability: Exploitability,
    
    /// Minimum detectable effect given noise level
    pub min_detectable_effect: MinDetectableEffect,
    
    /// CI gate result
    pub ci_gate: CiGate,
    
    /// Measurement quality assessment
    pub quality: MeasurementQuality,
    
    /// Fraction of samples trimmed as outliers
    pub outlier_fraction: f64,
    
    /// Metadata for debugging
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    /// Uniform shift in nanoseconds (positive = fixed class slower)
    pub shift_ns: f64,
    
    /// Tail effect in nanoseconds (positive = fixed has heavier upper tail)
    pub tail_ns: f64,
    
    /// 95% credible interval for total effect magnitude
    pub credible_interval_ns: (f64, f64),
    
    /// Dominant pattern description
    pub pattern: EffectPattern,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectPattern {
    /// Uniform shift across all quantiles (e.g., branch)
    UniformShift,
    /// Primarily affects upper tail (e.g., cache misses)
    TailEffect,
    /// Mixed pattern
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinDetectableEffect {
    /// Minimum detectable uniform shift (ns)
    pub shift_ns: f64,
    /// Minimum detectable tail effect (ns)
    pub tail_ns: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiGate {
    /// The alpha level used
    pub alpha: f64,
    /// Whether the test passed (no leak detected at this alpha)
    pub passed: bool,
    /// Per-quantile thresholds used (for debugging)
    pub thresholds: [f64; 9],
    /// Per-quantile observed differences
    pub observed: [f64; 9],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Exploitability {
    /// < 100 ns: Would require impractical number of measurements
    Negligible,
    /// 100–500 ns: Possible on local network with ~100k measurements
    PossibleLAN,
    /// 500 ns – 20 μs: Likely exploitable on local network
    LikelyLAN,
    /// > 20 μs: Possibly exploitable over internet
    PossibleRemote,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeasurementQuality {
    /// Low noise, high confidence
    Excellent,
    /// Normal noise levels
    Good,
    /// High noise, results less reliable
    Poor,
    /// Cannot produce meaningful results
    TooNoisy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchRationale {
    /// ticks_per_call >= 50, no batching needed
    NotNeeded,
    /// Batched to reach ~50 ticks
    Batched { target_ticks: f64 },
    /// K hit max but still below 50 ticks
    PartiallyBatched { achieved_ticks: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingInfo {
    /// Whether batching was enabled
    pub enabled: bool,
    /// Iterations per batch (1 if batching disabled)
    pub k: u32,
    /// Effective ticks per batch measurement
    pub achieved_ticks: f64,
    /// Why this batching decision was made
    pub rationale: BatchRationale,
}

/// Top-level outcome of a timing test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    /// Analysis completed successfully
    Completed(TestResult),
    
    /// Operation too fast to measure reliably on this platform
    Unmeasurable {
        /// Estimated operation duration in nanoseconds
        operation_ns: f64,
        /// Minimum measurable duration on this platform
        threshold_ns: f64,
        /// Platform description (e.g., "Apple Silicon (cntvct)")
        platform: String,
        /// Suggested actions
        recommendation: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// Samples per class after outlier filtering
    pub samples_per_class: usize,
    /// Cycles per nanosecond (for conversion)
    pub cycles_per_ns: f64,
    /// Timer type used
    pub timer: String,
    /// Timer resolution in nanoseconds
    pub timer_resolution_ns: f64,
    /// Total runtime in seconds
    pub runtime_secs: f64,
    /// Batching configuration and rationale
    pub batching: BatchingInfo,
}

/// Input class identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Fixed,
    Random,
}
```

### Builder API

```rust
impl TimingOracle {
    /// Create with default configuration
    pub fn new() -> Self;
    
    /// Set samples per class
    pub fn samples(mut self, n: usize) -> Self;
    
    /// Set warmup iterations
    pub fn warmup(mut self, n: usize) -> Self;
    
    /// Set CI false positive rate
    pub fn ci_alpha(mut self, alpha: f64) -> Self;
    
    /// Set minimum effect of concern (ns)
    pub fn min_effect_of_concern(mut self, ns: f64) -> Self;
    
    /// Run test with simple closures
    pub fn test<F, R, T>(self, fixed: F, random: R) -> TestResult
    where
        F: FnMut() -> T,
        R: FnMut() -> T;
    
    /// Run test with setup and state
    pub fn test_with_state<S, F, R, I, E>(
        self,
        setup: impl FnOnce() -> S,
        fixed_input: F,
        random_input: R,
        execute: E,
    ) -> TestResult
    where
        F: FnMut(&mut S) -> I,
        R: FnMut(&mut S, &mut impl Rng) -> I,
        E: FnMut(&mut S, I);
}

/// Convenience function for simple cases
pub fn test<F, R, T>(fixed: F, random: R) -> TestResult
where
    F: FnMut() -> T,
    R: FnMut() -> T,
{
    TimingOracle::new().test(fixed, random)
}
```

### Macro API (Optional Feature)

```rust
/// Declarative timing test macro
/// Generates a #[test] that fails if leak_probability > 0.9
#[macro_export]
macro_rules! timing_test {
    ($name:ident {
        $(setup: $setup:block,)?
        fixed: $fixed:expr,
        random: $random:expr,
        test: $test:expr $(,)?
    }) => { /* ... */ };
}
```

---

## 3. Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEASUREMENT PHASE                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Pre-flight checks (sanity, generator cost, autocorrelation) │
│  2. Warmup iterations (not measured)                            │
│  3. Interleaved randomized measurement                          │
│  4. Outlier filtering (pooled symmetric threshold)              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANTILE COMPUTATION                          │
├─────────────────────────────────────────────────────────────────┤
│  Compute deciles (10%, 20%, ..., 90%) for each class            │
│  Δ = q̂(Fixed) - q̂(Random) ∈ ℝ⁹                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│   LAYER 1: CI GATE       │      │   LAYER 2: BAYESIAN          │
├──────────────────────────┤      ├──────────────────────────────┤
│ RTLF-style bootstrap     │      │ Closed-form conjugate        │
│ Within-class resampling  │      │ Bayes factor                 │
│ Conservative max bounds  │      │                              │
│ Bonferroni correction    │      │ Bootstrap covariance Σ₀      │
│                          │      │ (single quantile vectors)    │
│ Output: passed/failed    │      │                              │
│         at α level       │      │ Output: P(leak), effect size │
└──────────────────────────┘      └──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                         TEST RESULT                              │
└─────────────────────────────────────────────────────────────────┘
```

### Why Two Layers?

**CI Gate** answers: "Should this block my build?"
- Needs bounded false positive rate
- Uses conservative max-based thresholds
- Binary pass/fail

**Bayesian Layer** answers: "What's the probability and how bad is it?"
- Needs interpretable quantities
- Uses proper statistical model
- Continuous probability + effect size

These require different null constructions, so they're separate.

---

## 4. Statistical Methodology

### 4.1 Measurement Protocol

#### Interleaved Randomized Design

Measurements alternate between Fixed (F) and Random (R) in randomized order:
```
Sequence: [F R R F R F F R F R F R R F ...]
```

This ensures drift affects both classes equally and supports near-IID assumptions.

Store measurements as sequence of `(Class, u64)` tuples to enable autocorrelation analysis.

#### Timer Implementation

Use platform-specific cycle counters with serialization barriers:

**x86_64:** `lfence; rdtsc` with compiler fence
**aarch64:** `isb; mrs cntvct_el0`  
**Fallback:** `std::time::Instant` (coarser but portable)

Wrap the function under test with `std::hint::black_box()` to prevent optimization.

#### Outlier Filtering

Apply **pooled symmetric** threshold to preserve distributional symmetry:

1. Pool all samples from both classes
2. Compute threshold at `outlier_percentile` (default 99.9%)
3. Apply **same threshold** to both classes
4. Report `outlier_fraction` for transparency

### 4.2 Test Statistic: Signed Quantile Differences

Compute deciles for each class:
```
p ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
```

Quantile difference vector:
```
Δₚ = q̂ₚ(Fixed) − q̂ₚ(Random)
Δ ∈ ℝ⁹
```

Use `slice.select_nth_unstable()` for O(n) quantile computation (not full sort).

### 4.3 Layer 1: CI Gate (RTLF-Style Bounded FPR)

#### Algorithm

**Step 1: Bootstrap within each class separately**

For b = 1 to B (B = 10,000):
- Block-resample within Fixed: F₁*, F₂* → δ_F*[b] = quantile_diffs(F₁*, F₂*)
- Block-resample within Random: R₁*, R₂* → δ_R*[b] = quantile_diffs(R₁*, R₂*)

**Step 2: Compute per-class thresholds, then take max**

For each quantile i:
```
threshold_F[i] = quantile({|δ_F*[b][i]| : b = 1..B}, 1 − α/9)
threshold_R[i] = quantile({|δ_R*[b][i]| : b = 1..B}, 1 − α/9)
threshold[i] = max(threshold_F[i], threshold_R[i])
```

**Step 3: Decision**

Reject H₀ (leak detected) if any |Δᵢ| > threshold[i]

> **Note:** This is `max(quantile, quantile)`, not `quantile(max)`. The latter (our original spec) is more conservative but has less power. This matches RTLF Algorithm 1 exactly.

#### Why 10,000 Bootstrap Iterations?

With α = 0.01 and Bonferroni correction (α/9 ≈ 0.0011), we need the 0.9989 quantile.
- With B=2,000: only ~2 samples in tail region → unstable
- With B=10,000: ~11 samples in tail region → stable

#### Block Bootstrap

Preserve autocorrelation by resampling contiguous blocks. Block size ≈ √n.

### 4.4 Layer 2: Bayesian Interpretation

> **Key Design Decision: Sample Splitting**
> 
> To make `leak_probability` a calibrated posterior (not just a score), we use **sample splitting**:
> - **Calibration set (30%)**: Estimate Σ₀, compute MDE, set prior hyperparameters
> - **Inference set (70%)**: Compute Δ and Bayes factor using fixed Σ₀ and prior
> 
> This avoids "double-dipping" where data informs both the prior and likelihood, which would make the posterior overconfident.

#### Sample Split Procedure

```
1. Randomly partition each class: Fixed → F_cal (30%), F_inf (70%)
                                  Random → R_cal (30%), R_inf (70%)

2. CALIBRATION PHASE (using F_cal, R_cal):
   - Estimate Σ_F, Σ_R via bootstrap → Σ₀ = Σ_F + Σ_R
   - Compute MDE from Σ₀
   - Set prior: σ_μ = max(2×MDE_μ, min_effect_of_concern_ns), same for σ_τ
   
3. INFERENCE PHASE (using F_inf, R_inf):
   - Compute Δ = quantiles(F_inf) - quantiles(R_inf)
   - Compute Bayes factor using Σ₀ and prior from step 2
   - Compute posterior probability and effect estimate
```

With 100k samples per class, the inference set has 70k samples—still plenty for stable quantile estimation.

#### Covariance Estimation (on calibration set)

Bootstrap **single quantile vectors** (not differences) to avoid factor-of-2 error:

1. For each class, do B=2000 block-bootstrap resamples of the calibration set
2. Compute quantile vector for each resample
3. Compute sample covariance matrix Σ_F and Σ_R
4. **Σ₀ = Σ_F + Σ_R** (variance of difference for independent samples)

#### Numerical Stability

Add small jitter to diagonal before Cholesky:
```rust
let jitter = 1e-9 * sigma.diagonal().iter().cloned().fold(0.0, f64::max).max(1.0);
for i in 0..9 { sigma[(i, i)] += jitter; }
```

#### Effect Model: Orthogonalized Basis

Design matrix **X** (9 × 2):
- Column 1: 𝟙 = [1, 1, 1, 1, 1, 1, 1, 1, 1] (uniform shift)
- Column 2: b_tail = [-0.5, -0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375, 0.5] (centered tail effect)

The centering makes μ (shift) and τ (tail) approximately orthogonal.

Model under H₁:
```
Δ = Xβ + ε,  where β = (μ, τ)ᵀ, ε ~ N(0, Σ₀)
```

#### Prior (set from calibration data)

```
β ~ N(0, Λ₀)
Λ₀ = diag(σ²_μ, σ²_τ)
σ_μ = max(2 × MDE_μ, min_effect_of_concern_ns)
σ_τ = max(2 × MDE_τ, min_effect_of_concern_ns)
```

The `min_effect_of_concern_ns` clamp prevents false positives on very quiet systems.

#### Minimum Detectable Effect (from calibration data)

Sample Δ* ~ N(0, Σ₀) from the null, fit β̂ to each sample, and take 95th percentile of |β̂|.

#### Closed-Form Bayes Factor (on inference data)

**Key insight:** Since both likelihood and prior are Gaussian:

Under H₀: Δ ~ N(0, Σ₀)
Under H₁: Δ ~ N(0, Σ₁) where Σ₁ = Σ₀ + X Λ₀ Xᵀ

Log Bayes factor:
```
log BF₁₀ = log N(Δ; 0, Σ₁) - log N(Δ; 0, Σ₀)
```

For MVN log-pdf with covariance Σ:
```
log N(x; 0, Σ) = -½(d×log(2π) + log|Σ| + xᵀΣ⁻¹x)
```

Use nalgebra's `Cholesky::ln_determinant()` for log|Σ| and `.solve()` for the quadratic form. Add small diagonal jitter (ε = 1e-9 × median(diag(Σ))) before decomposition for numerical stability.

#### Posterior Probability

```rust
fn compute_posterior(log_bf_10: f64, prior_no_leak: f64) -> f64 {
    let prior_odds = (1.0 - prior_no_leak) / prior_no_leak;
    1.0 / (1.0 + (-log_bf_10).exp() / prior_odds)
}
```

#### Effect Posterior

Standard Bayesian linear regression with Gaussian prior:

```
Posterior precision: Λ_post⁻¹ = Xᵀ Σ₀⁻¹ X + Λ₀⁻¹
Posterior mean: β_post = Λ_post Xᵀ Σ₀⁻¹ Δ
Posterior: β|Δ ~ N(β_post, Λ_post)
```

For the 95% CI on total effect magnitude √(μ² + τ²), sample 5000 draws from the posterior and take empirical quantiles (more accurate than delta method for this nonlinear function).

### 4.5 Adaptive Batching (Coarse Timer Platforms)

When timer resolution is coarse relative to operation duration (e.g., Apple Silicon's 
~41ns resolution), individual measurements may yield only a few distinguishable tick 
values. This section describes how to handle this gracefully.

#### Measurability Floor

There is a fundamental limit to what can be measured on coarse-timer platforms. 
Below a certain threshold, we cannot reliably distinguish timing differences from 
quantization noise—and batching trades one artifact (discretization) for another 
(microarchitectural state accumulation).

**Key constants:**

```rust
/// Minimum ticks per operation for reliable single-call measurement
const MIN_TICKS_SINGLE_CALL: f64 = 5.0;

/// Target ticks per batch for stable distributional inference
const TARGET_TICKS_PER_BATCH: f64 = 50.0;

/// Maximum batch size (low to limit microarchitectural artifacts)
const MAX_BATCH_SIZE: u32 = 20;

/// Pilot samples for adaptive K selection
const PILOT_SAMPLES: usize = 100;
```

#### Decision Logic

During warmup, run a pilot to measure operation duration:

```rust
let pilot_ns = measure_pilot(&fixed, &random, PILOT_SAMPLES);
let median_ns = median(&pilot_ns);
let ticks_per_call = median_ns / timer.resolution_ns();

let batch_decision = if ticks_per_call < MIN_TICKS_SINGLE_CALL {
    // Too fast to measure reliably on this platform
    BatchDecision::Unmeasurable {
        operation_ns: median_ns,
        threshold_ns: timer.resolution_ns() * MIN_TICKS_SINGLE_CALL,
    }
} else if ticks_per_call >= TARGET_TICKS_PER_BATCH {
    // Fine-grained enough, no batching needed
    BatchDecision::NoBatching
} else {
    // Batch moderately to improve tick resolution
    let k = ((TARGET_TICKS_PER_BATCH / ticks_per_call).ceil() as u32)
        .clamp(1, MAX_BATCH_SIZE);
    let achieved_ticks = ticks_per_call * k as f64;
    BatchDecision::Batch { k, achieved_ticks }
};
```

#### Decision Matrix

| ticks_per_call | K | Outcome | Rationale |
|----------------|---|---------|-----------|
| < 5 | — | `Unmeasurable` | Can't distinguish signal from quantization noise |
| 5–10 | 10–20 | Batch (partial) | May not reach 50 ticks; results are weaker |
| 10–50 | 2–5 | Batch | Reaches ~50 ticks with minimal μarch artifacts |
| ≥ 50 | 1 | No batching | Fine-grained enough |

#### Platform Implications

| Platform | Timer resolution | Measurable threshold | Notes |
|----------|------------------|---------------------|-------|
| x86_64 (rdtsc) | ~0.3ns | ~2ns | Can measure almost anything |
| Apple Silicon (cntvct) | ~41ns | ~200ns | Many crypto primitives too fast |
| std::time::Instant | ~100–1000ns | ~500ns–5μs | Fallback only |

#### Unmeasurable Outcome

When an operation is too fast to measure, return early with a clear explanation:

```rust
pub enum Outcome {
    /// Analysis completed successfully
    Completed(AnalysisResult),
    
    /// Operation too fast to measure reliably on this platform
    Unmeasurable {
        operation_ns: f64,
        threshold_ns: f64,
        platform: String,
        recommendation: String,
    },
}
```

Terminal output for unmeasurable operations:

```
timing-oracle · xor_compare (32 bytes)
──────────────────────────────────────────────────────────────

  ⚠️  Operation too fast to measure reliably
  
  Estimated duration: ~15ns
  Timer resolution:   41ns (Apple Silicon)
  Minimum measurable: ~200ns on this platform
  
  Recommendations:
  • Run on x86_64 with rdtsc (~0.3ns resolution)
  • Test a larger operation (more bytes, more rounds)
  • If testing crypto primitives, test full encrypt/decrypt call
  • Use dudect-bencher for ultra-fast operations (different methodology)

──────────────────────────────────────────────────────────────
```

#### Why MAX_BATCH_SIZE = 20?

Higher batch sizes cause false positives on constant-time code because:

1. **Branch predictor training**: K iterations of identical input trains predictors 
   differently than K varied inputs
2. **Cache state accumulation**: Same cache lines accessed K times vs K different 
   access patterns
3. **μop cache effects**: Same instruction sequence cached vs varied sequences

These are measurement artifacts, not timing leaks. With K ≤ 20, these effects are 
limited. If K=20 still doesn't reach 50 ticks, the operation is likely unmeasurable.

#### What Batching Tests

When batching is enabled, you're testing **amortized cost over K executions**. This 
is a valid threat model—real attackers often average repeated measurements. But it's 
different from single-call timing.

Report results clearly:

- `effect_ns` is per-call (batch effect ÷ K)
- Document that results reflect K-amortized timing
- For single-call guarantees, require a platform with finer resolution

#### Statistical Considerations

**What batching preserves:**

- Mean differences between classes (detected via shift component μ)
- SNR typically improves for mean-like effects (signal scales K, noise scales √K)

**What batching attenuates:**

Batching smooths the per-sample distribution; it reduces sensitivity to rare slow 
events. Our quantile-based statistic remains shape-sensitive and will detect mean 
shifts, but the τ (tail) component may not isolate rare events specifically.

**Inference operates on batch totals:**

Compute quantiles and covariances on raw batch measurements (in ticks or total ns),
not on per-call divided values:

```rust
// Correct: inference on batch totals
let quantiles_f = compute_quantiles(&batch_totals_f);
let effect_total = posterior.mu;
let effect_per_call = effect_total * timer.resolution_ns() / k as f64;

// Incorrect: division before inference reintroduces quantization
let per_call_f: Vec<f64> = batch_totals_f.iter()
    .map(|&t| t as f64 / k as f64)
    .collect();
```

#### Microarchitectural Mitigations

Even with K ≤ 20, be aware of measurement artifacts:

**Required:**

- Keep K identical across both classes
- Pre-generate random inputs outside the timed region

**Recommended:**

- Use `compiler_fence(SeqCst)` between iterations if cache-sensitive
- Consider memory barrier or dummy access to reset predictor state
- Document batching status in output

#### Output Annotation

When batching is enabled:

```rust
pub struct BatchingInfo {
    pub enabled: bool,
    pub k: u32,
    pub achieved_ticks: f64,
    pub rationale: BatchRationale,
}

pub enum BatchRationale {
    NotNeeded,                          // ticks_per_call >= 50
    Batched { target: f64 },            // reached ~50 ticks
    PartiallyBatched { achieved: f64 }, // K=20 but < 50 ticks
}
```

Terminal output examples:

```
timing-oracle: batching enabled (K=5, ~52 ticks/batch)
               results reflect amortized cost over 5 executions
```

Or for partial batching:

```
timing-oracle: batching enabled (K=20, ~35 ticks/batch)
               ⚠️ below target resolution; results may be less reliable
               results reflect amortized cost over 20 executions
```

#### Example: Apple Silicon

| Operation duration | Timer | Ticks/call | Decision | K | Achieved ticks |
|-------------------|-------|------------|----------|---|----------------|
| 15ns | 41ns | 0.4 | Unmeasurable | — | — |
| 100ns | 41ns | 2.4 | Unmeasurable | — | — |
| 250ns | 41ns | 6.1 | Batch (partial) | 20 | ~122 |
| 500ns | 41ns | 12.2 | Batch | 5 | ~61 |
| 2μs | 41ns | 48.8 | Batch | 2 | ~98 |
| 5μs | 41ns | 122 | No batching | 1 | 122 |

### 4.6 Pre-flight Checks

#### 1. Sanity Check (Fixed vs Fixed)
Split fixed samples in half and run analysis. If "leak" detected → broken harness.

#### 2. Generator Cost Check
For `test_with_state`, measure input generation separately. If fixed/random generators differ by >10%, warn about confounded timing.

#### 3. Autocorrelation Check
Compute ACF on **full interleaved sequence** (not per-class). If lag-1 or lag-2 > 0.3 → warn about periodic interference.

#### 4. CPU Frequency Governor (Linux)
Check `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`. If not "performance" → warn.

---

## 5. Implementation Details

### 5.1 Cycle-to-Nanosecond Conversion

Calibrate by sleeping 1ms and measuring cycles vs wall-clock time. Take median of ~100 iterations.

### 5.2 Exploitability Assessment

Based on Crosby et al. (2009):

| Effect Size | Assessment |
|-------------|------------|
| < 100 ns | Negligible |
| 100–500 ns | PossibleLAN (~10⁵ queries) |
| 500 ns – 20 μs | LikelyLAN (~10⁴ queries) |
| > 20 μs | PossibleRemote |

Include disclaimer: "Heuristic based on typical network conditions."

### 5.3 Quality Assessment

Based on MDE magnitude:
- < 5 ns: Excellent
- 5–20 ns: Good  
- 20–100 ns: Poor
- > 100 ns: TooNoisy

### 5.4 Terminal Output

Use `colored` crate. Format example (without batching):

```
timing-oracle · rsa_decrypt_timing
──────────────────────────────────────────────────────────────

  Samples: 100,000 per class
  Quality: Good
  Min detectable effect: 10 ns (shift), 15 ns (tail)
  
  ⚠️  Timing leak detected
  
    Probability of leak: 97%
    Effect: 850 ns UniformShift (95% CI: 790–920 ns)
    
    Exploitability (heuristic):
      Local network:  Likely (~10⁴ queries)
      Internet:       Unlikely

──────────────────────────────────────────────────────────────
```

Example with batching enabled (Apple Silicon):

```
timing-oracle · aes_ctr_encrypt
──────────────────────────────────────────────────────────────

  Samples: 100,000 per class
  Quality: Good
  Batching: K=21 (51 ticks/batch, timer res 41ns)
            Results reflect amortized cost over 21 executions
  Min detectable effect: 8 ns (shift), 12 ns (tail)
  
  ✓ No timing leak detected
  
    Probability of leak: 12%
    
──────────────────────────────────────────────────────────────
```

---

## 6. Module Structure

```
timing-oracle/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                    # Public API, re-exports
│   │
│   ├── oracle.rs                 # TimingOracle builder and main logic
│   ├── config.rs                 # Config struct
│   ├── result.rs                 # TestResult and related types
│   │
│   ├── measurement/
│   │   ├── mod.rs
│   │   ├── timer.rs              # Platform-specific timing (rdtsc, cntvct, etc.)
│   │   ├── collector.rs          # Sample collection with interleaving
│   │   ├── batching.rs           # Adaptive batching logic (K selection, pilot)
│   │   └── outlier.rs            # Outlier filtering
│   │
│   ├── statistics/
│   │   ├── mod.rs
│   │   ├── quantile.rs           # Quantile computation (selection-based)
│   │   ├── bootstrap.rs          # Block bootstrap implementation
│   │   ├── covariance.rs         # Covariance estimation
│   │   └── autocorrelation.rs    # ACF computation
│   │
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── ci_gate.rs            # Layer 1: RTLF-style CI gate
│   │   ├── bayes.rs              # Layer 2: Closed-form Bayesian inference
│   │   ├── effect.rs             # Effect decomposition and posterior
│   │   └── mde.rs                # Minimum detectable effect
│   │
│   ├── preflight/
│   │   ├── mod.rs
│   │   ├── sanity.rs             # Fixed-vs-Fixed check
│   │   ├── generator.rs          # Generator cost check
│   │   ├── autocorr.rs           # Autocorrelation check
│   │   └── system.rs             # CPU governor, etc.
│   │
│   └── output/
│       ├── mod.rs
│       ├── terminal.rs           # Human-readable output
│       └── json.rs               # JSON serialization
│
├── timing-oracle-macros/         # Proc macros (optional feature)
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                # timing_test! macro
│
├── examples/
│   ├── simple.rs                 # Basic usage
│   ├── aes.rs                    # AES encryption timing test
│   ├── compare.rs                # Constant-time comparison
│   └── ci_integration.rs         # CI usage patterns
│
└── tests/
    ├── known_leaky.rs            # Must detect known leaks
    ├── known_safe.rs             # Must not false positive
    ├── calibration.rs            # Verify statistical properties
    └── integration.rs            # End-to-end tests
```

---

## 7. Dependencies

### Crate Selection Rationale

| Purpose | Crate | Why |
|---------|-------|-----|
| **Linear algebra** | `nalgebra` 0.34 | Pure Rust, built-in Cholesky with `.solve()` and `.ln_determinant()`, static 9×9 sizing via `SMatrix`, 2.1M downloads/mo |
| **RNG** | `rand` 0.9 + `rand_distr` 0.5 | De facto standard, 48M downloads/mo |
| **Serialization** | `serde` + `serde_json` | De facto standard |
| **Terminal colors** | `colored` 3.0 | Simple API, respects NO_COLOR, actively maintained |
| **Parallelization** | `rayon` 1.11 (optional) | Easy `.par_iter()` for bootstrap if needed |

**Not using external crates for:**
- **Timing**: `core::arch::x86_64::_rdtsc()` and inline asm for ARM are in std. Crates like `quanta` add calibration overhead.
- **Quantile computation**: Use `select_nth_unstable` directly—O(n) vs O(n log n) for sort-based methods in `statrs`.
- **Statistics**: We're implementing custom methodology; general stats crates don't help.

### Cargo.toml

```toml
[package]
name = "timing-oracle"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
license = "MIT OR Apache-2.0"
description = "Detect timing side channels in cryptographic code"
repository = "https://github.com/..."
keywords = ["security", "cryptography", "timing", "side-channel"]
categories = ["cryptography", "development-tools::testing"]

[dependencies]
# Linear algebra - 9x9 matrices with Cholesky
nalgebra = { version = "0.34", default-features = false, features = ["std"] }

# Random number generation
rand = "0.9"
rand_distr = "0.5"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Terminal output
colored = "3.0"

# Optional parallelization for bootstrap
rayon = { version = "1.11", optional = true }

[dev-dependencies]
aes-gcm = "0.10"
subtle = "2.5"

[features]
default = []
parallel = ["rayon"]
macros = ["timing-oracle-macros"]
```

### nalgebra Type Aliases

```rust
use nalgebra::{SMatrix, SVector, Matrix2, Vector2, Cholesky};

pub type Matrix9 = SMatrix<f64, 9, 9>;
pub type Vector9 = SVector<f64, 9>;
pub type Matrix9x2 = SMatrix<f64, 9, 2>;
pub type Cholesky9 = Cholesky<f64, nalgebra::U9>;
```

nalgebra's `Cholesky` provides:
- `.l()` — lower triangular factor
- `.solve(&b)` — solve Σx = b
- `.ln_determinant()` — numerically stable log|Σ|
- `.inverse()` — compute Σ⁻¹

---

## 8. Testing Strategy

### 8.1 Known-Leaky Tests

Must detect:
- **Branch timing**: `if x == 0 { spin_loop(); }`
- **Early-exit comparison**: `a == b` (non-constant-time)

Threshold: `leak_probability > 0.9`, `ci_gate.passed == false`

### 8.2 Known-Safe Tests

Must not false positive on constant-time code (e.g., XOR-based comparison).

Threshold: `leak_probability < 0.5`, `ci_gate.passed == true`

### 8.3 Calibration Tests

**CI Gate FPR Calibration:**

Run ~100 trials on pure noise (split random data in half). Verify rejection rate ≤ 2×α.

> **Important:** RTLF uses IID resampling; we use block bootstrap to handle autocorrelation. This is a deliberate deviation. The calibration test verifies that our block bootstrap still achieves bounded FPR. If this test fails, reduce block size or fall back to IID resampling.

**Bayesian Layer Calibration:**

On known-null data, `leak_probability` should be roughly uniformly distributed (or at least not concentrated near 1.0). Run ~100 trials and check that <10% have `leak_probability > 0.9`.

### 8.4 Numerical Stability Tests

- Near-singular covariance (highly correlated quantiles)
- Zero variance (constant samples)

Must not crash or produce NaN.

### 8.5 Multiple Testing in CI

> **Warning for users:** Even with α=0.01 per test, running N tests gives P(≥1 false positive) ≈ 1 - (1-α)^N. For 500 tests: ~99% chance of at least one false positive.
>
> Recommendations:
> - Use smaller α (e.g., 0.001) for large test suites
> - Treat first failure as "warning" requiring confirmation on re-run
> - Use hierarchical gating: `leak_probability > 0.99` = fail, `> 0.9` = warn

---

## 9. References

### Academic Papers

1. **Reparaz, O., Balasch, J., & Verbauwhede, I. (2016)**. "Dude, is my code constant time?" Design, Automation & Test in Europe (DATE).
   - Original dudect methodology using Welch's t-test

2. **Dunsche, M., et al. (2024)**. "With Great Power Come Great Side Channels: Statistical Timing Side-Channel Analyses with Bounded Type-1 Errors." USENIX Security.
   - RTLF methodology: quantile-based, bootstrap thresholds, bounded FPR
   - **Key source for Layer 1 (CI Gate) design**

3. **Crosby, S. A., Wallach, D. S., & Riedi, R. H. (2009)**. "Opportunities and limits of remote timing attacks." ACM Transactions on Information and System Security (TISSEC).
   - Exploitability thresholds (100ns, 500ns, 20μs)

### Statistical Methods

4. **Künsch, H. R. (1989)**. "The Jackknife and the Bootstrap for General Stationary Observations." Annals of Statistics.
   - Block bootstrap for correlated data

5. **Bayesian Linear Regression**. Standard conjugate Gaussian model.
   - Posterior: β|Δ ~ N(β_post, Λ_post)
   - Marginal likelihood: Δ ~ N(0, Σ₀ + X Λ₀ Xᵀ)

### Existing Tools

6. **dudect** (C): https://github.com/oreparaz/dudect
   - Mean-only t-test, no shape detection

7. **dudect-bencher** (Rust): https://github.com/rozbb/dudect-bencher
   - Abandoned, security vulnerabilities

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| Δ | 9-dimensional vector of signed quantile differences |
| Δₚ | Quantile difference at percentile p |
| Σ₀ | Covariance matrix of Δ under H₀ (no leak) |
| Σ₁ | Covariance matrix of Δ under H₁ (leak) |
| X | 9×2 design matrix [𝟙 \| b_tail] |
| β = (μ, τ) | Effect parameters (shift, tail) |
| Λ₀ | Prior covariance for β |
| BF₁₀ | Bayes factor for H₁ vs H₀ |
| α | CI gate false positive rate |
| MDE | Minimum detectable effect |

## Appendix B: Constants

```rust
/// Decile percentiles
pub const DECILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Unit vector (for uniform shift)
pub const ONES: [f64; 9] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Centered tail basis vector
pub const B_TAIL: [f64; 9] = [-0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5];

/// ln(2π)
pub const LOG_2PI: f64 = 1.8378770664093453;

/// Minimum ticks per operation for reliable single-call measurement
/// Below this threshold, operations are considered unmeasurable
pub const MIN_TICKS_SINGLE_CALL: f64 = 5.0;

/// Target ticks per batch for stable distributional inference
pub const TARGET_TICKS_PER_BATCH: f64 = 50.0;

/// Maximum batch size to limit microarchitectural artifacts
/// Kept low (20) to avoid false positives from cache/predictor state
pub const MAX_BATCH_SIZE: u32 = 20;

/// Number of pilot samples for adaptive K selection
pub const PILOT_SAMPLES: usize = 100;
```
