# Spec Gap Checklist

Tracking divergences between the current implementation and the v1.0 spec.

## Core statistical pipeline
- [x] Covariance estimation bootstraps **single-class quantile vectors**, not differences (spec §4.4).
- [x] Sample splitting is **random** and used consistently (calibration/inference split).
- [x] Bayesian layer uses **Σ1 = Σ0 + X Λ0 Xᵀ** with log BF from MVN log-pdf ratio (spec §4.4).
- [x] Priors derived from **MDE + min_effect_of_concern_ns** clamp (spec §4.4).
- [x] MDE uses calibration Σ0 and **95th percentile of |β̂|** under null with adequate simulations (spec §4.4).
- [x] Effect credible interval computed from **posterior samples of √(μ²+τ²)** (spec §4.4).

## CI gate (Layer 1)
- [x] RTLF-style **within-class block bootstrap** for thresholds (not MVN simulation).
- [x] Use `Config::ci_bootstrap_iterations` (default 10,000) with Bonferroni α/9.
- [x] Thresholds computed as **max(quantile(|δ_F*|), quantile(|δ_R*|))**, not quantile(max).

## Measurement + preflight
- [x] Preserve **true interleaved sequence** for autocorrelation checks.
- [x] Implement generator cost timing and hook into preflight.
- [x] Sanity check runs full fixed-vs-fixed analysis (not mean/variance heuristic).
- [x] CPU governor + turbo + SMT + VM + load checks (spec §4.5).
- [x] Cycle-to-ns calibration: **~100 × 1ms** median (spec §5.1).

## Public API + macros
- [x] `Config::min_effect_of_concern_ns` available and used; align naming with spec.
- [x] `TimingOracle::test_with_state` signature uses `&mut impl Rng` (spec §2, user opted out of exact signature match).
- [x] `timing_test!` macro and optional `timing-oracle-macros` crate/feature.

## Output + examples + deps
- [x] Terminal output matches spec layout and exploitability breakdown (LAN vs Internet).
- [x] Examples missing: `aes.rs`, `ci_integration.rs`.
- [x] Dependency versions: nalgebra 0.34 (spec §7) and macros feature in `Cargo.toml`.

## Tests
- [ ] Calibration tests meaningful after RTLF/bootstrap implementation.
- [ ] Numerical stability tests for near-singular covariance and zero variance.
