# Timing-Oracle Test Adapter (Draft)

This draft focuses on a builder-first API that drops cleanly into `cargo test` and `cargo nextest` without requiring a proc macro. The aim: first-class CI ergonomics (env overrides, presets, seeds, reports) while keeping the surface simple and debuggable. A proc macro can arrive later if people want an attribute style.

## Minimal v1 (builder-first)

```rust
#[test]
fn compare_is_constant_time() {
    timing_oracle::TimingOracle::ci_test()
        .from_env() // TO_MODE, TO_SAMPLES, TO_ALPHA, TO_REPORT, TO_SEED, etc.
        .samples(50_000)
        .alpha(0.01)
        .mode(Mode::Full)
        .run(
            || fixed_compare(&FIXED),
            || random_compare(&rand_input()),
        )
        .unwrap_or_report(); // prints summary and panics if leak detected
}
```

Why builder-first?
- No proc-macro crate or compile-time cost.
- No redundant wrapper (`#[test]` + `run_pair`).
- Easier to debug and step through.
- Env handling and seeding live in the builder (`from_env`, `unwrap_or_report`).

## Builder surface (proposed)

- `from_env()` — merges env vars; explicit setters override:
  - `TO_SAMPLES`, `TO_ALPHA`, `TO_MODE`, `TO_REPORT`, `TO_SEED`, `TO_MAX_DURATION_MS`,
    `TO_CALIBRATION_FRAC`, `TO_EFFECT_PRIOR_NS`, `TO_EFFECT_THRESHOLD_NS`, `TO_FAIL_ON`,
    `TO_ASYNC_WORKLOAD` (true/1 to inflate priors/thresholds for async noise).
- Tunables:
  - `samples(u64)` — total samples; note 30/70 split means only ~70% feed inference unless overridden.
  - `calibration_fraction(f32)` — default 0.3 (preflight/calibration split); exposed so users know how samples are used.
  - `alpha(f64)` — FPR bound; keep ≤ 0.01 even in smoke mode to avoid flakiness.
  - `mode(Mode)` — preset; currently just sets samples/alpha/prior defaults.
    - `Mode::Smoke`: ~20k samples, alpha 0.01, effect prior ~20ns.
    - `Mode::Full`: ~100k samples, alpha 0.01, effect prior ~10ns.
  - `effect_prior_ns(f64)` — prior scale (σ_μ). Rename from “min_effect” to avoid implying pass/fail.
  - `effect_threshold_ns(f64)` — optional hard cutoff for reporting/panic, distinct from the prior.
  - `max_duration_ms(u64)` — guardrail to avoid hanging CI.
  - `seed(u64)` — deterministic RNG seed; if omitted, pick random and print on failure.
  - `report_path(PathBuf)` — write JSON/JUnit-lite.
  - `fail_on(FailCriterion)` — controls pass/fail logic:
    - `CiGate` (default, bounded FPR).
    - `Probability(f64)` — Bayesian posterior threshold.
    - `Either` — fail if any signal trips.
  - `async_workload(bool)` — mark workloads expected to be async/jittery; inflates priors/thresholds and logs the async flag.
- Execution:
  - `run(fixed, random) -> Result<TestResult, CiFailure>`
  - `unwrap_or_report()` — prints concise lines (seed/mode/samples + stats or error), writes report if configured, then panics on leak/ci-failure.

## Optional proc macro (v1.1+)

If demand exists, add `#[timing_oracle::test(...)]` that injects the builder call (no extra `run_pair`). Possible shape:

```rust
#[timing_oracle::test(samples = 50_000)]
fn compare_is_constant_time(t: &mut TimingTest) {
    t.compare(|| fixed_compare(&FIXED), || random_compare(&rand_input()));
}
```

The macro would expand to a normal `#[test]`, keeping discovery automatic. But the builder remains the primary path.

## Failure output shape

Example failure under `cargo test`:

```
---- compare_is_constant_time stdout ----
timing-oracle: seed=0x9f31c2a4 mode=Full samples=50_000 alpha=0.01 split=30/70
timing-oracle: ci_gate=tripped leak_probability=0.92 effect_ns=34.7 exploitability=LikelyLAN
timing-oracle: report saved to /tmp/timing-oracle/report-compare_is_constant_time.json

thread 'compare_is_constant_time' panicked at 'timing leak detected (fail_on=CiGate, p=0.92, effect=34.7ns, exploitability=LikelyLAN, seed=0x9f31c2a4)', tests/timing.rs:12:5
note: run with `RUST_LOG=timing_oracle=debug` for phase details
```

`cargo test -q` still shows the panic line; `nextest` shows the panic and any logged lines.

## Async warning

Async timing is noisier (scheduler/task switches). Defaults should adjust when an async helper is used:
- Raise `effect_prior_ns` and/or `effect_threshold_ns`.
- Maybe enforce single-threaded runtime by default.
- Emit a warning in logs when running in async mode so users know the SNR is different.

## Ergonomic behaviors (kept)

- Short, human-friendly panic message; logs stay quiet unless `RUST_LOG=timing_oracle=debug` or `TIMING_ORACLE_DEBUG=1`.
- Seeds printed on failure; `TO_SEED` forces determinism.
- Reports: opt-in via `report_path`/`TO_REPORT`; also auto-dump to a temp file when `--timing-oracle-dump` appears in `std::env::args`, logging the path.
- Time guard: `max_duration_ms` aborts with a clear error rather than hanging CI.
- Single-threaded measurement by default; opt-in parallel via feature/env if measurement supports it.
- Clear `CiFailure` variants (measurement error, timeout, statistical positive).

## CI and nextest patterns

- `cargo test` default: rely on `Mode::Smoke` + `from_env()` so CI can tighten via env:
  - `TO_MODE=full` in PR/nightly jobs.
  - `TO_REPORT=target/timing-oracle.json` to collect artifacts.
  - `TO_SEED=$(date +%s)` for reproducibility across reruns.
- `cargo nextest`: recommend a `timing` profile with `test-threads = 1` and maybe `--nocapture`:
  ```toml
  [profile.timing]
  test-threads = 1
  status-level = "fail"
  failure-output = "immediate"
  ```
  Run with `NEXTEST_PROFILE=timing TO_MODE=smoke`.

## Open questions to validate

- Async defaults: use `current_thread` or `multi_thread`? How much to inflate `effect_prior_ns`?
- Should `mode` adjust trimming/quantile params beyond samples/alpha/prior?
- Report format: JSON schema vs JUnit-lite for CI upload — likely both.
- Do we expose `calibration_fraction` or keep it fixed to avoid user footguns?

Feedback welcome; once settled, we can wire the builder API and consider the proc-macro shim if demand appears.***
