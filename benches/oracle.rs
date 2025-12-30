use criterion::{black_box, criterion_group, criterion_main, Criterion};
use timing_oracle::{FailCriterion, TimingOracle};

fn bench_oracle_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("timing_oracle");
    group.sample_size(20);
    group.bench_function("baseline_addition", |b| {
        b.iter(|| {
            // Empty timing oracle run on trivial closures; keeps samples small to avoid long benches.
            let result = TimingOracle::new()
                .samples(500)
                .test(|| black_box(1u64 + 1), || black_box(2u64 + 2));
            black_box(result.leak_probability)
        });
    });

    group.bench_function("ci_builder_probability_fail", |b| {
        b.iter(|| {
            let _ = TimingOracle::ci_test()
                .samples(500)
                .fail_on(FailCriterion::Probability(0.0))
                .run(|| black_box(3u64 + 3), || black_box(4u64 + 4));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_oracle_simple);
criterion_main!(benches);
