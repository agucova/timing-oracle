use timing_oracle::{CiTestBuilder, FailCriterion, Mode};

fn main() {
    let builder = CiTestBuilder::new()
        .mode(Mode::Smoke)
        .samples(20_000)
        .fail_on(FailCriterion::Either { probability: 0.9 });

    let result = builder
        .unwrap_or_report(
            || std::hint::black_box(1 + 1),
            || std::hint::black_box(2 + 2),
        );

    println!("Leak probability: {:.2}", result.leak_probability);
}
