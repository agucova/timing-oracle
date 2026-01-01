use timing_oracle::helpers::InputPair;
use timing_oracle::{CiTestBuilder, FailCriterion, Mode};

fn main() {
    let builder = CiTestBuilder::new()
        .mode(Mode::Smoke)
        .samples(20_000)
        .fail_on(FailCriterion::Either { probability: 0.9 });

    // Simulate timing difference via iteration count
    let inputs = InputPair::new(|| 1u32, || 2u32);

    let result = builder.unwrap_or_report(inputs, |x| {
        std::hint::black_box(*x);
    });

    println!("Leak probability: {:.2}", result.leak_probability);
}
