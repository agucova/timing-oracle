//! Example: Testing AES-256-GCM encryption for timing leaks.
//!
//! This demonstrates the CORRECT way to test crypto operations:
//! - Pre-generate all inputs before measurement
//! - Both closures execute identical code paths
//! - Only the input data differs

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use timing_oracle::helpers;

fn main() {
    let key = Key::<Aes256Gcm>::from_slice(&[0u8; 32]);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(&[0u8; 12]);

    // ✅ CORRECT: Pre-generate inputs outside closures
    // Both fixed and random generators are called BEFORE measurement
    let plaintexts = helpers::byte_vecs(1024);

    let result = timing_oracle::TimingOracle::new()
        .samples(50_000)
        .test(
            || {
                // Both closures do identical operations:
                // 1. Get next input from InputPair (identical indexing)
                // 2. Encrypt it
                let _ = cipher
                    .encrypt(nonce, plaintexts.fixed().as_slice())
                    .expect("encryption should succeed");
            },
            || {
                let _ = cipher
                    .encrypt(nonce, plaintexts.random().as_slice())
                    .expect("encryption should succeed");
            },
        );

    println!("Leak probability: {:.2}%", result.leak_probability * 100.0);
    println!("CI gate: {}", if result.ci_gate.passed { "PASS" } else { "FAIL" });

    if let Some(effect) = result.effect {
        println!("Effect size: {:.1} ns", effect.shift_ns);
    }
}
