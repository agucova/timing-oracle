use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};

fn main() {
    let key = Key::<Aes256Gcm>::from_slice(&[0u8; 32]);
    let cipher = Aes256Gcm::new(key);
    let nonce = Nonce::from_slice(&[0u8; 12]);

    let fixed_plaintext = vec![0u8; 1024];

    let result = timing_oracle::TimingOracle::new()
        .samples(50_000)
        .test(
            || {
                let _ = cipher
                    .encrypt(nonce, fixed_plaintext.as_slice())
                    .expect("encryption should succeed");
            },
            || {
                let mut random_plaintext = vec![0u8; 1024];
                for byte in &mut random_plaintext {
                    *byte = rand::random();
                }
                let _ = cipher
                    .encrypt(nonce, random_plaintext.as_slice())
                    .expect("encryption should succeed");
            },
        );

    println!("Leak probability: {:.2}", result.leak_probability);
}
