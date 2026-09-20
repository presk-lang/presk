//! Splittable random keys (ChaCha20), for reproducible initialization and
//! sampling: a key is 32 bytes; `split` derives children deterministically;
//! `uniform`/`normal` draw from the stream of a key.

use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Key(pub [u8; 32]);

impl Key {
    pub fn from_seed(seed: u64) -> Key {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut k = [0u8; 32];
        rng.fill_bytes(&mut k);
        Key(k)
    }

    pub fn split(&self, n: usize) -> Vec<Key> {
        let mut rng = ChaCha20Rng::from_seed(self.0);
        (0..n)
            .map(|_| {
                let mut k = [0u8; 32];
                rng.fill_bytes(&mut k);
                Key(k)
            })
            .collect()
    }

    /// The `i`-th child of a key, derived in constant time — `split(i + 1)`
    /// would cost `i` draws. Used for the per-step keys of a training run.
    pub fn child(&self, i: u64) -> Key {
        let mut seed = self.0;
        for (b, x) in seed[..8].iter_mut().zip(i.to_le_bytes()) {
            *b ^= x;
        }
        let mut rng = ChaCha20Rng::from_seed(seed);
        let mut k = [0u8; 32];
        rng.fill_bytes(&mut k);
        Key(k)
    }

    pub fn rng(&self) -> ChaCha20Rng {
        ChaCha20Rng::from_seed(self.0)
    }

    pub fn short(&self) -> String {
        self.0[..4].iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// Uniform double in [0, 1) with 53 random bits.
pub fn uniform01(rng: &mut ChaCha20Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal by Box–Muller.
pub fn normal(rng: &mut ChaCha20Rng) -> f64 {
    let u1 = loop {
        let u = uniform01(rng);
        if u > 0.0 {
            break u;
        }
    };
    let u2 = uniform01(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
