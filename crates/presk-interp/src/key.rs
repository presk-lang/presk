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

/// Standard normal by Box–Muller, in `f64`: `√(−2 ln u) · cos(2π u′)` with
/// the `libm` crate's `log` and `cos`, musl's, which the native backend and
/// the kernel interpreter compute with (D159), so that a key draws the same
/// doubles on every machine. Until D211 they were Rust's standard library's,
/// the platform's C library: glibc gave the same bits on x86-64 and aarch64,
/// Apple's did not.
pub fn normal(rng: &mut ChaCha20Rng) -> f64 {
    let u1 = loop {
        let u = uniform01(rng);
        if u > 0.0 {
            break u;
        }
    };
    let u2 = uniform01(rng);
    (-2.0 * libm::log(u1)).sqrt() * libm::cos(2.0 * std::f64::consts::PI * u2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The draws of a key are the same doubles on every machine (D211): the
    /// first four of the key seed 0 names, and an FNV-1a fold of the bits of
    /// the next 100 000. With glibc's `log` and `cos` the first is
    /// `0x3feac568a525e1d2` and the fold `0x25befa4d25506d93` (falicon,
    /// 2026-11-17); Apple's differ in other draws. A machine on which this
    /// fails draws other parameters and inputs than the manual's documents
    /// were written with.
    #[test]
    fn a_key_draws_the_same_normals_on_every_machine() {
        let mut rng = Key::from_seed(0).rng();
        let first: Vec<u64> = (0..4).map(|_| normal(&mut rng).to_bits()).collect();
        let fold = (0..100_000).fold(0xcbf2_9ce4_8422_2325u64, |h, _| (h ^ normal(&mut rng).to_bits()).wrapping_mul(0x0000_0100_0000_01b3));
        assert_eq!(first, [0x3feac568a525e1d1, 0xbfc884e0b5405abc, 0xbff1feb45f60e89a, 0x3fee552e3fbd0476]);
        assert_eq!(fold, 0xf10af503abdab221);
    }
}
