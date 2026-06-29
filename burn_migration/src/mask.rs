// Skip masks для selective backward / checkpointing.
// LCSB: random per-micro-batch. Ckpt: deterministic every-N.

/// Random LCSB mask (Fisher-Yates, n из n*ratio слоёв без backward).
/// seed обеспечивает разную маску на каждый micro-batch.
fn lcsb_mask(n: usize, ratio: f64, seed: u64) -> Vec<bool> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let threshold = (n as f64 * ratio).min(n as f64) as usize;
    if threshold == 0 { return vec![false; n]; }
    if threshold >= n { return vec![true; n]; }
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (n - threshold..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
    let mut mask = vec![false; n];
    for &i in idx.iter().skip(n - threshold) { mask[i] = true; }
    mask
}

/// Build skip mask: LCSB (random) or deterministic (ckpt_every).
/// - ckpt_every > 0: skip (detach) слои НЕ кратные ckpt_every.
/// - selective: LCSB случайная маска.
/// - last layer всегда keep (иначе градиент не течёт).
pub fn build_skip_mask(n_layers: usize, selective: bool, backward_ratio: f64, ckpt_every: usize, seed: u64) -> Vec<bool> {
    if ckpt_every > 0 {
        (0..n_layers).map(|i| i % ckpt_every != 0 && i != n_layers - 1).collect()
    } else if selective {
        lcsb_mask(n_layers, backward_ratio, seed)
    } else {
        vec![]
    }
}
