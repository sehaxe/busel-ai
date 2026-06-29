// DropBP + LCSB + progressive freeze — комбинированная skip-маска.

/// Комбинированная маска для forward_mask.
/// layer пропускает backward если `skip[i] == true`.
/// - dropbp_prob: DropBP — вероятность скипнуть слой (на каждый шаг, независимо)
/// - backward_ratio: LCSB — доля слоёв, которые держат backward
/// - freeze_progress: 0..1 — какая часть слоёв заморожена (0 = ничего, 1 = всё кроме последнего)
pub fn combined_mask(
    n_layers: usize,
    dropbp_prob: f64,
    backward_ratio: f64,
    freeze_progress: f64,
    seed: u64,
) -> Vec<bool> {
    let mut mask = vec![false; n_layers];

    // 1. DropBP: prob на каждый слой
    if dropbp_prob > 0.0 {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for m in &mut mask {
            let p = (rng.next_u64() % 1000000) as f64 / 1000000.0;
            if p < dropbp_prob {
                *m = true;
            }
        }
    }

    // 2. LCSB: переопределяет — оставляет backward_ratio слоёв
    if backward_ratio < 1.0 {
        let keep_n = (n_layers as f64 * backward_ratio).round() as usize;
        if keep_n < n_layers {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(42));
            let mut idx: Vec<usize> = (0..n_layers).collect();
            for i in (n_layers - keep_n..n_layers).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                idx.swap(i, j);
            }
            let mut keep = vec![false; n_layers];
            for &i in idx.iter().skip(n_layers - keep_n) {
                keep[i] = true;
            }
            // Пересечение: слой держит, только если LCSB разрешает
            for (m, &k) in mask.iter_mut().zip(keep.iter()) {
                *m = *m || !k;
            }
        }
    }

    // 3. Progressive freeze: заморозка первых freeze_progress слоёв
    if freeze_progress > 0.0 {
        let frozen_n = ((n_layers - 1) as f64 * freeze_progress.min(1.0)).round() as usize;
        for i in 0..frozen_n.min(n_layers.saturating_sub(1)) {
            mask[i] = true;
        }
    }

    // Последний слой всегда держит backward
    if let Some(last) = mask.last_mut() {
        *last = false;
    }

    mask
}
