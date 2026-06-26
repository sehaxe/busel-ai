// 🚀 CubeCL-ускоренные ядра для горячего пути.
// Все через флаги, fallback на Burn-операции.
// CubeCL ядра требуют ComputeClient — подключатся после компиляции основы.

use burn::tensor::{backend::Backend, Tensor, Int};

/// Histogram expert assignments — 1 kernel вместо ne проходов.
/// Fallback на Burn ops (ne итераций).
#[allow(dead_code)]
pub fn moe_histogram<B: Backend>(
    routes: Tensor<B, 1, Int>,
    ne: usize,
    dev: &B::Device,
) -> Tensor<B, 1> {
    let mut counts = Tensor::zeros([ne], dev);
    for ei in 0..ne {
        let mask = routes.clone().equal_elem(ei as i64).float();
        counts = counts + mask.sum().reshape([1]);
    }
    counts
}

/// Column norm: x / sqrt(sum(x^2, dim=-1)).
/// Burn fusion: x^2 → sum → sqrt в 2 кернела.
/// CubeCL: всё в 1 с shared memory reduction.
#[allow(dead_code)]
pub fn col_norm2<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let n = (x.clone() * x.clone()).sum_dim(0).sqrt().clamp_min(1e-8);
    x / n
}
