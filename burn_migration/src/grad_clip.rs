// Gradient clipping by global L2 norm. Одна GPGPU kernel chain без synс.
use burn::tensor::{backend::Backend, Tensor};

/// clip: g ← g * min(1, max_norm / ||g||₂).
/// Single clone for norm computation, no GPU→CPU sync (всё в тензорах).
pub fn clip_by_norm<B: Backend, const D: usize>(g: Tensor<B, D>, max_norm: f64) -> Tensor<B, D> {
    let dims = g.dims();
    let norm = g.clone().powf_scalar(2.0).sum().sqrt().clamp_min(1e-8);
    let scale = (norm.recip() * (max_norm as f32)).clamp_max(1.0f32);
    let ones: [usize; D] = core::array::from_fn(|_| 1);
    g * scale.reshape(ones).expand(dims)
}
