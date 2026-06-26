// SeRoPE: Stabilized Rotary Position Embedding for GDN-2.
use burn::tensor::{backend::Backend, Tensor};

/// Apply rotary embeddings to Q and K tensors.
/// q, k: [B, H, T, D] — already split into heads
pub fn apply_serope<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [b, h, t, d] = q.dims();
    let dev = q.device();

    // Frequencies: θ_i = 1 / 10000^(2i/d) for i = 0..d/2
    let d2 = d / 2;
    let freqs: Vec<f64> = (0..d2).map(|i| 1.0 / (10000.0_f64.powf(2.0 * i as f64 / d as f64))).collect();
    let pos: Vec<f64> = (0..t).map(|i| i as f64).collect();

    // Outer product: [T, d2] — cos/sin table
    let mut cos_vals = vec![0.0f32; t * d2];
    let mut sin_vals = vec![0.0f32; t * d2];
    for ti in 0..t { for fi in 0..d2 {
        let angle = pos[ti] * freqs[fi];
        cos_vals[ti * d2 + fi] = angle.cos() as f32;
        sin_vals[ti * d2 + fi] = angle.sin() as f32;
    }}

    // Duplicate for real/imag pairs: [d/2] → [d] by interleaving
    let mut cos_full = vec![0.0f32; t * d];
    let mut sin_full = vec![0.0f32; t * d];
    for ti in 0..t { for fi in 0..d2 {
        cos_full[ti * d + 2*fi] = cos_vals[ti * d2 + fi];
        cos_full[ti * d + 2*fi + 1] = cos_vals[ti * d2 + fi];
        sin_full[ti * d + 2*fi] = sin_vals[ti * d2 + fi];
        sin_full[ti * d + 2*fi + 1] = sin_vals[ti * d2 + fi];
    }}

    let cos_t: Tensor<B, 2> = Tensor::from_floats(cos_full.as_slice(), &dev).reshape([t, d]);
    let sin_t: Tensor<B, 2> = Tensor::from_floats(sin_full.as_slice(), &dev).reshape([t, d]);

    // Broadcast to [1, 1, T, D] for Q/K
    let cos = cos_t.reshape([1, 1, t, d]);
    let sin = sin_t.reshape([1, 1, t, d]);

    // Rotate: q_rot = q*cos + rotate_half(q)*sin, k_rot = k*cos + rotate_half(k)*sin
    let q_rot = q.clone() * cos.clone() + rotate_half(q) * sin.clone();
    let k_rot = k * cos + rotate_half(k) * sin;

    (q_rot, k_rot)
}

/// Rotate half: swap pairs, negate first of each pair
/// [..., 2i, 2i+1] → [..., -2i-1, 2i]
fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, t, d] = x.dims(); let d2 = d / 2;
    // Reshape to [B, H, T, d/2, 2] → swap last two dims → flatten
    let xp = x.reshape([b, h, t, d2, 2]); // [..., i, 0] = even, [..., i, 1] = odd
    let ev = xp.clone().narrow(4, 0, 1).reshape([b, h, t, d2]); // even
    let od = xp.narrow(4, 1, 1).reshape([b, h, t, d2]); // odd
    // Interleave: [-odd, even]
    let neg_od = od.clone() * (-1.0f64);
    // pony: interleave manually — [-od[0], ev[0], -od[1], ev[1], ...]
    let mut parts = Vec::with_capacity(d);
    for i in 0..d2 {
        parts.push(neg_od.clone().narrow(3, i, 1) * 1.0); // need actual insertion
    }
    // pony: simpler — just do paired reshape trick
    let x2 = x.reshape([b, h, t, d2, 2]);
    // Swap and negate: [..., i, 0] = -x[..., i, 1]; [..., i, 1] = x[..., i, 0]
    let swapped = Tensor::cat(vec![
        x2.clone().narrow(4, 1, 1) * (-1.0f64),
        x2.narrow(4, 0, 1),
    ], 4);
    swapped.reshape([b, h, t, d])
}
