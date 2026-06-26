// 🎯 mAR: learned residual gate (Sinkhorn-Knopp → Birkhoff → DTopK).
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct MAR<B: Backend> {
    gate: BitLinear<B>,
    bias: Tensor<B, 2>,
    pub dtopk_k: usize,
    pub sk_iter: usize,
}

impl<B: Backend> MAR<B> {
    pub fn new(dm: usize, k: usize, sk: usize, dev: &B::Device) -> Self {
        Self {
            gate: BitLinear::new(1, dm, false, dev),
            bias: Tensor::zeros([1, 1], dev),
            dtopk_k: k,
            sk_iter: sk,
        }
    }

    /// out = g * fx + (1-g) * x, где g ∈ [0,1] — doubly-stochastic + sparsified
    pub fn forward(&self, x: Tensor<B, 3>, fx: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _dm] = x.dims();
        // gate logits: [B, T, 1]
        let g = self.gate.forward(x.clone()) + self.bias.clone().reshape([1, 1, 1]);
        // Sinkhorn-Knopp: нормализация до doubly-stochastic [B, T, 1]
        // В 1D случае SK — просто softmax по T
        let mut g = g.reshape([b, t]);
        // exp для неотрицательности
        g = (g.clone() - g.clone().max_dim(1).reshape([b, 1])).exp().clamp_min(1e-8);

        // Sinkhorn-Knopp итерации (на матрице [b, t] → row/col norm)
        for _ in 0..self.sk_iter {
            let row_sum = g.clone().sum_dim(1).clamp_min(1e-8).reshape([b, 1]);
            g = g / row_sum;
            let col_sum = g.clone().sum_dim(0).clamp_min(1e-8).reshape([1, t]);
            g = g / col_sum;
        }

        // DTopK: нули кроме top-k по T
        if self.dtopk_k > 0 && self.dtopk_k < t {
            let k = self.dtopk_k.min(t);
            let topk_vals = g.clone().topk(k, 1);
            let thresh = topk_vals.narrow(1, k - 1, 1);
            let mask = g.clone().greater_equal(thresh).float();
            g = g * mask;
        }

        let g = g.reshape([b, t, 1]).clamp(0.0, 1.0);
        g.clone() * fx + (g.neg() + 1.0) * x
    }
}
