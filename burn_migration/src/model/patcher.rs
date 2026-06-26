// ByteFlow patcher: embed → GLU → boundary conv → pool → causal conv → norm.
// Burn nn::Conv1d вместо ручного conv1d — 1 cuDNN вызов вместо k.
use burn::{
    module::Module,
    nn::{RmsNorm, RmsNormConfig, conv::{Conv1d, Conv1dConfig}, PaddingConfig1d},
    tensor::{backend::Backend, Tensor, Int, Distribution},
};
use crate::config::BuselConfig;
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct ByteFlowPatcher<B: Backend> {
    embed: Tensor<B, 2>, gate_down: BitLinear<B>, gate_up: BitLinear<B>,
    boundary_conv: Conv1d<B>, conv: Conv1d<B>,
    norm: RmsNorm<B>, n_patches: usize,
}

impl<B: Backend> ByteFlowPatcher<B> {
    pub fn new(cfg: &BuselConfig, dev: &B::Device) -> Self {
        let d = cfg.d_byte; let dm = cfg.d_model; let g = (d / 4).max(1);
        Self {
            embed: Tensor::random([cfg.vocab_size, d], Distribution::Normal(0.0, 0.02), dev),
            gate_down: BitLinear::new(g, d, false, dev),
            gate_up: BitLinear::new(d, g, false, dev),
            // boundary: d→1, k=3, same padding
            boundary_conv: Conv1dConfig::new(d, 1, 3).with_padding(PaddingConfig1d::Same).with_bias(true).init(dev),
            // conv: d→dm, k=5, no padding
            conv: Conv1dConfig::new(d, dm, 5).init(dev),
            norm: RmsNormConfig::new(dm).init(dev),
            n_patches: cfg.n_patches,
        }
    }

    /// [B, T] byte IDs → [B, n_patches, d_model]
    pub fn forward(&self, byte_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [b, t] = byte_ids.dims(); let d = self.embed.dims()[1];
        let flat = byte_ids.reshape([b * t, 1]);
        let e = burn::tensor::module::embedding(self.embed.clone(), flat);
        let mut x: Tensor<B, 3> = e.reshape([b, t, d]);

        // GLU gate
        let g = burn::tensor::activation::silu(self.gate_down.forward(x.clone()));
        x = x * burn::tensor::activation::sigmoid(self.gate_up.forward(g));

        // Boundary conv: [B, d, T] → sigmoid → gate
        let x_t = x.swap_dims(1, 2);  // [B, d, T]
        let s = burn::tensor::activation::sigmoid(self.boundary_conv.forward(x_t.clone()));  // [B, 1, T]
        let x_t = x_t * s;  // [B, d, T]

        // Avg pool: [B, d, T] → [B, d, n_patches]
        let ks = (t / self.n_patches).max(1);
        let xp = burn::tensor::module::avg_pool1d(x_t, ks, ks, 0, false, false);

        // Causal padding (left side)
        let pad_len = 4;
        let pad = Tensor::zeros([b, d, pad_len], &xp.device());
        let xp_pad = Tensor::cat(vec![pad, xp], 2);

        // Conv: [B, d, T'] → [B, dm, T'']
        let p = self.conv.forward(xp_pad);
        let p = p.swap_dims(1, 2);  // [B, T'', dm]
        let max_p = p.dims()[1];
        let p = p.narrow(1, 0, self.n_patches.min(max_p));
        self.norm.forward(p)
    }
}
