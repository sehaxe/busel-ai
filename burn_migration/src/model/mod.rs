// BuselModel + DecoderLayer. GDN-2 (7/8) + MLA (1/8) attention, MoE, mAR, MTP.
pub mod bitlinear; pub mod patcher; pub mod gdn2; pub mod mla; pub mod sct; pub mod hybrid;
pub mod moe; pub mod mar; pub mod routing_free; pub mod mtp;

use burn::{
    module::Module,
    nn::{RmsNorm, RmsNormConfig},
    tensor::{backend::{Backend, AutodiffBackend}, Tensor, Int},
};
use crate::config::BuselConfig;
use self::{
    patcher::ByteFlowPatcher, gdn2::GDN2Attention, mla::MLA,
    mar::MAR, routing_free::RoutingFreeMoE, mtp::MTPHeads, sct::SCTBitLinear,
};

fn sct_ffn<B: Backend>(dm: usize, eh: usize, rank: usize, dev: &B::Device) -> [SCTBitLinear<B>; 3] {
    [SCTBitLinear::new(eh, dm, rank, false, false, dev),
     SCTBitLinear::new(eh, dm, rank, false, false, dev),
     SCTBitLinear::new(dm, eh, rank, false, false, dev)]
}

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    a_norm: RmsNorm<B>,
    gdn2: Option<GDN2Attention<B>>,
    mla: Option<MLA<B>>,
    pub is_global: bool,
    f_norm: RmsNorm<B>,
    mar: MAR<B>,
    moe: Option<RoutingFreeMoE<B>>,
    ffn: Option<[SCTBitLinear<B>; 3]>,
    pub use_moe: bool,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(dm: usize, nh: usize, eh: usize, rank: usize, ne: usize, tk: usize, d_c: usize, is_global: bool, dev: &B::Device) -> Self {
        let use_moe = ne > 1;
        Self {
            a_norm: RmsNormConfig::new(dm).init(dev),
            gdn2: if is_global { None } else { Some(GDN2Attention::new(dm, nh, dev)) },
            mla: if is_global { Some(MLA::new(dm, nh, d_c, dev)) } else { None },
            is_global,
            f_norm: RmsNormConfig::new(dm).init(dev),
            mar: MAR::new(dm, 0, 3, dev),
            moe: if use_moe { Some(RoutingFreeMoE::new(dm, eh, ne, tk, rank, dev)) } else { None },
            ffn: if use_moe { None } else { Some(sct_ffn(dm, eh, rank, dev)) },
            use_moe,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let r = x.clone();
        let xn = self.a_norm.forward(x);
        let attn_out = if self.is_global {
            self.mla.as_ref().unwrap().forward(xn)
        } else {
            self.gdn2.as_ref().unwrap().forward(xn)
        };
        let x = self.mar.forward(r, attn_out);
        let n = self.f_norm.forward(x);
        if self.use_moe {
            self.moe.as_ref().unwrap().forward(n)
        } else {
            let dev = n.device();
            let ffn = self.ffn.as_ref().unwrap();
            let g = burn::tensor::activation::gelu(ffn[0].forward(n.clone()));
            (ffn[2].forward(g * ffn[1].forward(n)), Tensor::zeros([1], &dev))
        }
    }
}

#[derive(Module, Debug)]
pub struct BuselModel<B: Backend> {
    patcher: ByteFlowPatcher<B>, layers: Vec<DecoderLayer<B>>,
    f_norm: RmsNorm<B>, mtp: MTPHeads<B>,
    vocab_size: usize, nmtp: usize,
}

impl<B: Backend> BuselModel<B> {
    pub fn new(cfg: &BuselConfig, dev: &B::Device) -> Self {
        Self {
            patcher: ByteFlowPatcher::new(cfg, dev),
            layers: (0..cfg.n_layers).map(|i| {
                let is_global = i % 4 == 3;
                DecoderLayer::new(cfg.d_model, cfg.n_heads, cfg.expert_hidden, cfg.sct_rank,
                    cfg.num_experts, cfg.top_k, cfg.d_c, is_global, dev)
            }).collect(),
            f_norm: RmsNormConfig::new(cfg.d_model).init(dev),
            mtp: MTPHeads::new(cfg.d_model, cfg.vocab_size, cfg.num_mtp_heads, dev),
            vocab_size: cfg.vocab_size, nmtp: cfg.num_mtp_heads,
        }
    }

    /// Call after optimizer step to update loss-free biases.
    pub fn update_moe_biases(&mut self) {
        for l in &mut self.layers {
            if let Some(ref mut m) = l.moe { m.update_bias(); }
        }
    }

    pub fn forward(
        &self, ids: Tensor<B, 2, Int>,
        mtp_targets: Option<&[Tensor<B, 1, Int>]>,
    ) -> (Vec<Tensor<B, 3>>, Tensor<B, 1>) {
        let mut x = self.patcher.forward(ids);
        let mut aux = Tensor::zeros([1], &x.device());
        for l in &self.layers { let (out, a) = l.forward(x); x = out; aux = aux + a; }
        let h = self.f_norm.forward(x);
        let out = self.mtp.forward(h, mtp_targets);
        (out, aux)
    }

}

impl<B: AutodiffBackend> BuselModel<B> {
    pub fn forward_mask(
        &self, ids: Tensor<B, 2, Int>, skip: &[bool],
        mtp_targets: Option<&[Tensor<B, 1, Int>]>,
    ) -> (Vec<Tensor<B, 3>>, Tensor<B, 1>) {
        let mut x = self.patcher.forward(ids);
        let mut aux = Tensor::zeros([1], &x.device());
        for (i, l) in self.layers.iter().enumerate() {
            let (out, a) = l.forward(x);
            let detach = skip.get(i).copied().unwrap_or(false);
            x = if detach { out.detach() } else { out };
            aux = if detach { aux + a.detach() } else { aux + a };
        }
        let h = self.f_norm.forward(x);
        let out = self.mtp.forward(h, mtp_targets);
        (out, aux)
    }
}
