// BuselModel + DecoderLayer. MoE + Routing-Free MoE + mAR residual gating + LCSB selective backward.
pub mod bitlinear; pub mod patcher; pub mod gdn2; pub mod sct; pub mod hybrid; pub mod moe; pub mod mar; pub mod cube_opt; pub mod routing_free;

use burn::{
    module::Module,
    nn::{RmsNorm, RmsNormConfig},
    tensor::{backend::{Backend, AutodiffBackend}, Tensor, Int},
};
use crate::config::BuselConfig;
use self::{bitlinear::BitLinear, patcher::ByteFlowPatcher, sct::SCTBitLinear, moe::MoE, mar::MAR};

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    a_norm: RmsNorm<B>, gdn2: gdn2::GDN2Attention<B>, f_norm: RmsNorm<B>,
    gate: SCTBitLinear<B>, up: SCTBitLinear<B>, down: SCTBitLinear<B>,
    moe: MoE<B>,
    rf_moe: Option<routing_free::RoutingFreeMoE<B>>,
    mar: Option<MAR<B>>,
    pub use_moe: bool, pub use_mar: bool, pub use_rf: bool,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(dm: usize, nh: usize, eh: usize, rank: usize, ne: usize, tk: usize, dtopk: usize, rf: bool, dev: &B::Device) -> Self {
        Self {
            a_norm: RmsNormConfig::new(dm).init(dev),
            gdn2: gdn2::GDN2Attention::new(dm, nh, dev),
            f_norm: RmsNormConfig::new(dm).init(dev),
            gate: SCTBitLinear::new(eh, dm, rank, false, false, dev),
            up: SCTBitLinear::new(eh, dm, rank, false, false, dev),
            down: SCTBitLinear::new(dm, eh, rank, false, false, dev),
            moe: MoE::new(dm, eh, ne, tk, rank, dev),
            rf_moe: if rf { Some(routing_free::RoutingFreeMoE::new(dm, eh, ne, tk, rank, dev)) } else { None },
            mar: if dtopk > 0 { Some(MAR::new(dm, dtopk, 3, dev)) } else { None },
            use_moe: ne > 1 && !rf, use_mar: dtopk > 0, use_rf: rf,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let r = x.clone();
        let attn_out = self.gdn2.forward(self.a_norm.forward(x));
        let x = if let Some(ref mar) = self.mar {
            mar.forward(r.clone(), attn_out)
        } else {
            attn_out + r
        };
        let n = self.f_norm.forward(x);
        if self.use_rf {
            self.rf_moe.as_ref().unwrap().forward(n)
        } else if self.use_moe {
            self.moe.forward(n)
        } else {
            let g = burn::tensor::activation::silu(self.gate.forward(n.clone()));
            let u = self.up.forward(n);
            let out = self.down.forward(g * u);
            let dev = out.device();
            (out, Tensor::zeros([1], &dev))
        }
    }
}

#[derive(Module, Debug)]
pub struct BuselModel<B: Backend> {
    patcher: ByteFlowPatcher<B>, layers: Vec<DecoderLayer<B>>,
    f_norm: RmsNorm<B>, head: BitLinear<B>, mtp: Vec<BitLinear<B>>,
}

impl<B: Backend> BuselModel<B> {
    pub fn new(cfg: &BuselConfig, dev: &B::Device) -> Self {
        Self {
            patcher: ByteFlowPatcher::new(cfg, dev),
            layers: (0..cfg.n_layers).map(|_|
                DecoderLayer::new(cfg.d_model, cfg.n_heads, cfg.expert_hidden, cfg.sct_rank,
                    cfg.num_experts, cfg.top_k, cfg.dtopk_k, cfg.routing_free, dev)
            ).collect(),
            f_norm: RmsNormConfig::new(cfg.d_model).init(dev),
            head: BitLinear::new(cfg.vocab_size, cfg.d_model, false, dev),
            mtp: (0..cfg.num_mtp_heads).map(|_| BitLinear::new(cfg.vocab_size, cfg.d_model, false, dev)).collect(),
        }
    }

    /// Call after optimizer step to update loss-free biases.
    pub fn update_moe_biases(&mut self) {
        for l in &mut self.layers {
            if let Some(ref mut rf) = l.rf_moe {
                rf.update_bias();
            }
        }
    }

    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> (Vec<Tensor<B, 3>>, Tensor<B, 1>) {
        let mut x = self.patcher.forward(ids);
        let mut aux = Tensor::zeros([1], &x.device());
        for l in &self.layers { let (out, a) = l.forward(x); x = out; aux = aux + a; }
        let h = self.f_norm.forward(x);
        let mut out = vec![self.head.forward(h.clone())];
        for m in &self.mtp { out.push(m.forward(h.clone())); }
        (out, aux)
    }
}

// LCSB selective backward: forward with skip mask. RequireAutodiffBackend for .detach().
impl<B: AutodiffBackend> BuselModel<B> {
    pub fn forward_mask(&self, ids: Tensor<B, 2, Int>, skip: &[bool]) -> (Vec<Tensor<B, 3>>, Tensor<B, 1>) {
        let mut x = self.patcher.forward(ids);
        let mut aux = Tensor::zeros([1], &x.device());
        for (i, l) in self.layers.iter().enumerate() {
            let (out, a) = l.forward(x);
            let detach = skip.get(i).copied().unwrap_or(false);
            x = if detach { out.detach() } else { out };
            aux = if detach { aux + a.detach() } else { aux + a };
        }
        let h = self.f_norm.forward(x);
        let mut out = vec![self.head.forward(h.clone())];
        for m in &self.mtp { out.push(m.forward(h.clone())); }
        (out, aux)
    }
}
