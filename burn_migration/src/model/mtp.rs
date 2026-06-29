// 🎯 MTP heads: авторегрессивная цепочка (DeepSeek, Meta 2024).
// h_0 = norm(x)
// for i in 0..=nmtp:
//   logits_i = head_i(h_i)
//   h_{i+1} = proj(embed(targets[i]) + h_i)   (teacher forcing)
use burn::{
    module::{Module, Param},
    nn::{RmsNorm, RmsNormConfig},
    tensor::{backend::Backend, Int, Tensor, Distribution},
};
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct MTPHeads<B: Backend> {
    embed: Param<Tensor<B, 2>>,
    norms: Vec<RmsNorm<B>>,
    heads: Vec<BitLinear<B>>,
    projs: Vec<BitLinear<B>>,
    nmtp: usize,
    vocab_size: usize,
}

impl<B: Backend> MTPHeads<B> {
    pub fn new(dm: usize, vocab: usize, nmtp: usize, dev: &B::Device) -> Self {
        let embed = Param::from_tensor(
            Tensor::random([vocab, dm], Distribution::Normal(0.0, 0.02), dev)
        );
        let n_all = 1 + nmtp;
        Self {
            embed,
            norms: (0..n_all).map(|_| RmsNormConfig::new(dm).init(dev)).collect(),
            heads: (0..n_all).map(|_| BitLinear::new(vocab, dm, false, dev)).collect(),
            projs: (0..nmtp).map(|_| BitLinear::new(dm, dm, false, dev)).collect(),
            nmtp, vocab_size: vocab,
        }
    }

    /// Forward с teacher forcing: targets[i] — ground-truth токены для головы i.
    /// Если targets = None, используем нулевое conditioning (инференс).
    pub fn forward(
        &self, h: Tensor<B, 3>,
        targets: Option<&[Tensor<B, 1, Int>]>,
    ) -> Vec<Tensor<B, 3>> {
        let mut h_i = h;
        let mut out = Vec::with_capacity(1 + self.nmtp);

        for i in 0..=self.nmtp {
            h_i = self.norms[i].forward(h_i);
            out.push(self.heads[i].forward(h_i.clone()));

            if i < self.nmtp {
                if let Some(tgt) = targets {
                    // Teacher forcing: embed(targets[i]) + h_i
                    let flat = tgt[i].clone().reshape([-1, 1]);
                    let e = burn::tensor::module::embedding(self.embed.val().clone(), flat);
                    let e_r = e.reshape([h_i.dims()[0], h_i.dims()[1], h_i.dims()[2]]);
                    h_i = self.projs[i].forward(e_r + h_i);
                } else {
                    h_i = self.projs[i].forward(h_i);
                }
            }
        }
        out
    }
}
