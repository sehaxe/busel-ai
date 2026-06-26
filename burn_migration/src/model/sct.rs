// SCT: factorized forward U @ diag(s) @ V^T без дорогой реконструкции.
// FLOP: rank*B*T*(d_in + d_out) вместо d_out*d_in*(rank + B*T).
// Для d_out=2048, dm=512, rank=32, B*T=2048: 13× быстрее.
use burn::{module::Module, tensor::{backend::Backend, Tensor, Distribution}};

#[derive(Module, Debug)]
pub struct SCTBitLinear<B: Backend> {
    u: Tensor<B, 2>, s: Tensor<B, 1>, v: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>, rank: usize, d_in: usize, d_out: usize,
    quantize: bool, // true = полная реконструкция + ternary (медленно, Python-compat)
}

impl<B: Backend> SCTBitLinear<B> {
    pub fn new(d_out: usize, d_in: usize, rank: usize, bias: bool, quantize: bool, dev: &B::Device) -> Self {
        let su = 1.0 / (d_in as f64).sqrt();
        let sv = 1.0 / (d_out as f64).sqrt();
        let sr = 1.0 / (rank as f64).sqrt();
        Self {
            u: Tensor::random([d_out, rank], Distribution::Normal(0.0, su), dev),
            s: Tensor::ones([rank], dev) * sr,
            v: Tensor::random([d_in, rank], Distribution::Normal(0.0, sv), dev),
            bias: if bias { Some(Tensor::zeros([d_out], dev)) } else { None },
            rank, d_in, d_out, quantize,
        }
    }

    /// y = ((x @ V) * s) @ U^T + bias  (factorized, ~13× быстрее)
    /// Если quantize=true: y = x @ quantize(U @ diag(s) @ V^T)^T * scale
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();

        if self.quantize {
            // Python-compat: полная реконструкция + ternary + boost
            let w_full = self.u.clone().matmul(
                (self.v.clone() * self.s.clone().reshape([1, self.rank])).transpose()
            );
            let scale = w_full.clone().abs().mean().clamp(1e-5, 1e9);
            let wq = (w_full / scale.clone().reshape([1, 1])).clamp(-1.0, 1.0).round();
            let x2 = x.reshape([b * t, self.d_in]);
            let mut out = x2.matmul(wq.transpose()).reshape([b, t, self.d_out]);
            out = out * scale.clone().reshape([1, 1, 1]);
            let boost = 0.02 * ((self.d_in * self.d_out) as f64).sqrt() / (self.rank as f64).sqrt();
            out = out * boost;
            if let Some(bias) = &self.bias { out = out + bias.clone().reshape([1, 1, self.d_out]); }
            return out;
        }

        // Factorized: h = (x @ V) * s; out = h @ U^T + bias
        // x: [B, T, d_in], V: [d_in, rank], s: [rank], U: [d_out, rank]
        let h = x.reshape([b * t, self.d_in]).matmul(self.v.clone()).reshape([b, t, self.rank]);
        let h = h * self.s.clone().reshape([1, 1, self.rank]);
        let mut out = h.reshape([b * t, self.rank]).matmul(self.u.clone().transpose()).reshape([b, t, self.d_out]);
        // pony: SCT boost — variance compensation
        // ponytail: boost — variance compensation
        let boost = 0.02 * ((self.d_in * self.d_out) as f64).sqrt() / (self.rank as f64).sqrt();
        out = out * boost;
        if let Some(bias) = &self.bias { out = out + bias.clone().reshape([1, 1, self.d_out]); }
        out
    }
}
