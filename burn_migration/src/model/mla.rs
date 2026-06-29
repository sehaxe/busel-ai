// Multi-head Latent Attention (DeepSeek, arXiv:2405.04434)
// KV compression into small latent d_c, decompress to full heads.
use burn::{module::Module, nn::{RmsNorm, RmsNormConfig}, tensor::{activation, backend::Backend, Int, Tensor}};
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct MLA<B: Backend> {
    kv_compress: BitLinear<B>, kv_norm: RmsNorm<B>,
    k_decompress: BitLinear<B>, v_decompress: BitLinear<B>,
    q_compress: BitLinear<B>, q_norm: RmsNorm<B>,
    q_decompress: BitLinear<B>,
    out_norm: RmsNorm<B>, o: BitLinear<B>,
    n_heads: usize, d_v: usize,
}

impl<B: Backend> MLA<B> {
    pub fn new(dm: usize, nh: usize, d_c: usize, dev: &B::Device) -> Self {
        let d_v = dm / nh;
        Self {
            kv_compress: BitLinear::new(d_c, dm, false, dev),
            kv_norm: RmsNormConfig::new(d_c).init(dev),
            k_decompress: BitLinear::new(nh * d_v, d_c, false, dev),
            v_decompress: BitLinear::new(nh * d_v, d_c, false, dev),
            q_compress: BitLinear::new(d_c, dm, false, dev),
            q_norm: RmsNormConfig::new(d_c).init(dev),
            q_decompress: BitLinear::new(nh * d_v, d_c, false, dev),
            out_norm: RmsNormConfig::new(nh * d_v).init(dev),
            o: BitLinear::new(dm, nh * d_v, false, dev),
            n_heads: nh, d_v,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        let dev = x.device();

        let kv = self.kv_norm.forward(self.kv_compress.forward(x.clone()));
        let k = self.k_decompress.forward(kv.clone())
            .reshape([b, t, self.n_heads, self.d_v]).swap_dims(1, 2);
        let v = self.v_decompress.forward(kv)
            .reshape([b, t, self.n_heads, self.d_v]).swap_dims(1, 2);

        let q = self.q_decompress.forward(
            self.q_norm.forward(self.q_compress.forward(x))
        ).reshape([b, t, self.n_heads, self.d_v]).swap_dims(1, 2);

        let scale = (self.d_v as f64).sqrt().recip();
        let scores = q.matmul(k.transpose()) * scale;
        let causal = Tensor::<B, 2>::zeros([t, t], &dev).mask_fill(
            Tensor::<B, 2, Int>::ones([t, t], &dev).triu(1).bool(),
            f32::NEG_INFINITY,
        );
        let attn = activation::softmax(scores + causal.reshape([1, 1, t, t]), 3);
        let ctx = attn.matmul(v).swap_dims(1, 2).reshape([b, t, self.n_heads * self.d_v]);
        self.o.forward(self.out_norm.forward(ctx))
    }
}
