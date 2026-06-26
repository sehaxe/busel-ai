// GDN-2 kernelized — all-GPU decay matrix.
use burn::{module::Module, tensor::{backend::Backend, Tensor}};
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct GDN2Attention<B: Backend> {
    q: BitLinear<B>, k: BitLinear<B>, v: BitLinear<B>, o: BitLinear<B>,
    alpha_proj: BitLinear<B>, alpha_a: Tensor<B, 2>,
    n_heads: usize, d_head: usize,
}

impl<B: Backend> GDN2Attention<B> {
    pub fn new(dm: usize, nh: usize, dev: &B::Device) -> Self {
        let dx = dm / nh; let ad = nh * dx;
        Self { q: BitLinear::new(ad, dm, false, dev), k: BitLinear::new(ad, dm, false, dev),
            v: BitLinear::new(ad, dm, false, dev), o: BitLinear::new(dm, ad, false, dev),
            alpha_proj: BitLinear::new(nh, dm, false, dev),
            alpha_a: Tensor::ones([nh, 1], dev) * (-3.0),
            n_heads: nh, d_head: dx }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let r = x.clone(); let [b, t, _] = r.dims(); let h = self.n_heads; let d = self.d_head;
        let q = self.q.forward(x.clone()).reshape([b, t, h, d]).swap_dims(1, 2);
        let k = self.k.forward(x.clone()).reshape([b, t, h, d]).swap_dims(1, 2);
        let v = self.v.forward(x).reshape([b, t, h, d]).swap_dims(1, 2);

        let alpha = burn::tensor::activation::softplus(self.alpha_proj.forward(r.clone()), 1.0);
        let decay = (-alpha) * self.alpha_a.clone().exp().reshape([1, 1, h]);
        let d_exp = decay.exp();                                    // [B, H, T]

        // Decay matrix: reverse cumprod + upper-triangular mask — all on GPU.
        let d_rev = d_exp.flip([2]).cumprod(2).flip([2]);           // [B, H, T]
        let scores = q.matmul(k.transpose())                            // [B, H, T, T]
            * d_rev.reshape([b, h, 1, t]);                          // broadcast [B,H,1,T]
        let scores = scores.triu(0);                                 // zero lower triangle

        let attn = scores.matmul(v).swap_dims(1, 2).reshape([b, t, h * d]);
        self.o.forward(attn) + r
    }
}
