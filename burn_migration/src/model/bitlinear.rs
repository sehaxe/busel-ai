// Ternary weight quantization with STE. BitNet v2 spec.
use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor, Distribution},
};

#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}

impl<B: Backend> BitLinear<B> {
    pub fn new(d_out: usize, d_in: usize, bias: bool, dev: &B::Device) -> Self {
        Self {
            weight: Tensor::random([d_out, d_in], Distribution::Normal(0.0, 0.02), dev),
            bias: if bias { Some(Tensor::zeros([d_out], dev)) } else { None },
        }
    }

    /// x @ ternary(weight)^T + bias.  STE through round().
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let s = self.weight.clone().abs().mean().clamp(1e-5, 1e9);
        let wq = (self.weight.clone() / s.clone().reshape([1, 1])).clamp(-1.0, 1.0).round();
        let [b, t, di] = x.dims();
        let [dout, _] = wq.dims();
        let mut out = x.reshape([b * t, di]).matmul(wq.transpose()).reshape([b, t, dout]);
        out = out * s.clone().reshape([1, 1, 1]);
        if let Some(bias) = &self.bias { out = out + bias.clone().reshape([1, 1, dout]); }
        out
    }
}
