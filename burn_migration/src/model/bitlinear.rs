// BitLinear: ternary quant {−1,0,+1} × fp16. Single clone for scale.
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, Distribution},
};

#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> BitLinear<B> {
    pub fn new(d_out: usize, d_in: usize, bias: bool, dev: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::random([d_out, d_in], Distribution::Normal(0.0, 0.02), dev)),
            bias: if bias { Some(Param::from_tensor(Tensor::zeros([d_out], dev))) } else { None },
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let w = self.weight.val().clone();
        let s = w.clone().abs().mean().clamp(1e-5, 1e9);
        let [b, t, di] = x.dims();
        let [dout, _] = w.dims();
        let mut out = x.reshape([b * t, di]).matmul(
            (w / s.clone().reshape([1, 1])).clamp(-1.0, 1.0).round().transpose()
        ).reshape([b, t, dout]);
        out = out * s.clone().reshape([1, 1, 1]);
        if let Some(bias) = &self.bias {
            out = out + bias.val().clone().reshape([1, 1, dout]);
        }
        out
    }
}
