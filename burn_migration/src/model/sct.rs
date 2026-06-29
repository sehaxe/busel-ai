// SCT: factorized forward U @ diag(s) @ V^T. Pre-multiplied Vs = V*s.
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, Distribution},
};

#[derive(Module, Debug)]
pub struct SCTBitLinear<B: Backend> {
    pub u: Param<Tensor<B, 2>>, pub vs: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub rank: usize, pub d_in: usize, pub d_out: usize,
    pub quantize: bool, pub boost: f64,
}

impl<B: Backend> SCTBitLinear<B> {
    pub fn new(d_out: usize, d_in: usize, rank: usize, bias: bool, quantize: bool, dev: &B::Device) -> Self {
        let su: f64 = 1.0 / (d_in as f64).sqrt();
        let sv: f64 = 1.0 / (d_out as f64).sqrt();
        let sr: f64 = 1.0 / (rank as f64).sqrt();
        let boost: f64 = 0.02 * ((d_in * d_out) as f64).sqrt() / (rank as f64).sqrt();
        let u = Tensor::random([d_out, rank], Distribution::Normal(0.0, su), dev);
        let s: Tensor<B, 1> = Tensor::ones([rank], dev) * sr;
        let v = Tensor::random([d_in, rank], Distribution::Normal(0.0, sv), dev);
        let vs = v * s.reshape([1, rank]);
        Self {
            u: Param::from_tensor(u),
            vs: Param::from_tensor(vs),
            bias: if bias { Some(Param::from_tensor(Tensor::zeros([d_out], dev))) } else { None },
            rank, d_in, d_out, quantize, boost,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = x.dims();
        if self.quantize { return self.forward_quantize(x, b, t); }
        let h = x.reshape([b * t, self.d_in]).matmul(self.vs.val().clone())
            .reshape([b, t, self.rank]);
        let mut out = h.reshape([b * t, self.rank]).matmul(self.u.val().clone().transpose())
            .reshape([b, t, self.d_out]);
        out = out * self.boost;
        if let Some(bias) = &self.bias {
            out = out + bias.val().clone().reshape([1, 1, self.d_out]);
        }
        out
    }

    fn forward_quantize(&self, x: Tensor<B, 3>, b: usize, t: usize) -> Tensor<B, 3> {
        let w_full = self.u.val().clone().matmul(self.vs.val().clone().transpose());
        let scale = w_full.clone().abs().mean().clamp(1e-5, 1e9);
        let wq = (w_full / scale.clone().reshape([1, 1])).clamp(-1.0, 1.0).round();
        let x2 = x.reshape([b * t, self.d_in]);
        let mut out = x2.matmul(wq.transpose()).reshape([b, t, self.d_out]);
        out = out * scale.clone().reshape([1, 1, 1]) * self.boost;
        if let Some(bias) = &self.bias {
            out = out + bias.val().clone().reshape([1, 1, self.d_out]);
        }
        out
    }
}
