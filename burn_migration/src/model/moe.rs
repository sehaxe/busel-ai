// MoE: top-1 routing, SCTBitLinear experts, aux balance loss (всё тензорно).
use burn::{module::Module, nn, tensor::{backend::Backend, Tensor}};
use super::sct::SCTBitLinear;
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct MoE<B: Backend> {
    router: BitLinear<B>, experts: Vec<[SCTBitLinear<B>; 3]>,
    pub ne: usize, tk: usize, eh: usize, dm: usize,
}

impl<B: Backend> MoE<B> {
    pub fn new(dm: usize, eh: usize, ne: usize, tk: usize, rank: usize, dev: &B::Device) -> Self {
        Self {
            router: BitLinear::new(ne, dm, false, dev),
            experts: (0..ne).map(|_| [
                SCTBitLinear::new(eh, dm, rank, false, false, dev),
                SCTBitLinear::new(eh, dm, rank, false, false, dev),
                SCTBitLinear::new(dm, eh, rank, false, false, dev),
            ]).collect(),
            ne, tk, eh, dm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [b, t, _] = x.dims(); let nt = b * t; let dev = x.device();
        let logits = self.router.forward(x.clone()).reshape([nt, self.ne]);
        let routes = logits.argmax(1).reshape([nt]);

        let x2 = x.clone().reshape([nt, 1, self.dm]);
        let mut out = Tensor::zeros([nt, 1, self.dm], &dev);

        for ei in 0..self.ne {
            let mask = routes.clone().equal_elem(ei as i64).float().reshape([nt, 1, 1]);
            let [g, u, d] = &self.experts[ei];
            let gate = nn::Gelu::new().forward(g.forward(x2.clone()));
            let down = d.forward(gate * u.forward(x2.clone()));
            out = out + down * mask;
        }

        // aux balance loss: one_hot histogram — 1 kernel вместо ne
        let one_hot: Tensor<B, 2> = routes.clone().one_hot::<2>(self.ne).float();
        let counts = one_hot.sum_dim(0);
        let m = counts / (nt as f64);
        let diff = m - (1.0 / self.ne as f64);
        let aux = (diff.clone() * diff).sum().reshape([1]);

        (out.reshape([b, t, self.dm]), aux * 0.01)
    }
}
