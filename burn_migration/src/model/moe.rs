// Fused MoE: stacked params + GPU-native batched dispatch. No CPU loop, no stale copies.
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, Tensor, Distribution},
};
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct MoE<B: Backend> {
    router: BitLinear<B>,
    gate_vs: Param<Tensor<B, 2>>, gate_ut: Param<Tensor<B, 2>>,
    up_vs: Param<Tensor<B, 2>>, up_ut: Param<Tensor<B, 2>>,
    down_vs: Param<Tensor<B, 3>>, down_u: Param<Tensor<B, 3>>,
    down_boost: Param<Tensor<B, 1>>,
    pub ne: usize, tk: usize, eh: usize, dm: usize, rank: usize,
}

impl<B: Backend> MoE<B> {
    pub fn new(dm: usize, eh: usize, ne: usize, tk: usize, rank: usize, dev: &B::Device) -> Self {
        let sr: f64 = 1.0 / (rank as f64).sqrt();

        let mut gvs = Vec::with_capacity(ne);
        let mut uvs = Vec::with_capacity(ne);
        let mut gut = Vec::with_capacity(ne);
        let mut uut = Vec::with_capacity(ne);
        let mut dvs = Vec::with_capacity(ne);
        let mut duu = Vec::with_capacity(ne);

        let bg: f64 = 0.02 * ((dm * eh) as f64).sqrt() / (rank as f64).sqrt();
        let bd: f64 = 0.02 * ((eh * dm) as f64).sqrt() / (rank as f64).sqrt();

        for _ in 0..ne {
            let sv: f64 = 1.0 / (dm as f64).sqrt();
            let su: f64 = 1.0 / (eh as f64).sqrt();
            let s: Tensor<B, 1> = Tensor::ones([rank], dev) * sr;

            let v: Tensor<B, 2> = Tensor::random([dm, rank], Distribution::Normal(0.0, sv), dev);
            gvs.push(v * s.clone().reshape([1, rank]));
            let u: Tensor<B, 2> = Tensor::random([eh, rank], Distribution::Normal(0.0, su), dev);
            gut.push(u.transpose() * bg);

            let v: Tensor<B, 2> = Tensor::random([dm, rank], Distribution::Normal(0.0, sv), dev);
            uvs.push(v * s.clone().reshape([1, rank]));
            let u: Tensor<B, 2> = Tensor::random([eh, rank], Distribution::Normal(0.0, su), dev);
            uut.push(u.transpose() * bg);

            let sv: f64 = 1.0 / (eh as f64).sqrt();
            let v: Tensor<B, 2> = Tensor::random([eh, rank], Distribution::Normal(0.0, sv), dev);
            dvs.push(v * s.clone().reshape([1, rank]));
            let su: f64 = 1.0 / (dm as f64).sqrt();
            let u: Tensor<B, 2> = Tensor::random([dm, rank], Distribution::Normal(0.0, su), dev);
            duu.push(u.transpose());
        }

        Self {
            router: BitLinear::new(ne, dm, false, dev),
            gate_vs: Param::from_tensor(Tensor::cat(gvs, 1)),
            gate_ut: Param::from_tensor(Tensor::cat(gut, 0)),
            up_vs: Param::from_tensor(Tensor::cat(uvs, 1)),
            up_ut: Param::from_tensor(Tensor::cat(uut, 0)),
            down_vs: Param::from_tensor(Tensor::stack(dvs, 0)),
            down_u: Param::from_tensor(Tensor::stack(duu, 0)),
            down_boost: Param::from_tensor(Tensor::ones([ne], dev) * bd),
            ne, tk, eh, dm, rank,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [b, t, _] = x.dims(); let nt = b * t;
        let logits = self.router.forward(x.clone()).reshape([nt, self.ne]);
        let routes = logits.argmax(1).reshape([nt]);
        let x_2d: Tensor<B, 2> = x.reshape([nt, self.dm]);

        let hg: Tensor<B, 2> = x_2d.clone().matmul(self.gate_vs.val().clone());
        let hu: Tensor<B, 2> = x_2d.matmul(self.up_vs.val().clone());

        let g3: Tensor<B, 3> = self.gate_ut.val().clone().reshape([self.ne, self.rank, self.eh]);
        let u3: Tensor<B, 3> = self.up_ut.val().clone().reshape([self.ne, self.rank, self.eh]);
        let gate_pre: Tensor<B, 3> = hg.reshape([nt, self.ne, self.rank]).swap_dims(0, 1)
            .matmul(g3).swap_dims(0, 1);
        let up_pre: Tensor<B, 3> = hu.reshape([nt, self.ne, self.rank]).swap_dims(0, 1)
            .matmul(u3).swap_dims(0, 1);

        let gu_all: Tensor<B, 3> = burn::nn::Gelu::new().forward(gate_pre) * up_pre;

        let hd: Tensor<B, 3> = gu_all.swap_dims(0, 1).matmul(self.down_vs.val().clone()).swap_dims(0, 1);
        let dx: Tensor<B, 3> = hd.swap_dims(0, 1).matmul(self.down_u.val().clone()).swap_dims(0, 1)
            * self.down_boost.val().clone().reshape([1, self.ne, 1]);

        let oh: Tensor<B, 2> = routes.clone().one_hot::<2>(self.ne).float();
        let out: Tensor<B, 2> = (dx * oh.clone().reshape([nt, self.ne, 1])).sum_dim(1).squeeze_dim(1);

        let counts: Tensor<B, 2> = oh.sum_dim(0);
        let m: Tensor<B, 2> = counts / (nt as f64);
        let diff: Tensor<B, 2> = m - (1.0 / self.ne as f64);
        let aux: Tensor<B, 1> = (diff.clone() * diff).sum().reshape([1]);

        (out.reshape([b, t, self.dm]), aux * 0.01)
    }
}
