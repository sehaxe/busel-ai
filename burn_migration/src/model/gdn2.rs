// GDN-2: per-token recurrent linear attention (NVlabs/GatedDeltaNet-2).
use burn::{
    module::{Module, Param},
    nn::{
        conv::{Conv1d, Conv1dConfig},
        PaddingConfig1d, RmsNorm, RmsNormConfig,
    },
    tensor::{activation, backend::Backend, Tensor},
};
use super::bitlinear::BitLinear;

#[derive(Module, Debug)]
pub struct GDN2Attention<B: Backend> {
    q_proj: BitLinear<B>,
    k_proj: BitLinear<B>,
    v_proj: BitLinear<B>,
    b_proj: BitLinear<B>,
    w_proj: BitLinear<B>,
    alpha_proj: BitLinear<B>,
    q_conv: Conv1d<B>,
    k_conv: Conv1d<B>,
    v_conv: Conv1d<B>,
    g_down: BitLinear<B>,
    g_up: BitLinear<B>,
    out_norm: RmsNorm<B>,
    o_proj: BitLinear<B>,
    alpha_a: Param<Tensor<B, 2>>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> GDN2Attention<B> {
    pub fn new(dm: usize, nh: usize, dev: &B::Device) -> Self {
        let ad = nh * (dm / nh);
        let dh = dm / nh;
        let ccfg = |c| {
            Conv1dConfig::new(c, c, 4)
                .with_padding(PaddingConfig1d::Explicit(3, 0))
                .with_bias(false)
                .with_groups(c)
                .init(dev)
        };
        Self {
            q_proj: BitLinear::new(ad, dm, false, dev),
            k_proj: BitLinear::new(ad, dm, false, dev),
            v_proj: BitLinear::new(ad, dm, false, dev),
            b_proj: BitLinear::new(ad, dm, false, dev),
            w_proj: BitLinear::new(ad, dm, false, dev),
            alpha_proj: BitLinear::new(ad, dm, false, dev),
            q_conv: ccfg(ad), k_conv: ccfg(ad), v_conv: ccfg(ad),
            g_down: BitLinear::new(dm / 4, dm, false, dev),
            g_up: BitLinear::new(dm, dm / 4, false, dev),
            out_norm: RmsNormConfig::new(dm).init(dev),
            o_proj: BitLinear::new(dm, dm, false, dev),
            alpha_a: Param::from_tensor(Tensor::ones([nh, 1], dev).mul_scalar(-3.0)),
            n_heads: nh, d_head: dh,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _dm] = x.dims();
        let h = self.n_heads;
        let d = self.d_head;

        let q = activation::silu(
            self.q_conv.forward(self.q_proj.forward(x.clone()).swap_dims(1, 2)),
        ).swap_dims(1, 2).reshape([b, t, h, d]);
        let k = activation::silu(
            self.k_conv.forward(self.k_proj.forward(x.clone()).swap_dims(1, 2)),
        ).swap_dims(1, 2).reshape([b, t, h, d]);
        let v = activation::silu(
            self.v_conv.forward(self.v_proj.forward(x.clone()).swap_dims(1, 2)),
        ).swap_dims(1, 2).reshape([b, t, h, d]);

        let qn = q.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5).sqrt().reshape([b, t, h, 1]);
        let kn = k.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5).sqrt().reshape([b, t, h, 1]);
        let q = q / qn;
        let k = k / kn;

        let b_gate = activation::sigmoid(self.b_proj.forward(x.clone()))
            .reshape([b, t, h, d]).mul_scalar(2.0);
        let w_gate = activation::sigmoid(self.w_proj.forward(x.clone()))
            .reshape([b, t, h, d]);
        let g_raw = self.alpha_proj.forward(x.clone())
            .reshape([b, t, h, d]);

        let dev = q.device();
        let base_decay = self.alpha_a.val().clamp(-10.0, 0.0).exp();

        // pony: precompute all per-step values before the loop
        // избегаем slice autodiff nodes внутри рекуррентного цикла
        let mut decays: Vec<Tensor<B, 4>> = Vec::with_capacity(t);
        let mut kts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        let mut vts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        let mut bts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        let mut wts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        let mut qts: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        for ti in 0..t {
            let decay_exp = activation::softplus(
                g_raw.clone().narrow(1, ti, 1).squeeze_dim::<3>(1), 1.0);
            decays.push((-base_decay.clone().unsqueeze_dim::<3>(0) * decay_exp)
                .exp().reshape([b, h, d, 1]));
            let sel = |ref_t: &Tensor<B, 4>| ref_t.clone().narrow(1, ti, 1).squeeze_dim::<3>(1);
            kts.push(sel(&k)); vts.push(sel(&v));
            bts.push(sel(&b_gate)); wts.push(sel(&w_gate)); qts.push(sel(&q));
        }

        let mut out_slices: Vec<Tensor<B, 3>> = Vec::with_capacity(t);
        let mut state = Tensor::<B, 4>::zeros([b, h, d, d], &dev);

        for ti in 0..t {
            state = state * decays[ti].clone();
            let bk = bts[ti].clone() * kts[ti].clone();
            let v_new = wts[ti].clone() * vts[ti].clone()
                - bk.clone().unsqueeze_dim::<4>(2).matmul(state.clone()).squeeze_dim::<3>(2);
            state = state
                + kts[ti].clone().unsqueeze_dim::<4>(3).matmul(v_new.clone().unsqueeze_dim::<4>(2));
            let out_t = state.clone().transpose()
                .matmul(qts[ti].clone().unsqueeze_dim::<4>(3)).squeeze_dim::<3>(3);
            out_slices.push(out_t.reshape([b, 1, h * d]));
            state = state.detach();
        }

        // детач после цикла — не храним граф предыдущего слоя
        let out = Tensor::cat(out_slices, 1);
        let gate = activation::sigmoid(self.g_up.forward(self.g_down.forward(x)));
        self.o_proj.forward(self.out_norm.forward(out) * gate)
    }
}
