// Routing-Free MoE: 2 shared + N routed experts, blackboard, loss-free bias, entmax.
use burn::{
    module::Module,
    tensor::{backend::Backend, IndexingUpdateOp, Tensor},
};
use super::bitlinear::BitLinear;
use super::sct::SCTBitLinear;

fn sct<B: Backend>(i: usize, o: usize, r: usize, d: &B::Device) -> SCTBitLinear<B> {
    SCTBitLinear::new(o, i, r, false, false, d)
}
fn bl<B: Backend>(i: usize, o: usize, d: &B::Device) -> BitLinear<B> {
    BitLinear::new(o, i, false, d)
}
fn gelu<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    burn::tensor::activation::gelu(x)
}

/// Entmax — sparse softmax on last dim.
fn entmax<B: Backend, const D: usize>(logits: Tensor<B, D>) -> Tensor<B, D> {
    let sorted = logits.clone().sort_descending(D - 1);
    let css = sorted.clone().cumsum(D - 1);
    let ks = Tensor::<B, D>::ones(logits.shape(), &logits.device()).cumsum(D - 1);
    let tau = (css.clone() - 1.0) / ks.clamp_min(1.0);
    let gt = sorted.greater(tau).float();
    let sup = gt.clone().sum_dim(D - 1).clamp_min(1.0);
    let tau_s = (css * gt).sum_dim(D - 1) - 1.0;
    (logits - tau_s / sup).clamp_min(0.0)
}

#[derive(Module, Debug)]
pub struct RoutingFreeMoE<B: Backend> {
    shared_a: [SCTBitLinear<B>; 3],
    shared_b: [SCTBitLinear<B>; 3],
    routed: Vec<[SCTBitLinear<B>; 3]>,
    w_bb_gate: BitLinear<B>,
    w_bb_read: BitLinear<B>,
    w_sh_gate: BitLinear<B>,
    w_router: BitLinear<B>,
    pub expert_bias: Tensor<B, 1>,
    pub bias_delta: Option<Tensor<B, 1>>,
    pub ne: usize, pub tk: usize, pub eh: usize, pub dm: usize,
}

impl<B: Backend> RoutingFreeMoE<B> {
    pub fn new(dm: usize, eh: usize, ne: usize, tk: usize, rank: usize, dev: &B::Device) -> Self {
        Self {
            shared_a: [sct(dm, eh, rank, dev), sct(dm, eh, rank, dev), sct(eh, dm, rank, dev)],
            shared_b: [sct(dm, eh, rank, dev), sct(dm, eh, rank, dev), sct(eh, dm, rank, dev)],
            routed: (0..ne).map(|_| [sct(dm, eh, rank, dev), sct(dm, eh, rank, dev), sct(eh, dm, rank, dev)]).collect(),
            w_bb_gate: bl(dm, dm, dev), w_bb_read: bl(dm, dm, dev),
            w_sh_gate: bl(dm, dm, dev), w_router: bl(dm, ne, dev),
            expert_bias: Tensor::zeros([ne], dev),
            bias_delta: None, ne, tk, eh, dm,
        }
    }

    fn shared_ffn(e: &[SCTBitLinear<B>; 3], x: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = gelu(e[0].forward(x.clone()));
        e[2].forward(g * e[1].forward(x))
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [b, t, dm] = x.dims();
        let nt = b * t; let dev = x.device();

        // 1. Gated Shared Experts
        let gs = burn::tensor::activation::sigmoid(
            self.w_sh_gate.forward(x.clone().detach()).mean_dim(2).unsqueeze::<3>()
        );
        let h_sh = gs.clone() * Self::shared_ffn(&self.shared_a, x.clone())
            + (gs.neg().add_scalar(1.0)) * Self::shared_ffn(&self.shared_b, x.clone());

        // 2. Blackboard enrichment
        let gate_sig = burn::tensor::activation::sigmoid(self.w_bb_gate.forward(x.clone()));
        let read_sig = self.w_bb_read.forward(h_sh.clone());
        let xe = x + gate_sig * read_sig;

        // 3. Router + loss-free bias
        let bias_bc = self.expert_bias.clone().unsqueeze::<3>();
        let logits = self.w_router.forward(xe.clone()) + bias_bc;
        let rw = entmax(logits);
        let (vals, idx) = rw.topk_with_indices(self.tk, 2);
        let w = vals.clone() / (vals.clone().sum_dim(2).unsqueeze::<3>() + 1e-8);

        // 4. Pre-sort dispatch
        let n_total = nt * self.tk;
        let all_tokens = xe.clone().unsqueeze::<4>().expand([b, t, self.tk, dm]).reshape([n_total, dm]);
        let flat_idx = idx.reshape([n_total]);
        let flat_w = w.reshape([n_total]);

        let sort_idx = flat_idx.clone().argsort(0);
        let st = all_tokens.clone().select(0, sort_idx.clone());
        let sw = flat_w.clone().select(0, sort_idx.clone());

        let counts = flat_idx.one_hot::<2>(self.ne).float().sum_dim(0);
        // ponytail: CPU sync to get per-expert sizes for slice-based dispatch
        let counts_cpu: Vec<i64> = counts.clone().reshape([self.ne]).int().into_data().to_vec().unwrap();
        let mut out = Tensor::zeros([n_total, dm], &dev);
        let mut offset = 0usize;

        for ei in 0..self.ne {
            let n = counts_cpu[ei] as usize;
            if n == 0 { continue; }
            let batch = st.clone().slice([offset..(offset + n), 0..dm]);
            let wgt = sw.clone().slice([offset..(offset + n)]).reshape([n, 1]);
            let batch3 = batch.clone().unsqueeze::<3>();
            let g = gelu(self.routed[ei][0].forward(batch3.clone()));
            let result = self.routed[ei][2].forward(g * self.routed[ei][1].forward(batch3)).squeeze::<2>() * wgt;
            let pos = sort_idx.clone().slice([offset..(offset + n)])
                .reshape([n, 1]).expand([n, dm]);
            out = out.scatter(0, pos, result, IndexingUpdateOp::Assign);
            offset += n;
        }

        let dispatched = out.reshape([b, t, self.tk, dm]).sum_dim(2).squeeze::<3>();

        // 5. Loss-free bias delta (stored for update_bias)
        let f_i: Tensor<B, 1> = counts.reshape([self.ne]) / (nt as f64 + 1e-8);
        let target = 1.0 / self.ne as f64;
        let delta = (Tensor::ones_like(&f_i) * target - f_i) * 0.01;
        unsafe {
            let ptr = &self.bias_delta as *const Option<Tensor<B, 1>> as *mut Option<Tensor<B, 1>>;
            ptr.write(Some(delta));
        }

        (h_sh + dispatched, Tensor::zeros([1], &dev))
    }

    pub fn update_bias(&mut self) {
        if let Some(delta) = self.bias_delta.take() {
            self.expert_bias = self.expert_bias.clone() * 0.99 + delta;
        }
    }
}
