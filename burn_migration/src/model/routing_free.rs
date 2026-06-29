// Routing-Free MoE: 2 shared + N routed experts, blackboard, loss-free bias.
// ponytail: CPU argmax вместо topk_with_indices/argsort/one_hot — не реализованы в autodiff.
use burn::{
    module::{Module, Param},
    tensor::{backend::Backend, DType, Int, Tensor},
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

#[derive(Module, Debug)]
pub struct RoutingFreeMoE<B: Backend> {
    shared_a: [SCTBitLinear<B>; 3],
    shared_b: [SCTBitLinear<B>; 3],
    routed: Vec<[SCTBitLinear<B>; 3]>,
    w_bb_gate: BitLinear<B>,
    w_bb_read: BitLinear<B>,
    w_sh_gate: BitLinear<B>,
    w_router: BitLinear<B>,
    pub expert_bias: Param<Tensor<B, 1>>,
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
            expert_bias: Param::from_tensor(Tensor::zeros([ne], dev)),
            bias_delta: None, ne, tk, eh, dm,
        }
    }

    fn shared_ffn(e: &[SCTBitLinear<B>; 3], x: Tensor<B, 3>) -> Tensor<B, 3> {
        let g = gelu(e[0].forward(x.clone()));
        e[2].forward(g * e[1].forward(x))
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [b, t, dm] = x.dims();
        let ne = self.ne;
        let nt = b * t; let dev = x.device();

        // 1. Gated Shared Experts
        let gs = burn::tensor::activation::sigmoid(
            self.w_sh_gate.forward(x.clone()).mean_dim(2).unsqueeze::<3>()
        );
        let h_sh = gs.clone() * Self::shared_ffn(&self.shared_a, x.clone())
            + (gs.neg().add_scalar(1.0)) * Self::shared_ffn(&self.shared_b, x.clone());

        // 2. Blackboard enrichment
        let gate_sig = burn::tensor::activation::sigmoid(self.w_bb_gate.forward(x.clone()));
        let read_sig = self.w_bb_read.forward(h_sh.clone());
        let xe = x + gate_sig * read_sig;

        // 3. Softmax router
        let bias_bc = self.expert_bias.val().clone().unsqueeze::<3>();
        let logits = self.w_router.forward(xe.clone()) + bias_bc;
        let rw = burn::tensor::activation::softmax(logits, 2); // [B, T, ne]

        // 4. Top-1 hard routing with CPU argmax
        // differentiable weight = max of softmax per token
        let max_w = rw.clone().max_dim(2); // [B, T, 1] — Burn сохраняет размерность

        // CPU argmax + sort (breaks autograd on routing decision)
        let flat_rw: Vec<f32> = rw.clone().reshape([nt, ne])
            .into_data().convert_dtype(DType::F32).to_vec().unwrap();
        let mut expert_ids = vec![0i32; nt];
        for i in 0..nt {
            let base = i * ne;
            let (ei, _) = (0..ne).map(|e| (e as i32, flat_rw[base + e]))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            expert_ids[i] = ei;
        }
        // sort tokens by expert_id → contiguous blocks per expert
        let mut order: Vec<usize> = (0..nt).collect();
        order.sort_by_key(|&i| expert_ids[i]);
        let perm: Vec<i32> = order.iter().map(|&i| i as i32).collect();
        let inv_perm: Vec<i32> = {
            let mut inv = vec![0i32; nt];
            for (pos, &orig) in perm.iter().enumerate() {
                inv[orig as usize] = pos as i32;
            }
            inv
        };
        // per-expert counts (CPU, used for slice ranges)
        let mut counts = vec![0i32; ne];
        for &e in &expert_ids { counts[e as usize] += 1; }

        // dispatch tokens by permutation
        let perm_t = Tensor::<B, 1, Int>::from_ints(perm.as_slice(), &dev);
        let all_tokens = xe.reshape([nt, dm]);
        let sorted_tokens = all_tokens.select(0, perm_t.clone());
        // differentiable weights sorted the same way
        let flat_max_w = max_w.reshape([nt, 1]);
        let sorted_w = flat_max_w.select(0, perm_t.clone());

        // per-expert FFN (contiguous slices in sorted order → cat avoids scatter/slice_assign)
        let mut slices: Vec<Tensor<B, 2>> = Vec::new();
        let mut offset = 0usize;
        for ei in 0..ne {
            let n = counts[ei] as usize;
            if n == 0 { continue; }
            let batch = sorted_tokens.clone().slice([offset..(offset + n), 0..dm]);
            let wgt = sorted_w.clone().slice([offset..(offset + n)]).reshape([n, 1]);
            let batch3 = batch.clone().unsqueeze_dim::<3>(1);
            let g = gelu(self.routed[ei][0].forward(batch3.clone()));
            let result = self.routed[ei][2].forward(g * self.routed[ei][1].forward(batch3)).squeeze_dim::<2>(1) * wgt;
            slices.push(result);
            offset += n;
        }
        let sorted_out = Tensor::cat(slices, 0);

        // unsort back to original token order
        let inv_t = Tensor::<B, 1, Int>::from_ints(inv_perm.as_slice(), &dev);
        let dispatched = sorted_out.select(0, inv_t).reshape([b, t, dm]);

        (h_sh + dispatched, Tensor::zeros([1], &dev))
    }

    pub fn update_bias(&mut self) {
        // ponytail: bias обновляется через softmax gradients — удалено CPU sync
    }
}
