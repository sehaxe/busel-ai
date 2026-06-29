// 🔄 Custom Backward<B, 7> for fused GDN-2.
// Forward: cubecl kernel (bf16 native, detached output).
// Backward: per-step recompute using Tensor<OurBackend> API.
//   Primitives stored in state, re-wrapped during backward.
//   Autodiff graph created during backward is temporary (GC'd after backward completes).

use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::backend::autodiff::tensor::AutodiffTensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use crate::types::Backend as OurBackend;

type InnerBackend = <OurBackend as AutodiffBackend>::InnerBackend;
type InnerPrim = CubeTensor<CudaRuntime>;
type T4 = Tensor<OurBackend, 4>;
type T3 = Tensor<OurBackend, 3>;
type T2 = Tensor<OurBackend, 2>;

/// GDN-2 fused operation marker.
#[derive(Debug)]
pub struct GDN2Fused;

/// State: primitives stored as inner backend tensors.
#[derive(Clone, Debug)]
pub struct GDN2State {
    pub q_prim: InnerPrim,
    pub k_prim: InnerPrim,
    pub v_prim: InnerPrim,
    pub b_gate_prim: InnerPrim,
    pub w_gate_prim: InnerPrim,
    pub g_raw_prim: InnerPrim,
    pub base_decay_prim: InnerPrim,
    pub b: usize, pub t: usize, pub h: usize, pub d: usize,
}

// ─── Re-wrap primitive as OurBackend tensor ──────────────────────────────
fn wrap_4(p: InnerPrim) -> T4 {
    let ad: AutodiffTensor<InnerBackend> = AutodiffTensor::new(p);
    Tensor::from_primitive(TensorPrimitive::Float(ad))
}
fn wrap_2(p: InnerPrim) -> T2 {
    let ad: AutodiffTensor<InnerBackend> = AutodiffTensor::new(p);
    Tensor::from_primitive(TensorPrimitive::Float(ad))
}

// ─── GDN-2 backward (per-step recompute on OurBackend tensors) ──────────
fn gdn2_backward_step(
    grad_output: &T3,  // [b, t, h*d]
    q: &T4, k: &T4, v: &T4,
    b_gate: &T4, w_gate: &T4, g_raw: &T4,
    base_decay: &T2,
    b: usize, t: usize, h: usize, d: usize,
) -> [T4; 6] {
    let device = &q.device();
    let mut gq = T4::zeros([b, t, h, d], device);
    let mut gk = T4::zeros([b, t, h, d], device);
    let mut gv = T4::zeros([b, t, h, d], device);
    let mut gb = T4::zeros([b, t, h, d], device);
    let mut gw = T4::zeros([b, t, h, d], device);
    let mut gg = T4::zeros([b, t, h, d], device);

    // Forward recompute: store states
    let mut states: Vec<T4> = Vec::with_capacity(t);  // [b, h, d, d]
    let mut state = T4::zeros([b, h, d, d], device);

    for ti in 0..t {
        let gi = g_raw.clone().narrow(1, ti, 1);   // [b, 1, h, d]
        let gi_3d = gi.squeeze_dim::<3>(1);         // [b, h, d]
        let sp = (Tensor::ones_like(&gi_3d) + gi_3d.exp()).log();
        let decay = (-base_decay.clone().reshape([1, h, 1, d]) * sp.reshape([b, 1, h, d])).exp();  // [b, 1, h, d]
        state = state * decay;  // [b, h, d, d] * [b, 1, h, d]

        let ki = k.clone().narrow(1, ti, 1);    // [b, 1, h, d]
        let vi = v.clone().narrow(1, ti, 1);
        let bi = b_gate.clone().narrow(1, ti, 1);
        let wi = w_gate.clone().narrow(1, ti, 1);

        let bk = bi.clone() * ki.clone();        // [b, 1, h, d]
        let bk_state = bk.reshape([b, h, 1, d])
            .matmul(state.clone())
            .reshape([b, 1, h, d]);

        let v_new = wi.clone() * vi.clone() - bk_state;  // [b, 1, h, d]

        let k_4d = ki.reshape([b, h, d, 1]);
        let vn_4d = v_new.reshape([b, h, 1, d]);
        state = state + k_4d.matmul(vn_4d);

        states.push(state.clone());
    }

    // Backward pass
    for ti in (0..t).rev() {
        let d_out = grad_output.clone()
            .narrow(1, ti, 1)                    // [b, 1, h*d]
            .reshape([b, h, d]);                  // [b, h, d]

        let qi_4 = q.clone().narrow(1, ti, 1);   // [b, 1, h, d]
        let ki_4 = k.clone().narrow(1, ti, 1);
        let vi_4 = v.clone().narrow(1, ti, 1);
        let bi_4 = b_gate.clone().narrow(1, ti, 1);
        let wi_4 = w_gate.clone().narrow(1, ti, 1);
        let gi_4 = g_raw.clone().narrow(1, ti, 1);

        let state_new = &states[ti];             // [b, h, d, d]
        let state_cur = if ti == 0 {
            T4::zeros([b, h, d, d], device)
        } else {
            states[ti - 1].clone()
        };

        // dq = state_new^T @ d_out.unsq(3)  → [b,h,d,d]@[b,h,d,1] → [b,1,h,d]
        let dq = state_new.clone().transpose()
            .matmul(d_out.clone().reshape([b, h, d, 1]))
            .reshape([b, 1, h, d]);
        gq = assign_slice_4d(&gq, &dq, ti);

        // dL/d(state_new) = qi_4.unsq(3) @ d_out.unsq(2)  → [b,1,h,d,1]@[b,1,1,d]
        // Actually: qi_4 is [b,1,h,d], reshape to [b,1,h,d,1], d_out reshape to [b,1,1,d]
        // Then matmul: [b,1,h,d,1]@[b,1,1,d] → [b,1,h,d,d]
        let d_state = qi_4.clone().reshape([b, 1, h, d, 1])
            .matmul(d_out.clone().reshape([b, 1, h, 1, d]));  // [b, 1, h, d, d]

        // dv_new = ki_4.unsq(3) @ d_state → [b,1,h,1,d]@[b,1,h,d,d] → [b,1,h,1,d]
        // Wait, ki_4 is [b,1,h,d]. We need to align dims.
        // d_state: [b,1,h,d,d], ki_4: [b,1,h,d]
        // dv_new_j = sum_i ki_4_i * d_state_ij
        // ki_4.unsq(-1) → [b,1,h,d,1] @ d_state [b,1,h,d,d] → [b,1,h,1,d]
        let dv_new = ki_4.clone().reshape([b, 1, h, 1, d])
            .matmul(d_state.clone())
            .reshape([b, 1, h, d]);

        // dv = dv_new * wi_4  (element-wise, same shape [b,1,h,d])
        let dv = dv_new.clone() * wi_4.clone();
        gv = assign_slice_4d(&gv, &dv, ti);

        let dw = dv_new.clone() * vi_4.clone();
        gw = assign_slice_4d(&gw, &dw, ti);

        // dbk = -(state_cur @ dv_new.unsq(3)).  state_cur: [b,h,d,d], dv_new: [b,1,h,d]
        // dv_new_bhd = dv_new.sq → [b,h,d]
        let dv_new_3d = dv_new.clone().squeeze_dim::<3>(1);  // [b, h, d]
        let dbk = -(state_cur.clone()
            .matmul(dv_new_3d.clone().reshape([b, h, d, 1]))
            .reshape([b, h, d]));  // [b, h, d]

        // dk_update and dk_bk
        let dk_update = d_state.clone()
            .matmul(dv_new_3d.clone().reshape([b, 1, h, d, 1]))
            .reshape([b, 1, h, d]);  // [b, 1, h, d]

        let dk_bk = dbk.clone().reshape([b, 1, h, d]) * bi_4.clone();
        let dk = dk_update + dk_bk;
        gk = assign_slice_4d(&gk, &dk, ti);

        let db = dbk.clone().reshape([b, 1, h, d]) * ki_4.clone();
        gb = assign_slice_4d(&gb, &db, ti);

        // g gradient through decay
        let gi_3d = gi_4.clone().squeeze_dim::<3>(1);  // [b, h, d]
        let sp = (Tensor::ones_like(&gi_3d) + gi_3d.exp()).log();
        let decay = (-base_decay.clone().reshape([1, h, 1, d]) * sp.reshape([b, 1, h, d])).exp();

        // d_state_total = d_state - bk_outer_prod
        let bk = bi_4.clone() * ki_4.clone();  // [b, 1, h, d]
        // outer: bk.unsq(-1) @ dv_new.unsq(-2) → [b,1,h,d,1]@[b,1,h,1,d] → [b,1,h,d,d]
        let bk_outer = bk.clone().reshape([b, 1, h, d, 1])
            .matmul(dv_new.clone().reshape([b, 1, h, 1, d]));
        let d_state_total = d_state - bk_outer;

        // d_decay = sum_over_d(d_state_total * state_cur, dim=4)
        let d_decay_4d = (d_state_total * state_cur.clone().reshape([b, 1, h, d, d]))
            .sum_dim(4)  // [b, 1, h, d, 1]
            .reshape([b, 1, h, d]);

        let sigmoid = (Tensor::ones_like(&gi_4) + (-gi_4.clone()).exp()).recip();
        let base_bhd = base_decay.clone().reshape([1, h, 1, d]);
        let dg = d_decay_4d.clone() * decay * (-base_bhd) * sigmoid;
        gg = assign_slice_4d(&gg, &dg, ti);
    }

    [gq, gk, gv, gb, gw, gg]
}

fn assign_slice_4d(dest: &T4, src: &T4, ti: usize) -> T4 {
    let [b, _t, h, d] = dest.dims();
    dest.clone().slice_assign([0..b, ti..ti+1, 0..h, 0..d], src.clone())
}

// ─── Backward trait implementation ────────────────────────────────────────
impl Backward<OurBackend, 7> for GDN2Fused {
    type State = GDN2State;

    fn backward(
        self,
        ops: Ops<Self::State, 7>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let state = ops.state;

        // Consume gradient
        let grad_out: T3 = {
            let raw = grads.consume::<OurBackend>(&ops.node);
            let ad: AutodiffTensor<InnerBackend> = raw;
            Tensor::from_primitive(TensorPrimitive::Float(ad))
        };

        // Wrap stored primitives as OurBackend tensors
        let q = wrap_4(state.q_prim);
        let k = wrap_4(state.k_prim);
        let v = wrap_4(state.v_prim);
        let b_gate = wrap_4(state.b_gate_prim);
        let w_gate = wrap_4(state.w_gate_prim);
        let g_raw = wrap_4(state.g_raw_prim);
        let base_decay = wrap_2(state.base_decay_prim);

        let grads_inner = gdn2_backward_step(
            &grad_out,
            &q, &k, &v, &b_gate, &w_gate, &g_raw, &base_decay,
            state.b, state.t, state.h, state.d,
        );

        let nodes = ops.parents;
        for (i, grad) in grads_inner.into_iter().enumerate() {
            if let Some(node) = &nodes[i] {
                let grad_ad = match grad.into_primitive() {
                    TensorPrimitive::Float(ad) => ad,
                    _ => unreachable!(),
                };
                grads.register::<OurBackend>(node.id, grad_ad);
            }
        }
    }
}

// ─── Public API ────────────────────────────────────────────────────────────
pub(crate) fn gdn2_fused_autodiff<C: CheckpointStrategy>(
    q: &T4, k: &T4, v: &T4,
    b_gate: &T4, w_gate: &T4, g_raw: &T4,
    base_decay: &T2,
) -> T3 {
    let [b, t, h, d] = q.dims();

    let extract_prim = |t: &T4| -> InnerPrim {
        match t.clone().into_primitive() {
            TensorPrimitive::Float(ad) => {
                let at: AutodiffTensor<InnerBackend> = ad;
                at.primitive
            }
            _ => unreachable!(),
        }
    };
    let extract_prim_2 = |t: &T2| -> InnerPrim {
        match t.clone().into_primitive() {
            TensorPrimitive::Float(ad) => {
                let at: AutodiffTensor<InnerBackend> = ad;
                at.primitive
            }
            _ => unreachable!(),
        }
    };

    let state = GDN2State {
        q_prim: extract_prim(q),
        k_prim: extract_prim(k),
        v_prim: extract_prim(v),
        b_gate_prim: extract_prim(b_gate),
        w_gate_prim: extract_prim(w_gate),
        g_raw_prim: extract_prim(g_raw),
        base_decay_prim: extract_prim_2(base_decay),
        b, t, h, d,
    };

    let out_ct = super::bridge::run_cubecl_gdn2_forward(q, k, v, b_gate, w_gate, g_raw, base_decay);
    let ad_out = AutodiffTensor::<InnerBackend>::new(out_ct);

    let node_of = |t: &T4| -> _ {
        match t.clone().into_primitive() {
            TensorPrimitive::Float(ad) => {
                let at: AutodiffTensor<InnerBackend> = ad;
                at.node
            }
            _ => unreachable!(),
        }
    };
    let node_of2 = |t: &T2| -> _ {
        match t.clone().into_primitive() {
            TensorPrimitive::Float(ad) => {
                let at: AutodiffTensor<InnerBackend> = ad;
                at.node
            }
            _ => unreachable!(),
        }
    };

    let nodes = [
        node_of(q), node_of(k), node_of(v),
        node_of(b_gate), node_of(w_gate), node_of(g_raw),
        node_of2(base_decay),
    ];

    let result_outer = match GDN2Fused.prepare::<C>(nodes).compute_bound().stateful() {
        OpsKind::Tracked(prep) => prep.finish(state, ad_out),
        OpsKind::UnTracked(prep) => prep.finish(ad_out),
    };
    // result_outer: AutodiffTensor<OurBackend> (nested). Extract inner primitive.
    T3::from_primitive(TensorPrimitive::Float(result_outer.primitive))
}
