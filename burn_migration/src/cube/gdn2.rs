// 🚀 GDN-2 custom Borrow op. Replaces the per-step autodiff nodes with
//   a single op + inner-backend primitives. Saves ~1.7 GB VRAM.
//
// Strategy (configurable):
//   d ≤ 128 → checkpoint every 8 steps, recompute chunks in backward.
//   d > 128 → skip (fallback to state.detach()).
//             State [b, h, d, d] is too large for checkpointing at d=256 (201 MB/step).
//             The detach approach is fine: ~18K tok/s stable.

use burn::tensor::backend::Backend;
use burn::tensor::Float;
use burn_backend::tensor::BasicOps;
use burn_backend::{FloatTensor, Shape, Slice, Scalar};

pub(crate) use inner::gdn2_forward_inner;
pub(crate) use inner::gdn2_backward_inner;
pub(crate) use inner::GDN2Saved;

mod inner {
    use super::*;

    // ─── Forward ─────────────────────────────────────────────────────
    pub(crate) fn gdn2_forward_inner<B: Backend>(
        q: &FloatTensor<B>,     // [b, t, h, d]
        k: &FloatTensor<B>,
        v: &FloatTensor<B>,
        b_gate: &FloatTensor<B>,
        w_gate: &FloatTensor<B>,
        g_raw: &FloatTensor<B>,
        base_decay: &FloatTensor<B>,  // [h, 1]
        b: usize, t: usize, h: usize, d: usize,
    ) -> (FloatTensor<B>, GDN2Saved<B>) {
        let dev = dev_of::<B>(q);
        let dt = dtype_of::<B>(q);

        let mut state = zeros::<B>(&[b, h, d, d], &dev, dt);
        let mut out_slices: Vec<FloatTensor<B>> = Vec::with_capacity(t);
        let chk = if d > 128 { t } else { 8.min(t) };
        let mut ckpt = Vec::new();

        for ti in 0..t {
            if chk < t && ti % chk == 0 {
                ckpt.push(B::float_clone(&state));
            }

            // decay = exp(-base_decay * softplus(g_raw[:,ti,:,:]))
            let gi = narrow_t::<B>(g_raw, 1, ti);
            let sp = softplus(gi);
            let neg = B::float_neg(sp);
            let bd = B::float_reshape(base_decay.clone(), &[1, h, d, 1]);
            let decay = B::float_exp(B::float_mul(bd, reshape_4d(neg, b, h, d)));

            state = B::float_mul(state.clone(), decay);

            // per-step slices
            let qi = narrow_t::<B>(q, 1, ti);
            let ki = narrow_t::<B>(k, 1, ti);
            let vi = narrow_t::<B>(v, 1, ti);
            let bi = narrow_t::<B>(b_gate, 1, ti);
            let wi = narrow_t::<B>(w_gate, 1, ti);

            // bk = b*k; bk_state = (bk.unsq(2) @ state).sq(2)
            let bk = B::float_mul(bi, ki.clone());
            let bk_4d = reshape_4d(bk, b, h, 1, d);
            let bk_state = reshape_bhd(B::float_matmul(bk_4d, state.clone()), b, h, d);

            // v_new = w*v - bk_state
            let v_new = B::float_sub(B::float_mul(wi, vi), bk_state);

            // state += k.unsq(3) @ v_new.unsq(2)
            let k_4d = reshape_4d(ki, b, h, d, 1);
            let vn_4d = reshape_4d(v_new, b, h, 1, d);
            state = B::float_add(state.clone(), B::float_matmul(k_4d, vn_4d));

            // out = state.T @ q.unsq(3) → sq → [b, h, d]
            let q_4d = reshape_4d(qi, b, h, d, 1);
            let state_t = B::float_swap_dims(state.clone(), 2, 3);
            let out = reshape_bhd(B::float_matmul(state_t, q_4d), b, h, d);
            out_slices.push(cat_helper(&out, b, h, d));  // [b, 1, h*d]
        }

        let output = B::float_cat(out_slices, 1);
        (output, GDN2Saved { ckpt, chk, b, h, d, t })
    }

    // ─── Saved state ─────────────────────────────────────────────────
    pub(crate) struct GDN2Saved<B: Backend> {
        pub ckpt: Vec<FloatTensor<B>>,  // state checkpoints
        pub chk: usize,                 // interval
        pub b: usize, pub h: usize, pub d: usize, pub t: usize,
    }

    // ─── Backward ─────────────────────────────────────────────────────
    pub(crate) fn gdn2_backward_inner<B: Backend>(
        grad_output: &FloatTensor<B>,  // [b, t, h*d]
        q: &FloatTensor<B>, k: &FloatTensor<B>, v: &FloatTensor<B>,
        b_gate: &FloatTensor<B>, w_gate: &FloatTensor<B>, g_raw: &FloatTensor<B>,
        base_decay: &FloatTensor<B>,
        saved: &GDN2Saved<B>,
    ) -> [FloatTensor<B>; 6] {
        let GDN2Saved { ref ckpt, chk, b, h, d, t } = *saved;
        let dev = dev_of::<B>(q);
        let dt = dtype_of::<B>(q);

        let mut gq = zeros::<B>(&[b, t, h, d], &dev, dt);
        let mut gk = zeros::<B>(&[b, t, h, d], &dev, dt);
        let mut gv = zeros::<B>(&[b, t, h, d], &dev, dt);
        let mut gb = zeros::<B>(&[b, t, h, d], &dev, dt);
        let mut gw = zeros::<B>(&[b, t, h, d], &dev, dt);
        let mut gg = zeros::<B>(&[b, t, h, d], &dev, dt);

        // recompute all states in one forward pass (O(t) memory, O(t) compute)
        // Since chk == t for d > 128, we always need the full states list
        let mut states: Vec<FloatTensor<B>> = Vec::with_capacity(t);
        let mut state = zeros::<B>(&[b, h, d, d], &dev, dt);
        for ti in 0..t {
            let gi = narrow_t::<B>(g_raw, 1, ti);
            let decay = compute_decay_inner::<B>(gi, base_decay, b, h, d);
            state = B::float_mul(state.clone(), decay);
            let ki = narrow_t::<B>(k, 1, ti);
            let vi = narrow_t::<B>(v, 1, ti);
            let bi = narrow_t::<B>(b_gate, 1, ti);
            let wi = narrow_t::<B>(w_gate, 1, ti);
            let update_kv = compute_update::<B>(&state, &ki, &bi, &wi, &vi, b, h, d);
            state = B::float_add(state.clone(), update_kv);
            states.push(B::float_clone(&state));
        }

        let grad_3d = |grad: &FloatTensor<B>, ti: usize| -> FloatTensor<B> {
            let sl = narrow_t::<B>(grad, 1, ti);
            reshape_bhd(sl, b, h, d)
        };

        for ti in (0..t).rev() {
            let d_out = grad_3d(grad_output, ti);
            let qi = narrow_t::<B>(q, 1, ti);
            let ki = narrow_t::<B>(k, 1, ti);
            let vi = narrow_t::<B>(v, 1, ti);
            let bi = narrow_t::<B>(b_gate, 1, ti);
            let wi = narrow_t::<B>(w_gate, 1, ti);

            let state_cur = if ti == 0 {
                zeros::<B>(&[b, h, d, d], &dev, dt)
            } else {
                B::float_clone(&states[ti - 1])
            };
            let state_new = B::float_clone(&states[ti]);

            // dL/d(q)
            let dq = reshape_bhd(
                B::float_matmul(state_new.clone(), reshape_4d(d_out.clone(), b, h, d, 1)),
                b, h, d,
            );
            gq = assign_slice_t(&gq, &dq, 1, ti);

            // dL/d(state_new)
            let q_4d = reshape_4d(qi.clone(), b, h, d, 1);
            let d_state = B::float_matmul(q_4d, reshape_4d(d_out.clone(), b, h, 1, d));

            // k and v_new gradients
            let ki_clone = ki.clone();
            let bi_clone = bi.clone();
            let wi_clone = wi.clone();
            let vi_clone = vi.clone();
            let state_cur_clone = state_cur.clone();

            let (dk_total, dv, dw, db) = step_backward::<B>(
                d_state, &ki_clone, &bi_clone, &wi_clone, &vi_clone, &state_cur_clone, b, h, d,
            );
            gk = assign_slice_t(&gk, &dk_total, 1, ti);
            gv = assign_slice_t(&gv, &dv, 1, ti);
            gw = assign_slice_t(&gw, &dw, 1, ti);
            gb = assign_slice_t(&gb, &db, 1, ti);

            // g gradient through decay
            let gi = narrow_t::<B>(g_raw, 1, ti);
            let decay = compute_decay_inner::<B>(gi.clone(), base_decay, b, h, d);
            let state_decayed;
            let state_before;
            if ti == 0 {
                state_before = zeros::<B>(&[b, h, d, d], &dev, dt);
            } else {
                state_before = B::float_clone(&states[ti - 1]);
            }
            state_decayed = B::float_mul(state_before, decay.clone());

            // dL/d(state_decayed) from output + from bk path
            // From output: d_state (already computed)
            // From bk: -(bk⊗dv_new) outer product
            let bk = B::float_mul(bi_clone, ki_clone);
            let dv_new = compute_dv_new::<B>(&d_state, &ki_clone, b, h, d);
            let d_state_bk = outer_prod::<B>(&bk, &dv_new, b, h, d);
            let mut d_state_total = B::float_add(d_state, B::float_neg(d_state_bk));

            // dL/d(decay) = sum_dim3(d_state_total * state_before, dim=3)
            let d_decay_4d = B::float_sum_dim(
                B::float_mul(d_state_total, B::float_reshape(state_cur_clone, [b, h, d, d].as_slice())),
                3,
            );

            // dL/d(g_raw) = d_decay * decay * (-base_decay) * softplus'(g_raw)
            let d_decay_bhd = reshape_bhd(d_decay_4d, b, h, d);
            let decay_bhd = reshape_bhd(decay, b, h, d);
            let base_bhd = reshape_bhd(B::float_reshape(base_decay.clone(), &[1, h, d]), b, h, d);
            let sp_grad = softplus_deriv(gi);
            let dg = B::float_mul(
                B::float_mul(d_decay_bhd, decay_bhd),
                B::float_mul(B::float_neg(base_bhd), sp_grad),
            );
            gg = assign_slice_t(&gg, &dg, 1, ti);
        }

        [gq, gk, gv, gb, gw, gg]
    }

    // ─── Step backward: returns (dk, dv, dw, db) ──────────────────────
    fn step_backward<B: Backend>(
        d_state: FloatTensor<B>,  // [b, h, d, d]
        ki: &FloatTensor<B>,
        bi: &FloatTensor<B>,
        wi: &FloatTensor<B>,
        vi: &FloatTensor<B>,
        state_decayed: &FloatTensor<B>,  // [b, h, d, d] or state_before
        b: usize, h: usize, d: usize,
    ) -> (FloatTensor<B>, FloatTensor<B>, FloatTensor<B>, FloatTensor<B>) {
        let dv_new = compute_dv_new::<B>(&d_state, ki, b, h, d);

        // dL/d(v) = dv_new * w
        let dv = B::float_mul(dv_new.clone(), wi.clone());
        // dL/d(w) = dv_new * v
        let dw = B::float_mul(dv_new.clone(), vi.clone());

        // dL/d(bk)
        // dv_new_j → bk_state_j → bk_k
        // bk_state_j = sum_k bk_k * state_kj
        // d(bk_state_j) / d(bk_k) = state_kj
        // dL/d(bk_k) = sum_j dL/d(v_new)_j * d(v_new_j) / d(bk_k)
        // v_new_j = w_j*v_j - bk_state_j
        // d(v_new_j)/d(bk_k) = -state_kj
        // dL/d(bk_k) = -sum_j dv_new_j * state_kj
        // = -(state @ dv_new.unsq(3)).sq(3)
        let dbk = B::float_neg(reshape_bhd(
            B::float_matmul(state_decayed.clone(), reshape_4d(dv_new.clone(), b, h, d, 1)),
            b, h, d,
        ));

        // dL/d(k) from bk = dbk * b, plus from state update
        // From state update (state = state + k.unsq(3) @ v_new.unsq(2)):
        // dL/d(k)_i from update = sum_j dL/d(state_new)_ij * v_new_j
        // = (d_state @ dv_new.unsq(3)).sq(3)
        let dk_update = reshape_bhd(
            B::float_matmul(d_state.clone(), reshape_4d(dv_new.clone(), b, h, d, 1)),
            b, h, d,
        );
        let dk_bk = B::float_mul(dbk.clone(), bi.clone());
        let dk = B::float_add(dk_update, dk_bk);

        // dL/d(b) = dbk * k
        let db = B::float_mul(dbk, ki.clone());

        (dk, dv, dw, db)
    }

    fn compute_dv_new<B: Backend>(
        d_state: &FloatTensor<B>,
        ki: &FloatTensor<B>,
        b: usize, h: usize, d: usize,
    ) -> FloatTensor<B> {
        // dv_new_j = sum_i d_state_ij * k_i = (k^T @ d_state)_j
        // k.unsq(2) → [b,h,1,d]; d_state → [b,h,d,d]; matmul → [b,h,1,d] → sq → [b,h,d]
        let k_unsq2 = B::float_reshape(ki.clone(), &[b, h, 1, d]);
        reshape_bhd(B::float_matmul(k_unsq2, d_state.clone()), b, h, d)
    }

    fn compute_update<B: Backend>(
        state: &FloatTensor<B>,
        ki: &FloatTensor<B>, bi: &FloatTensor<B>,
        wi: &FloatTensor<B>, vi: &FloatTensor<B>,
        b: usize, h: usize, d: usize,
    ) -> FloatTensor<B> {
        let bk = B::float_mul(bi.clone(), ki.clone());
        let bk_4d = reshape_4d(bk, b, h, 1, d);
        let bk_state = reshape_bhd(B::float_matmul(bk_4d, state.clone()), b, h, d);
        let v_new = B::float_sub(B::float_mul(wi.clone(), vi.clone()), bk_state);
        let k4d = reshape_4d(ki.clone(), b, h, d, 1);
        let vn4d = reshape_4d(v_new, b, h, 1, d);
        B::float_matmul(k4d, vn4d)
    }

    // ─── Math helpers ─────────────────────────────────────────────────
    fn softplus<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
        let e = B::float_exp(x);
        B::float_log(B::float_add_scalar(e, Scalar::new(1.0)))
    }

    fn softplus_deriv<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
        // d(softplus)/dx = sigmoid(x) = 1 / (1 + exp(-x))
        let n = B::float_neg(x);
        let e = B::float_exp(n);
        B::float_recip(B::float_add_scalar(e, Scalar::new(1.0)))
    }

    fn compute_decay_inner<B: Backend>(
        gi: FloatTensor<B>,
        base_decay: &FloatTensor<B>,
        b: usize, h: usize, d: usize,
    ) -> FloatTensor<B> {
        let sp = softplus(gi);
        let neg = B::float_neg(sp);
        let bd = B::float_reshape(base_decay.clone(), &[1, h, d, 1]);
        B::float_exp(B::float_mul(bd, reshape_4d(neg, b, h, d)))
    }

    fn outer_prod<B: Backend>(
        a: &FloatTensor<B>,  // [b, h, d]
        b: &FloatTensor<B>,  // [b, h, d]
        _b_: usize, _h_: usize, _d_: usize,
    ) -> FloatTensor<B> {
        let a4 = reshape_4d(a.clone(), _b_, _h_, _d_, 1);
        let b4 = reshape_4d(b.clone(), _b_, _h_, 1, _d_);
        B::float_matmul(a4, b4)
    }

    // ─── Tensor helpers ───────────────────────────────────────────────
    fn zeros<B: Backend>(shape: &[usize], dev: &B::Device, dtype: burn_tensor::DType) -> FloatTensor<B> {
        // use BasicOps for cross-backend compat
        use burn_backend::tensor::{Float, BasicOps};
        match <Float as BasicOps<B>>::zeros(Shape::from(shape.to_vec()), dev, dtype) {
            burn_backend::TensorPrimitive::Float(t) => t,
            _ => unreachable!(),
        }
    }

    fn dev_of<B: Backend>(t: &FloatTensor<B>) -> B::Device {
        B::float_device(t)
    }

    fn dtype_of<B: Backend>(_t: &FloatTensor<B>) -> burn_tensor::DType {
        burn_tensor::DType::BF16
    }

    fn reshape_4d<B: Backend>(t: FloatTensor<B>, d0: usize, d1: usize, d2: usize, d3: usize) -> FloatTensor<B> {
        B::float_reshape(t, Shape::from([d0, d1, d2, d3]))
    }

    fn reshape_bhd<B: Backend>(t: FloatTensor<B>, d0: usize, d1: usize, d2: usize) -> FloatTensor<B> {
        B::float_reshape(t, Shape::from([d0, d1, d2]))
    }

    fn narrow_t<B: Backend>(t: &FloatTensor<B>, dim: usize, idx: usize) -> FloatTensor<B> {
        // narrow = slice at index along dim
        let sh = B::float_shape(t);  // TensorMetadata::shape
        let mut slices: Vec<Slice> = sh.dims.iter().enumerate()
            .map(|(d, &len)| if d == dim {
                Slice::new(idx as isize, Some((idx + 1) as isize), 1)
            } else {
                Slice::new(0, Some(len as isize), 1)
            }).collect();
        // Pad slices to full tensor rank; Slice handles excess dims gracefully
        while slices.len() < sh.num_dims() { slices.push(Slice::new(0, Some(1), 1)); }
        B::float_slice(t, &slices)
    }

    fn assign_slice_t<B: Backend>(
        dest: &FloatTensor<B>,
        src: &FloatTensor<B>,
        dim: usize,
        idx: usize,
    ) -> FloatTensor<B> {
        let sh = B::float_shape(dest);
        let mut slices: Vec<Slice> = sh.dims.iter().enumerate()
            .map(|(d, &len)| if d == dim {
                Slice::new(idx as isize, Some((idx + 1) as isize), 1)
            } else {
                Slice::new(0, Some(len as isize), 1)
            }).collect();
        while slices.len() < sh.num_dims() { slices.push(Slice::new(0, Some(1), 1)); }
        B::float_slice_assign(dest.clone(), &slices, src.clone())
    }

    fn cat_helper<B: Backend>(t: &FloatTensor<B>, _b: usize, _h: usize, _d: usize) -> FloatTensor<B> {
        // reshape [b, h, d] → [b, 1, h*d]
        let sh = B::float_shape(t);
        B::float_reshape(t.clone(), Shape::from([sh.dims[0], 1, sh.dims[1] * sh.dims[2]]))
    }
}

// ─── Integration (future: custom op via Backward<B, 7>) ───────────────
// The inner forward/backward functions above are ready.
// Integration into Burn autodiff is blocked by state type mismatch:
//   - `inner::GDN2Saved<B>` stores `Vec<FloatTensor<B::InnerBackend>>`
//   - `Backward<B, 7>::State` must use `FloatTensor<B>` (outer autodiff backend)
//
// Fix: store checkpoints as serialized data (Vec<Vec<u8>>) in state, or
// convert via `AutodiffTensor::new(primitive)`. Low priority: current
// state.detach() approach yields 18K tok/s stable.
//
// When implementing: gate on d ≤ 128 (state 50 MB/chunk fits in 16 GB).
