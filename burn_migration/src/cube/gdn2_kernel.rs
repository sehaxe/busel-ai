// 🚀 Fused GDN-2 forward kernel (cubecl `#[cube]`).
// One launch handles all T steps for all (B, H) heads.
//
// Launch config:
//   CubeCount = B * H  (one cube per batch-head pair)
//   CubeDim   = D      (one thread per state column)
//
// Each thread handles one column j of state[b, h, :, :].
// Two-pass per step: (1) decay + bk_state, (2) output + outer product.
// sync_cube() between steps ensures column writes visible to all threads.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn gdn2_forward_kernel<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    b_gate: &Tensor<F>,
    w_gate: &Tensor<F>,
    g_raw: &Tensor<F>,
    base_decay: &Tensor<F>,
    state: &mut Tensor<F>,
    output: &mut Tensor<F>,
    batch: usize,
    time: usize,
    nhead: usize,
    dim: usize,
) {
    let b = CUBE_POS / nhead;
    let h_val = CUBE_POS % nhead;
    let j = UNIT_POS as usize;

    if b >= batch || h_val >= nhead || j >= dim {
        terminate!();
    }

    for t in 0..time {
        let base4 = b * time * nhead * dim + t * nhead * dim + h_val * dim;
        let base_state = b * nhead * dim * dim + h_val * dim * dim;

        let mut bk_state_j = F::from_int(0);

        for i in 0..dim {
            let g_i = g_raw[base4 + i];
            let sp = (F::from_int(1) + g_i.exp()).ln();
            let bd_i = base_decay[h_val * dim + i];
            let decay_i = (-bd_i * sp).exp();
            let bk_i = b_gate[base4 + i] * k[base4 + i];

            let s_idx = base_state + i * dim + j;
            let old_s = state[s_idx];
            let new_s = old_s * decay_i;
            state[s_idx] = new_s;
            bk_state_j += bk_i * new_s;
        }

        let wv_j = w_gate[base4 + j] * v[base4 + j];
        let v_new_j = wv_j - bk_state_j;

        let mut out_j = F::from_int(0);

        for i in 0..dim {
            let s_idx = base_state + i * dim + j;
            let s_val = state[s_idx];
            let k_i = k[base4 + i];
            let q_i = q[base4 + i];

            let new_s = s_val + k_i * v_new_j;
            out_j += new_s * q_i;
            state[s_idx] = new_s;
        }

        output[base4 + j] = out_j;
        sync_cube();
    }
}

// ─── Reference (CPU) ────────────────────────────────────────────────────
fn gdn2_ref(
    q: &[f32], k: &[f32], v: &[f32],
    b_gate: &[f32], w_gate: &[f32], g_raw: &[f32],
    base_decay: &[f32],
    batch: usize, time: usize, nhead: usize, dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * time * nhead * dim];
    for b in 0..batch {
        for h in 0..nhead {
            let mut state = vec![0.0f32; dim * dim];
            for t in 0..time {
                let b4 = b * time * nhead * dim + t * nhead * dim + h * dim;
                for i in 0..dim {
                    let g = g_raw[b4 + i];
                    let sp = (1.0 + g.exp()).ln();
                    let bd = base_decay[h * dim + i];
                    let decay_i = (-bd * sp).exp();
                    for j in 0..dim {
                        state[i * dim + j] *= decay_i;
                    }
                }
                for j in 0..dim {
                    let mut s = 0.0;
                    for i in 0..dim {
                        s += b_gate[b4 + i] * k[b4 + i] * state[i * dim + j];
                    }
                    let v_new = w_gate[b4 + j] * v[b4 + j] - s;
                    for i in 0..dim {
                        state[i * dim + j] += k[b4 + i] * v_new;
                    }
                }
                for j in 0..dim {
                    let mut s = 0.0;
                    for i in 0..dim {
                        s += state[i * dim + j] * q[b4 + i];
                    }
                    out[b * time * nhead * dim + t * nhead * dim + h * dim + j] = s;
                }
            }
        }
    }
    out
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cuda::CudaRuntime;
    use cubecl::cuda::CudaDevice;
    use cubecl::zspace::{Shape, Strides};

    fn contig_strides(shape: &[usize]) -> Strides {
        let mut st = vec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            st[i] = st[i + 1] * shape[i + 1];
        }
        Strides::new(&st)
    }

    unsafe fn as_u8_slice(v: &[f32]) -> &[u8] {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    }
    unsafe fn as_f32_slice(v: &[u8]) -> &[f32] {
        std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len() / 4)
    }
    unsafe fn u8_of_bf16(v: &[half::bf16]) -> &[u8] {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2)
    }
    unsafe fn bf16_of_u8(v: &[u8]) -> &[half::bf16] {
        std::slice::from_raw_parts(v.as_ptr() as *const half::bf16, v.len() / 2)
    }

    fn ref_out_f32(
        q: &[f32], k: &[f32], v: &[f32], bg: &[f32], wg: &[f32], gg: &[f32], bd: &[f32],
        batch: usize, time: usize, nhead: usize, dim: usize,
    ) -> Vec<f32> {
        gdn2_ref(q, k, v, bg, wg, gg, bd, batch, time, nhead, dim)
    }

    fn ref_out_bf16(
        q: &[half::bf16], k: &[half::bf16], v: &[half::bf16],
        bg: &[half::bf16], wg: &[half::bf16], gg: &[half::bf16], bd: &[half::bf16],
        batch: usize, time: usize, nhead: usize, dim: usize,
    ) -> Vec<half::bf16> {
        let to_f32 = |v: &[half::bf16]| -> Vec<f32> { v.iter().map(|x| f32::from(*x)).collect() };
        let f32_ref = ref_out_f32(
            &to_f32(q), &to_f32(k), &to_f32(v),
            &to_f32(bg), &to_f32(wg), &to_f32(gg), &to_f32(bd),
            batch, time, nhead, dim,
        );
        f32_ref.iter().map(|x| half::bf16::from_f32(*x)).collect()
    }

    fn lcg_f32(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f32) * (1.0 / (1u64 << 53) as f32)
    }

    #[test]
    fn test_gdn2_fused_small() {
        let batch = 1usize;
        let time = 2usize;
        let nhead = 2usize;
        let dim = 4usize;

        let device = CudaDevice::default();
        let client = CudaRuntime::client(&device);

        let total = batch * time * nhead * dim;
        let mut seed: u64 = 42;

        let q: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 2.0).collect();
        let k: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 2.0).collect();
        let v: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 2.0).collect();
        let bg: Vec<f32> = (0..total).map(|_| lcg_f32(&mut seed)).collect();
        let wg: Vec<f32> = (0..total).map(|_| lcg_f32(&mut seed) + 0.5).collect();
        let gg: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 0.5).collect();
        let bd: Vec<f32> = (0..nhead * dim).map(|_| lcg_f32(&mut seed) * 0.1).collect();

        let q_h = client.create_from_slice(unsafe { as_u8_slice(&q) });
        let k_h = client.create_from_slice(unsafe { as_u8_slice(&k) });
        let v_h = client.create_from_slice(unsafe { as_u8_slice(&v) });
        let bg_h = client.create_from_slice(unsafe { as_u8_slice(&bg) });
        let wg_h = client.create_from_slice(unsafe { as_u8_slice(&wg) });
        let gg_h = client.create_from_slice(unsafe { as_u8_slice(&gg) });
        let bd_h = client.create_from_slice(unsafe { as_u8_slice(&bd) });

        let state_bytes = batch * nhead * dim * dim * 4;
        let out_bytes = total * 4;
        let state_h = client.empty(state_bytes);
        let out_h = client.empty(out_bytes);

        let shape_4d: Shape = vec![batch, time, nhead, dim].into();
        let strides_4d = contig_strides(&[batch, time, nhead, dim]);
        let shape_bd: Shape = vec![nhead, dim].into();
        let strides_bd = contig_strides(&[nhead, dim]);
        let shape_state: Shape = vec![batch, nhead, dim, dim].into();
        let strides_state = contig_strides(&[batch, nhead, dim, dim]);

        unsafe {
            gdn2_forward_kernel::launch_unchecked::<f32, CudaRuntime>(
                &client,
                CubeCount::Static((batch * nhead) as u32, 1, 1),
                CubeDim::new_3d(dim as u32, 1, 1),
                TensorArg::from_raw_parts(q_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(k_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(v_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(bg_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(wg_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(gg_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(bd_h, strides_bd, shape_bd),
                TensorArg::from_raw_parts(state_h.clone(), strides_state, shape_state),
                TensorArg::from_raw_parts(out_h.clone(), strides_4d, shape_4d),
                batch, time, nhead, dim,
            );
        }

        let out_bytes = client.read_one(out_h).expect("read_one failed");
        let out_f32 = unsafe { as_f32_slice(&out_bytes) };
        let gpu_out = out_f32.to_vec();

        let ref_out = gdn2_ref(&q, &k, &v, &bg, &wg, &gg, &bd, batch, time, nhead, dim);

        let max_diff = gpu_out.iter().zip(ref_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-3, "f32 kernel max_diff={max_diff:.6} > 1e-3");
        println!("PASS f32: max_diff={max_diff:.6}");
    }

    #[test]
    fn test_gdn2_fused_bf16() {
        let batch = 1usize;
        let time = 2usize;
        let nhead = 2usize;
        let dim = 4usize;

        let device = CudaDevice::default();
        let client = CudaRuntime::client(&device);

        let total = batch * time * nhead * dim;
        let mut seed: u64 = 42;

        let to_bf16 = |v: &[f32]| -> Vec<half::bf16> {
            v.iter().map(|x| half::bf16::from_f32(*x)).collect()
        };

        let q_f32: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 2.0).collect();
        let k_f32: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 2.0).collect();
        let v_f32: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 2.0).collect();
        let bg_f32: Vec<f32> = (0..total).map(|_| lcg_f32(&mut seed)).collect();
        let wg_f32: Vec<f32> = (0..total).map(|_| lcg_f32(&mut seed) + 0.5).collect();
        let gg_f32: Vec<f32> = (0..total).map(|_| (lcg_f32(&mut seed) - 0.5) * 0.5).collect();
        let bd_f32: Vec<f32> = (0..nhead * dim).map(|_| lcg_f32(&mut seed) * 0.1).collect();

        let q = to_bf16(&q_f32);
        let k = to_bf16(&k_f32);
        let v = to_bf16(&v_f32);
        let bg = to_bf16(&bg_f32);
        let wg = to_bf16(&wg_f32);
        let gg = to_bf16(&gg_f32);
        let bd = to_bf16(&bd_f32);

        let q_h = client.create_from_slice(unsafe { u8_of_bf16(&q) });
        let k_h = client.create_from_slice(unsafe { u8_of_bf16(&k) });
        let v_h = client.create_from_slice(unsafe { u8_of_bf16(&v) });
        let bg_h = client.create_from_slice(unsafe { u8_of_bf16(&bg) });
        let wg_h = client.create_from_slice(unsafe { u8_of_bf16(&wg) });
        let gg_h = client.create_from_slice(unsafe { u8_of_bf16(&gg) });
        let bd_h = client.create_from_slice(unsafe { u8_of_bf16(&bd) });

        let state_bytes = batch * nhead * dim * dim * 2;  // bf16 = 2 bytes
        let out_bytes = total * 2;
        let state_h = client.empty(state_bytes);
        let out_h = client.empty(out_bytes);

        let shape_4d: Shape = vec![batch, time, nhead, dim].into();
        let strides_4d = contig_strides(&[batch, time, nhead, dim]);
        let shape_bd: Shape = vec![nhead, dim].into();
        let strides_bd = contig_strides(&[nhead, dim]);
        let shape_state: Shape = vec![batch, nhead, dim, dim].into();
        let strides_state = contig_strides(&[batch, nhead, dim, dim]);

        unsafe {
            gdn2_forward_kernel::launch_unchecked::<half::bf16, CudaRuntime>(
                &client,
                CubeCount::Static((batch * nhead) as u32, 1, 1),
                CubeDim::new_3d(dim as u32, 1, 1),
                TensorArg::from_raw_parts(q_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(k_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(v_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(bg_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(wg_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(gg_h, strides_4d.clone(), shape_4d.clone()),
                TensorArg::from_raw_parts(bd_h, strides_bd, shape_bd),
                TensorArg::from_raw_parts(state_h.clone(), strides_state, shape_state),
                TensorArg::from_raw_parts(out_h.clone(), strides_4d, shape_4d),
                batch, time, nhead, dim,
            );
        }

        let out_bytes = client.read_one(out_h).expect("read_one failed");
        let out_bf16 = unsafe { bf16_of_u8(&out_bytes) };
        let gpu_out: Vec<half::bf16> = out_bf16.to_vec();

        let ref_out = ref_out_bf16(&q, &k, &v, &bg, &wg, &gg, &bd, batch, time, nhead, dim);

        let max_diff: f32 = gpu_out.iter().zip(ref_out.iter())
            .map(|(a, b)| (f32::from(*a) - f32::from(*b)).abs())
            .fold(0.0f32, f32::max);

        // bf16 has ~7.8 mantissa bits → ~0.02 relative. 0.05 covers 2-3 ULPs.
        assert!(max_diff < 0.05, "bf16 kernel max_diff={max_diff:.6} > 0.05");
        println!("PASS bf16: max_diff={max_diff:.6}");
    }
}
