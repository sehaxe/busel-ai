// 🚀 GDN-2 cubecl fused kernel bridge (bf16 native, no CPU round-trip).
// Low-level: cubecl kernel launch + handle management.
// High-level: used by gdn2_op.rs for autodiff integration.

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubecl::zspace::{Shape, Strides};
use burn_cubecl::tensor::CubeTensor;
use cubecl_zspace::metadata::Metadata;
use crate::types::Backend;
use super::gdn2_kernel::gdn2_forward_kernel;

use burn_backend::TensorPrimitive;

type InnerBackend = <Backend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

/// Extract CubeTensor from a 4D burn input tensor.
fn ct4(t: &burn::tensor::Tensor<Backend, 4>) -> CubeTensor<CudaRuntime> {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(ad) => {
            use burn::backend::autodiff::tensor::AutodiffTensor;
            let at: AutodiffTensor<InnerBackend> = ad;
            at.primitive
        }
        _ => unreachable!(),
    }
}

/// Extract CubeTensor from a 2D burn input tensor.
fn ct2(t: &burn::tensor::Tensor<Backend, 2>) -> CubeTensor<CudaRuntime> {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(ad) => {
            use burn::backend::autodiff::tensor::AutodiffTensor;
            let at: AutodiffTensor<InnerBackend> = ad;
            at.primitive
        }
        _ => unreachable!(),
    }
}

fn contig_strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len()-1).rev() { s[i] = s[i+1] * shape[i+1]; }
    s
}

/// Run the bf16 cubecl GDN-2 kernel, return output CubeTensor (no autodiff).
/// Called by gdn2_op.rs which wraps the result in an AutodiffTensor.
pub(crate) fn run_cubecl_gdn2_forward(
    q: &burn::tensor::Tensor<Backend, 4>,
    k: &burn::tensor::Tensor<Backend, 4>,
    v: &burn::tensor::Tensor<Backend, 4>,
    b_gate: &burn::tensor::Tensor<Backend, 4>,
    w_gate: &burn::tensor::Tensor<Backend, 4>,
    g_raw: &burn::tensor::Tensor<Backend, 4>,
    base_decay: &burn::tensor::Tensor<Backend, 2>,
) -> CubeTensor<CudaRuntime> {
    let q_ct = ct4(q);
    let k_ct = ct4(k);
    let v_ct = ct4(v);
    let bg_ct = ct4(b_gate);
    let wg_ct = ct4(w_gate);
    let gg_ct = ct4(g_raw);
    let bd_ct = ct2(base_decay);
    let client = q_ct.client.clone();

    let s4: Shape = q_ct.meta.shape().clone();
    let str4: Strides = q_ct.meta.strides().clone();
    let [b, t, h, d] = <[usize; 4]>::try_from(s4.to_vec()).ok().unwrap();

    let state = client.empty(b * h * d * d * 2);
    let out = client.empty(b * t * h * d * 2);

    let sbd: Shape = bd_ct.meta.shape().clone();
    let strbd: Strides = bd_ct.meta.strides().clone();
    let sst: Shape = vec![b, h, d, d].into();
    let strst: Strides = Strides::new(&contig_strides(&[b, h, d, d]));

    unsafe {
        gdn2_forward_kernel::launch_unchecked::<half::bf16, CudaRuntime>(
            &client,
            CubeCount::Static((b * h) as u32, 1, 1),
            CubeDim::new_3d(d as u32, 1, 1),
            TensorArg::from_raw_parts(q_ct.handle.clone(), str4.clone(), s4.clone()),
            TensorArg::from_raw_parts(k_ct.handle.clone(), str4.clone(), s4.clone()),
            TensorArg::from_raw_parts(v_ct.handle.clone(), str4.clone(), s4.clone()),
            TensorArg::from_raw_parts(bg_ct.handle.clone(), str4.clone(), s4.clone()),
            TensorArg::from_raw_parts(wg_ct.handle.clone(), str4.clone(), s4.clone()),
            TensorArg::from_raw_parts(gg_ct.handle.clone(), str4.clone(), s4.clone()),
            TensorArg::from_raw_parts(bd_ct.handle.clone(), strbd, sbd),
            TensorArg::from_raw_parts(state, strst, sst),
            TensorArg::from_raw_parts(out.clone(), str4, s4),
            b, t, h, d,
        );
    }

    let out_shape: Shape = vec![b * t, h * d].into();
    let out_strides: Strides = Strides::new(&[h * d, 1]);
    let meta = Metadata::new(out_shape, out_strides);
    CubeTensor::<CudaRuntime>::new(client, out, meta, q_ct.device, q_ct.dtype)
}
