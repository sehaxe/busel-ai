pub mod gdn2_kernel;
pub mod bridge;
pub mod gdn2_op;

// 🚀 Custom cubecl kernels — WIP.
// Integration requires: access to ComputeClient via CubeTensor::client, then
// launch via cubecl::prelude::* with TensorArg::from_raw_parts.
// For now: Burn-op fallbacks. CubeCL kernels are the next iterative upgrade.
//
// Pattern for a #[cube(launch_unchecked)] kernel:
//   #[cube(launch_unchecked)]
//   fn my_kernel<T: Numeric>(input: &Tensor<T>, output: &mut Tensor<T>, scale: T) {
//       let pos = ABSOLUTE_POS;
//       if pos < input.len() { output[pos] = input[pos] * scale; }
//   }
// Launch:
//   my_kernel::launch_unchecked::<T, R>(client, cube_count, cube_dim,
//       TensorArg::from_raw_parts(handle, strides, shape),
//       TensorArg::from_raw_parts(out_handle, strides, shape),
//       ScalarArg::new(2.0));

/// Fused column norm (Burn-op fallback, cubecl kernel = future):
/// out_ij = x_ij / sqrt(clamp_min(Σ_k x_kj², 1e-8))
pub fn col_norm_fused<B: burn::tensor::backend::Backend, const D: usize>(
    x: burn::tensor::Tensor<B, D>,
) -> burn::tensor::Tensor<B, D> {
    let n = (x.clone() * x.clone()).sum_dim(0).sqrt().clamp_min(1e-8);
    x / n
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Tensor, Distribution};
    use crate::types::Backend;

    #[test]
    fn test_col_norm_fused() {
        let dev = Default::default();
        let x = Tensor::<Backend, 2>::random([8, 4], Distribution::Default, &dev);
        let y = super::col_norm_fused(x.clone());
        let n = (x.clone() * x.clone()).sum_dim(0).sqrt().clamp_min(1e-8);
        let y_ref = x / n;
        let diff: f32 = (y - y_ref).abs().sum().into_scalar().into();
        assert!(diff < 1e-4, "diff={diff}");
    }
}
