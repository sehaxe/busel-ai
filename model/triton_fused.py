"""Fused Triton kernels for busel: RMSNorm+ternary matmul, BitLinear matmul."""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if not HAS_TRITON:
    def fused_rmsnorm_ternary_linear(x, w): raise ImportError("triton not installed")
    def fused_bitlinear(x, w): raise ImportError("triton not installed")
else:

    @triton.jit
    def _ternary_matmul_kernel(
        X_ptr, W_ptr, Y_ptr, W_SCALE,
        N, K,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        ALPHA: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """RMSNorm + ternary matmul. One program per row."""
        row = tl.program_id(0)
        n_start = tl.program_id(1) * BLOCK_N
        off_n = n_start + tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        # Pass 1: RMS
        acc_sq = tl.zeros((1,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            x = tl.load(X_ptr + row * K + k, mask=k < K, other=0.0)
            acc_sq += tl.sum(x.to(tl.float32) * x.to(tl.float32))
        rms_inv = 1.0 / tl.sqrt(acc_sq / K + 1e-6)

        # Pass 2: matmul
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            k_mask = k < K
            x = tl.load(X_ptr + row * K + k, mask=k_mask, other=0.0)
            xn = x.to(tl.float32) * rms_inv * ALPHA
            w_ptrs = W_ptr + k[:, None] * stride_wk + off_n[None, :] * stride_wn
            w = tl.load(w_ptrs, mask=k_mask[:, None] & (off_n[None, :] < N), other=0.0)
            ws = w.to(tl.float32) * W_SCALE
            w3 = tl.where(ws > 0.35, 1.0, tl.where(ws < -0.35, -1.0, 0.0))
            acc += tl.sum(xn[:, None] * w3.to(tl.float32), axis=0)

        y_ptrs = Y_ptr + row * stride_ym + off_n * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=off_n < N)


    @triton.jit
    def _bitlinear_kernel(
        X_ptr, W_ptr, W_SCALE, Y_ptr,
        N, K,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        ALPHA: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused BitLinear: center + abs_mean + INT4 + ternary + matmul + rescale."""
        row = tl.program_id(0)
        n_start = tl.program_id(1) * BLOCK_N
        off_n = n_start + tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        # Pass 1: mean
        acc_m = tl.zeros((1,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            x = tl.load(X_ptr + row * K + k, mask=k < K, other=0.0)
            acc_m += tl.sum(x.to(tl.float32))
        row_mean = acc_m / K

        # Pass 2: abs_mean of centered
        acc_ab = tl.zeros((1,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            x = tl.load(X_ptr + row * K + k, mask=k < K, other=0.0)
            acc_ab += tl.sum(tl.abs(x.to(tl.float32) - row_mean))
        beta = tl.maximum(1e-5, acc_ab / K)
        inv_beta = 2.6457 / beta

        # Pass 3: quantize + matmul
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            k_mask = k < K
            x = tl.load(X_ptr + row * K + k, mask=k_mask, other=0.0)
            xc = x.to(tl.float32) - row_mean
            xq = tl.clamp(xc * inv_beta, -8.0, 7.0)
            w_ptrs = W_ptr + k[:, None] * stride_wk + off_n[None, :] * stride_wn
            w = tl.load(w_ptrs, mask=k_mask[:, None] & (off_n[None, :] < N), other=0.0)
            ws = w.to(tl.float32) * W_SCALE
            w3 = tl.where(ws > 0.35, 1.0, tl.where(ws < -0.35, -1.0, 0.0))
            acc += tl.sum(xq[:, None] * w3.to(tl.float32), axis=0)

        acc = acc * ALPHA * beta / 2.6457
        y_ptrs = Y_ptr + row * stride_ym + off_n * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=off_n < N)


def fused_rmsnorm_ternary_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm + ternary matmul in one fused kernel."""
    if not HAS_TRITON or not x.is_cuda:
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        xn = x / rms * 2.6457
        a = weight.abs().mean() + 1e-5
        wt = torch.clamp(torch.round(torch.clamp(weight / a, -1, 1)), -1, 1)
        return xn.to(x.dtype) @ wt.to(x.dtype)
    M, K = x.shape; N = weight.size(1); w = weight.to(x.dtype).contiguous()
    ws = 1.0 / (weight.abs().mean().item() + 1e-5)
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BK, BN = 64, 256
    _ternary_matmul_kernel[(M, triton.cdiv(N, BN))](
        x.contiguous(), w, y, ws, N, K, w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        ALPHA=2.6457, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=2)
    return y


def fused_bitlinear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Full BitLinear forward in 1 kernel: center + abs_mean + INT4 + ternary + matmul + rescale."""
    if not HAS_TRITON or not x.is_cuda:
        a = weight.abs().mean() + 1e-5
        wt = torch.clamp(torch.round(torch.clamp(weight / a, -1, 1)), -1, 1)
        xc = x - x.mean(dim=-1, keepdim=True)
        b = xc.abs().mean(dim=-1, keepdim=True) + 1e-5
        xq = torch.round(torch.clamp(xc * (2.6457 / b), -8, 7))
        return (xq.float() @ wt.float()) * (a * b.float() / 2.6457)
    M, K = x.shape; N = weight.size(1); w = weight.to(x.dtype).contiguous()
    ws = 1.0 / (weight.abs().mean().item() + 1e-5)
    alpha = weight.abs().mean().item() + 1e-5
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    BK, BN = 64, 256
    _bitlinear_kernel[(M, triton.cdiv(N, BN))](
        x.contiguous(), w, ws, y, N, K, w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        ALPHA=alpha, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=2)
    return y


__all__ = ["fused_rmsnorm_ternary_linear", "fused_bitlinear", "HAS_TRITON"]
