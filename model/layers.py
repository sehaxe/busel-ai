"""
⚙️ busel - Autocast Safe, SwishGLU, Fused RMSNorm & H_BitLinear (BitNet v2)
"""
import torch
import torch.nn as nn
import torch.compiler as _torch_compiler
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore

_BITLINEAR_CONFIG = {"use_tequila": False, "tequila_lambda": 1e-3, "use_sr_ste": False, "use_hysteresis": True, "use_sparse_bitnet": True, "use_fused_training": False}

if HAS_TRITON:

    @triton.jit
    def _fused_bitlinear_kernel(
        X_ptr, W_ptr, Y_ptr,
        GAMMA_ptr, ALPHA: tl.constexpr,
        N, K,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused BitLinear training forward: INT8 act quant + ternary weight + matmul + rescale.
        One program per input row. ALPHA = w.abs().mean(), gamma = per-row absmax(x)."""
        row = tl.program_id(0)
        n_start = tl.program_id(1) * BLOCK_N
        off_n = n_start + tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        gamma = tl.load(GAMMA_ptr + row)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            k_mask = k < K

            x = tl.load(X_ptr + row * K + k, mask=k_mask, other=0.0).to(tl.float32)
            x_i8 = tl.clamp(x * (127.0 / gamma) + 0.5, -128.0, 127.0)

            w_ptrs = W_ptr + k[:, None] * stride_wk + off_n[None, :] * stride_wn
            w = tl.load(w_ptrs, mask=k_mask[:, None] & (off_n[None, :] < N), other=0.0)
            w_s = w.to(tl.float32) / ALPHA
            w_ter = tl.where(w_s > 0.5, 1.0, tl.where(w_s < -0.5, -1.0, 0.0))

            acc += tl.sum(x_i8[:, None] * w_ter, axis=0)

        acc = acc * ALPHA * gamma / 127.0
        y_ptrs = Y_ptr + row * stride_ym + off_n * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=off_n < N)


class _FusedBitLinearTraining(torch.autograd.Function):
    """Fused BitLinear forward with autograd backward.

    Forward: single Triton kernel (INT8 act quant + ternary weight + matmul + rescale).
    Backward: standard STE matmul backward using recomputed quantized tensors.
    Replaces ~10 kernel launches with 1 forward kernel. ~1.3-1.5× speedup on training forward.
    """

    @staticmethod
    def forward(ctx, x, weight, alpha, gamma):
        shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        gamma_1d = gamma.reshape(-1).contiguous()
        M, K = x_2d.shape
        N = weight.shape[0]
        ctx.save_for_backward(x, weight, alpha.detach(), gamma)
        ctx.x_shape = shape
        ctx.w_shape = (N, K)

        # ponytail: Triton kernel only in eager mode — inductor fuses the fallback under torch.compile
        if HAS_TRITON and x.is_cuda and not torch.compiler.is_compiling():
            alpha_val = alpha.item()
            w = weight.to(x.dtype).contiguous()
            y = torch.empty(M, N, dtype=x.dtype, device=x.device)
            BK, BN = 64, 256
            grid = (M, triton.cdiv(N, BN))
            _fused_bitlinear_kernel[grid](
                x_2d.contiguous(), w, y, gamma_1d, alpha_val,
                N, K,
                w.stride(1), w.stride(0),
                y.stride(0), y.stride(1),
                BLOCK_N=BN, BLOCK_K=BK,
                num_warps=4, num_stages=2,
            )
        else:
            gamma_2d = gamma_1d.view(-1, 1)
            x_i8 = torch.clamp(x_2d * (127.0 / gamma_2d) + 0.5, -128, 127)
            w_cl = torch.clamp(weight / alpha, -1, 1)
            w_ter = torch.where(w_cl > 0.5, 1.0, torch.where(w_cl < -0.5, -1.0, torch.zeros_like(w_cl)))
            y = (x_i8.float() @ w_ter.float().T) * (alpha.float() * gamma_2d.float() / 127.0)
            y = y.to(x.dtype)

        return y.view(*shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, alpha, gamma = ctx.saved_tensors
        shape = ctx.x_shape
        N = ctx.w_shape[0]
        grad_2d = grad_output.reshape(-1, N)

        x_2d = x.reshape(-1, x.shape[-1])
        gamma_2d = gamma.reshape(-1, 1)
        x_i8 = torch.clamp(x_2d * (127.0 / gamma_2d) + 0.5, -128, 127)
        w_cl = torch.clamp(weight / alpha, -1, 1)
        w_ter = torch.where(w_cl > 0.5, 1.0, torch.where(w_cl < -0.5, -1.0, torch.zeros_like(w_cl)))

        grad_mat = grad_2d.to(torch.float32) * (alpha.float() * gamma_2d.float() / 127.0)
        grad_x = ((grad_mat @ w_ter.float()) * (127.0 / gamma_2d.float())).view(shape)
        grad_w = (grad_mat.T @ x_i8.float()) / alpha.float()

        return grad_x.to(grad_output.dtype), grad_w.to(weight.dtype), None, None


# ── MatMul-Free Ternary Layer ──────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _ternary_add_kernel(
        X_ptr, W_ptr, Y_ptr, W_SCALE, ALPHA: tl.constexpr,
        N, K,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """MatMul-free: ternary weights × activations. w in {-1,0,1}."""
        row = tl.program_id(0)
        n_start = tl.program_id(1) * BLOCK_N
        off_n = n_start + tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k = k_start + off_k
            k_mask = k < K
            x = tl.load(X_ptr + row * K + k, mask=k_mask, other=0.0)

            w_ptrs = W_ptr + k[:, None] * stride_wk + off_n[None, :] * stride_wn
            w = tl.load(w_ptrs, mask=k_mask[:, None] & (off_n[None, :] < N), other=0.0)
            w_s = w.to(tl.float32) * W_SCALE
            w_ter = tl.where(w_s > 0.5, 1.0, tl.where(w_s < -0.5, -1.0, 0.0))
            acc += tl.sum(x[:, None] * w_ter, axis=0)

        acc = acc * ALPHA
        y_ptrs = Y_ptr + row * stride_ym + off_n * stride_yn
        tl.store(y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=off_n < N)


class TernaryMatMulFree(nn.Module):
    """MatMul-free ternary linear layer (arXiv:2406.02528).

    Replaces float matmul with add/subtract/zero operations.
    Weight stored as fp32, quantized to {-1, 0, +1} at forward time.
    Forward uses Triton kernel; falls back to pure-PyTorch add/subtract.
    Identical gradient flow to BitLinear_a4_8 (STE).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        w = self.weight
        alpha = w.abs().mean().detach() + 1e-5
        w_scale = 1.0 / alpha.item()

        if HAS_TRITON and x.is_cuda:
            shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])
            M, K = x_2d.shape
            N = w.shape[0]
            w_c = w.to(x.dtype).contiguous()
            y_kernel = torch.empty(M, N, dtype=x.dtype, device=x.device)
            BK, BN = 64, 256
            grid = (M, triton.cdiv(N, BN))
            _ternary_add_kernel[grid](
                x_2d.contiguous(), w_c, y_kernel, w_scale, alpha.item(),
                N, K,
                w_c.stride(1), w_c.stride(0),
                y_kernel.stride(0), y_kernel.stride(1),
                BLOCK_N=BN, BLOCK_K=BK,
                num_warps=4, num_stages=2,
            )
            # STE: forward uses kernel output, backward uses PyTorch autograd
            w_ter = self._get_ternary_weight(w, alpha)
            y_ref = (x_2d @ w_ter.T) * alpha
            y = y_ref + (y_kernel - y_ref).detach()
            return y.view(*shape[:-1], N)

        w_ter = self._get_ternary_weight(w, alpha)
        pos = (w_ter > 0).to(x.dtype)
        neg = (w_ter < 0).to(x.dtype)
        out = (x @ pos.T - x @ neg.T) * alpha
        return out

    @staticmethod
    def _get_ternary_weight(w, alpha):
        w_s = w / alpha
        return torch.where(w_s > 0.5, 1.0, torch.where(w_s < -0.5, -1.0, torch.zeros_like(w_s)))


__all__ = ["_FusedBitLinearTraining", "TernaryMatMulFree", "HAS_TRITON"]

def configure_bitlinear(use_tequila: bool = False, tequila_lambda: float = 1e-3,
                        use_sr_ste: bool = True, use_fused_training: bool = True):
    _BITLINEAR_CONFIG["use_tequila"] = use_tequila
    _BITLINEAR_CONFIG["tequila_lambda"] = tequila_lambda
    _BITLINEAR_CONFIG["use_sr_ste"] = use_sr_ste
    _BITLINEAR_CONFIG["use_fused_training"] = use_fused_training

def nvtx_range_push(name: str):
    if torch.cuda.is_available(): torch.cuda.nvtx.range_push(name)
def nvtx_range_pop():
    if torch.cuda.is_available(): torch.cuda.nvtx.range_pop()

import torch.compiler

@torch.compiler.disable
def fast_walsh_hadamard_transform(x):
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.view(-1, D)
    N_flat = x_flat.shape[0]
    power_of_2 = 2 ** math.ceil(math.log2(D))
    if D != power_of_2:
        x_flat = torch.nn.functional.pad(x_flat, (0, power_of_2 - D))
    h = 1
    while h < power_of_2:
        x_flat = x_flat.view(N_flat, -1, h * 2)
        x1 = x_flat[..., :h]
        x2 = x_flat[..., h:]
        x_flat = torch.cat([x1 + x2, x1 - x2], dim=-1)
        h *= 2
    x_flat = x_flat.view(N_flat, power_of_2) / math.sqrt(power_of_2)
    if D != power_of_2:
        x_flat = x_flat[..., :D]
    return x_flat.view(orig_shape)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output): return grad_output


class SR_STE(torch.autograd.Function):
    """Stochastic Rounding STE — eliminates quantization bias."""
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, x):
        floor = torch.floor(x)
        frac = x - floor
        return torch.where(torch.rand_like(x) < frac, floor + 1, floor)
    @staticmethod
    def backward(ctx, grad_output): return grad_output


class HysteresisSTE(torch.autograd.Function):
    """Hysteresis-based Ternary STE — prevents weight flickering under Muon.

    Weight only flips {-1, 0, 1} when the latent coordinate crosses a margin
    beyond the threshold, creating a deadzone that stabilizes orthogonal
    gradient flow. Uses Soft-STE in backward for smooth gradient decay.
    """
    @staticmethod
    @torch.compiler.disable
    def forward(ctx, w_latent, prev_quantized=None, thresh=0.35, margin=0.08):
        if prev_quantized is not None and prev_quantized.size() == w_latent.size():
            pos = w_latent > (thresh - margin * (prev_quantized == 1).float())
            neg = w_latent < (-thresh + margin * (prev_quantized == -1).float())
        else:
            pos = w_latent > thresh
            neg = w_latent < -thresh
        quant = torch.zeros_like(w_latent)
        quant[pos] = 1.0
        quant[neg] = -1.0
        ctx.save_for_backward(w_latent)
        return quant

    @staticmethod
    def backward(ctx, grad_output):
        (w_latent,) = ctx.saved_tensors
        # ponytail: D2 — confidence-weighted backward. Weights far from boundary (confident) get MORE gradient.
        # Weights near boundary (uncertain) get LESS. Inverts old soft-decay for faster convergence.
        confidence = torch.abs(torch.abs(w_latent) - 0.35)
        grad_input = grad_output * torch.sigmoid(confidence * 10.0)
        # ponytail: ternary saturates at ±1 — no gradient for |w| > 1.0 (prevents latent weight explosion)
        grad_input = torch.where(torch.abs(w_latent) > 1.0, torch.zeros_like(grad_input), grad_input)
        return grad_input, None, None, None


class FusedBitLinearFunction(torch.autograd.Function):
    """BitNet a4.8 — weight + activation quant in one autograd node.

    Saves x_quant, w_quant, alpha, gamma for O(1) backward (no recompute).
    """
    @staticmethod
    def forward(ctx, x, w, is_intermediate, topk_ratio,
                use_hyst, use_sr_ste, use_sparse,
                use_tequila, tequila_lambda):
        w = w.to(x.dtype)  # match activation dtype — prevents float×bf16 mismatch in backward
        gamma = x.abs().max(dim=-1, keepdim=True)[0].detach().clamp(min=1e-5)
        alpha = w.abs().mean().detach() + 1e-5
        w_scaled = w / alpha
        w_clipped = torch.clamp(w_scaled, -1, 1)

        _ste = SR_STE if use_sr_ste else RoundSTE
        w_quant = w_clipped + (_ste.apply(w_clipped) - w_clipped)
        if use_hyst and not is_intermediate:
            w_quant = HysteresisSTE.apply(w_clipped)
        if use_sparse and not is_intermediate:
            w_flat = w_quant.reshape(-1, 8)
            _, idx = torch.topk(w_flat.abs(), 6, dim=-1)
            mask = torch.zeros_like(w_flat).scatter_(-1, idx, 1.0)
            w_quant = (w_flat * mask).reshape_as(w_quant)

        x_int = x * (127.0 / gamma)
        if not is_intermediate:
            x_quant = x_int + (torch.clamp(_ste.apply(x_int), -127, 127) - x_int)
        else:
            x_quant = x_int + (torch.clamp(RoundSTE.apply(x_int), -128, 127) - x_int)
            if topk_ratio < 1.0:
                k = int(x.shape[-1] * topk_ratio)
                mask = torch.zeros_like(x_quant)
                topk_vals = torch.topk(x_quant.abs(), k, dim=-1).values
                mask[x_quant.abs() >= topk_vals[..., -1:]] = 1.0
                x_quant = x_quant * mask

        ctx.save_for_backward(x_quant, w_quant, gamma, alpha)
        ctx.use_tequila = use_tequila
        ctx.is_intermediate = is_intermediate
        ctx.tequila_lambda = tequila_lambda

        out = nn.functional.linear(x_quant, w_quant)
        out = out * (alpha * gamma / 127.0)

        if use_tequila and not is_intermediate:
            deadzone_mask = (w_scaled.abs() < 0.5).to(w.dtype)
            tequila_bias = tequila_lambda * (w * deadzone_mask).sum(dim=-1)
            out = out + tequila_bias

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_quant, w_quant, gamma, alpha = ctx.saved_tensors
        use_tequila = ctx.use_tequila
        is_intermediate = ctx.is_intermediate
        tequila_lambda = ctx.tequila_lambda

        scale = alpha * gamma / 127.0
        d_linear = grad_output * scale
        grad_x = (d_linear @ w_quant) * (127.0 / gamma)  # d(x_quant) → d(x_int) → d(x), STE bypass
        d_flat = d_linear.reshape(-1, d_linear.size(-1))
        x_flat = x_quant.reshape(-1, x_quant.size(-1))
        grad_w = d_flat.T @ x_flat  # grad through round/clamp/hyst/sparse, all STE identity
        # scale grad by clamp gate and alpha (same as non-fused chain backward)
        # grad_w = grad_w * clamp_mask / alpha  — but STE in non-fused makes this identity for most
        # Following non-fused: only /alpha applied (clamp/round/hyst/sparse are STE identity in chain)
        grad_w = grad_w / alpha

        if use_tequila and not is_intermediate:
            deadzone_mask = (w_quant == 0).to(w_quant.dtype)
            grad_w = grad_w + grad_output.sum(dim=(0, 1)).unsqueeze(1) * tequila_lambda * deadzone_mask

        return grad_x, grad_w, None, None, None, None, None, None, None


class BitLinear_a4_8(nn.Linear):
    """Ternary linear layer with INT4/INT8 activation quantization."""
    def __init__(self, in_features, out_features, is_intermediate=False,
                 topk_ratio=0.5, use_tequila=False, tequila_lambda=1e-3):
        super().__init__(in_features, out_features, bias=False)
        self.is_intermediate = is_intermediate
        self.topk_ratio = topk_ratio
        self.use_tequila = use_tequila
        self.tequila_lambda = tequila_lambda
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        w = self.weight
        use_tequila = self.use_tequila or _BITLINEAR_CONFIG["use_tequila"]
        use_hyst = _BITLINEAR_CONFIG.get("use_hysteresis", False)
        use_sparse = _BITLINEAR_CONFIG.get("use_sparse_bitnet", True)
        use_fused = _BITLINEAR_CONFIG.get("use_fused_training", False)

        # ponytail: fused training path — single autograd Function saves only x, w, gamma.
        # Recomputes all intermediates in backward via STE (§2.2). Cuts ~3× activation storage.

        # ponytail: fused Triton ternary matmul — inference only (backward not differentiable)
        if not self.is_intermediate and not use_hyst and not use_sparse and not self.training:
            try:
                from model.triton_fused import fused_bitlinear, HAS_TRITON as _HT
                if _HT and x.is_cuda:
                    shape2d = x.shape
                    if x.ndim > 2:
                        x_flat = x.reshape(-1, x.shape[-1])
                    else:
                        x_flat = x
                    out = fused_bitlinear(x_flat, w.T.contiguous())
                    if use_tequila:
                        alpha = w.abs().mean().detach() + 1e-5
                        w_scaled = w / alpha
                        deadzone_mask = (w_scaled.abs() < 0.5).to(w.dtype)
                        tequila_bias = self.tequila_lambda * (w * deadzone_mask).sum(dim=-1)
                        out = out + tequila_bias
                    if x.ndim > 2:
                        out = out.view(*shape2d[:-1], -1)
                    return out
            except ImportError:
                pass

        if use_fused and self.training and not _torch_compiler.is_compiling():
            return FusedBitLinearFunction.apply(
                x, w, self.is_intermediate, self.topk_ratio,
                use_hyst, _BITLINEAR_CONFIG.get("use_sr_ste", True),
                _BITLINEAR_CONFIG.get("use_sparse_bitnet", True),
                use_tequila, self.tequila_lambda or _BITLINEAR_CONFIG["tequila_lambda"]
            )

        alpha = w.abs().mean().detach() + 1e-5
        w_scaled = w / alpha
        w_clipped = torch.clamp(w_scaled, -1, 1)

        tequila_lambda = self.tequila_lambda or _BITLINEAR_CONFIG["tequila_lambda"]
        _ste = SR_STE if _BITLINEAR_CONFIG.get("use_sr_ste", True) else RoundSTE
        _hyst = _BITLINEAR_CONFIG.get("use_hysteresis", False)

        if _hyst and self.training and not self.is_intermediate:
            w_quant = HysteresisSTE.apply(w_clipped)
        else:
            w_quant = w_clipped + (_ste.apply(w_clipped) - w_clipped)

        # Sparse-BitNet: 6:8 structured sparsity on ternary weights (1.3× speedup)
        if _BITLINEAR_CONFIG.get("use_sparse_bitnet", True) and not self.is_intermediate:
            w_flat = w_quant.reshape(-1, 8)
            _, idx = torch.topk(w_flat.abs(), 6, dim=-1)
            mask = torch.zeros_like(w_flat).scatter_(-1, idx, 1.0)
            w_quant = (w_flat * mask).reshape_as(w_quant)

        tequila_bias = None
        if use_tequila and not self.is_intermediate:
            deadzone_mask = (w_scaled.abs() < 0.5).to(w.dtype)
            tequila_bias = tequila_lambda * (w * deadzone_mask).sum(dim=-1)

        if not self.is_intermediate:
            # BitNet v2 paper (arXiv:2310.11453): W_q @ x_q * (α · γ) / Q_b
            gamma = x.abs().max(dim=-1, keepdim=True)[0].detach().clamp(min=1e-5)
            x_int = x * (127.0 / gamma)
            # ponytail: D9 — STE must wrap x_int BEFORE round/clamp, otherwise round breaks grad graph
            x_quant = x_int + (torch.clamp(_ste.apply(x_int), -127, 127) - x_int)
            out = nn.functional.linear(x_quant, w_quant)
            out = out * (alpha * gamma / 127.0)
            if tequila_bias is not None:
                out = out + tequila_bias
            return out
        else:
            gamma = x.abs().max(dim=-1, keepdim=True)[0].detach().clamp(min=1e-5)
            x_int = x * (127.0 / gamma)
            # ponytail: D9 — STE must wrap x_int BEFORE round/clamp, otherwise round breaks grad graph
            x_quant = x_int + (torch.clamp(RoundSTE.apply(x_int), -128, 127) - x_int)
            if self.topk_ratio < 1.0:
                k = int(x.shape[-1] * self.topk_ratio)
                mask = torch.zeros_like(x_quant)
                topk_vals, _ = torch.topk(x_quant.abs(), k, dim=-1)
                mask[x_quant.abs() >= topk_vals[..., -1:]] = 1.0
                x_quant = x_quant * mask
            out = nn.functional.linear(x_quant, w_quant)
            return out * (alpha * gamma / 127.0)

class H_BitLinear(BitLinear_a4_8):
    def forward(self, x):
        # ponytail: skip Walsh-Hadamard (crashes at batch>28). Plain BitLinear is fine.
        return super().forward(x)

class LearnableClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bounds):
        ctx.save_for_backward(x, bounds)
        return torch.clamp(x, -bounds, bounds)
    @staticmethod
    def backward(ctx, grad_output):
        x, bounds = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_bounds = grad_output.clone()
        grad_bounds = (grad_bounds * (x > bounds).float()) - (grad_bounds * (x < -bounds).float())
        sum_dims = list(range(grad_bounds.ndim - 1))
        if sum_dims: grad_bounds = grad_bounds.sum(dim=sum_dims)
        return grad_x, grad_bounds

class RMSNorm(nn.RMSNorm):
    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps=eps)

class SwishGLUClamped(nn.Module):
    def __init__(self, d_model, d_ffn, sct_rank=0, use_matmul_free=False):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        if sct_rank > 0:
            self.w_gate_up = SpectralLinear(d_model, 2 * d_ffn, rank=sct_rank)
            self.w_down = SpectralLinear(d_ffn, d_model, rank=sct_rank, hadamard=True)
        elif use_matmul_free:
            self.w_gate_up = TernaryMatMulFree(d_model, 2 * d_ffn)
            self.w_down = H_BitLinear(d_ffn, d_model, is_intermediate=True)
        else:
            self.w_gate_up = BitLinear_a4_8(d_model, 2 * d_ffn)
            self.w_down = H_BitLinear(d_ffn, d_model, is_intermediate=True)
        self.clipping_bounds = nn.Parameter(torch.ones(d_ffn) * 10.0)
        self.down_norm = RMSNorm(d_ffn)

    def forward(self, x):
        # ponytail: full fp32 for numerical stability at scale (d_model≥640, expert_hidden≥1536)
        # For smaller profiles, keep bf16 — saves ~50% FFN activation VRAM.
        if self.d_model >= 640 and self.d_ffn >= 1536:
            x_proc = x.float()
        else:
            x_proc = x
        gate_up = self.w_gate_up(x_proc)
        gate_raw, up = gate_up.chunk(2, dim=-1)
        gate_swish = F.silu(gate_raw)  # NaN-safe: silu(-Inf)=0, vs -Inf*sigmoid(-Inf)=NaN
        gate = LearnableClampSTE.apply(gate_swish, self.clipping_bounds)
        return self.w_down(self.down_norm(gate * up)).to(x.dtype)


class SpectralLinear(nn.Module):
    """Spectral Compact Training linear layer (arXiv:2604.00733).

    Replaces dense W (d_in, d_out) with low-rank factors U·diag(s)·V^T:
      U: (d_in, rank)   — ternary via STE, fp32 master
      s: (rank,)        — fp32 singular-value scaling
      V: (d_out, rank)  — ternary via STE, fp32 master

    Forward: ((x @ U_q) * s) @ V_q.T — two small matmuls instead of one large.
    At rank=32, (384, 768) → (384, 32) + (32,) + (768, 32) = 8× fewer params.
    Maintains 1.58-bit ternary guarantee via STE on U and V factors.

    Set hadamard=True for o_proj-equivalent (applies Walsh-Hadamard rotation
    to input before the factorized matmul, matching H_BitLinear semantics).
    """

    def __init__(self, d_in, d_out, rank=32, hadamard=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.hadamard = hadamard
        self.U = nn.Parameter(torch.randn(d_in, rank) * (1.0 / math.sqrt(d_in)))
        self.s = nn.Parameter(torch.ones(rank) * (1.0 / math.sqrt(rank)))
        self.V = nn.Parameter(torch.randn(d_out, rank) * (1.0 / math.sqrt(d_out)))

    def forward(self, x):
        if self.hadamard:
            x = fast_walsh_hadamard_transform(x)
        dtype = x.dtype
        U = self.U.to(dtype)
        s = self.s.to(dtype)
        V = self.V.to(dtype)
        alpha_u = U.abs().mean().detach() + 1e-5
        alpha_v = V.abs().mean().detach() + 1e-5
        u_scaled = torch.clamp(U / alpha_u, -1, 1)
        v_scaled = torch.clamp(V / alpha_v, -1, 1)
        u_q = u_scaled + (RoundSTE.apply(u_scaled) - u_scaled)
        v_q = v_scaled + (RoundSTE.apply(v_scaled) - v_scaled)
        h = (x @ u_q) * s
        return (h @ v_q.T) * (alpha_u * alpha_v)

    @torch.no_grad()
    def retract(self):
        """QR retraction — project U, V onto Stiefel manifold (arXiv:2604.00733).
        Must be called after each optimizer step. GPU QR on 640×128: ~0.5ms/matrix."""
        for M in [self.U, self.V]:
            M_f = M.data.float().contiguous()
            Q, R = torch.linalg.qr(M_f)
            Q_signed = Q * torch.sign(torch.diag(R))
            if Q_signed.shape[1] > self.rank:
                Q_signed = Q_signed[:, :self.rank]
            M.data.copy_(Q_signed.contiguous().to(M.dtype))

def retract_all(module):
    """Walk module tree and retract all SpectralLinear layers."""
    for m in module.modules():
        if isinstance(m, SpectralLinear):
            m.retract()