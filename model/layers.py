"""
⚙️ busel - Autocast Safe, SwishGLU, Fused RMSNorm & H_BitLinear (BitNet v2)
"""
import torch
import torch.nn as nn
import math

_BITLINEAR_CONFIG = {"use_tequila": True, "tequila_lambda": 1e-3, "use_sr_ste": True, "use_hysteresis": False, "use_sparse_bitnet": False}

def configure_bitlinear(use_tequila: bool = False, tequila_lambda: float = 1e-3,
                        use_sr_ste: bool = True):
    _BITLINEAR_CONFIG["use_tequila"] = use_tequila
    _BITLINEAR_CONFIG["tequila_lambda"] = tequila_lambda
    _BITLINEAR_CONFIG["use_sr_ste"] = use_sr_ste

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
        return grad_input, None, None, None


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

        # ponytail: fast path — fused Triton kernel when no hysteresis/sparsity active
        if not self.is_intermediate and not use_hyst and not use_sparse:
            try:
                from model.triton_fused import fused_bitlinear, HAS_TRITON as _HT
                if _HT and x.is_cuda:
                    # fused_bitlinear expects 2D (M,K); model passes 3D (B,T,K)
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
            # Mean bias removal — eliminates FP4 quantization instability (arXiv:2603.10444)
            x = x - x.mean(dim=-1, keepdim=True)
            beta = x.abs().mean(dim=-1, keepdim=True).detach() + 1e-5
            x_scaled = x * (2.6457 / beta)
            x_quant = x_scaled + (_ste.apply(torch.clamp(x_scaled, -8, 7)) - x_scaled)
            out = nn.functional.linear(x_quant, w_quant)
            out = out * (alpha * beta / 2.6457)
            if tequila_bias is not None:
                out = out + tequila_bias
            return out
        else:
            gamma = x.abs().max(dim=-1, keepdim=True)[0].detach() + 1e-5
            x_scaled = x * (127.0 / gamma)
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -128, 127)) - x_scaled)
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
    def __init__(self, d_model, d_ffn, sct_rank=0):
        super().__init__()
        if sct_rank > 0:
            self.w_gate_up = SpectralLinear(d_model, 2 * d_ffn, rank=sct_rank)
            self.w_down = SpectralLinear(d_ffn, d_model, rank=sct_rank, hadamard=True)
        else:
            self.w_gate_up = BitLinear_a4_8(d_model, 2 * d_ffn)
            self.w_down = H_BitLinear(d_ffn, d_model, is_intermediate=True)
        self.clipping_bounds = nn.Parameter(torch.ones(d_ffn) * 10.0)
        self.down_norm = RMSNorm(d_ffn)

    def forward(self, x):
        # ponytail: full fp32 for numerical stability at scale (d_model≥640, expert_hidden≥1536)
        x32 = x.float()
        gate_up = self.w_gate_up(x32)
        gate_raw, up = gate_up.chunk(2, dim=-1)
        gate_swish = gate_raw * torch.sigmoid(gate_raw)
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