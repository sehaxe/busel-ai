"""
💡 busel - Gated DeltaNet-2 & MLA (Stabilized Broadcasting)
Интегрирован раздельный закон стирания и записи GDN-2,
когерентный логарифмический распад alpha (Eq. 12) с выверенным бродкастом.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BitLinear_a4_8, H_BitLinear, RMSNorm, nvtx_range_push, nvtx_range_pop
from busel_registry import register

# GDN-2 Triton kernel: NVIDIA chunk_gdn2 (Blackwell-stable) or PyTorch fallback
try:
    from model.gdn2_chunk import chunk_gdn2 as _chunk_gdn2
    _GDN2_TRITON = True
except Exception:
    from model.gdn2 import gdn2_recurrent as _gdn2_recurrent
    _GDN2_TRITON = False


def _sdpa(q, k, v, is_causal: bool = False):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


@register("attention", "gdn2")
class BulbaGDN2SeRoPEBlock(nn.Module):
    def __init__(self, d_model=1536, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.q_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.k_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.v_proj = BitLinear_a4_8(d_model, n_heads * self.d_v)
        
        # Каузальные depthwise свертки (NVIDIA GDN-2 Spec)
        self.q_conv = nn.Conv1d(n_heads * self.d_k, n_heads * self.d_k, kernel_size=4, groups=n_heads * self.d_k, padding=0)
        self.k_conv = nn.Conv1d(n_heads * self.d_k, n_heads * self.d_k, kernel_size=4, groups=n_heads * self.d_k, padding=0)
        self.v_conv = nn.Conv1d(n_heads * self.d_v, n_heads * self.d_v, kernel_size=4, groups=n_heads * self.d_v, padding=0)
        
        self.b_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.w_proj = BitLinear_a4_8(d_model, n_heads * self.d_v)
        
        # ЛОГАРИФМИЧЕСКИЙ РАСПАД NVIDIA GDN-2 (Формула 12)
        self.alpha_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        # Обучаемый вектор масштаба логарифмического затухания, инициализируемый отрицательным числом
        self.alpha_a = nn.Parameter(torch.ones(n_heads, 1) * -3.0)
        
        # Низкоранговый выходной гейт (Output Gating — Формула 10)
        self.g_proj_down = BitLinear_a4_8(d_model, d_model // 4)
        self.g_proj_up = BitLinear_a4_8(d_model // 4, d_model)
        self.out_norm = RMSNorm(d_model)
        
        # o_proj заменена на H_BitLinear по спецификации BitNet v2
        self.o_proj = H_BitLinear(d_model, d_model)
        self.register_buffer("freqs", 10000 ** (-torch.arange(0, self.d_k, 2).float() / self.d_k))
        self.rope_scale: float = 1.0  # ponytail: YaRN scaling factor. 1.0→32.0 for 4K→128K

    def apply_serope(self, T, q, k):
        B, _, H, _ = q.shape
        q_real, q_imag = q[..., 0::2], q[..., 1::2]
        k_real, k_imag = k[..., 0::2], k[..., 1::2]
        
        # ponytail: YaRN scaling — multiply freqs by rope_scale for context extension
        scaled_freqs = self.freqs * self.rope_scale
        angles = torch.arange(T, device=q.device).view(1, T, 1, 1) * scaled_freqs.view(1, 1, 1, -1)
        cos, sin = torch.cos(angles), torch.sin(angles)
        
        q_out = torch.zeros_like(q)
        k_out = torch.zeros_like(k)
        q_out[..., 0::2], q_out[..., 1::2] = q_real * cos - q_imag * sin, q_real * sin + q_imag * cos
        k_out[..., 0::2], k_out[..., 1::2] = k_real * cos + k_imag * sin, -k_real * sin + k_imag * cos
        return q_out, k_out

    def forward(self, x):
        nvtx_range_push("busel_GDN2_SeRoPE_Forward")
        B, T, C = x.shape
        
        q_proj = self.q_proj(x).transpose(1, 2)
        q_conv = self.q_conv(F.pad(q_proj, (3, 0)))
        q = F.silu(q_conv).transpose(1, 2).view(B, T, self.n_heads, self.d_k)
        
        k_proj = self.k_proj(x).transpose(1, 2)
        k_conv = self.k_conv(F.pad(k_proj, (3, 0)))
        k = F.silu(k_conv).transpose(1, 2).view(B, T, self.n_heads, self.d_k)
        
        v_proj = self.v_proj(x).transpose(1, 2)
        v_conv = self.v_conv(F.pad(v_proj, (3, 0)))
        v = F.silu(v_conv).transpose(1, 2).view(B, T, self.n_heads, self.d_v)
        
        q, k = self.apply_serope(T, q, k)
        
        b = torch.sigmoid(self.b_proj(x)).view(B, T, self.n_heads, self.d_k) * 2.0  # [0,2] — Gated DeltaNet-2 negative eigenvalues
        w = torch.sigmoid(self.w_proj(x)).view(B, T, self.n_heads, self.d_v)         # [0,1] — write gate unchanged
        g = self.alpha_proj(x).view(B, T, self.n_heads, self.d_k)
        if _GDN2_TRITON:
            out, _ = _chunk_gdn2(q.float(), k.float(), v.float(), g.float(), b.float(), w.float(),
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=self.alpha_a)
        else:
            out = _gdn2_recurrent(q, k, v, g, b, w, self.alpha_a)
        out = out.view(B, T, -1)
            
        gate = torch.sigmoid(self.g_proj_up(self.g_proj_down(x)))
        out_gated = self.out_norm(out) * gate
        
        res = self.o_proj(out_gated)
        nvtx_range_pop()
        return res


@register("attention", "mla")
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=1536, n_heads=12, d_c=128, use_differential=False, use_qknorm_l2=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_c = d_c
        self.d_v = d_model // n_heads
        self.use_differential = use_differential
        self.use_qknorm_l2 = use_qknorm_l2

        self.kv_compress = BitLinear_a4_8(d_model, d_c)
        self.kv_norm = RMSNorm(d_c)
        self.k_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        self.v_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)

        self.q_compress = BitLinear_a4_8(d_model, d_c)
        self.q_norm = RMSNorm(d_c)
        self.q_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)

        if use_differential:
            self.q2_compress = BitLinear_a4_8(d_model, d_c)
            self.q2_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
            self.k2_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
            self.diff_lambda = nn.Parameter(torch.tensor(1.0))

        self.out_norm = RMSNorm(n_heads * self.d_v)
        # o_proj заменена на H_BitLinear по спецификации BitNet v2
        self.o_proj = H_BitLinear(n_heads * self.d_v, d_model)

    def forward(self, x):
        nvtx_range_push("busel_MLA_Forward")
        B, T, C = x.shape
        kv_latent = self.kv_norm(self.kv_compress(x))
        k = self.k_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        v = self.v_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)

        q_latent = self.q_norm(self.q_compress(x))
        q = self.q_decompress(q_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)

        if self.use_qknorm_l2:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        attn1 = _sdpa(q, k, v)
        if self.use_differential:
            q2 = self.q2_decompress(self.q_norm(self.q2_compress(x))).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
            k2 = self.k2_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
            if self.use_qknorm_l2:
                q2 = F.normalize(q2, p=2, dim=-1)
                k2 = F.normalize(k2, p=2, dim=-1)
            attn2 = _sdpa(q2, k2, v)
            context = (attn1 - attn2) * self.diff_lambda
        else:
            context = attn1
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(self.out_norm(context))
        nvtx_range_pop()
        return out


# ── NSA — Native Sparse Attention (DeepSeek 2025) ─────────────────────────

@register("attention", "nsa")
class BulbaNSAAttention(nn.Module):
    """Native Sparse Attention — hardware-aligned, natively trainable sparse attention.
    DeepSeek 2025 (arXiv:2502.11089). 3× faster than dense MLA, matches/exceeds quality.
    Requires n_heads % 16 == 0 (FlA constraint).
    """
    def __init__(self, d_model=1536, n_heads=16):
        super().__init__()
        assert n_heads % 16 == 0, f"NSA requires n_heads divisible by 16, got {n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.k_proj = BitLinear_a4_8(d_model, self.d_k)  # 1 key head (GQA)
        self.v_proj = BitLinear_a4_8(d_model, self.d_k)
        # 3 branch gates: compression, selection, sliding window
        self.g_cmp = BitLinear_a4_8(d_model, n_heads)
        self.g_slc = BitLinear_a4_8(d_model, n_heads)
        self.g_swa = BitLinear_a4_8(d_model, n_heads)
        self.o_proj = H_BitLinear(d_model, d_model)
        self.out_norm = RMSNorm(d_model)

    @torch.compiler.disable  # ponytail: fla parallel_nsa Triton kernel incompatible with dynamo trace
    def forward(self, x):
        nvtx_range_push("busel_NSA_Forward")
        B, T, C = x.shape
        from fla.ops.nsa import parallel_nsa
        # ponytail: chunk large batches to avoid OOM in fla's parallel_nsa
        _MAX_NSA_BATCH = 64
        if B > _MAX_NSA_BATCH:
            chunks = []
            for i in range(0, B, _MAX_NSA_BATCH):
                xc = x[i:i + _MAX_NSA_BATCH]
                q = self.q_proj(xc).view(-1, T, self.n_heads, self.d_k)
                k = self.k_proj(xc).view(-1, T, 1, self.d_k)
                v = self.v_proj(xc).view(-1, T, 1, self.d_k)
                g_cmp = self.g_cmp(xc).view(-1, T, self.n_heads)
                g_slc = self.g_slc(xc).view(-1, T, self.n_heads)
                g_swa = self.g_swa(xc).view(-1, T, self.n_heads)
                ctx = parallel_nsa(q, k, v, g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa, block_size=32)
                chunks.append(ctx.reshape(ctx.shape[0], T, -1))
            context = torch.cat(chunks, dim=0)
        else:
            # ponytail: at extreme context (>128K), NSA selection branch collapses → use sliding window only
            if T > 131072:
                q = self.q_proj(x).view(B, T, self.n_heads, self.d_k)
                k = self.k_proj(x).view(B, T, 1, self.d_k)
                v = self.v_proj(x).view(B, T, 1, self.d_k)
                # pure sliding window attention — O(1) memory, no selection overhead
                context = parallel_nsa(q, k, v, block_size=32)
            else:
                q = self.q_proj(x).view(B, T, self.n_heads, self.d_k)
                k = self.k_proj(x).view(B, T, 1, self.d_k)
                v = self.v_proj(x).view(B, T, 1, self.d_k)
                g_cmp = self.g_cmp(x).view(B, T, self.n_heads)
                g_slc = self.g_slc(x).view(B, T, self.n_heads)
                g_swa = self.g_swa(x).view(B, T, self.n_heads)
                context = parallel_nsa(q, k, v, g_cmp=g_cmp, g_slc=g_slc, g_swa=g_swa, block_size=32)
            context = context.reshape(B, T, -1)
        out = self.o_proj(self.out_norm(context))
        nvtx_range_pop()
        return out

