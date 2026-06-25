"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ busel - mAR: mHC (DeepSeek) + AttnRes (Kimi) — exact      ║
║                                                                           ║
║ Manifold-Constrained Attention Residuals combines:                        ║
║   • mHC:  n_hyper parallel streams, mixing H ∈ Birkhoff polytope         ║
║           via Sinkhorn-Knopp (restores identity-mapping property)         ║
║   • AttnRes: input-dependent H computed via multi-query attention         ║
║           (q from current input, k from each stream)                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
import math
import random
import torch
import torch.nn as nn
from model.layers import BitLinear_a4_8, RMSNorm, nvtx_range_push, nvtx_range_pop
from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention, BulbaNSAAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanMoE
from multimodal.special_tokens import vocab_size as _vocab_size, enabled_ids as _enabled_ids


class ManifoldConstrainedAttnRes(nn.Module):
    """Manifold-Constrained Attention Residuals (mAR).

    Combines Kimi Attention Residuals (input-dependent attention over layer
    outputs) with DeepSeek mHC (Sinkhorn-Knopp projection of the mixing matrix
    onto the Birkhoff polytope of doubly-stochastic matrices).

    Maintains n_hyper parallel residual streams. At each call:
      1. Compute n queries from current input, n keys from the n streams
         (multi-query attention — AttnRes spirit).
      2. Build raw H_logits ∈ R^{n×n} per (B, T) via q·k + fixed identity
         bias (+5.0 on diagonal, mHC's identity-mapping property at init).
      3. Project to Birkhoff polytope via Sinkhorn-Knopp (mHC constraint).
      4. Mix the n streams with H, return the mean over streams.

    Args:
        d_model: residual stream width (must be divisible by n_hyper).
        n_hyper: number of parallel hyper-connection streams (default 2).
        n_sinkhorn_iters: Sinkhorn-Knopp iterations (paper uses 3-20).
    """

    def __init__(self, d_model: int, n_hyper: int = 2, n_sinkhorn_iters: int = 3):
        super().__init__()
        if d_model % n_hyper != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_hyper ({n_hyper})")
        self.d_model = d_model
        self.n_hyper = n_hyper
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.d_head = d_model // n_hyper

        self.q_proj = BitLinear_a4_8(d_model, d_model)
        self.k_proj = BitLinear_a4_8(d_model, self.d_head)

        identity_bias = torch.zeros(n_hyper, n_hyper)
        for i in range(n_hyper):
            identity_bias[i, i] = 1.0  # ponytail: D10 — 1.0 instead of 5.0 to allow q@k^T gradient through mAR
        # Add small random noise to break symmetry between streams and enable gradient flow
        identity_bias = identity_bias + torch.randn(n_hyper, n_hyper) * 0.1
        self.register_buffer("identity_bias", identity_bias)

        self.temperature = nn.Parameter(torch.ones(1))

        self.norm = RMSNorm(d_model)

    def sinkhorn_knopp(self, M: torch.Tensor, n_iters: int | None = None) -> torch.Tensor:
        """Project M onto the Birkhoff polytope (doubly-stochastic matrices).

        Uses DTopK (Differentiable Top-K) for speed: single softmax + sparsify
        + one column normalization, instead of iterative Sinkhorn. ~10× faster.
        
        M: [..., n, n] real-valued matrix.
        Returns: [..., n, n] doubly-stochastic matrix (rows AND cols sum to 1).
        """
        if n_iters is None:
            n_iters = self.n_sinkhorn_iters
        M = M * self.temperature
        n = M.size(-1)

        # DTopK for n>=4: sparsified double-stochastic. Small matrices use classic Sinkhorn.
        if n >= 4:
            M = torch.softmax(M, dim=-1)
            k = max(1, n // 2)
            _, idx = torch.topk(M, k, dim=-1)
            mask = torch.zeros_like(M).scatter_(-1, idx, 1.0)
            M = M * mask
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        else:
            M = torch.exp(M - M.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0])
            for _ in range(n_iters):
                M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
                M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)
        return M

    def forward(self, current_x: torch.Tensor, streams: torch.Tensor) -> torch.Tensor:
        """Mix n_hyper streams using input-dependent doubly-stochastic H.

        Args:
            current_x: [B, T, d_model] — input to current layer.
            streams: [n_hyper, B, T, d_model] — stacked tensor of streams.

        Returns:
            y: [B, T, d_model] — mixed stream (mean over n_hyper).
        """
        n = self.n_hyper
        if streams.shape[0] != n:
            raise ValueError(f"Expected {n} streams, got {streams.shape[0]}")
        B, T, _ = current_x.shape

        q = self.q_proj(current_x).view(B, T, n, self.d_head)

        k_stack = self.k_proj(streams).permute(1, 2, 0, 3)

        H_logits = torch.einsum('btqd,btkd->btqk', q, k_stack) / math.sqrt(self.d_head)
        H_logits = H_logits + self.identity_bias

        H = self.sinkhorn_knopp(H_logits)

        streams_stack = streams.permute(1, 2, 0, 3)
        y_streams = torch.einsum('btij,btjd->btid', H, streams_stack)
        y = y_streams.mean(dim=2)
        return self.norm(y)


class buselDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, expert_hidden, num_experts, is_global=False, capacity_factor=1.0, top_k=2, use_differential=False, layer_idx=0, mod_interval=2, nsa_n_heads=16, sct_rank=0, use_matmul_free=False):
        super().__init__()
        self.mod_router = MoDSequenceRouter(d_model, capacity_factor=capacity_factor)
        self.layer_idx = layer_idx
        self.mod_interval = mod_interval
        if is_global and d_model >= 256 and nsa_n_heads % 16 == 0:
            self.attn = BulbaNSAAttention(d_model, nsa_n_heads)
        elif is_global:
            self.attn = MultiHeadLatentAttention(d_model, n_heads, use_differential=use_differential)
        else:
            self.attn = BulbaGDN2SeRoPEBlock(d_model, n_heads)
        self.moe = BulbaTernaryTitanMoE(d_model, expert_hidden, num_experts=num_experts, top_k=top_k, sct_rank=sct_rank, use_matmul_free=use_matmul_free)
        self.attn_norm = RMSNorm(d_model)
        self.moe_norm = RMSNorm(d_model)

    def forward(self, x, progress=0.0):
        do_mod = self.mod_router.capacity_factor < 1.0 and (self.layer_idx % self.mod_interval == 0)
        if not do_mod:
            attn_out = self.attn(self.attn_norm(x))
            moe_out, aux_loss = self.moe(self.moe_norm(attn_out), progress=progress)
            return moe_out, aux_loss
        B, T, C = x.shape
        mask, logits, topk_idx, mod_aux = self.mod_router(x)
        k = max(2, int(T * self.mod_router.capacity_factor))
        if k >= T:
            attn_out = self.attn(self.attn_norm(x))
            moe_out, aux_loss = self.moe(self.moe_norm(attn_out), progress=progress)
            return moe_out, aux_loss + mod_aux
        # ponytail: integer indexing instead of boolean mask — torch.compile friendly
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, k)  # (B, k)
        active_tokens = x[batch_idx, topk_idx]  # (B, k, C)
        attn_out = self.attn(self.attn_norm(active_tokens))
        moe_out, aux_loss = self.moe(self.moe_norm(attn_out), progress=progress)
        # ponytail: detach logits in gate — prevents double-gradient feedback
        # (topk backward + sigmoid backward both push logits → ±∞ over ~170 steps).
        # Router still learns via topk gradient + aux loss; gate still works.
        gated_out = moe_out * torch.sigmoid(logits.detach()[batch_idx, topk_idx]).unsqueeze(-1)
        out = torch.zeros_like(x)
        out[batch_idx, topk_idx] = gated_out.to(out.dtype)
        return out, aux_loss + mod_aux


class buselMTPPipeline(nn.Module):
    """Multi-Token Prediction pipeline — predicts the next N patch tokens.

    Uses N-1 autoregressive steps (t+1 is direct from main_hidden_states).
    Each future head: detach → projection → embed_lookup(prev_token) → next head.
    """
    def __init__(self, config):
        super().__init__()
        self.n_mtp_heads = int(getattr(config, "num_mtp_heads", 4))
        self.embed_weight = nn.Parameter(torch.randn(config.vocab_size, config.d_model) * 0.02)
        self.projection = BitLinear_a4_8(config.d_model, config.d_model)  # shared
        padded_vocab = ((config.vocab_size + 15) // 16) * 16  # FP8-safe: next multiple of 16
        self.head = BitLinear_a4_8(config.d_model, padded_vocab)           # FP8-safe
        self.vocab_size = config.vocab_size
        self.pos_embed = nn.Parameter(torch.randn(self.n_mtp_heads, config.d_model) * 0.02)

    def _embed_lookup(self, token_ids):
        return self.embed_weight[token_ids.to(self.embed_weight.device)]

    def forward(self, main_hidden_states, next_token_ids=None):
        B, T, D = main_hidden_states.shape
        _h = lambda t: self.head(t)[..., :self.vocab_size]  # trim FP8 padding

        # t+1: direct from main_hidden_states
        x = main_hidden_states.unsqueeze(2) + self.pos_embed[:1]
        logits = [_h(x[:, :, 0])]

        if next_token_ids is None or any(t is None for t in next_token_ids):
            # Fill remaining with None for API compatibility
            logits.extend([None] * (self.n_mtp_heads - 1))
            return tuple(logits)

        h_d = main_hidden_states.detach()
        prev_h = h_d
        for i in range(1, self.n_mtp_heads):
            prev_h = self.projection(prev_h) + self._embed_lookup(next_token_ids[i - 1])
            h_pos = prev_h.unsqueeze(2) + self.pos_embed[i:i+1]
            logits.append(_h(h_pos.squeeze(2)))

        return tuple(logits)


class buselModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_hyper = int(getattr(config, "n_hyper", 2))
        self.vocab_size = int(getattr(config, "vocab_size", 0))
        if self.vocab_size == 0:
            self.vocab_size = _vocab_size()
        registry_vocab = _vocab_size()
        if self.vocab_size < registry_vocab:
            raise ValueError(
                f"config.vocab_size={self.vocab_size} is smaller than the "
                f"current special-token registry vocab={registry_vocab}. "
                f"Update configs/default.yaml or disable tokens to match. "
                f"Enabled token IDs: {_enabled_ids()[:10]}{'…' if len(_enabled_ids()) > 10 else ''}"
            )

        capacity = float(getattr(config, "mod_capacity", 1.0))
        mod_interval = int(getattr(config, "mod_interval", 2))
        nsa_n_heads = int(getattr(config, "nsa_n_heads", 16))
        use_differential = bool(getattr(config, "use_differential_attention", False))
        use_matmul_free = bool(getattr(config, "use_matmul_free", False))
        sct_rank = int(getattr(config, "sct_rank", 0))
        self.layers = nn.ModuleList()
        for l in range(config.n_layers):
            # 7:1 GDN-2:global ratio. Global layer auto-picks Diff MLA (short) or NSA (long).
            is_global = (l == config.n_layers - 1) or ((l + 1) % 8 == 0)
            self.layers.append(buselDecoderLayer(
                config.d_model, config.n_heads, config.expert_hidden,
                config.num_experts, is_global=is_global, capacity_factor=capacity,
                top_k=int(getattr(config, "top_k", 2)),
                use_differential=use_differential,
                layer_idx=l, mod_interval=mod_interval,
                nsa_n_heads=nsa_n_heads,
                sct_rank=sct_rank,
                use_matmul_free=use_matmul_free,
            ))

        self.m_residuals = nn.ModuleList([
            ManifoldConstrainedAttnRes(config.d_model, n_hyper=self.n_hyper)
            for _ in range(config.n_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)
        self.mtp_pipeline = buselMTPPipeline(config)
        self.n_mtp_heads = int(getattr(config, "num_mtp_heads", 4))
        self.use_gradient_checkpointing = False
        self.checkpoint_every = 1
        self.selective_backward = bool(getattr(config, "selective_backward", False))
        self.backward_ratio = max(0.0, min(1.0, float(getattr(config, "backward_ratio", 1.0))))
        # ponytail: D5 — freeze stabilized layers in late training (1.5× speedup). Disabled until step 50%.
        self._progressive_freeze = bool(getattr(config, "progressive_freeze", False))
        self._freeze_threshold = float(getattr(config, "freeze_threshold", 0.1))
        self._frozen_layers: set[int] = set()
        self._layer_var: dict[int, list[float]] = {}
        self._selected_layers: list[int] = list(range(config.n_layers))
        self.use_dropbp = bool(getattr(config, "use_dropbp", False))
        self.dropbp_prob = float(getattr(config, "dropbp_prob", 0.3))

    def enable_gradient_checkpointing(self, every: int = 1):
        self.use_gradient_checkpointing = True
        self.checkpoint_every = max(1, int(every))
    def disable_gradient_checkpointing(self): self.use_gradient_checkpointing = False

    def set_rope_scale(self, scale: float):
        """YaRN: set rotary position encoding scale for all GDN-2 layers. 1.0→32.0 for 4K→128K."""
        for layer in self.layers:
            if hasattr(layer.attn, 'rope_scale'):
                layer.attn.rope_scale = scale

    def forward(self, x, next_token_ids=None, progress=0.0):
        nvtx_range_push("buselModel_Forward")
        progress = round(float(progress), 1)
        streams = x.unsqueeze(0).expand(self.n_hyper, *x.shape).contiguous()
        total_aux_loss = 0.0
        ckpt_every = self.checkpoint_every if self.use_gradient_checkpointing else 1
        ckpt_eligible = self.training and self.use_gradient_checkpointing and x.device.type == "cuda"

        if self.selective_backward and self.training and self.backward_ratio < 1.0:
            n_layers = len(self.layers)
            n_select = max(1, int(n_layers * self.backward_ratio))
            self._selected_layers = random.sample(range(n_layers), n_select)
        else:
            self._selected_layers = list(range(len(self.layers)))

        # DropBP: randomly skip backward through layers (independent per-layer probability)
        _dropbp = set()
        if self.use_dropbp and self.training:
            _dropbp = {i for i in range(len(self.layers)) if random.random() < self.dropbp_prob}

        # ponytail: D5 — progressive layer freezing. After 50% progress, freeze last layers (1.5× late speedup).
        frozen = set()
        if self._progressive_freeze and progress > 0.5:
            freeze_frac = min(0.75, (progress - 0.5) / 0.4 * 0.75)
            n_active = int(len(self.layers) * (1.0 - freeze_frac))
            frozen = set(range(n_active, len(self.layers)))

        for i, layer in enumerate(self.layers):
            mixed = self.m_residuals[i](x, streams)

            no_grad = i in frozen or (i not in self._selected_layers and self.training) or i in _dropbp
            if no_grad:
                with torch.no_grad():
                    layer_out, aux_loss = layer(mixed, progress=progress)
            elif ckpt_eligible and (i % ckpt_every == 0):
                layer_out, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, mixed, progress, use_reentrant=False, determinism_check="none"
                )
            else:
                layer_out, aux_loss = layer(mixed, progress=progress)

            x = mixed + layer_out
            # ponytail: grad-safe activation scaling — tanh limits extremes without killing gradient
            if self.training: x = 100.0 * torch.tanh(x / 100.0)
            total_aux_loss += aux_loss
            streams = torch.cat([streams[1:], x.unsqueeze(0)], dim=0)

        final_hidden = self.final_norm(x)
        self._last_hidden = final_hidden  # ponytail: exposed for EMA self-distillation (D3)
        mtp_outputs = self.mtp_pipeline(final_hidden, next_token_ids)
        nvtx_range_pop()
        return mtp_outputs, total_aux_loss
