"""MoE with Routing-Free + Loss-Free Balancing + Gated Shared Experts."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BitLinear_a4_8, nvtx_range_push, nvtx_range_pop

def _entmax(logits, dim=-1):
    """Sparse softmax — produces exact zeros."""
    if dim < 0: dim = logits.dim() + dim
    z = logits.sort(dim=dim, descending=True).values
    css = z.cumsum(dim=dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device).float()
    for _ in range(dim): k = k.unsqueeze(0)
    tau = ((css - 1) / k.clamp(min=1))
    support = (z > tau).float()
    tau_star = ((css * support).sum(dim=dim, keepdim=True) - 1) / support.sum(dim=dim, keepdim=True).clamp(min=1)
    return (logits - tau_star).clamp(min=0)


class MoDSequenceRouter(nn.Module):
    """Token-level routing mask with load-balancing aux loss."""
    def __init__(self, d_model, capacity_factor=1.0):
        super().__init__()
        self.capacity_factor = capacity_factor
        # ponytail: nn.Linear not BitLinear — (d_model, 1) is ~4 KB, ternary
        # quant provides zero benefit but its INT8 act-quant unstable with GRAD
        # accumulation over ~170 steps → NaN at step 177.
        self.router = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        if self.capacity_factor >= 1.0:
            return None, None, None, 0.0
        x_detached = x.detach().float()
        x_detached = torch.nan_to_num(x_detached, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = self.router(x_detached).squeeze(-1)
        # ponytail: Gumbel noise for exploration — prevents router from
        # converging to same tokens (→ logit ceiling → zero gradient → NaN @ step 170)
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            logits = logits - torch.log(-torch.log(u)) * 0.1
        k = max(2, int(x.shape[1] * self.capacity_factor))
        _, indices = torch.topk(logits, max(1, k), dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
        kept_frac = mask.float().mean()
        aux_loss = ((kept_frac - self.capacity_factor) ** 2) * 0.1
        return mask, logits, indices, aux_loss


class BulbaTernaryTitanExpertFFN(nn.Module):
    """Single MoE expert: SwishGLU FFN."""
    def __init__(self, d_model, d_ffn, sct_rank=0, use_matmul_free=False):
        super().__init__()
        from model.layers import SwishGLUClamped
        self.ffn = SwishGLUClamped(d_model, d_ffn, sct_rank=sct_rank, use_matmul_free=use_matmul_free)
    def forward(self, x):
        return self.ffn(x)


class BulbaTernaryTitanMoE(nn.Module):
    def __init__(self, d_model, d_ffn, num_experts=64, top_k=2, sct_rank=0, use_matmul_free=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.shared_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn, sct_rank=sct_rank, use_matmul_free=use_matmul_free)
            for _ in range(2)
        ])
        self.routed_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn, sct_rank=sct_rank, use_matmul_free=use_matmul_free)
            for _ in range(num_experts)
        ])
        
        self.w_gate_blackboard = BitLinear_a4_8(d_model, d_model)
        self.w_read_blackboard = BitLinear_a4_8(d_model, d_model)
        self.shared_gate = BitLinear_a4_8(d_model, d_model)
        # Routing-Free: learnable projection instead of raw features
        self.w_router = BitLinear_a4_8(d_model, num_experts)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self._bias_delta = None

    def forward(self, x, progress=0.0):
        nvtx_range_push("busel_MoE_Experts_Forward")
        B, T, D = x.shape
        
        # Gated Shared Experts
        gate = torch.sigmoid(self.shared_gate(x.detach()).mean(dim=-1, keepdim=True))
        h_bb = gate * self.shared_experts[0](x) + (1 - gate) * self.shared_experts[1](x)
        
        # Blackboard enrichment
        gate_sig = torch.sigmoid(self.w_gate_blackboard(x))
        read_sig = self.w_read_blackboard(h_bb)
        x_enriched = x + gate_sig * read_sig
        
        # Routing-Free: learnable router projection on enriched hidden state
        router_logits = self.w_router(x_enriched)
        
        # ponytail: Gumbel disabled (crashes at batch>28). Entmax + loss-free bias is sufficient.
        # if self.training:
        #     u = torch.rand_like(router_logits).clamp(1e-8, 1 - 1e-8)
        #     router_logits = router_logits - torch.log(-torch.log(u)) * 0.1
        
        # Loss-Free bias
        router_logits = router_logits + self.expert_bias
        
        # EntMax sparse routing + Top-K
        routing_weights = _entmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Expert dispatch — ponytail: pre-sort tokens by expert for batched FFN calls
        routed_output = torch.zeros_like(x_enriched)
        B, T, D = x_enriched.shape
        flat_indices = topk_indices.reshape(-1)          # (B*T*top_k,)
        flat_tokens = x_enriched.unsqueeze(2).expand(-1, -1, self.top_k, -1).reshape(-1, D)
        flat_weights = topk_weights.reshape(-1)           # (B*T*top_k,)

        sort_idx = flat_indices.argsort(stable=True)
        sorted_tokens = flat_tokens[sort_idx]
        sorted_weights = flat_weights[sort_idx]
        sorted_indices = flat_indices[sort_idx]

        counts = torch.bincount(flat_indices, minlength=self.num_experts)
        offset = 0
        for i in range(self.num_experts):
            n = int(counts[i].item())
            if n > 0:
                batch = sorted_tokens[offset:offset + n]
                out = self.routed_experts[i](batch)
                w = sorted_weights[offset:offset + n].unsqueeze(-1)
                routed_output.view(-1, D)[sort_idx[offset:offset + n]] = (out * w).to(routed_output.dtype)
                offset += n
        
        # Load-balancing: Loss-Free bias delta (no aux loss — Wang et al. 2024)
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            n_tokens = counts.float()
            total = n_tokens.sum() + 1e-8
            f_i = n_tokens / total
            target = 1.0 / self.num_experts
            self._bias_delta = 0.01 * (target - f_i)
        
        nvtx_range_pop()
        return h_bb + routed_output, aux_loss
    
    def update_bias(self):
        if self._bias_delta is not None:
            self.expert_bias.data.mul_(0.99).add_(self._bias_delta)
            self._bias_delta = None
