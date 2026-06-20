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
    """Token-level routing mask."""
    def __init__(self, d_model, capacity_factor=1.0):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, 1)

    def forward(self, x):
        if self.capacity_factor >= 1.0:
            return None, None
        logits = self.router(x.detach()).squeeze(-1)
        k = int(x.shape[1] * self.capacity_factor)
        _, indices = torch.topk(logits, max(1, k), dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, indices, True)
        return mask, logits


class BulbaTernaryTitanExpertFFN(nn.Module):
    """Single MoE expert: SwishGLU FFN."""
    def __init__(self, d_model, d_ffn, sct_rank=0):
        super().__init__()
        from model.layers import SwishGLUClamped
        self.ffn = SwishGLUClamped(d_model, d_ffn, sct_rank=sct_rank)
    def forward(self, x):
        return self.ffn(x)


class BulbaTernaryTitanMoE(nn.Module):
    def __init__(self, d_model, d_ffn, num_experts=64, top_k=2, sct_rank=0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.shared_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn, sct_rank=sct_rank)
            for _ in range(2)
        ])
        self.routed_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn, sct_rank=sct_rank)
            for _ in range(num_experts)
        ])
        
        self.w_gate_blackboard = BitLinear_a4_8(d_model, d_model)
        self.w_read_blackboard = BitLinear_a4_8(d_model, d_model)
        self.shared_gate = BitLinear_a4_8(d_model, d_model)
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
        
        # Routing-Free: last N dims of hidden state = expert logits
        router_logits = x_enriched[..., -self.num_experts:].to(dtype=x_enriched.dtype)
        
        # Gumbel exploration
        if self.training:
            u = torch.rand_like(router_logits).clamp(1e-8, 1 - 1e-8)
            router_logits = router_logits - torch.log(-torch.log(u)) * 0.1
        
        # Loss-Free bias
        router_logits = router_logits + self.expert_bias
        
        # EntMax sparse routing + Top-K
        routing_weights = _entmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Expert dispatch
        routed_output = torch.zeros_like(x_enriched)
        expert_masks = []
        expert_weights_list = []
        for i in range(self.num_experts):
            mask = (topk_indices == i)
            weight_sum = torch.where(mask, topk_weights, torch.zeros_like(topk_weights)).sum(dim=-1)
            expert_masks.append(mask.any(dim=-1))
            expert_weights_list.append(weight_sum)
        
        for i in range(self.num_experts):
            mask = expert_masks[i]
            if mask.any():
                tokens = x_enriched[mask]
                out = self.routed_experts[i](tokens)
                weights = expert_weights_list[i][mask].unsqueeze(-1)
                routed_output[mask] = (out * weights).to(routed_output.dtype)
        
        # Loss-Free: compute bias delta
        if self.training:
            n_tokens = torch.tensor([m.sum().float() for m in expert_masks], device=x.device)
            load = n_tokens / (n_tokens.sum() + 1e-8)
            target = 1.0 / self.num_experts
            self._bias_delta = 0.01 * (target - load)
        
        nvtx_range_pop()
        return h_bb + routed_output, torch.tensor(0.0, device=x.device)
    
    def update_bias(self):
        if self._bias_delta is not None:
            self.expert_bias.data.mul_(0.99).add_(self._bias_delta)
            self._bias_delta = None
