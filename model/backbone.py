"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL BACKBONE v3.6 - Stable Cross-Platform Architecture                 ║
║                                                                           ║
║ Компоненты:                                                               ║
║  • ByselDecoderLayer: Attention + MoD + MoE Blackboard                   ║
║  • ManifoldConstrainedAttnRes (mAR): межслойные связи без аллокаций      ║
║  • ByselMTP4Pipeline: Multi-Token Prediction (4 головы)                  ║
║  • ByselModel: оркестратор с безопасным Gradient Checkpointing            ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
from model.layers import BitLinear_a4_8, RMSNorm, nvtx_range_push, nvtx_range_pop
from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanMoE


class ManifoldConstrainedAttnRes(nn.Module):
    """
    Улучшенный mAR с проекцией на Birkhoff Polytope.
    
    Решает проблему Attention Sink через 3 итерации Sinkhorn-Knopp,
    вычисляя логиты на лету без промежуточного выделения гигабайтных тензоров.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = RMSNorm(d_model)
        nn.init.zeros_(self.proj.weight)

    def forward(self, current_x, all_prev_outputs):
        if not all_prev_outputs:
            return current_x
        
        # Вычисляем логиты для каждого слоя индивидуально во избежание оверхеда памяти
        logits_list = []
        proj_weight = self.proj.weight.squeeze()
        
        for prev_x in all_prev_outputs:
            K_part = self.norm(prev_x)
            logit_part = torch.einsum('d, b t d -> b t', proj_weight, K_part)
            logits_list.append(logit_part)
        
        # Стек логитов [L, B, T]
        M = torch.stack(logits_list, dim=0)
        
        # Стабилизация экспоненты под float16
        M_stable = M - M.max(dim=0, keepdim=True)[0]
        M = torch.exp(M_stable)
        
        # 3 итерации Sinkhorn-Knopp
        for _ in range(3):
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Взвешенная сумма без промежуточных аллокаций
        h = torch.zeros_like(current_x)
        for l in range(len(all_prev_outputs)):
            h = h + M[l].unsqueeze(-1) * all_prev_outputs[l]
            
        return current_x + h


class ByselDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, expert_hidden, num_experts, is_global=False):
        super().__init__()
        self.mod_router = MoDSequenceRouter(d_model, capacity_factor=1.0)
        
        if is_global:
            self.attn = MultiHeadLatentAttention(d_model, n_heads)
        else:
            self.attn = BulbaGDN2SeRoPEBlock(d_model, n_heads)
        
        self.moe = BulbaTernaryTitanMoE(d_model, expert_hidden, num_experts=num_experts)
        
        self.attn_norm = RMSNorm(d_model)
        self.moe_norm = RMSNorm(d_model)

    def forward(self, x):
        # Временно все токены проходят через слои для стабильного pretraining
        active_tokens = x
        
        attn_out = self.attn(self.attn_norm(active_tokens))
        moe_out, aux_loss = self.moe(self.moe_norm(attn_out))
        
        out = x + moe_out
        return out, aux_loss


class ByselMTP4Pipeline(nn.Module):
    """
    Multi-Token Prediction с 4 головами.
    Использует прямую индексацию для обхода MPS багов.
    """
    def __init__(self, config):
        super().__init__()
        
        # Parameter вместо nn.Embedding для стабильности выравнивания
        self.embed_weight = nn.Parameter(
            torch.randn(config.vocab_size, config.d_model) * 0.02
        )
        
        # Проекции для передачи состояния между головами
        self.projections = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.d_model) 
            for _ in range(3)
        ])
        
        # 4 головы для предсказания t+1, t+2, t+3, t+4
        self.heads = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.vocab_size) 
            for _ in range(4)
        ])

    def _embed_lookup(self, token_ids):
        return self.embed_weight[token_ids.to(self.embed_weight.device)]

    def forward(self, main_hidden_states, next_token_ids=None):
        logits_t1 = self.heads[0](main_hidden_states)
        
        if next_token_ids is None or any(t is None for t in next_token_ids):
            return logits_t1, None, None, None
        
        h_detached = main_hidden_states.detach()
        
        # t+2
        combined_t2 = self.projections[0](h_detached) + self._embed_lookup(next_token_ids[0])
        logits_t2 = self.heads[1](combined_t2)
        
        # t+3
        combined_t3 = self.projections[1](combined_t2) + self._embed_lookup(next_token_ids[1])
        logits_t3 = self.heads[2](combined_t3)
        
        # t+4
        combined_t4 = self.projections[2](combined_t3) + self._embed_lookup(next_token_ids[2])
        logits_t4 = self.heads[3](combined_t4)
        
        return logits_t1, logits_t2, logits_t3, logits_t4


class ByselModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for l in range(config.n_layers):
            is_global = (l + 1) % 4 == 0
            self.layers.append(ByselDecoderLayer(
                config.d_model, 
                config.n_heads, 
                config.expert_hidden, 
                config.num_experts, 
                is_global=is_global
            ))
        
        self.m_residuals = nn.ModuleList([
            ManifoldConstrainedAttnRes(config.d_model) 
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.d_model)
        self.mtp_pipeline = ByselMTP4Pipeline(config)
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.use_gradient_checkpointing = False

    def forward(self, x, next_token_ids=None):
        nvtx_range_push("ByselModel_Forward")
        
        prev_outputs = []
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            m_res = self.m_residuals[i]
            
            # 🎯 БЕЗОПАСНЫЙ ЧЕКПОИНТИНГ: Оборачиваем только чистую функцию декодера.
            # mAR связи и добавление в списки выполняются снаружи во избежание изменения состояния при повторном запуске.
            if self.training and self.use_gradient_checkpointing and x.device.type == "cuda":
                attn_out, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, x,
                    use_reentrant=False
                )
            else:
                attn_out, aux_loss = layer(x)
            
            total_aux_loss += aux_loss
            prev_outputs.append(attn_out)
            x = m_res(attn_out, prev_outputs)
            
        hidden_states = self.final_norm(x)
        mtp_outputs = self.mtp_pipeline(hidden_states, next_token_ids)
        
        nvtx_range_pop()
        return mtp_outputs, total_aux_loss