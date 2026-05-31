"""
⚙️ ОПТИМИЗИРОВАННЫЙ BYSEL LOSS ENGINE v3.6
Поддерживает вычисления в низком разрешении (bfloat16) и Liger Kernel для CUDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Импорт эффективного Fused Cross Entropy из Liger Kernel для Linux/CUDA
    from liger_kernel.transformers.functional import liger_cross_entropy
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


class ByselLossEngine:
    def __init__(self, vocab_size=259):
        self.vocab_size = vocab_size

    def compute_pretrain_loss(self, logits, targets, mtp_logits_list=None, mtp_targets_list=None):
        """
        Вычисление основного лосса в низком разрешении (bfloat16 / float16).
        """
        # Гарантируем, что таргеты находятся на одном устройстве с логитами (MPS/CUDA)
        targets_device = targets.to(logits.device).long()
        
        if HAS_LIGER and logits.device.type == "cuda":
            # Быстрое ядро Liger без лишних аллокаций на GPU
            loss = liger_cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets_device.reshape(-1)
            )
        else:
            # Нативный fallback для Mac (MPS) и CPU в исходной точности
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets_device.reshape(-1)
            )
        return loss

    def compute_sft_loss(self, logits, targets, thought_mask):
        masked_targets = targets.clone()
        masked_targets[thought_mask == 0] = -100
        
        mask = masked_targets != -100
        return F.cross_entropy(
            logits[mask].reshape(-1, self.vocab_size),
            masked_targets[mask].reshape(-1)
        )

    def compute_kto_loss(self, policy_logps, reference_logps, labels, beta=0.1, kl_weight=0.1):
        log_ratios = policy_logps - reference_logps
        kl = torch.clamp(log_ratios, min=0.0).mean()
        
        losses = []
        for log_ratio, label in zip(log_ratios, labels):
            if label == 1:
                losses.append(-F.logsigmoid(beta * (log_ratio - kl)))
            else:
                losses.append(-F.logsigmoid(beta * (kl - log_ratio)))
        
        kto_loss = torch.stack(losses).mean() + kl_weight * kl
        return kto_loss