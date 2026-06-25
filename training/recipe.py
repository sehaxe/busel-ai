"""
⚙️ busel (MTP-4 STABILIZED + SALT KD)
Вычисляет многоголовый причинный лосс MTP-4 с затухающим взвешиванием,
а также Knowledge Distillation loss для SALT (Small model Aided Large model Training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from liger_kernel.transformers.functional import liger_cross_entropy
    from liger_kernel.transformers import LigerCrossEntropyLoss
    HAS_LIGER = True
    _LIGER_CE = LigerCrossEntropyLoss(return_z_loss=True)
except ImportError:
    HAS_LIGER = False
    _LIGER_CE = None


def validate_training_schedule(max_steps, warmup_steps):
    """Runtime guard for the training schedule (ISSUES.md #7).

    Rejects `max_steps <= warmup_steps` (would cause NaN spikes in
    `autopilot.update_parameters` when computing `progress / 0` or `progress > 1.0`).
    Also rejects `warmup_steps < 1` (autopilot needs the first 50 steps to
    be free of predictive dampening).

    Returns the validated `(max_steps, warmup_steps)` as a tuple of ints, or
    raises `ValueError` with a helpful message.
    """
    if max_steps is None or warmup_steps is None:
        raise ValueError(
            f"max_steps ({max_steps}) and warmup_steps ({warmup_steps}) must be "
            f"set to integers before calling validate_training_schedule"
        )
    max_steps = int(max_steps)
    warmup_steps = int(warmup_steps)
    if max_steps <= warmup_steps:
        raise ValueError(
            f"max_steps ({max_steps}) must be strictly greater than "
            f"warmup_steps ({warmup_steps}). Either raise max_steps or "
            f"lower warmup_steps in configs/default.yaml."
        )
    if warmup_steps < 1:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be >= 1. The first 50 "
            f"steps also need to be free of predictive dampening (autopilot.py)."
        )
    return max_steps, warmup_steps


class buselLossEngine:
    def __init__(self, vocab_size=259):
        self.vocab_size = vocab_size

    def compute_pretrain_loss(self, logits, targets, mtp_logits_list=None, mtp_targets_list=None):
        """
        Compute main loss for head t+1 (weight 1.0) + weighted MTP-N losses.
        Pass logits=None to skip T1 and only compute MTP losses.
        """
        loss = torch.tensor(0.0, device=(mtp_logits_list[0].device if mtp_logits_list else torch.device('cpu')))
        
        if logits is not None:
            targets_device = targets.to(logits.device).long()
            if _LIGER_CE is not None and logits.device.type == "cuda":
                ce_out = _LIGER_CE(logits.reshape(-1, self.vocab_size), targets_device.reshape(-1))
                loss = loss + ce_out.loss + 0.001 * ce_out.z_loss
            elif HAS_LIGER and logits.device.type == "cuda":
                loss = loss + liger_cross_entropy(logits.reshape(-1, self.vocab_size), targets_device.reshape(-1))
            else:
                loss = loss + F.cross_entropy(logits.reshape(-1, self.vocab_size), targets_device.reshape(-1))
            
        # 2. 🎯 РАСЧЕТ И ВЗВЕШИВАНИЕ ПОТЕРЬ MTP-N:
        # Если переданы логиты и таргеты для дополнительных голов предсказания будущего
        # (t+2, t+3, ..., t+N), мы рассчитываем лосс для каждой головы и суммируем
        # их с затухающим геометрическим коэффициентом 0.5^(i+1).
        if mtp_logits_list is not None and mtp_targets_list is not None:
            for i, (m_logits, m_targets) in enumerate(zip(mtp_logits_list, mtp_targets_list)):
                if m_logits is None or m_targets is None:
                    continue

                m_targets_device = m_targets.to(m_logits.device).long()

                if _LIGER_CE is not None and m_logits.device.type == "cuda":
                    ce_out = _LIGER_CE(m_logits.reshape(-1, self.vocab_size), m_targets_device.reshape(-1))
                    m_loss = ce_out.loss
                elif HAS_LIGER and m_logits.device.type == "cuda":
                    m_loss = liger_cross_entropy(m_logits.reshape(-1, self.vocab_size), m_targets_device.reshape(-1))
                else:
                    m_loss = F.cross_entropy(m_logits.reshape(-1, self.vocab_size), m_targets_device.reshape(-1))

                # Геометрически затухающий вес: T2=0.5, T3=0.25, T4=0.125, ...
                weight = 0.5 ** (i + 1)
                loss = loss + m_loss * weight
        return loss

    def compute_sft_loss(self, logits, targets, thought_mask):
        masked_targets = targets.clone()
        masked_targets[thought_mask == 0] = -100
        
        mask = masked_targets != -100
        return F.cross_entropy(
            logits[mask].reshape(-1, self.vocab_size),
            masked_targets[mask].reshape(-1)
        )

    @staticmethod
    def compute_dpo_loss(
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float = 0.1,
    ) -> torch.Tensor:
        """Direct Preference Optimization loss (Rafailov et al. 2023).

        L_DPO = -E[log_sigmoid(β · (log π_θ(y_w|x)/π_θ(y_l|x)
                                   - log π_ref(y_w|x)/π_ref(y_l|x)))]

        Args:
            policy_chosen_logps: (B,) sum of log P(chosen | prompt) under policy
            policy_rejected_logps: (B,) sum of log P(rejected | prompt) under policy
            reference_chosen_logps: (B,) sum under frozen reference (SFT model)
            reference_rejected_logps: (B,) sum under frozen reference
            beta: KL penalty coefficient. Default 0.1 (Rafailov paper).

        Returns:
            Scalar DPO loss.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = beta * (pi_logratios - ref_logratios)
        return -F.logsigmoid(logits).mean()

    @staticmethod
    def compute_sequence_logprob(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sum log P(target[t] | logits[t]) over positions where mask=1.

        Args:
            logits: (B, T, V) output of the model head.
            targets: (B, T) integer target token IDs.
            mask: (B, T) float/int 0/1; only mask=1 positions contribute.

        Returns:
            (B,) tensor of per-sequence log-probability sums.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1).long()).squeeze(-1)
        return (target_log_probs * mask.float()).sum(dim=-1)

    @staticmethod
    def compute_dispersion_loss(
        embedding: torch.Tensor,
        weight: float = 0.1,
        temperature: float = 2.0,
        sample_size: int = 4096,
    ) -> torch.Tensor:
        """Uniformity loss (Wang & Isola 2020) on L2-normalised embeddings.

        Counter the token-embedding condensation that hurts small LMs
        (Wang et al. 2026, arXiv:2602.00217 — +1.17 % avg on 10 benchmarks,
        +3.3 % over baseline).  L = weight · log E[exp(-t·‖z_i−z_j‖²)] over
        a `sample_size` random subset of embeddings.  Backprop drives the
        bytes apart on the unit hypersphere.
        """
        e = embedding.reshape(-1, embedding.size(-1))
        n = min(sample_size, e.size(0))
        idx = torch.randperm(e.size(0), device=e.device)[:n]
        z = F.normalize(e[idx], dim=-1)
        sq_dists = torch.cdist(z, z, p=2).pow(2)
        return weight * torch.log(torch.exp(-temperature * sq_dists).mean() + 1e-8)
