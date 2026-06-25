"""
🎯 busel — GRPO (Group Relative Policy Optimization) RL stage.

Post-SFT reinforcement learning à la DeepSeek-R1.
For each prompt, generates N completions, scores via reward fn,
computes group-relative advantages, and updates via clipped policy gradient.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from busel_logging import log_event
from training.stages.base import (
    StageState,
    _apply_model_profile,
    register_stage,
)


def _build_optimizer(model, lr: float, wd: float):
    """Minimal AdamW for GRPO — avoids Muon/OptimizerEngine dependency."""
    from training.optimizer import buselOptimizerEngine
    from training.stages.pretrain_config import buselPretrainConfig
    cfg = buselPretrainConfig()
    cfg.learning_rate_muon = lr * 10
    cfg.learning_rate_adamw = lr
    cfg.weight_decay = wd
    return buselOptimizerEngine(model, None, lr_muon=lr * 10, lr_adamw=lr, lotus_rank=8)


@torch.no_grad()
def _sample_completions(model, patcher, prompts, max_new: int = 256, temperature: float = 0.8, top_p: float = 0.95):
    """Generate N completions for a batch of prompts via argmax sampling."""
    device = next(model.parameters()).device
    B = prompts.shape[0]
    completions = torch.full((B, max_new), 0, dtype=torch.long, device=device)
    logprobs = torch.zeros(B, device=device)

    x = patcher(prompts)
    for t in range(max_new):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            mtp_logits, _ = model(x, progress=0.0)
            logits = mtp_logits[0][:, -1, :2048]  # last position, all bytes
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        # top-p
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # sample
        idx = torch.multinomial(sorted_probs, 1).squeeze(-1)
        next_token = sorted_idx[torch.arange(B, device=device), idx]
        completions[:, t] = next_token
        logprobs += torch.log(sorted_probs[torch.arange(B, device=device), idx].clamp(min=1e-8))
        # feed next token
        next_byte = next_token.unsqueeze(1).float().unsqueeze(-1)
        # ponytail: grow patches by 1 token — embed next byte and append
        next_patch = patcher(next_byte)
        x = torch.cat([x, next_patch], dim=1)
    return completions, logprobs


def default_reward(completion: torch.Tensor, target: torch.Tensor | None = None) -> float:
    """Default reward: length penalty + format bonus + target match if provided."""
    tokens = completion.tolist()
    score = 0.0
    # length reward (favor concise)
    score += min(1.0, 128.0 / max(1.0, float(len(tokens))))
    # special-token format: prefer completions that use byte space (32) properly
    if 32 in tokens:
        score += 0.1
    # target match
    if target is not None and len(tokens) >= len(target):
        matches = sum(1 for a, b in zip(tokens, target) if a == b)
        score += 2.0 * matches / len(target)
    return score


@register_stage("grpo")
class buselGRPOStage:
    """Group Relative Policy Optimization (DeepSeek-R1 style).

    Lifecycle:
        setup(cfg, profile_name, resume) → loads SFT model + optimizer
        run(state)                      → GRPO training loop
        finalize(state)                 → saves final checkpoint
    """

    name: str = "grpo"

    def __init__(self) -> None:
        self.model: Any = None
        self.patcher: Any = None
        self.opt: Any = None
        self.device: str = "cuda"
        self.profile_name: str = "busel-200m"

    def setup(self, profile: dict | str, profile_name: str = "busel-200m", *, resume: str | None = None, **kwargs):
        if not torch.cuda.is_available():
            raise RuntimeError("GRPO requires CUDA")
        self.device = "cuda"
        self.profile_name = profile_name

        if isinstance(profile, str):
            with open("configs/default.yaml", encoding="utf-8") as f:
                full = yaml.safe_load(f)
            profile_dict = full["profiles"][profile]
        else:
            profile_dict = profile

        from training.stages.pretrain_config import buselPretrainConfig
        self.cfg = buselPretrainConfig.from_profile(profile_dict)

        from model.backbone import buselModel
        from model.patching import StridedFastBLTPatcher
        from model.layers import configure_bitlinear

        configure_bitlinear(use_fused_training=True)
        self.patcher = StridedFastBLTPatcher(d_model=self.cfg.d_model).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)
        self.model.train()

        if resume and os.path.exists(resume):
            checkpoint = torch.load(resume, map_location=self.device)
            from model.checkpoint import load_state_dict_safely
            load_state_dict_safely(self.model, checkpoint["model_state_dict"])
            load_state_dict_safely(self.patcher, checkpoint["patcher_state_dict"])
            log_event("grpo_resumed", checkpoint=resume)

        self.opt = _build_optimizer(self.model, lr=1e-7, wd=0.01)

        stage_params = kwargs.get("stage_params", {}) or {}
        self._n_completions = int(stage_params.get("n_completions", 4))
        self._max_new = int(stage_params.get("max_new_tokens", 256))
        self._temperature = float(stage_params.get("temperature", 0.8))
        self._top_p = float(stage_params.get("top_p", 0.95))
        self._clip_eps = float(stage_params.get("grpo_clip", 0.2))
        self._kl_coef = float(stage_params.get("kl_coef", 0.01))
        self._max_steps = int(stage_params.get("max_steps", 500))
        self._data_glob = stage_params.get("data_glob", "data_train/sft/**/*.jsonl")
        self._reward_fn = stage_params.get("reward_fn", default_reward)

    def _load_prompts(self):
        import glob
        prompts = []
        for path in sorted(glob.glob(self._data_glob, recursive=True)):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        text = data.get("text", data.get("messages", ""))
                        if isinstance(text, list):
                            text = "\n".join(m.get("content", "") for m in text if m.get("role") == "user")
                        if text:
                            # encode to bytes
                            prompts.append(torch.tensor(list(text.encode("utf-8")), dtype=torch.long))
                            if len(prompts) >= 1000:
                                break
            except Exception:
                pass
            if len(prompts) >= 1000:
                break
        return prompts

    def run(self, state: StageState) -> StageState:
        prompts = self._load_prompts()
        if not prompts:
            log_event("grpo_no_data", profile=self.profile_name)
            return state

        print(f"🎯 GRPO: {len(prompts)} prompts, {self._n_completions} completions/prompt, {self._max_steps} steps")
        log_event("grpo_start", prompts=len(prompts), n_completions=self._n_completions)

        for step in range(self._max_steps):
            idx = step % len(prompts)
            prompt = prompts[idx].unsqueeze(0).to(self.device)

            # keep reference model for KL
            with torch.no_grad():
                ref_model = copy.deepcopy(self.model)
                ref_model.eval()

            # generate N completions
            completions = []
            logprobs_old = []
            for n in range(self._n_completions):
                with torch.no_grad():
                    comp, lp = _sample_completions(ref_model, self.patcher, prompt,
                                                   max_new=self._max_new,
                                                   temperature=self._temperature,
                                                   top_p=self._top_p)
                completions.append(comp)
                logprobs_old.append(lp)

            # compute rewards
            rewards = []
            for comp in completions:
                r = self._reward_fn(comp.squeeze(0))
                rewards.append(r)
            rewards = torch.tensor(rewards, device=self.device)

            # group-relative advantages: (r - mean(r)) / std(r)
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # policy gradient update per completion
            total_loss = 0.0
            for n in range(self._n_completions):
                comp = completions[n]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    # forward through model to get new logprobs
                    patches = self.patcher(comp)
                    logits, _ = self.model(patches, progress=float(step) / self._max_steps)
                    logp = F.log_softmax(logits[0].float(), dim=-1)
                    # gather logprobs of generated tokens
                    new_lp = logp.gather(-1, comp.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
                # KL penalty
                kl = F.kl_div(F.log_softmax(logits[0].float().view(-1, logits[0].shape[-1]), dim=-1),
                              F.softmax(logits[0].detach().float().view(-1, logits[0].shape[-1]), dim=-1),
                              reduction="batchmean")
                ratio = torch.exp(new_lp - logprobs_old[n])
                clipped = torch.clamp(ratio, 1 - self._clip_eps, 1 + self._clip_eps)
                pg_loss = -torch.min(ratio * adv[n], clipped * adv[n]).mean()
                loss = pg_loss + self._kl_coef * kl
                (loss / self._n_completions).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

            if step % 10 == 0:
                print(f"  GRPO step {step}/{self._max_steps} | reward {rewards.mean().item():.3f} ± {rewards.std().item():.3f} | adv {adv.abs().mean().item():.3f}")

            state.step = step

        log_event("grpo_complete", steps=self._max_steps)
        return state

    def finalize(self, state: StageState) -> None:
        os.makedirs("checkpoints", exist_ok=True)
        from model.checkpoint import strip_compile_prefix
        torch.save(
            {
                "model_state_dict": strip_compile_prefix(self.model.state_dict()),
                "patcher_state_dict": strip_compile_prefix(self.patcher.state_dict()),
                "step": state.step,
                "profile_name": self.profile_name,
            },
            "checkpoints/busel_grpo_final.pt",
        )
        log_event("grpo_saved", path="checkpoints/busel_grpo_final.pt")
