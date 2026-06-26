# training/ — Optimizer, AutoPilot, Loss, **Stage Framework**

**Scope:** SF-NorLotusMuon + FP8 AdamW hybrid optimizer (v8.5 cleanup: single path, no dead branches), predictive AutoPilot v6.0, MTP-4 weighted loss engine, multi-stage pipeline framework (pretrain → SFT → DPO → eval).

## STRUCTURE
```
training/
├── optimizer.py    # SF-NorLotusMuon + FP8 AdamW (213 lines, single clean path)
├── autopilot.py    # buselAutoPilot v6.0 — predictive dampening, adaptive AGC, dynamic WD, sub-group LR push
├── recipe.py       # buselLossEngine — pretrain, SFT, KTO, DPO losses
└── stages/         # multi-stage pipeline framework
    ├── __init__.py  # Public API exports; eager-imports all 4 stage modules
    ├── base.py      # BaseStage Protocol, StageState/StageSpec/PipelineConfig, register_stage, load_pipeline_yaml
    ├── pretrain.py  # buselPretrainStage — pretrain stage. buselPretrainConfig lives here.
    ├── sft.py       # buselSFTStage (chat-format SFT with masked CE)
    ├── dpo.py       # buselDPOStage (Rafailov et al. 2023 DPO)
    └── eval.py      # buselEvalStage (4-metric eval suite)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Change optimizer | `optimizer.py` | SF-NorLotusMuon (2D params, excl. `router`/`embed`) + FP8 AdamW (rest) |
| Tune LR schedule | `autopilot.py` → `update_parameters` | Cosine decay w/ warmup, spike recovery (35% LR × 15 steps) |
| Change grad clipping | `autopilot.py` → `before_step` | First 50 steps: max_norm=2.0 free; later: rolling_avg × 1.5 |
| Add loss term | `recipe.py` | `compute_pretrain_loss` already handles MTP-4 weighted sum |
| Change dampening | `autopilot.py` line ~50 | 3σ rule on last 15 grad norms (predictive) |
| Add dispersion loss | `recipe.py` → `compute_dispersion_loss` | Wang 2026 uniformity loss on L2-normalised token embeddings; counters condensation in small LMs. |
| **Add a new stage** | `stages/<name>.py` → `@register_stage("<name>")` class | Auto-discovered; `__init__.py` eager-imports for registration |
| **Define a pipeline** | `configs/pipelines/<name>.yaml` | Read by `tools/orchestrator.py:pipeline()` |
| **Change stage protocol** | `stages/base.py` → `BaseStage` | Has `setup/run/finalize`; `StageState` is shared state across stages |

## KEY CLASSES
| Symbol | Type | Location | Role |
|---|---|---|---|
| `_newton_schulz_core` | function | optimizer.py | Quintic NS iteration, 5 steps, transposed for tall matrices. **v8.5 bugfix:** removed the erroneous `return X * scale` rescaling that inflated singular values. |
| `_compiled_newton_schulz` | function | optimizer.py | `@torch.compile(reduce-overhead)` on Linux+CUDA; eager fallback |
| `_MuonBase` | base class | optimizer.py | **v8.5 refactored.** Base optimizer with momentum + NS orthogonalization + Muon+ column norm. Single `step()` loop. |
| `LotusMuon` | Optimizer | optimizer.py | Rank-`lotus_rank` factorised Muon momentum. Reconstructs `m ≈ buf_p @ buf_q.T` on the fly before NS. **~20× less optimizer state** than full Muon at rank=32. |
| `NorLotusMuon` | Optimizer | optimizer.py | **Default Muon path.** Extends LotusMuon with `cautious_wd` (sign-aware weight decay masking). Always wrapped in `_ScheduleFreeWrapper`. |
| `_ScheduleFreeWrapper` | wrapper | optimizer.py | Wraps any optimizer with SF-SGD (Defazio et al. 2024). Three internal states per param: x (Polyak avg), z (gradient state), y (interpolated forward). Always ON in `buselOptimizerEngine` via `sf_beta=0.9, sf_gamma_factor=2.0`. |
| `buselOptimizerEngine` | class | optimizer.py | Splits params: 2D+!`router`+!`embed`→SF-NorLotusMuon; rest→FP8 AdamW. **6 sub-groups** for **decoupled per-layer LR multipliers**. SF always ON, FP8 AdamW always ON. |
| `buselAutoPilot` | class | autopilot.py | Wraps engine; tracks loss/grad history; recovery countdown; **re-pushes per-subgroup LR multipliers every step** |
| `buselLossEngine` | class | recipe.py | Liger-CE on CUDA; vanilla `F.cross_entropy` elsewhere. MTP weights `[0.5, 0.25, 0.125]`. Also exposes `compute_dispersion_loss` static method. |
| `compute_dispersion_loss` | staticmethod | recipe.py | Uniformity loss (Wang & Isola 2020 / Wang et al. 2026, arXiv:2602.00217) on L2-normalised byte embeddings. |
| `validate_training_schedule` | function | recipe.py | Runtime guard for `max_steps > warmup_steps` and `warmup >= 1` |
| `BaseStage` | Protocol | stages/base.py | Stage lifecycle contract: `setup(cfg)` → `run(state)` → `finalize(state)` |
| `StageState` | dataclass | stages/base.py | Shared mutable state between stages |
| `StageSpec` | dataclass | stages/base.py | One entry in a pipeline YAML |
| `PipelineConfig` | dataclass | stages/base.py | Top-level pipeline: `name`, `stages`, `global_params` |
| `register_stage(name)` | decorator | stages/base.py | Wraps `busel_registry.register("stage", name)` |
| `load_pipeline_yaml(path)` | function | stages/base.py | Validates YAML shape |
| `buselPretrainStage` | class | stages/pretrain.py | Pretrain stage: setup → run → finalize |
| `buselPretrainConfig` | dataclass | stages/pretrain.py | **Single source of truth for default values**. Constructed via `from_profile(profile_dict)`. |
| `EMA` | class | optimizer.py | EMA shadow of weights, `decay=0.999`. Saved alongside model state in checkpoints. Smooths loss + gives 10-15 % fewer steps. |

## CONVENTIONS
- **Param routing rule (SF-NorLotusMuon vs FP8 AdamW):** `param.ndim == 2 and "router" not in name and "embed" not in name` → SF-NorLotusMuon; else → FP8 AdamW.
- **Param sub-group classification (for decoupled per-layer LR):** every param is sorted into one of 6 sub-groups by name pattern (`_classify_param` in `optimizer.py`):
  - `"router"` → `router` group (always FP8 AdamW; LR multiplier 0.5)
  - `"embed"` → `embed` group (always FP8 AdamW; LR multiplier 0.5)
  - `"norm"` → `norm` group (always FP8 AdamW; LR multiplier 1.0)
  - `"mtp"` → `mtp` group (FP8 AdamW; LR multiplier 1.0)
  - `"ffn"` / `"blackboard"` / `"moe"` → `ffn` group (mostly SF-NorLotusMuon; LR multiplier 1.0)
  - Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `qkv`, `wk`, `wv`, `wq`) → `attn` group (SF-NorLotusMuon; LR multiplier 1.0); default fallback
  - LR multipliers: `{attn: 1.0, ffn: 1.0, mtp: 1.0, norm: 1.0, embed: 0.5, router: 0.5}` by default. Override via `buselPretrainConfig.lr_multipliers` or in the profile YAML.
- **LOTUS rank (`lotus_rank`):** default 32. 16 = ~95 % quality, 32 = ~98 %, 64 = ~99.5 %. Affects only SF-NorLotusMuon group.
- **LOTUS LR scale (`lotus_lr_scale`):** default 0.5. LOTUS effective LR = `lr_muon × lotus_lr_scale`. Compensates for the rank-r approximation.
- **Muon momentum:** 0.95; NS steps: 5; weight_decay: dynamic (set by AutoPilot)
- **FP8 AdamW weight_decay:** dynamic (set by AutoPilot on every step from `target_wd × wd_factor` curve). lr_adamw is 10× smaller than lr_muon.
- **Dampening threshold:** `mean(history[:-1]) + 3σ` over last 15 grad norms (predictive)
- **Spike detection:** `current_loss > 1.35 × rolling_avg(loss[:-1])` over 15 steps
- **Recovery:** LR scaled to 35% for 15 steps after spike; noise scale ×1.5
- **LR cosine:** `min_lr_ratio + (1-min_lr_ratio) × 0.5 × (1+cos(π·progress))` after warmup
- **Weight decay curve:** `wd_factor = 0.1` warmup, `0.1 + 0.9·progress` mid, `0.5` last 10%
- **Liger kernel:** Only if `HAS_LIGER` and CUDA; `liger_cross_entropy` for CE/MTP
- **Selective activation checkpointing (`grad_ckpt_every`):** default 2 — recompute every other block during backward. Halves activation memory at <5 % step-time cost. Set 0 to disable, 1 to checkpoint every block.
- **Gram NS package:** Uses `gram_newton_schulz.StandardNewtonSchulz` if available (`ns_use_kernels=False`). Falls back to `_compiled_newton_schulz` (quintic, 5 steps).
- **SF always ON:** `_ScheduleFreeWrapper` always wraps the Muon AND AdamW optimizers in `buselOptimizerEngine`. No opt-out. `sf_beta=0.9`, `sf_gamma_factor=2.0`.
- **FP8 AdamW always ON:** `torchao.optim.AdamWFp8` for all non-Muon params. ~75% memory reduction vs fp32 AdamW. No opt-out.
- **Stage registration:** Put a class with `@register_stage("name")` in `stages/<name>.py`; the `__init__.py` already does imports to trigger registration.
- **Pipeline YAML schema:** Top-level: `name` (string, required), `stages` (list, non-empty, required), `global_params` (dict, optional). Per-stage: `name`, `data_preset`, `resume`, `checkpoint_out`, `params`.
- **Stage state contract:** Stages receive a `StageState` instance on every call. `StageState.artifact` passes checkpoints between stages.

## ANTI-PATTERNS
- **NEVER** apply SF-NorLotusMuon to 1D params (norms, biases) — `buselOptimizerEngine` filters them
- **NEVER** apply SF-NorLotusMuon to anything with `router` or `embed` in name — `_MUON_EXCLUDE = ("router", "embed")`. Routers are policy, not value; embed is categorical.
- **NEVER** disable `use_ema`, decoupled LR (`lr_multipliers`), or selective ckpt (`grad_ckpt_every=2`) without measuring — all three help, all are on by default
- **NEVER** change `momentum=0.95` without testing — Muon spec is brittle
- **NEVER** disable predictive dampening in first 50 steps — gradients must be free then
- **NEVER** set `noise_scale > 0` after progress > 0.90 — final phase is noise-free
- **NEVER** add `@torch.compile` to the whole `step()` — only to inner NS function
- **NEVER** use `F.cross_entropy` on CUDA when Liger is available — 2-3× slower
- **NEVER** save KTO labels as float — must be `0` or `1` (integer label)
- **NEVER** register a stage in a runtime-loaded module without re-triggering `__init__.py` — the registry is populated only at import time
- **NEVER** swallow `KeyError` from `get_stage()` in production — orchestrator treats it as a hard config error
- **NEVER** revert the v8.5 optimizer cleanup (Muon/NorMuon/SOAP/MuonQ/Adafactor/Cautious/QuEST/FlashMuon were all deleted — they're dead branches)
- **NEVER** run `uv run train.py` — `train.py` was deleted in v8.5. Use `cli.py` instead.
- **NEVER** use `return X * scale` in `_newton_schulz_core` — the NS bugfix changed it to `return X`. The scale rescaling blows up singular values.

## NOTES
- **Muon scale formula:** `0.2 * sqrt(max(A, B))` per Muon paper (Keller Jordan)
- **NS coefficients:** `(3.4445, -4.7750, 2.0315)` — optimal for 5-step quintic iteration
- **v8.5 optimizer cleanup:** `optimizer.py` went from 863 lines → 213 lines. Removed: `Muon` (merged into base), `NorMuon` (merged into `NorLotusMuon._apply_weight_decay` as cautious_wd), `LotusMuon` (preserved as base + subclass), `SOAP`, `MuonQ`, `Adafactor`, `Cautious`, `QuEST`, `FlashMuon`. The single path is now `_MuonBase → LotusMuon → NorLotusMuon`, always SF-wrapped.
- **Gram NS as primary:** `gram_newton_schulz.StandardNewtonSchulz` package is the primary NS path (detected at import). Falls back to `_newton_schulz_core` (quintic, 5 steps) if the package is unavailable.
- **SF always ON:** `buselOptimizerEngine` always wraps both optimizers in `_ScheduleFreeWrapper`. No opt-out. For best results set `min_lr_ratio=1.0` in the profile to disable cosine LR decay (cosine interferes with SF's implicit schedule). MLCommons 2024 AlgoPerf self-tuning track winner.
- **`inject_noise`:** Gaussian `noise_scale × grad_norm` per param (only if `grad_norm > 1e-5`)
- **Loss API contract:** `compute_pretrain_loss(self, logits, targets, mtp_logits_list=None, mtp_targets_list=None)` — MTP logits and targets are passed as **separate lists**. T1 has implicit 1.0, T2/T3/T4 use `[0.5, 0.25, 0.125]`.
- **Liger fallback:** `importlib.util.find_spec("liger_kernel")` or try-except — auto-fallback to vanilla
- **LOTUS rank memory (measured):** on Shpak (52.8 M params, 1024×1024 max weight), full Muon momentum = 2 MB; LOTUS rank-32 = 128 KB → **~16× per-param reduction, ~20× total**. Quality ~98% of full Muon.
- **EMA cost (measured):** checkpoint size grows by ~model_size in EMA shadow (fp32). For shpak = ~211 MB per checkpoint. Eval-quality improvement: 10-15 % fewer steps to same loss.
- **Decoupled LR empirical multipliers:** `{attn: 1.0, ffn: 1.0, mtp: 1.0, norm: 1.0, embed: 0.5, router: 0.5}`. The 0.5 on `embed` is critical — full-LR embed updates cause catastrophic forgetting in 1.58-bit. The 0.5 on `router` prevents router collapse.
- **Stage framework phases:** pretrain → SFT → DPO → eval stages all implemented.
- **Registry kind `stage`:** Uses `busel_registry.register("stage", name)`. `get_stage("pretrain")` returns the class.
- **LCSB wiring to `buselPretrainConfig`:** Two fields — `selective_backward`, `backward_ratio`. Default ON in shpak/zubr/chyzh.
- **Dispersion Loss wiring:** Three `buselPretrainConfig` fields — `use_dispersion_loss`, `dispersion_weight`, `dispersion_temperature`. All default OFF. Validation on shpak 52.8M: dispersion alone → −6.3 % loss at +2.5 % step. Cost: +519 MB VRAM.
