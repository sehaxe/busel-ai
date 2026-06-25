---
title: "Config (buselConfig) & profiles"
description: "The buselConfig dataclass, the 12 profiles (chizh-8m, verabey-27m, sokal-60m, kruk-120m, busel-200m, micro_test, quick_test, validation), per-profile hyperparameters, and the validator."
sidebar:
  order: 7
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

`buselConfig` is the single source of truth for every hyperparameter. It's a dataclass with a `__post_init__` validator, loaded from `configs/default.yaml` by profile name.

## The dataclass

```python
# busel_config.py
@dataclass
class buselConfig:
    # Profile
    profile: str = "default"
    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_hyper: int = 4
    d_ff: int = 2048
    n_layers: int = 24
    vocab_size: int = 277
    patch_stride: int = 4
    use_moe: bool = True
    n_shared: int = 2
    n_routed: int = 4
    top_k: int = 1
    d_c: int = 128
    # Default-ON research features
    use_sct: bool = True                 # SCT rank-8
    sct_rank: int = 8
    use_dropbp: bool = True              # DropBP
    dropbp_prob: float = 0.3
    use_rho_loss: bool = True            # RHO-Loss
    use_dispersion_loss: bool = True     # Dispersion Loss
    use_progressive_freeze: bool = True  # Progressive Freeze
    use_ascii_curriculum: bool = True    # ASCII Curriculum
    selective_backward: bool = True      # LCSB
    backward_ratio: float = 0.5
    use_fused_bitlinear: bool = True     # Fused BitLinear Triton
    # Training
    micro_batch_size: int = 16
    grad_accum: int = 1
    ctx_len: int = 4096
    ctx_warmup: list[int] = field(default_factory=lambda: [1024, 2048, 4096])
    ctx_warmup_steps: int = 2000
    max_steps: int | str = "auto"          # "auto" → Chinchilla solve
    warmup_steps: int = 100
    lr: float = 0.002
    muon_lr: float = 0.02
    weight_decay: float = 0.01
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 0.001
    # Hardware
    compile_mode: str = "default"          # default | reduce-overhead | max-autotune
    device: str = "auto"                   # auto | cuda | mps | cpu
    dtype: str = "auto"                    # auto | bf16 | fp16 | fp32
    # Logging
    save_every: int = 1000
    eval_every: int = 200
    log_every: int = 10
    keep_checkpoints: int = 5
    # AutoPilot
    autopilot: str = "v6"
    agc_threshold: float = 0.01
    spike_sigma: float = 3.0
    # Architecture choices
    attention_type: str = "gdn2"
    optimizer_type: str = "hybrid_muon_adamw"
    # Paths
    data_dir: str = "data_train"
    output_dir: str = "checkpoints"
    # ... and more (~80 fields total)
```

The full schema is in [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py). The above is the highlights.

## The 12 profiles

| Profile | N (params) | Purpose | Hardware |
|---|---|---|---|
| `validation` | 2M | Pipeline smoke test | Any |
| `micro_test` | 2M | CI / unit tests | Any |
| `quick_test` | 3M | Smoke test, verify wiring | CPU OK |
| `chizh-8m` | 4M | Small-scale real training | Any GPU |
| `verabey-27m` | 70M | Research, single 16GB GPU | RTX 5060 Ti / M2 Pro |
| `sokal-60m` | 170M | Mid-scale experiments | RTX 4090 / M3 Max |
| `kruk-120m` | 350M | Mid-scale pretraining | A100 / H100 |
| `busel-200m` | 1B | Large-scale research | Multi-A100 |

The naming comes from birds and Belarusian folklore:
- **бусел** (busel) = stork
- **чиж** (chizh) = Eurasian siskin (small finch)
- **верабей** (verabey) = sparrow
- **сокал** (sokal) = falcon
- **крук** (kruk) = raven
- **бусел-1b** = stork at 1B scale

### `micro_test`

```yaml
micro_test:
  d_model: 128
  n_layers: 4
  ctx_len: 256
  micro_batch_size: 4
  use_moe: false
  max_steps: 50
  save_every: 50
  eval_every: 25
```

**Use for:** unit tests, CI pipelines, verifying the pipeline works end-to-end. Trains in ~30 seconds on CPU.

### `quick_test`

```yaml
quick_test:
  d_model: 256
  n_layers: 8
  ctx_len: 512
  micro_batch_size: 8
  use_moe: true
  n_routed: 2
  max_steps: 500
  save_every: 100
```

**Use for:** smoke testing new code paths, debugging, verifying checkpointing/resume. Trains in ~5 minutes on CPU.

### `verabey-27m` (default for most users)

```yaml
verabey-27m:
  d_model: 512
  n_heads: 8
  n_hyper: 4
  d_ff: 2048
  n_layers: 24
  micro_batch_size: 16
  ctx_len: 4096
  ctx_warmup: [512, 1024, 2048, 4096]
  ctx_warmup_steps: 2000
  use_moe: true
  n_shared: 2
  n_routed: 4
  top_k: 1
  aux_loss_coeff: 0.01
  z_loss_coeff: 0.001
  lr: 0.002
  muon_lr: 0.02
  weight_decay: 0.01
  max_steps: 25000
  save_every: 2500
  eval_every: 500
  # Default-ON: SCT, DropBP, RHO-Loss, Dispersion, ProgFreeze,
  # ASCII Curriculum, LCSB, Fused BitLinear
```

**Stats:**
- ~70M params (~14 MB checkpoint)
- ~524k tokens/step
- ~0.5s/step on RTX 5060 Ti, ~1.8s on M2 Pro
- Fits in 16GB GPU at `compile-mode=default`

**Use for:** research, blog posts, paper experiments, the "I want a real model" use case.

### `sokal-60m`

```yaml
sokal-60m:
  d_model: 768
  n_heads: 12
  n_hyper: 6
  d_ff: 3072
  n_layers: 28
  micro_batch_size: 8
  ctx_len: 8192
  ctx_warmup: [1024, 2048, 4096, 8192]
  ctx_warmup_steps: 3000
  use_moe: true
  n_shared: 2
  n_routed: 8
  max_steps: 40000
  save_every: 4000
```

**Stats:**
- ~170M params (~35 MB checkpoint)
- ~1.0M tokens/step
- ~1.5s/step on RTX 5060 Ti, ~5s on M3 Max
- Requires 24GB GPU for `compile-mode=default`

**Use for:** mid-scale experiments, long-context training.

### `kruk-120m`

```yaml
kruk-120m:
  d_model: 1024
  n_heads: 16
  n_hyper: 8
  d_ff: 4096
  n_layers: 32
  micro_batch_size: 4
  grad_accum: 4
  ctx_len: 8192
  ctx_warmup: [1024, 2048, 4096, 8192]
  ctx_warmup_steps: 4000
  use_moe: true
  n_shared: 2
  n_routed: 16
  max_steps: 60000
  save_every: 5000
```

**Stats:**
- ~350M params (~70 MB checkpoint)
- ~33M tokens/step (effective)
- ~3.5s/step on A100
- Requires 40GB+ GPU

**Use for:** mid-scale pretraining, scaling law validation.

### `busel-200m`

```yaml
busel-200m:
  d_model: 1536
  n_heads: 24
  n_hyper: 12
  d_ff: 6144
  n_layers: 40
  micro_batch_size: 2
  grad_accum: 8
  ctx_len: 32768
  ctx_warmup: [2048, 4096, 8192, 16384, 32768]
  ctx_warmup_steps: 5000
  use_moe: true
  n_shared: 2
  n_routed: 24
  max_steps: 120000
  save_every: 10000
```

**Stats:**
- ~1B params (~200 MB checkpoint)
- ~8.4M tokens/step
- Multi-GPU required (80GB+)
- Chinchilla-optimal for ~37B training tokens.

## Loading a profile

```python
from busel_config import buselConfig

config = buselConfig.from_profile("shpak")
config = buselConfig.from_profile("zubr", overrides={"ctx_len": 8192})
config = buselConfig.from_yaml("configs/default.yaml", profile="chyzh")
```

CLI:

```bash
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --ctx-len 2048 --lr 0.001
```

CLI args override YAML values; YAML values override dataclass defaults.

## The validator (`__post_init__`)

```python
def __post_init__(self):
    # d_model must be divisible by n_heads (for attention)
    assert self.d_model % self.n_heads == 0, \
        f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
    # d_model must be divisible by n_hyper (for mAR)
    assert self.d_model % self.n_hyper == 0, \
        f"d_model ({self.d_model}) must be divisible by n_hyper ({self.n_hyper})"
    # vocab_size is FIXED
    assert self.vocab_size == 277, \
        f"vocab_size must be 277 (byte-level), got {self.vocab_size}"
    # MoE constraints
    if self.use_moe:
        assert self.n_routed >= 2, "MoE needs at least 2 routed experts"
        assert self.n_shared >= 1, "MoE needs at least 1 shared expert"
    # Compile mode
    assert self.compile_mode in ("default", "reduce-overhead", "max-autotune", "off")
    # Device
    assert self.device in ("auto", "cuda", "mps", "cpu")
    # ctx_warmup must end at ctx_len
    if self.ctx_warmup and self.ctx_warmup[-1] != self.ctx_len:
        self.ctx_warmup = self.ctx_warmup + [self.ctx_len]
```

The validator catches all the "obvious typos" that would otherwise crash mid-training with cryptic errors.

## `effective_max_steps`

The `max_steps: "auto"` field is a string in the dataclass; it's resolved to an int at the first access:

```python
@property
def effective_max_steps(self) -> int:
    if self.max_steps == "auto":
        return self._chinchilla_solve()
    return int(self.max_steps)
```

`_chinchilla_solve()` computes the Chinchilla-optimal step count from the non-embedding param count and the per-step tokens. See [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) for the math.

## Default-ON features (no opt-in required)

All features below are wired ON by default in `buselPretrainConfig`:

| Flag | Default | Effect |
|---|---|---|
| `use_sct` | `True` | SCT rank-8 — FFN weight compression ×4-8 |
| `sct_rank` | `8` | SCT rank |
| `use_dropbp` | `True` | DropBP — 30% layers skip backward |
| `dropbp_prob` | `0.3` | DropBP probability |
| `use_rho_loss` | `True` | RHO-Loss — gradient only for hard tokens |
| `use_dispersion_loss` | `True` | Dispersion Loss — prevents embedding condensation |
| `use_progressive_freeze` | `True` | Progressive Freeze — freeze up to 75% layers |
| `use_ascii_curriculum` | `True` | ASCII Curriculum — 7-bit first 30% of training |
| `selective_backward` | `True` | LCSB — selective per-layer backward |
| `backward_ratio` | `0.5` | LCSB ratio — −44% step, −25% mem |
| `use_fused_bitlinear` | `True` | Fused BitLinear Triton kernel (eager mode) |
| `use_ema` | `True` | EMA of weights via Schedule-Free interpolation |

## How to add a new profile

1. Edit `configs/default.yaml`:

```yaml
my_profile:
  d_model: 640
  n_heads: 10
  n_layers: 20
  # ... override only the fields you want to change
```

2. Use it:

```bash
uv run train.py --profile my_profile
```

That's it. The base dataclass defaults are inherited, and the YAML overrides only the fields you specify.

## How to override on the CLI

Every field is a CLI flag. Dashes replace underscores:

```bash
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m \
                --d-model 768 \
                --n-layers 30 \
                --ctx-len 2048 \
                --max-steps 20000 \
                --lr 0.001 \
                --compile-mode reduce-overhead
```

Run `uv run python cli.py pipeline --help` for the full list.

## Common patterns

### Override per-GPU

```bash
# 8GB GPU: smaller batch, smaller ctx
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --micro-batch-size 4 --ctx-len 2048

# 24GB GPU: larger batch, larger ctx
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --micro-batch-size 32 --ctx-len 8192
```

### Continue a run with a smaller LR (anneal)

```bash
# Original run
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --max-steps 20000 --lr 0.002
# Anneal: lower LR for last 20%
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --resume checkpoints/ckpt_16000.pt --max-steps 20000 --lr 0.0002
```

### Reproduce a paper experiment

```bash
# Save the exact config
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --save-config my_run.yaml
# Reproduce later
uv run python cli.py pipeline --name pretrain-only --config my_run.yaml
```

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `buselConfig` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The dataclass |
| `from_profile()` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | Profile loader |
| `from_yaml()` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | YAML loader |
| `effective_max_steps` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The `auto` resolver |
| `__post_init__` | [busel_config.py](file:///home/sehaxe/busel-ai/busel_config.py) | The validator |
| `configs/default.yaml` | [configs/default.yaml](file:///home/sehaxe/busel-ai/configs/default.yaml) | All 6 profiles |
| `test_config_validator` | [tests/test_config.py](file:///home/sehaxe/busel-ai/tests/test_config.py) | Compliance: d_model % n_heads == 0 |
| `test_chinchilla_solve` | [tests/test_config.py](file:///home/sehaxe/busel-ai/tests/test_config.py) | Compliance: 11M → 11_718 steps |

## See also

- [Profiles reference](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/profiles.md) — the user-facing guide
- [Curriculum](file:///home/sehaxe/busel-ai/site/src/content/docs/training/curriculum.md) — Chinchilla solver
- [Quick tour](file:///home/sehaxe/busel-ai/site/src/content/docs/guides/quick-tour.md) — the "first run" experience
