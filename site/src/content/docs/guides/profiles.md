---
title: Choosing a profile
description: Which of the bundled profiles (micro_test, quick_test, validation, chizh-8m, verabey-27m, sokal-60m, kruk-120m, busel-200m) is right for you?
---

The `configs/default.yaml` file ships with 12 profiles. They trade off
**training time** vs **model capacity** vs **VRAM cost**. Pick one
based on what you want to do with the run.

## The profiles at a glance

| Profile       | Total params | Bit-size | Context | VRAM (bf16) | What it's for |
|---------------|-------------:|---------:|--------:|------------:|---------------|
| `validation`  | ~2 M         | ~1 MB    | 256 B   | ~0.5 GB     | pipeline smoke test |
| `micro_test`  | ~2 M         | ~1 MB    | 256 B   | ~0.5 GB     | CI / unit tests |
| `quick_test`  | ~3 M         | ~1 MB    | 256 B   | ~0.6 GB     | quick sanity check |
| `chizh-8m`    | ~4 M         | ~1 MB    | 1024 B  | ~1.0 GB     | small-scale real training |
| `verabey-27m` | ~70 M        | 14 MB    | 4096 B  | ~8 GB       | the "real" 70 M run |
| `sokal-60m`  | ~170 M       | 35 MB    | 8192 B  | ~16 GB      | long-context demo |
| `kruk-120m`   | ~350 M       | 70 MB    | 8192 B  | ~24 GB      | mid-scale pretraining |
| `busel-200m`    | ~1 B         | 200 MB   | 32 768 B| ~40 GB      | large-scale research |

## How to pick

### You just want to see *something* train

→ **`validation`**

```bash
uv run python cli.py pipeline --name pretrain-only --profile validation
```

200 steps, ~1 min on a 5060 Ti, loss drops ~10→7, prints a
checkpoint. This is the smoke test for the whole pipeline: byte
loader, model, mAR, MTP, AutoPilot, optimizer, JSON log, checkpointing.

### You want a CI test

→ **`micro_test`** (faster than `quick_test`)

```bash
uv run python cli.py pipeline --name pretrain-only --profile micro_test
```

No real checkpoint produced. Designed to fail fast if any of the
imports or shape math is broken.

### You want a small model that actually does something

→ **`chizh-8m`** (~4 M, ~15 min)

```bash
uv run python cli.py pipeline --name pretrain-only --profile chizh-8m
```

Small enough to fit on a laptop, large enough to start producing
non-trivial perplexity on a small corpus.

### You want the "main" run

→ **`verabey-27m`** (~70 M, ~8 h on a 5060 Ti)

```bash
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m
```

This is the profile the README advertises. Context grows 512→1024
→2048→4096 patches, batch adapts, Chinchilla target ~5.6 B
byte-tokens. End result: a 14 MB checkpoint you can run inference on
with `cli.py infer`.

### You want a long-context demo

→ **`sokal-60m`** (170 M, context 8192)

```bash
uv run python cli.py pipeline --name pretrain-only --profile sokal-60m
```

The MLA latent cache makes 8 K contexts tractable on consumer
hardware. This profile is *expensive* — ~20 h on a 5060 Ti, ~16 GB
VRAM. Use it when you actually need long context.

## Tweak a profile without making a new one

All knobs are CLI flags:

```bash
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m \
    --no-compile            # disable torch.compile
    --no-checkpointing      # disable gradient checkpointing (more VRAM, faster)
    --compile-mode max-autotune   # try harder at compile time
    --resume checkpoints/verabey-27m_step_10000.pt
```

Or edit `configs/default.yaml` directly — the format is plain YAML
and any field can be overridden. The auto-planner picks up your
`max_steps: "auto"` setting and computes the rest from the scaling law.

## Making your own profile

Copy the closest existing profile in `configs/default.yaml` and tweak:

```yaml
my_profile:
  model:
    d_model: 256
    n_layers: 6
    n_heads: 4
    expert_hidden: 512
    num_experts: 4
    top_k: 1
    vocab_size: 277      # DO NOT change this — vocab=277 is byte-level
  data:
    data_path: "data_train"
    chunk_size: 1024
    batch_size: 64
  training:
    max_steps: "auto"    # computed from Chinchilla
    warmup_steps: "auto"
    min_lr_ratio: 0.1
    learning_rate_muon: 0.001
    learning_rate_adamw: 0.0001
    weight_decay: 0.1
    grad_accum_steps: 1
    checkpoint_interval: 500
```

    Then run it with `uv run python cli.py pipeline --name pretrain-only --profile my_profile`.

:::caution
Do not change `vocab_size` from 277. The byte-level patcher is wired
to exactly 256 UTF-8 byte values + 21 multimodal specials. Any other
value will silently misbehave.
:::

## Profile gotchas

- **`d_model % n_heads == 0`** — required for the multi-head layouts
  in `BusbaGDN2SeRoPEBlock` and `MultiHeadLatentAttention`.
- **`d_model % n_hyper == 0`** — required for the mAR `d_head` split
  (`d_head = d_model / n_hyper`).
- **`d_model` × `num_experts`** — budget VRAM. Each routed expert is
  a `d_model → expert_hidden → d_model` BitLinear pair, so memory is
  roughly `2 × d_model × expert_hidden × num_experts × 1.58 bits`.
- **`chunk_size % 4 == 0`** — the byte-to-patch stride is 4.
  The curriculum switches between `chunk_size ∈ {256, 1024, 4096}`.

The validator at the top of `buselConfig.__init__` (in `train.py`)
catches the first two of these and raises a `ValueError` with the
exact numbers, so you find out at startup, not at step 1 000.
