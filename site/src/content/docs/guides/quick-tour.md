---
title: Quick tour
description: A 5-minute walkthrough of what Busel is and what it can do.
---

This page is the elevator pitch for someone who has 5 minutes and
already knows what an LLM is. For the full install-and-run, see
[Installation & quick start](/busel-ai/guides/getting-started/).

## What is Busel?

A from-scratch **1.58-bit LLM** written in ~3 000 lines of Python +
~140 lines of Rust, with a custom architecture that combines:

- **BitNet v2** 1.58-bit weights and H-BitLinear output projection
- **Fused BitLinear** Triton kernel — replaces 10+ ops in one launch (eager mode)
- **mAR** — input-dependent, doubly-stochastic layer-mixing residuals
  (Busel's own combination of Kimi AttnRes + DeepSeek mHC)
- **3:1 GDN-2 / MLA / NSA** — linear attention (75%), latent-KV
  attention (25%), Native Sparse Attention for dynamic sparsity
- **SCT rank-8** — Spectral Compact Training, FFN compression ×4-8
- **MatMul-free FFN** — ternary weights replace all multiplications
- **MoE with Blackboard Memory** — 2 always-on shared + N routed experts
- **MTP-4** — 4 parallel heads predicting `t+1` … `t+4`
- **SF-NorLotusMuon + FP8 AdamW** — Schedule-Free + NorMuon + LOTUS
  rank-8 with Gram Newton-Schulz orthogonalization, FP8 AdamW for
  1D params, AutoPilot v6.0 to glue them together
- **Byte-level tokens** — vocab=277, no BPE, ByteFlow patcher
- **GRPO stage** — Group Relative Policy Optimization for RL-based fine-tuning
- **Curriculum + scaling-law auto-planner** — `D ≈ 37 × N` byte-tokens (small models)

Everything in one repo, with a Typer CLI, an Astro+Starlight wiki, and
a Teto Vocaloid emoticon in the training log.

## Who is it for?

- **Researchers** who want to study 1-bit LLM dynamics, mAR residuals,
  or the GDN-2/MLA interaction without the 200 k-LoC of a
  reference implementation.
- **Hobbyists** with a single RTX 5060 Ti or a MacBook who want to
  actually finish a from-scratch pretrain run.
- **Engineers** evaluating 1-bit inference for a constrained device
  (CPU, edge, mobile) where the 11 MB Shpak checkpoint and pure-add
  forward pass are a real win.

## What is it *not*?

- Not a state-of-the-art base model. The architecture is the
  interesting part; the absolute quality is bounded by the parameter
  count and data (Shpak ≈ 50 M).
- Not a commercial product. **CC BY-NC-SA 4.0.** No commercial use
  without written permission.

## The numbers

All numbers are from a single RTX 5060 Ti (16 GB, sm_120, PyTorch 2.12
+ CUDA 13.0). The validation profile is a 2 M-param toy used to
exercise the full pipeline; verabey-27m is the "real" 70 M-param
training target.

| Profile      | Total params | Bit-size | Context | Planned tokens |
|--------------|-------------:|---------:|--------:|---------------:|
| chizh-8m     | ~4 M         | ~1 MB    | 1024 B  | ~0.15 B        |
| verabey-27m  | ~70 M        | 14 MB    | 4096 B  | **~2.6 B**     |
| sokal-60m   | ~170 M       | 35 MB    | 8192 B  | ~6.3 B         |
| kruk-120m    | ~350 M       | 70 MB    | 8192 B  | ~13 B          |
| busel-200m     | ~1 B         | 200 MB   | 32 768 B| ~37 B          |

### Inference cost (CPU, ternary matmul via Rust)

- verabey-27m forward: 100+ tok/s on a modern laptop CPU.
- Memory: 14 MB for weights, plus KV cache (~98 MB at 128 K ctx for MLA).

### Training cost (RTX 5060 Ti, validation profile)

| Mode (compile)                | tok/s     | vs eager |
|-------------------------------|----------:|---------:|
| Eager (Fused BitLinear Triton)| 215 000   | 1.00×    |
| `torch.compile` (default)     | **578 255** | **2.7×** |
| End-to-end training (200 steps) | 33 575 avg | — |

The 2.7× compile speedup applies to the raw forward/backward/step. The
end-to-end number includes per-step Python overhead (optimizer, JSON
logging every 10 steps, dataloader handoffs) which is the bottleneck
on small profiles. On verabey-27m the compute fraction dominates and you
get back close to the bench number.

See [Performance → torch.compile modes](/busel-ai/performance/compile-modes/)
for the full guide.

## What's in the repo?

The whole project fits in one screen of `tree -L 2`:

```
busel-ai/
├── model/              # BitLinear, mAR, attention mix, MoE, MTP
├── training/           # SF-NorLotusMuon + FP8 AdamW, AutoPilot, stages/ (pretrain→SFT→DPO→eval→GRPO)
├── data/               # Stream-interleaving byte loader
├── multimodal/         # Any-to-token encoders (image/video/audio/PDF/docx)
├── ui/                 # Teto Vocaloid + rich terminal
├── tools/              # CLI, data manager, orchestrator, plotter, inference, tool_executor
├── tests/              # 175 unit tests + ultra-stable profiler
├── busel_rust_io/      # PyO3 Rust: mmap streamer, ternary matmul, packer
├── configs/            # default.yaml — 12 profiles
├── site/               # Astro+Starlight wiki (you are here)
├── busel_registry.py   # Plug-in extension-point registry
├── busel_logging.py    # JSONL event stream
├── cli.py              # Typer entrypoint
└── pyproject.toml      # uv-managed, maturin backend
```

Each module has its own **`AGENTS.md`** with a knowledge-base dump
(structure, where-to-look, key classes, conventions, anti-patterns).
Those are the "code archaeology" files; this wiki is the "human tour"
of the same material.

## A complete training loop, top to bottom

The diagram below traces one full step of the validation profile. It
is the canonical picture to keep in your head while reading the rest
of the docs.

```text
  ┌──────────────────┐
  │ raw bytes        │  (B, T) uint8 tensor — text, code, JSON, etc.
  │ data_train/*     │  Rust mmap streamer or Python fallback
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ ByteFlow          │  vocab=277 → d_byte=128 → d_model
  │ Patcher           │  adaptive pooling + boundary detection
  │ (model/patching)  │  + sigmoid-gated mini-SwishGLU
  └────────┬─────────┘
           │  patches  (B, T/4, d_model)
           ▼
  ┌──────────────────┐
  │ buselModel       │  n_layers × (mAR + decoder layer)
  │ (model/backbone) │  is_global = (l+1) % 4 == 0  ← 3:1 GDN-2/MLA/NSA
  │                  │  mAR: n_hyper=2 parallel streams
  │                  │  + buselDecoderLayer (attn + MatMul-free MoE)
  │                  │  + buselMTP4Pipeline (4 heads)
  └────────┬─────────┘
           │  logits_t1, _t2, _t3, _t4
           ▼
  ┌──────────────────┐
  │ buselLossEngine  │  MTP-4 weighted sum, [1.0, 0.5, 0.25, 0.125]
  │ (training/recipe)│  Liger-CE on CUDA, vanilla elsewhere
  └────────┬─────────┘
           │  loss
           ▼
  ┌──────────────────┐
  │ backward         │  autocast(bf16), gradient checkpointing
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ buselAutoPilot   │  3σ predictive dampening, AGC, dynamic WD
  │ (training/auto)  │  spike recovery (35% LR × 15 steps)
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ buselOptimizer   │  2D proj weights → SF-NorLotusMuon (Gram NS)
  │ (training/opt)   │  everything else → FP8 AdamW
  └──────────────────┘
           │
           ▼
  checkpoints/busel_validation_step_N.pt  (every N steps)
  checkpoints/busel.log.jsonl             (every event, JSONL)
```

## Where to go next

- **Read the [Architecture overview](/busel-ai/architecture/overview/)**
  for the design philosophy and the "why" behind every choice.
- **Run a [quick training](/busel-ai/guides/getting-started/)** in under
  10 minutes.
- **Skim the [API Reference](/busel-ai/reference/model/)** to find the
  class you need.
- **Hit [Performance → compile modes](/busel-ai/performance/compile-modes/)**
  if you want to squeeze more out of your GPU.
