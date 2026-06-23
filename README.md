# Busel (Бусел) — Sovereign 1-bit Any-to-Text LLM

> *Pronounced **[ˈbusɛl]** — from Belarusian **бусел** (stork).*
>
> Token-free 1.58-bit LLM with **mAR** residuals, **GDN-2 + NSA** attention mix,
> **MoD + Top-1 MoE**, **MTP-12**, **SF-NorLotusMuon** optimizer, byte-level
> patching, and **24 training optimizations**.
> Trains on consumer hardware (RTX 5060 Ti 16 GB) without tokenizers.

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)
[![CUDA](https://img.shields.io/badge/device-CUDA-green.svg)](#)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-orange.svg)](./LICENSE)

---

## Why Busel exists

1. **1.58-bit ternary weights** — every linear layer quantizes to `{-1, 0, +1}`.
   At inference: pure additions on CPU, model ~30× smaller than fp16.
2. **Byte-level tokens (vocab=326)** — no BPE, no tokenizer. Same stream carries
   text, code, JSON, images, video, audio, PDF.
3. **mAR residuals** — input-dependent mixing of layer outputs via Sinkhorn-Knopp
   projection onto the Birkhoff polytope (doubly-stochastic).
4. **7:1 GDN-2/NSA attention** — linear attention (O(n)) on 87% of layers,
   Native Sparse Attention on global layers.
5. **MoD cap=0.5** — 50% of tokens skip the FFN entirely (−40% FLOPs).
6. **DropBP** — 30% of layers drop backward pass (−25% VRAM).
7. **SF-NorLotusMuon** — Schedule-Free + LOTUS rank-8 (~85× less state than full Muon).
8. **WSD schedule** — flat LR 90% + sqrt decay 10% for +5-10% quality.
9. **SCT rank-8** — Spectral Compact Training in FFN layers.
10. **24 optimizations** active — per-layer compile, fused Triton kernel,
    data mixing 60/10/30, configurable grad clip, MTP-12, EMA, dispersion loss.

~3,000 LoC Python + ~140 LoC Rust. Readable in an afternoon.

---

## Quick start

```bash
# 1. Install (auto-detect GPU: NVIDIA → cu130, AMD → rocm63, else → cpu)
./scripts/setup.sh

# 2. Compile Rust extension
uv run maturin develop --release

# 3. Test (3 min, 4M params)
uv run python cli.py train --profile sovereign_test --max-steps 50

# 4. Train minimal chatbot (70M params, ~3h)
uv run python cli.py pipeline --name sovereign-5h

# 5. Train big model (1B params, ~3 days)
uv run python cli.py pipeline --name sovereign-3d
```

---

## Profiles

| Profile | d_model×L | Params | Time | Use case |
|---------|-----------|--------|------|----------|
| `sovereign_test` | 128×2 | 4M | 3 min | CI / smoke test |
| `sovereign_5h` | 512×8 | 84M | ~3h | Minimal chatbot |
| `sovereign_12h` | 768×8 | 170M | ~12h | Good model |
| `sovereign_24h` | 768×16 | 350M | ~24h | Strong model |
| `sovereign_3d` | 1024×18 | ~1B | ~3d | Sovereign |

All profiles: NSA, MoD, DropBP, SCT-8, WSD, MTP-12, compile (per-layer),
data mixing 60/10/30 (FineWeb/Wiki/Cosmopedia).

`max_steps` is `"auto"` — computed from Busel Scaling Law (37 tok/param for <3B models).

---

## Architecture

```
        text / image / video / audio / PDF
                      │
                      ▼
        ┌──────────────────────────┐
        │ StridedFastBLTPatcher   │  byte→patch, stride=4
        │  vocab=326, d_byte=128  │  + GLU gate + boundary conv
        └──────────────────────────┘
                      │ patches (B × n_patches × d_model)
                      ▼
        ╔══════════════════════════════╗
        ║  buselModel (n_layers)      ║
        ║                             ║
        ║  for each layer:            ║
        ║    mixed = mAR(x, streams)  ║  Sinkhorn-Knopp
        ║    h = GDN-2/NSA(mixed)     ║  + MoD routing
        ║    h = MoE(h)               ║  6 experts, top-1
        ║    x = mixed + h            ║  residual
        ║    [LCSB: 50% no_grad]     ║  −44% step
        ║    [DropBP: 30% no_bwd]    ║  −25% VRAM
        ╚══════════════════════════════╝
                      │ hidden
                      ▼
        ┌──────────────────────────┐
        │ buselMTPPipeline (12)    │  predict t+1..t+12
        │  decay [1.0, .5, .25...] │
        └──────────────────────────┘
                      │
                      ▼
               logits (B × T × 326)
```

---

## Project layout

```
busel-ai/
├── model/              # BitNet architecture (BitLinear, GDN-2, NSA, MoE, mAR)
├── training/           # SF-NorLotusMuon + AdamW, AutoPilot, WSD, loss engine
├── data/               # Stream-interleaving loader (Rust mmap + weighted mixing)
├── multimodal/         # Image/video/audio/PDF encoders, special tokens
├── ui/                 # Teto Vocaloid emoticon + rich terminal
├── tools/              # CLI (typer), orchestrator, inference, tool executor
├── tests/              # 172-test unittest suite + profiler
├── busel_rust_io/      # PyO3 Rust ext: mmap, ternary matmul, binary packer
├── configs/            # 5 sovereign profiles + pipeline YAMLs
├── busel_registry.py   # Plug-in registry (attention, optimizer, encoder...)
├── busel_logging.py    # Structured JSONL event stream
└── cli.py              # Typer entrypoint
```

---

## Performance

| Model | Batch | tok/s | VRAM | GPU util |
|-------|-------|-------|------|----------|
| sovereign_test (4M) | 128 | ~500K | 1GB | 30% |
| sovereign_5h (84M) | 288 | ~130K | 6GB | 70% |
| sovereign_12h (170M) | 96 | ~80K | 8GB | 85% |
| 280M (intermediate) | 288 | ~135K | 6GB | 60% |

**Optimizations enabled:**
- Per-layer `torch.compile` with `dynamic=False` (avoids layer_idx recompilation)
- Fused Triton BitLinear kernel (1 kernel instead of 5-6 for ternary matmul)
- MoD cap=0.5 (50% tokens skip FFN)
- DropBP (30% layers skip backward)
- Gradient checkpointing every 3-4 layers
- 4 data workers with prefetch
- Inline Tequila deadzone bias

---

## CLI

```bash
# Train
uv run python cli.py train --profile sovereign_5h
uv run python cli.py pipeline --name sovereign-5h

# Chat (after training)
uv run python cli.py chat --checkpoint checkpoints/busel_5h_FINAL.pt

# Data
uv run python cli.py download-preset --name sft-shpak-fable5

# Monitoring
tail -f checkpoints/training_5h.log
```

---

## Busel Scaling Laws

Ternary weights hold ~30× less info per param than fp16:
- **<3B params:** 37 tok/param (empirical, from 2.68M benchmark)
- **≥3B params:** 80 tok/param (matches BitNet/Chinchilla for fp16)

The auto-planner selects the right law based on `total_params`.

---

## License

**CC BY-NC-SA 4.0** — see [LICENSE](./LICENSE). Commercial use requires written permission from `sehaxe`.
