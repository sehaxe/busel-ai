# PROJECT KNOWLEDGE BASE — busel (Бусел)

**Generated:** 2026-06-03 17:23 UTC
**Commit:** d731f9a
**Branch:** main

## OVERVIEW
**busel v5.1** — Sovereign 1-bit (1.58b) Any-to-Text LLM with hybrid linear attention (GDN-2/MLA), mAR (Manifold Constrained Attention Residuals), MoE, byte-level patching (no BPE), and MTP-4. Targets consumer HW (RTX 5060 Ti 16GB / Apple Silicon). Hybrid Python + Rust (PyO3 via maturin). Trained/inferred via CLI. Docs site is Astro+Starlight (Bun).

## STRUCTURE
```
busel-ai/
├── model/              # BitNet v2 architecture (patching/layers/attention/routing/backbone)
├── training/           # Muon+AdamW hybrid optimizer, AutoPilot v6.0, MTP-4 loss
├── data/               # Stream-interleaving byte loader (Rust mmap or Python fallback)
├── tools/              # Typer CLI, data_manager, orchestrator, plotter, inference
├── services/           # FastAPI inference server (uvicorn)
├── tests/              # unittest suite + ultra-stable profiler v2.0
├── busel_rust_io/      # PyO3 Rust ext: mmap ByteStreamer, ternary matmul, binary packer
├── configs/            # default.yaml — Shpak/Zubr/Chyzh/MicroTest/QuickTest profiles
├── site/               # Astro+Starlight docs (GitHub Pages)
├── checkpoints/        # *.pt training state (gitignored)
├── data_train/         # Raw training data (gitignored)
├── train.py            # Cybernetic training orchestrator (curriculum + Chinchilla auto-planner)
├── cli.py              # Typer entrypoint (all user commands)
└── pyproject.toml      # uv-managed, maturin build backend
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model layer | [model/](file:///home/sehaxe/busel-ai/model/AGENTS.md) | Must use BitLinear_a4_8 / H_BitLinear / RMSNorm |
| Modify training loop | [train.py](file:///home/sehaxe/busel-ai/train.py) + [training/](file:///home/sehaxe/busel-ai/training/AGENTS.md) | Cybernetic curriculum is here |
| Add CLI command | [cli.py](file:///home/sehaxe/busel-ai/cli.py) → register in `tools.orchestrator` | Typer-based |
| Modify data loader | [data/pipeline.py](file:///home/sehaxe/busel-ai/data/pipeline.py) | Prefers Rust `ByteStreamer`; Python fallback exists |
| Add API endpoint | [services/inference_api.py](file:///home/sehaxe/busel-ai/services/inference_api.py) | FastAPI + uvicorn |
| Tune model size | [configs/default.yaml](file:///home/sehaxe/busel-ai/configs/default.yaml) | Profile: shpak/zubr/chyzh/micro_test/quick_test |
| Profile step perf | [tests/profiler_run.py](file:///home/sehaxe/busel-ai/tests/profiler_run.py) | No torch.profiler (MPS hangs) |
| Edit docs site | [site/](file:///home/sehaxe/busel-ai/site/) | `bun install && bun run build` |

## ARCHITECTURE (1-bit LLM)
- **Weights:** 1.58-bit ternary `{-1, 0, +1}` via STE (`BitLinear_a4_8` in [model/layers.py](file:///home/sehaxe/busel-ai/model/layers.py))
- **Tokens:** Raw UTF-8 bytes (vocab=259), `stride=4` conv → patches (`StridedFastBLTPatcher` in [model/patching.py](file:///home/sehaxe/busel-ai/model/patching.py))
- **Attention:** 3:1 GDN-2 (linear, O(1) cache) : MLA (latent KV, d_c=128)
- **Residuals:** mAR — Birkhoff-polytope projection (Sinkhorn-Knopp ×3) over all previous layers
- **MoE:** 2 shared + N routed (Top-2), Blackboard Memory bus to prevent collapse
- **Heads:** MTP-4 (predict t+1, t+2, t+3, t+4) with decaying loss weights [1.0, 0.5, 0.25, 0.125]
- **Optimizer:** Hybrid Muon (2D `proj` params, Newton-Schulz ×5) + AdamW (rest)
- **Curriculum:** 1024 → 2048 → 4096 ctx warmup; AutoPilot v6.0 spike dampening

## CONVENTIONS
- **Build:** `uv` for Python+deps, `maturin develop --release` for Rust ext, `bun` for site
- **Device:** Auto-detect (CUDA → MPS → CPU). MPS uses `bf16`/`fp16`, CUDA uses `bf16`
- **Stability:** Seed 42, TF32 on, cuDNN benchmark on, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **NVTX:** All major ops wrapped in `nvtx_range_push/pop` for profiling (CUDA only)
- **Env vars:** `INFERENCE_API_URL` (default `http://127.0.0.1:8000`), `DEFAULT_PROFILE`
- **Config loading:** profiles from `configs/default.yaml`; `max_steps`/`warmup_steps` support `"auto"`
- **License:** CC BY-NC-SA 4.0 (NC clause — NO commercial use)

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER** use BPE/tokenizers — model is byte-level (vocab=259 only)
- **NEVER** add new `nn.Linear` to model — must use `BitLinear_a4_8` or `H_BitLinear`
- **NEVER** add `Lock` in `state_manager.py` — use `RLock` (re-entrant) to avoid deadlock
- **NEVER** checkpoint `*.pt < 10MB` — auto-rejected as corrupted by `services/inference_api.py`
- **NEVER** use `torch.profiler` on macOS — known to hang; use `tests/profiler_run.py` instead
- **NEVER** set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` > 0.0 — `train.py` enforces 0.0
- **NEVER** mix `H_BitLinear` for non-`o_proj` outputs — reserved for output projection per BitNet v2 spec
- **NEVER** bypass `BitLinear_a4_8` `is_intermediate=True` path in FFN experts — needed for INT8 TopK quantization
- **NEVER** commit `data_train/`, `checkpoints/`, `.env`, `Cargo.lock`, `uv.lock`

## UNIQUE STYLES
- **Emoji-prefixed module headers:** Every Python file starts with `"""🦩 / ⚙️ / 💡 / 📚 / 🤖 / 🎯 / 🛸 ..."""` docstring
- **Russian-language comments:** Heavy use of Cyrillic comments throughout (technical)
- **`busel*` prefix:** All custom classes (`buselModel`, `buselOptimizerEngine`, `buselLossEngine`, `buselAutoPilot`, `buselOmnivoreTextExtractor`)
- **`cfg.profile` in checkpoint dict:** Every saved `.pt` carries its profile name for auto-detect
- **Rust parallel iterators:** `rayon::prelude::*` for `ternary_matmul_cpu` (no GPU on inference)
- **Subprocess CLI orchestration:** `tools/orchestrator.py` shells out to `train.py`, `profiler_run.py` via `subprocess.run`

## COMMANDS
```bash
# Setup
uv sync
uv add docling              # PDF support for data loader
uv run maturin develop --release   # Build Rust ext into venv

# Data
uv run python cli.py download-all --preset shpak
# (or copy PDFs/JSONL into data_train/ — auto-detected)

# Train
uv run python cli.py autopilot --profile shpak   # one-click: data + profiler + train
uv run train.py --profile shpak                   # manual
uv run python cli.py profile                      # hardware profiler only

# Serve
uv run python cli.py serve --port 8000            # FastAPI

# Docs
cd site && bun install && bun run build           # GitHub Pages deploy
```

## NOTES
- **Checkpoint size guard:** Reject `<10MB` `.pt` (corrupt) in both `services/inference_api.py` and `tools/inference.py`
- **Target bit size:** 11MB (Shpak) / 30MB (Zubr) — 1.58-bit weights compress ~10x vs fp16
- **Metrics log:** `checkpoints/metrics.jsonl` (one JSON per step, for ETA calc)
- **macOS Rust flag:** `.cargo/config.toml` uses `link-arg=-undefined,dynamic_lookup` for macOS
- **License:** Commercial use requires written permission from `sehaxe`
