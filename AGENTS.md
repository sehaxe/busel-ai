# PROJECT KNOWLEDGE BASE — busel (Бусел)

**Last updated:** 2026-06-25
**Branch:** main
**Test count:** 176 (unittest, no pytest)

## [PRIORITY] — read first
1. **Performance + LOC** — when in doubt, the faster + shorter option wins.
2. **Stage of project: EARLY.** Compatibility is NOT a constraint. Breaking changes are fine.
3. **Ship working code > elegant code.** We iterate on what runs, refactor later.
4. **Code is small** — entire model + training + data is ~3,000 LoC Python + ~140 LoC Rust. Do not add files without justification.

## OVERVIEW
**busel** — Sovereign 1-bit (1.58b) Any-to-Text LLM. Hybrid Python + Rust (PyO3 via maturin). Targets consumer HW (RTX 5060 Ti 16 GB / Apple Silicon). Trained via CLI, documented in `site/` (Astro+Starlight, Bun).

**Architecture:** 1.58-bit ternary weights · byte-level vocab=277 · **ByteFlow** patching (adaptive pooling + boundary detection) · 3:1 GDN-2:MLA attention · mAR residuals (Sinkhorn-Knopp on Birkhoff polytope, DTopK) · **Top-1 MoE** with Blackboard Memory · MTP-6 heads · **SF-NorLotusMuon** (Schedule-Free + NorMuon + LOTUS rank-128) + **FP8 AdamW** hybrid · **Gram NS** (`gram_newton_schulz` package) · **Muon+** column normalization · **EMA of weights** · **selective activation checkpointing** (every=2) · **LCSB selective per-layer backward** (default ON in shpak/zubr/chyzh) · **decoupled per-layer LR** (6 sub-groups) · **multi-stage pipeline** (pretrain → SFT → DPO → eval) · **REPL tool executor** · **compile-safe checkpoint loader** · **research features** (Dispersion Loss, Rho-1 Loss, DropBP, Routing-Free, SCT Spectral Compact Training, Tequila, FlexAttention, CLA, Hestia).

## STRUCTURE
```
busel-ai/
├── model/             # BitNet v2 architecture (layers/attention/routing/backbone/patching/checkpoint)
├── training/          # SF-NorLotusMuon + FP8 AdamW, EMA, AutoPilot, MTP-4 loss, **stages/** framework
├── data/              # Stream-interleaving token loader (list[int], Rust mmap or Python fallback)
├── multimodal/        # Any-to-token encoders (image/video/audio/PDF/docx) + 18-token special vocab
├── ui/                # Teto Vocaloid emoticon + rich terminal helpers (animation.py, cli.py)
├── tools/             # Typer CLI (orchestrator, data_manager, plotter, inference, **tool_executor**)
├── tests/             # unittest suite (175) + ultra-stable profiler + 3 profile scripts
├── busel_rust_io/     # PyO3 Rust ext: mmap ByteStreamer, ternary matmul, binary packer
├── configs/           # default.yaml — 12 profiles (validation/MicroTest/QuickTest/Chyzh/Scale_m/Shpak/Zubr/IMU1/Noc/Kruk/Byvol)
├── site/              # Astro+Starlight docs (GitHub Pages)
├── checkpoints/       # *.pt + busel.log.jsonl (gitignored)
├── data_train/        # Raw training data (gitignored)
├── busel_registry.py  # 🛸 Plug-in extension-point registry (attention/optimizer/encoder/autopilot/curriculum/loss/stage)
├── busel_logging.py   # 📚 Structured JSONL event stream
├── cli.py             # Typer entrypoint (root-level — all user commands)
└── pyproject.toml     # uv-managed, maturin build backend
```

## DEFAULTS — single source of truth (wired in code, not YAML)
These are hardcoded defaults in `buselOptimizerEngine` and `NorLotusMuon`:

| Feature | Default | Why |
|---|---|---|---|
| **SF-NorLotusMuon** (Muon path) | Always ON | Schedule-Free + NorMuon + LOTUS rank-128 + column norm. Single path, no opt-out. |
| **FP8 AdamW** (AdamW path) | Always ON | `torchao.optim.AdamWFp8` — 75% memory reduction vs fp32 AdamW. |
| **Gram NS** (orthogonalization) | Fallback | `gram_newton_schulz` package disabled — aggressive coefficients unstable on SCT. Uses built-in `_newton_schulz_core` (quintic, 5 steps). |
| **Muon+** column normalization | Always ON | `O_t / (O_t.norm(dim=0, keepdim=True) + 1e-8)` after NS. |
| **LOTUS column norm** (§3.2) | Always ON | `bp.norm(dim=0)`, `bq.norm(dim=0)` — prevents exponential buffer growth → NaN. |
| **top_k** (MoE) | `1` | 1 of N experts per token. −35% routed FFN FLOPs. |
| **EMA** | `True` | `ema_decay=0.999`. 10-15% fewer steps to same loss. |
| **SCT rank** | `128` | All profiles. SCT+Muon stable with LOTUS column norm (prevents buffer explosion → NS NaN). |
| **LOTUS rank** | `128` | Match SCT rank — column L2 norm prevents unbounded growth. |
| **Selective ckpt** | `every=2` | Halves activation memory at <5% step-time cost. |
| **LCSB** | `True` (all profiles) | `backward_ratio=0.5`. −44% step, −25% mem, +80% tok/s. |
| **Decoupled LR** | `{attn:1.0, ffn:1.0, mtp:1.0, norm:1.0, embed:0.5, router:0.5}` | Embed/router get half LR. |
| **SCT rank-8** | `True` (rank=8) | Spectral Compact Training — FFN compression ×4-8, same quality. |
| **DropBP** | `True` (prob=0.3) | 30% layers skip backward — regularization, +speed. |
| **Dispersion Loss** | `False` | Optional uniformity loss on embeddings — see opt-in table. |
| **Progressive Freeze** | `True` | Freeze up to 75% layers in late training — +speed. |
| **ASCII Curriculum** | `True` | 7-bit ASCII first 30% training, then full 8-bit. |
| **Fused BitLinear** | `True` (eager mode) | Triton kernel replaces 10+ ops in one launch. Disabled under torch.compile (inductor fuses automatically). |
| **Chunk Curriculum** | `True` | Context growth: 1/16→1/8→1/4→1/2→full (512→1024→2048→4096→8192). |

## REMOVED OPTIMIZERS (v8.5 cleanup)
All dead branches deleted. The optimizer is now a single clean path:
- **Muon** — merged into `_MuonBase`
- **NorMuon** — merged into `NorLotusMuon._apply_weight_decay`
- **LotusMuon** — preserved as `_MuonBase` subclass
- **SOAP** — deleted (Shampoo eigenspace was never the winner)
- **MuonQ** — deleted (4-bit quantization too lossy)
- **Adafactor** — deleted (FP8 AdamW is strictly better)
- **Cautious** — deleted (SF handles gradient noise)
- **QuEST** — deleted (trust gradient proven unnecessary with SR-STE)
- **FlashMuon** — deleted (Gram NS package handles CUDA path)

## OPT-IN RESEARCH FEATURES (defaults OFF — profile before flipping)

**Flipped to default ON:**
- **LCSB selective per-layer backward** (`selective_backward=True, backward_ratio=0.5`) — validated at all 3 sizes (−57.7% step at 2M, −44.4% at 52.8M, −39.1% at 120M; 0 quality regression at 10 steps). Don't disable without measuring.

All profiles in `configs/default.yaml` (validation, micro_test, quick_test, chyzh, scale_m, shpak, noc, kruk, byvol) inherit these defaults. The CLI `tests/profiler_run.py` defaults are aligned.

## OPT-IN RESEARCH FEATURES (defaults OFF — profile before flipping)
| Field | Default | Why OFF by default | Measured on shpak 52.8M |
|---|---|---|---|
| `use_dispersion_loss` (training) | `False` | — Dispersion Loss (Wang et al. 2026, arXiv:2602.00217). Uniformity loss (Wang & Isola 2020) on L2-normalised token embeddings. L = weight · log E[exp(−t·‖z_i−z_j‖²)] over a `sample_size` random subset. Backprop drives byte embeddings apart on the unit hypersphere — counters embedding condensation that hurts small LMs. Paper: +1.17 % avg on 10 benchmarks, +3.3 % over baseline. | Dispersion alone: −6.3 % loss@10, +2.5 % step, +9 MB. winner (DA+Cautious+LCSB+Dispersion): −4.8 % loss@10 vs winner, +1.8 % step, +519 MB VRAM (offsets LCSB's gain). |
| `dispersion_weight` | `0.1` | Multiplicative scale on the uniformity loss. | — |
| `dispersion_temperature` | `2.0` | t in the uniformity loss exp(−t·d²). | — |

## OPT-IN RESEARCH FEATURES (defaults OFF — profile before flipping)
| Field | Default | Why OFF by default | Measured on shpak 52.8M |
|---|---|---|---|
| `optimizer_type: "soap"` (training) | `False` | — SOAP (Vyas et al. 2025, ICLR 2025). Shampoo eigenspace + Adam. Maintains factored second-moment estimates L, R per 2D param; periodically eigendecomposes and applies Adam-like update in eigenspace. −40 % iterations vs AdamW; overhead from eigendecomposition every N steps. For 1D params falls back to Adam. Memory: O(d1² + d2²) per 2D param. | — (needs 50+ step validation) |
| `use_quest` (training) | `False` | — QuEST (Panferov et al. 2025, ICML 2025). Trust gradient estimator for ternary training. Hadamard rotation whitens weight distribution, MSE-optimal ternary grid fitting, trust gradient correction reduces bias vs naive STE. ~5 % step-time overhead. Wraps any base optimizer. | — (needs 50+ step validation) |
| `quest_bits` (training) | `1.58` | Bit-width for QuEST ternary grid fitting. | — |
| `lr_schedule: "wsd"` (training) | `"cosine"` | — Warmup-Stable-Decay schedule. Flat LR for first (1−decay_fraction) of training, then sqrt decay. | — |
| `lr_schedule: "wd33"` (training) | `"cosine"` | — Warmdown-to-33 % (Mapping Schedule × Bit-Width, 2026). Cosine + warmdown to 33 % of peak LR in last 33 % of training. Optimal at all bit-widths for sub-100 M models. | — |
| `wsd_s_enabled` (training) | `False` | — WSD-S checkpoint reuse (ICLR 2025). Reuses decay-phase checkpoints for the next cycle. Outperforms WSD and Cyclic-Cosine. | — (needs 50+ step validation) |
| `wsd_s_interval` (training) | `1000` | Steps in stable phase before switching to decay. | — |
| `wsd_s_decay_steps` (training) | `200` | Steps in decay phase. | — |
| `use_tequila` (training) | `False` | — Tequila (Huang et al. 2025, ICLR 2026). Deadzone trapping fix. Reactivates weights trapped at quantization boundary (|w| < Δ) as dynamic biases, providing direct gradients. >4 % accuracy gain on ARC. Zero inference overhead. | — (needs 50+ step validation) |
| `tequila_lambda` (training) | `0.001` | Tequila reactivation scale (λ in paper). | — |
| `use_hestia` (model) | `False` | — Hestia (Wang et al. 2026). Hessian-guided QAT. Temperature-controlled softmax relaxation replaces STE. Hessian trace drives per-layer temperature annealing. 5.39 % avg zero-shot improvement on Llama-3.2-1B. | — (needs 50+ step validation) |
| `hestia_init_temp` (training) | `6.0` | Initial temperature for softmax relaxation. | — |
| `hestia_end_temp` (training) | `0.0` | Final temperature (0 = hard quantization). | — |
| `optimizer_type: "muonq"` (training) | `False` | — MuonQ (Su et al. 2025). 4-bit Muon via directional fidelity optimization. Pre-quantization normalization, power-iteration structural decomposition, μ-law companding. 7.3× memory reduction vs full-precision Muon. | — (needs 50+ step validation) |

**Scale gate (50 M+):** automatically disables heavy optimizations (SOAP, Adafactor, QuEST, QK-Norm L2, NorMuon, MuonQ, Hestia) when model params < 50 M. Falls back to lotus_muon. These optimizations have overhead that outweighs benefits at small scale. Threshold: `_SCALE_THRESHOLD = 50_000_000` in `training/stages/pretrain.py`.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new model layer | [model/](file:///home/sehaxe/busel-ai/model/AGENTS.md) | Must use `BitLinear_a4_8` or `H_BitLinear` (no raw `nn.Linear`) |
| Modify training loop | [training/](file:///home/sehaxe/busel-ai/training/AGENTS.md) | Pretrain is `buselPretrainStage`; SFT/DPO/eval in `training/stages/` |
| **Debug LOTUS NaN** | `training/optimizer.py` line 94 | Column norm fix (LOTUS §3.2): `bp/bq.norm(dim=0)` after momentum update — prevents buffer explosion |
| Add CLI command | [tools/](file:///home/sehaxe/busel-ai/tools/AGENTS.md) | Typer `@app.command`; subprocess pattern, never `import train` |
| Modify data loader | [data/AGENTS.md](file:///home/sehaxe/busel-ai/data/AGENTS.md) | Prefers Rust `ByteStreamer`; Python fallback exists |
| Add a new modality | [multimodal/AGENTS.md](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) | `@register("encoder", ...)`; return `list[int]` in `[0, 277)` |
| Tune model size | [configs/default.yaml](file:///home/sehaxe/busel-ai/configs/default.yaml) | 12 profiles: validation / micro_test / quick_test / chyzh / scale_m / shpak / imu1 / noc / kruk / byvol / soap / quest |
| Profile step perf | [tests/AGENTS.md](file:///home/sehaxe/busel-ai/tests/AGENTS.md) | No `torch.profiler` (MPS hangs); use `tests/profiler_run.py` |
| Edit docs site | [site/AGENTS.md](file:///home/sehaxe/busel-ai/site/AGENTS.md) | Bun + Starlight; 7 stable URL sections |
| Add attention/optimizer/etc. | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | `@register("kind", "name")` — auto-discovered, no switch stmt |
| Customise terminal UX | [ui/](file:///home/sehaxe/busel-ai/ui/) | Teto emoticon + rich panels/spinners/trees |
| Consume event stream | [busel_logging.py](file:///home/sehaxe/busel-ai/busel_logging.py) | `checkpoints/busel.log.jsonl` — one JSON per event |
| Add a new pipeline stage | [training/AGENTS.md](file:///home/sehaxe/busel-ai/training/AGENTS.md) | `@register_stage("name")` in `training/stages/<name>.py` |
| Load a checkpoint | [model/checkpoint.py](file:///home/sehaxe/busel-ai/model/checkpoint.py) | `load_state_dict_safely(model, sd)` — never raw `load_state_dict` |

## DO
- **Use the registry** for any plug-in point (attention, optimizer, encoder, autopilot, curriculum, loss, stage). `@register("kind", "name")` — no central switch statements.
- **Run 175 tests before pushing:** `uv run python -m unittest tests.test_suite` — all must pass.
- **Update README.md AND site/ docs** when a feature changes. The two-track rule: code change → README change → site/ change. Site/ is the human-friendly tour, README is the elevator pitch.
- **Match existing patterns.** Sample 2-3 similar files before adding a new one. Busel is small — patterns are visible at a glance.
- **Profile with `tests/profiler_run.py`** before claiming a speedup. Numbers, not vibes.
- **Use `uv run python cli.py`** — never `python cli.py` (maturin ext needs venv).
- **Add new files sparingly.** Default bias: extend an existing module. A 3,000 LoC codebase doesn't need more files.
- **Document anti-patterns in per-module AGENTS.md** when you discover one. The DO/NEVER list is the institutional memory.
- **Subprocess for training runs, direct call for everything else.** `tools/orchestrator.py` shells out; `model/`, `training/`, `data/`, `multimodal/` are in-process.
- **Single source of truth for checkpoint loading:** `model.checkpoint.load_state_dict_safely`. Four cross-config cases (compiled↔eager, save↔load) — the helper handles all of them.
- **Keep tests in `tests/test_suite.py`.** Never spawn a second test file. The suite is one big `unittest.TestCase` class.

## NEVER
- **BPE / tokenizers** — model is byte-level, vocab=277 (256 raw bytes + 21 specials). See [multimodal/AGENTS.md](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) for the 18-token breakdown.
- **Raw `nn.Linear` in model** — must use `BitLinear_a4_8` or `H_BitLinear`. Breaks the 1.58-bit guarantee.
- **`H_BitLinear` for non-`o_proj`** — BitNet v2 spec mandates it for output projection only (massive-activation mitigation).
- **Disable any default in [DEFAULTS](#defaults-buselpretrainconfig--single-source-of-truth) without measuring** — LOTUS+Muon, EMA, Top-1, decoupled LR, selective ckpt, LCSB all help. Re-enabling them is the right move, not disabling.
- **`torch.profiler` on macOS** — known to hang; use `tests/profiler_run.py` (`time.perf_counter` + `cuda.max_memory_allocated`).
- **`model.load_state_dict(sd)` directly** — always `load_state_dict_safely(model, sd)`. Direct loads fail when saved with `--compile` (the default). The helper strips `_orig_mod.` and unwraps `OptimizedModule`.
- **Drop the `File` handle in `ByteStreamer`** — macOS mmap segfaults without it. Comment in `busel_rust_io/lib.rs:13` explains.
- **`cargo build`** — only `uv run maturin develop --release` produces a working Python ext.
- **PyO3 `unsafe` outside `Mmap::map`** — keep memory safety intact.
- **`import train` from `orchestrator.py`** — always `subprocess.run([sys.executable, "cli.py", ...])`. The legacy `train.py` was deleted in v8.5; use `cli.py pipeline`.
- **`python-dotenv`** — project has its own `load_env()` parser in `tools/orchestrator.py`.
- **`assertTrue(x == y)` in tests** — use `assertEqual`. Better failure messages.
- **`nn.Embedding` for tokens** — use `nn.Parameter(torch.randn(vocab_size(), d_byte))`. Size auto-tracks the special-token registry.
- **Hardcoded vocab IDs** — import from `multimodal.special_tokens`. IDs are auto-allocated and may shift when tokens are added/disabled.
- **Hardcoded `259` in embedding shape** — use `vocab_size()`. (Old constant from before the 18-token expansion.)
- **Softmax on mAR logits** — Sinkhorn-Knopp projects to the Birkhoff polytope (doubly-stochastic), not softmax.
- **Muon on 1D params** (norms, biases) — `buselOptimizerEngine` filters them.
- **Muon on anything with `router` or `embed` in name** — `_MUON_EXCLUDE = ("router", "embed")`. Routers are policy, not value; embed is categorical.
- **Add `@torch.compile` to the whole `step()`** — only to inner Newton-Schulz function. Outer compile breaks the mAR FIFO buffer aliasing.
- **Change `momentum=0.95` of Muon** — spec is brittle; verify on validation profile if you must.
- **`F.cross_entropy` on CUDA when Liger is available** — 2-3× slower. Liger auto-falls back to vanilla.
- **`max_steps < warmup_steps` in any preset** — produces NaN spikes.
- **Skip the `Profile` step in `autopilot`** — HW profiling catches VRAM/RNG bugs early.
- **Execute a model-emitted tool call without user confirmation** — `tools/tool_executor.py:interactive_confirm` is the gate. `/auto on` is a per-session opt-in only.
- **Shrink `config.vocab_size` below `multimodal.special_tokens.vocab_size()`** — `buselModel.__init__` raises `ValueError`. Catches stale YAML.
- **Commit `data_train/`, `checkpoints/`, `.env`, `Cargo.lock`, `target/`** — all gitignored. The `.gitignore` is the single source of truth for what to never commit; do not duplicate that list in a NEVER rule.
- **Commit without explicit user request** — always wait for `commit` / `push` / `merge` instruction.
- **Use `sys_platform` markers in `[tool.uv.sources]`** — the 6 explicit extras (`cpu` / `cu118` / `cu126` / `cu128` / `cu130` / `rocm63`) are the supported way to pick hardware. Markers force every Linux user to CUDA 13.0, which fails the "different cards" requirement (Linux may have any NVIDIA / AMD / Intel / no GPU). If you need to add a new variant, add an extra + index, not a marker. **For the `rocm63` extra** the `pytorch-triton-rocm` package must be declared in `[tool.uv.sources]` pointing to the same `pytorch-rocm*` index — with `explicit = true` on the index, uv only searches the named index for packages listed in `[tool.uv.sources]`, and `pytorch-triton-rocm` is not on PyPI. The same trick may be needed for any future rocm / xpu / custom-torch extra whose transitive deps live only on the matching pytorch.org index.

## UNIQUE STYLES
- **Emoji-prefixed module headers:** every Python file starts with `"""🦩 / ⚙️ / 💡 / 📚 / 🤖 / 🎯 / 🛸 ..."""` docstring.
- **Russian-language comments:** heavy Cyrillic throughout (technical).
- **`busel*` prefix** on all custom classes (`buselModel`, `buselOptimizerEngine`, `buselLossEngine`, `buselAutoPilot`, `buselPretrainStage`, …).
- **`cfg.profile` in checkpoint dict** — every saved `.pt` carries its profile name for auto-detect.
- **Rust parallel iterators** (`rayon::prelude::*`) for `ternary_matmul_cpu` (no GPU on inference).
- **Pipeline orchestration** in `tools/orchestrator.py` — runs stages in-process via `get_stage()` → `setup/run/finalize`. No subprocess for training; profiler is still subprocess-based.

## COMMANDS
```bash
# Setup (extras are mutually exclusive — pick one per machine)
./scripts/setup.sh                    # auto-detect: NVIDIA→cu130, AMD→rocm63, else→cpu + maturin
./scripts/setup.sh cu128              # or pick a specific extra
uv sync --extra cu130                 # explicit, modern NVIDIA
uv sync --extra rocm63                # AMD GPU (RX 6000/7000/9000, gfx900-gfx1201)
uv sync --extra cpu                   # no GPU / Apple Silicon
uv sync --extra cu118                 # legacy NVIDIA (driver ≥ 470)
uv add docling                        # PDF support for data loader
uv run maturin develop --release      # Build Rust ext into venv (auto-run by setup.sh)

# Data
uv run python cli.py download-all --preset shpak
# (or copy PDFs/JSONL into data_train/ — auto-detected)

# Train
uv run python cli.py autopilot --profile shpak   # one-click: data + profiler + train
uv run python cli.py pipeline --name pretrain-only
uv run python cli.py profile                      # hardware profiler only

# Docs
cd site && bun install && bun run build           # GitHub Pages deploy

# Profile research features (2 modes: cumulative + dispersion)
uv run python tests/v58_profile.py --mode shpak-v60    #  cumulative on shpak 52.8M (5 runs adding DA, Cautious, SF, LCSB)
uv run python tests/v58_profile.py --mode shpak-disp   #  dispersion on shpak 52.8M (4 runs: baseline, +Dispersion, winner, winner)

# Quick IMU-1 vs baseline profiler (~5 min on RTX 5060 Ti)
uv run python tests/quick_imu1_profile.py              # baseline vs imu1 comparison (2M params)

# v8.5 kruk A/B profiler (12 experiments, ~65M params)
uv run python tests/v63_profile.py --mode kruk-v85      #  kruk 65M: 12 A/B experiments
```

## PER-MODULE RULES
This file is the project-level summary. Module-specific rules, anti-patterns, and the per-class API live in per-module AGENTS.md:

- [model/AGENTS.md](file:///home/sehaxe/busel-ai/model/AGENTS.md) — BitLinear, H_BitLinear, GDN-2, MLA, mAR, MoE, MTP, compile-safe checkpoint loader
- [training/AGENTS.md](file:///home/sehaxe/busel-ai/training/AGENTS.md) — SF-NorLotusMuon + FP8 AdamW, AutoPilot, loss engine, **stages/ framework** 
- [data/AGENTS.md](file:///home/sehaxe/busel-ai/data/AGENTS.md) — Rust mmap streamer, Python fallback, multimodal dispatch
- [multimodal/AGENTS.md](file:///home/sehaxe/busel-ai/multimodal/AGENTS.md) — 18-token vocab, 6 encoders (image/video/audio/PDF/docx/text)
- [ui/AGENTS.md](file:///home/sehaxe/busel-ai/ui/AGENTS.md) — Teto emoticon frames, live animation, rich terminal helpers
- [tools/AGENTS.md](file:///home/sehaxe/busel-ai/tools/AGENTS.md) — Typer CLI, pipeline orchestrator, **REPL tool executor**
- [tests/AGENTS.md](file:///home/sehaxe/busel-ai/tests/AGENTS.md) — 175-test unittest suite, custom profiler, 3 profile scripts
- [busel_rust_io/AGENTS.md](file:///home/sehaxe/busel-ai/busel_rust_io/AGENTS.md) — PyO3 extension, mmap safety, Rayon threading
- [site/AGENTS.md](file:///home/sehaxe/busel-ai/site/AGENTS.md) — Astro+Starlight, build commands, URL structure

## NOTES
- **Busel Scaling Laws (two-tier, experimental):** Ternary weights (1.58-bit) hold ~30× less info per param than fp16.
  - **Small models (<3B params):** 37 tok/param (empirical, from 2.68M-param benchmark). Model saturates earlier due to limited capacity per param.
  - **Large models (≥3B params):** 80 tok/param (matches BitNet/chinchilla for fp16). Large 1.58-bit models regain fp16-equivalent scaling per Microsoft BitNet findings.
  - For shpak (52.8M < 3B): optimal D ≈ 2B tokens, not 4.2B. Training time: ~10h, not ~20h.
  - See `tests/scaling_laws.py` and README "Busel Scaling Laws" section for full derivation.
- **Checkpoint size guard:** reject `<10MB` `.pt` (corrupt) in `tools/inference.py`.
- **Target bit size:** 11 MB (Shpak) / 30 MB (Zubr) — 1.58-bit weights compress ~10× vs fp16.
- **Metrics log:** `checkpoints/metrics.jsonl` (one JSON per step, for ETA).
- **Event stream:** `checkpoints/busel.log.jsonl` — structured JSON for downstream (TG bot, web). Events: `training_start`, `model_initialized`, `busel_scaling_planned`, `curriculum_upgrade`, `step_complete`, `checkpoint_saved`/`checkpoint_rejected`/`checkpoint_failed`, `emergency_save_requested`, `emergency_checkpoint`, `stage_complete`, `pipeline_start`/`pipeline_complete`, `stage_failed`, `training_complete`, `autopilot`.
- **Registry kinds:** `attention` (`gdn2`, `mla`), `optimizer` (`lotus_muon`, `norlotus_muon`, `hybrid_muon_adamw`), `autopilot` (`v6`), `curriculum` (`doubling`), `encoder` (`image`, `video`, `audio`, `pdf`, `docx`, `text`), `loss`, `stage` (`pretrain`, `sft`, `dpo`, `eval`).
- **Teto emoticon cycle:** 12-frame kawaii idle loop (`(ᗜˬᗜ)`, `ξ(｡•̀ᴗ-)✧ξ`, `ξ(≧◡≦)ξ`, `▼ᗜˬᗜ▼`, …) — see `ui.teto.frames()`. States: `idle`, `blink`, `smile`, `think`, `wave`, `training`, `done`.
- **macOS Rust flag:** `.cargo/config.toml` uses `link-arg=-undefined,dynamic_lookup` for macOS.
- **License:** CC BY-NC-SA 4.0 (NC clause — NO commercial use). Contact `sehaxe` for commercial licence.
- **LOTUS paper:** arXiv:2602.01233. Rank-r factorised Muon momentum.
- **Muon repo:** github.com/KellerJordan/Muon — original implementation.
