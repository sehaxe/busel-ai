# tools/ — CLI, Data Manager, Orchestrator

**Scope:** User-facing entrypoints (Typer CLI), dataset management, training orchestration, plotting, standalone inference.

## STRUCTURE
```
tools/
├── orchestrator.py    # Typer: autopilot/train/profile (all shell out via subprocess)
├── data_manager.py    # Typer: download-all/-vision/-text/-sft, label-vision (HF datasets, COCO)
├── inference.py       # Standalone CLI chat (subprocess entry from cli.py chat)
└── plotter.py         # matplotlib loss/lr/metrics visualization
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add CLI command | `cli.py` (root) + `tools/orchestrator.py` or `tools/data_manager.py` | All commands registered as `@app.command` in cli.py |
| Change dataset presets | `tools/data_manager.py:PRESETS` | shpak/chyzh defined; per-Chinchilla 80 bytes/param |
| Change auto-download behavior | `tools/orchestrator.py:autopilot` | Downloads if `data_train/` empty; profiles HW first |
| Add plot type | `tools/plotter.py` | Reads `checkpoints/metrics.jsonl` |
| Standalone inference | `tools/inference.py` | Checkpoint-first, profile-fallback config loading |

## KEY FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `autopilot` | Typer command | orchestrator.py | One-click: load .env → ensure data → profile HW → launch train.py |
| `train` | Typer command | orchestrator.py | Shell-out: `python train.py --profile X [--resume Y]` |
| `profile` | Typer command | orchestrator.py | Shell-out: `python tests/profiler_run.py` |
| `load_env` | function | orchestrator.py | `.env` file parser (no dotenv lib) |
| `download_all`/`-text`/`-sft`/`-vision` | Typer command | data_manager.py | HF streaming → Parquet/JSONL in `data_train/` |
| `label_vision` | Typer command | data_manager.py | Auto-label local image dir via local Ollama vision model |
| `_download_vision` (COCO fix) | function | data_manager.py | Auto-rewrites `HuggingFaceM4/COCO` → `jxie/coco_captions` (has PIL) |
| `InferenceConfig` | class | inference.py | Checkpoint-first, profile-fallback config loading (self-contained) |
| `resolve_config` | function | inference.py | Checkpoint-first, profile-fallback config loading |

## CONVENTIONS
- **CLI framework:** Typer (not Click). All commands defined as `@app.command(name="...")`
- **Subprocess pattern:** `orchestrator.py` ALWAYS `subprocess.run([sys.executable, "train.py", ...])` — never imports train.py directly
- **Path resolution:** `tools/X.py` prepends `project_root` to `sys.path` (handles subdir execution)
- **Env var loading:** `load_env()` parses `.env` manually (no `python-dotenv` dep)
- **HF dataset choice:** `jxie/coco_captions` preferred for vision (returns PIL images natively, not base64)
- **PyArrow threading:** `pyarrow.set_cpu_count(1)` set on import to avoid GIL conflicts on shutdown
- **Chinchilla preset math:** `text_limit = 80 × N_params ÷ avg_bytes_per_token`; vision_limit = 1k/200 for shpak/chyzh
- **Plotter input:** Reads `checkpoints/metrics.jsonl` (one JSON per training step)

## ANTI-PATTERNS
- **NEVER** `import train` from `orchestrator.py` — always `subprocess.run([sys.executable, "train.py", ...])`
- **NEVER** use `python-dotenv` — project has its own `load_env()` parser
- **NEVER** add `if __name__ == "__main__"` blocks that auto-run training — these are CLI tools
- **NEVER** hardcode HuggingFace dataset names that don't return PIL — use `jxie/coco_captions` for COCO
- **NEVER** skip the `Profile` step in `autopilot` — HW profiling catches VRAM/RNG bugs early
- **NEVER** call `cli.py` with Python directly — use `uv run python cli.py` (maturin ext needs venv)
- **NEVER** exceed `chunk_size` from config — it's the model context boundary (no padding logic in loader)
- **NEVER** set `max_steps` < `warmup_steps` in any preset — produces NaN spikes
- **NEVER** modify `data_train/` directly without gitignoring first — dirs are already in .gitignore

## NOTES
- **`autopilot` data bootstrap:** If `data_train/` empty, downloads text/SFT/vision in sequence before profiling
- **Profile gate:** `subprocess.run(profiler)` MUST return 0 or `autopilot` exits with code 1
- **`chat` CLI command:** Defined in `cli.py` directly (subprocesses `tools/inference.py`)
- **Data path:** `data_train/` is both source and processed output; PDFs auto-converted by `data/pipeline.py` via Docling
- **Plotter file:** See `tools/plotter.py` for exact metrics.jsonl schema (loss, lr, step, grad_norm, moe_aux)
- **Typer `help`:** All commands have emoji-prefixed `help="..."` for rich output
