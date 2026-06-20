# tools/ — CLI, Data Manager, Orchestrator

**Scope:** User-facing entrypoints (Typer CLI), dataset management, training orchestration, plotting, standalone inference, multi-stage pipeline runner, **REPL tool executor**.

## STRUCTURE
```
tools/
├── orchestrator.py    # Typer: autopilot / train / train-all / profile / pipeline / escalate
├── data_manager.py    # Typer: download-* / list-presets / label-vision (HF datasets, COCO)
├── inference.py       # Standalone CLI chat (subprocess entry from cli.py chat)
├── tool_executor.py   # REPL tool registry, sandboxed bash/read, interactive confirm
└── plotter.py         # matplotlib loss/lr/metrics visualization
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add CLI command | `cli.py` (root) + `tools/orchestrator.py` or `tools/data_manager.py` | All commands registered as `@app.command` in cli.py |
| Change dataset presets | `tools/data_manager.py:PRESETS` | shpak/chyzh defined; per-Chinchilla 80 bytes/param |
| Change auto-download behavior | `tools/orchestrator.py:autopilot` | Downloads if `data_train/` empty; profiles HW first |
| Add plot type | `tools/plotter.py` | Reads `checkpoints/metrics.jsonl` |
| Standalone inference | `tools/inference.py` | Checkpoint-first, profile-fallback config loading; multi-pass REPL tool executor |
| **Wire a new tool into the REPL** | `tools/tool_executor.py:default_tool_registry` | Register with `r.register("TOOL_X", fn)`; auto-exposed via `/tools` |
| **Add a new pipeline preset** | `configs/pipelines/<name>.yaml` | Read by `pipeline()` command below |
| **Change pipeline orchestration** | `tools/orchestrator.py:pipeline` | Loads YAML, instantiates stages via `get_stage()`, calls `setup → run → finalize` |

## KEY FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `autopilot` | Typer command | orchestrator.py | One-click: load .env → ensure data → profile HW → launch cli.py pipeline |
| `train` | Typer command | orchestrator.py | Shell-out: `python cli.py pipeline --name pretrain-only --profile X [--resume Y]` |
| `profile` | Typer command | orchestrator.py | Shell-out: `python tests/profiler_run.py` |
| `pipeline` | Typer command | orchestrator.py | Runs a multi-stage pipeline from `configs/pipelines/<name>.yaml`. Loads YAML, instantiates each stage via `get_stage(name)`, runs `setup → run → finalize` in order, logs `pipeline_start`/`stage_start`/`stage_complete`/`pipeline_complete` events to `checkpoints/busel.log.jsonl`. Supports `--start-stage` to resume mid-pipeline. |
| `load_env` | function | orchestrator.py | `.env` file parser (no dotenv lib) |
| `download_all`/`-text`/`-sft`/`-vision` | Typer command | data_manager.py | HF streaming → Parquet/JSONL in `data_train/` |
| `label_vision` | Typer command | data_manager.py | Auto-label local image dir via local Ollama vision model |
| `_download_vision` (COCO fix) | function | data_manager.py | Auto-rewrites `HuggingFaceM4/COCO` → `jxie/coco_captions` (has PIL) |
| `InferenceConfig` | class | inference.py | Checkpoint-first, profile-fallback config loading (self-contained) |
| `resolve_config` | function | inference.py | Checkpoint-first, profile-fallback config loading |
| `interactive_loop` | function | inference.py | Multi-pass REPL. Generates → parses `<function_calls>` → asks user (default N = deny) → executes/denies → injects `<function_results>` → re-prompts. Up to `MAX_TOOL_CALLS_PER_TURN=5` rounds per user turn. |
| `default_tool_registry` | function | tool_executor.py | `ToolRegistry` pre-loaded with `TOOL_BASH` (sandboxed `subprocess.run(shell=True, timeout=10s)`) and `TOOL_READ` (32 KB cap). To add a tool: `r.register("TOOL_X", fn)`. |
| `parse_tool_calls` | function | tool_executor.py | Finds `<function_calls>...<invoke>...</invoke>...</function_calls>` envelopes (handles MULTIPLE invokes per envelope). Returns `[{name, params}, ...]`. |
| `interactive_confirm` | function | tool_executor.py | Sync `input("[?] Execute TOOL_X(args) ? [y/N]: ")`. Returns `True` only on `'y'`/`'yes'`. EOF/KeyboardInterrupt/`'n'`/empty return `False` (safe default). |
| `format_tool_call_for_user` | function | tool_executor.py | Pretty-print a tool call for the `[?]` prompt, truncating long values to 120 chars and stripping ANSI escapes. |
| `denied_result` | function | tool_executor.py | Synthetic `{output: "ERROR: tool execution denied by user"}` so the model sees its call was rejected. |
| `strip_ansi` | function | tool_executor.py | Remove CSI / OSC / lone-ESC sequences from tool output (defensive against escape injection). |

## CONVENTIONS
- **CLI framework:** Typer (not Click). All commands defined as `@app.command(name="...")`
- **Subprocess pattern:** `orchestrator.py` uses `subprocess.run([sys.executable, "tests/profiler_run.py", ...])` for the profiler; training runs in-process via stage framework
- **Path resolution:** `tools/X.py` prepends `project_root` to `sys.path` (handles subdir execution)
- **Env var loading:** `load_env()` parses `.env` manually (no `python-dotenv` dep)
- **HF dataset choice:** `jxie/coco_captions` preferred for vision (returns PIL images natively, not base64)
- **PyArrow threading:** `pyarrow.set_cpu_count(1)` set on import to avoid GIL conflicts on shutdown
- **Chinchilla preset math:** `text_limit = 80 × N_params ÷ avg_bytes_per_token`; vision_limit = 1k/200 for shpak/chyzh
- **Plotter input:** Reads `checkpoints/metrics.jsonl` (one JSON per training step)

## ANTI-PATTERNS
- **NEVER** `import train` from `orchestrator.py` — always `subprocess.run([sys.executable, "cli.py", ...])`
- **NEVER** use `python-dotenv` — project has its own `load_env()` parser
- **NEVER** add `if __name__ == "__main__"` blocks that auto-run training — these are CLI tools
- **NEVER** hardcode HuggingFace dataset names that don't return PIL — use `jxie/coco_captions` for COCO
- **NEVER** skip the `Profile` step in `autopilot` — HW profiling catches VRAM/RNG bugs early
- **NEVER** call `cli.py` with Python directly — use `uv run python cli.py` (maturin ext needs venv)
- **NEVER** exceed `chunk_size` from config — it's the model context boundary (no padding logic in loader)
- **NEVER** set `max_steps` < `warmup_steps` in any preset — produces NaN spikes
- **NEVER** modify `data_train/` directly without gitignoring first — dirs are already in .gitignore
- **NEVER** execute a model-emitted tool call without user confirmation — the REPL's `interactive_confirm` is the gate. The only escape hatch is the per-session `/auto on` toggle, which the user must opt into.
- **NEVER** call `model.load_state_dict(sd)` directly in `tools/inference.py` or `tools/eval.py` — always use `model.checkpoint.load_state_dict_safely`. `inference.py` uses `strict=False` to print a diagnostic of any missing/unexpected keys (most common: vocab-size mismatch when a pre-v5.4 checkpoint is loaded).
- **NEVER** set `stop_on_newline=True` inside the multi-pass tool loop in `interactive_loop` — tool-call envelopes span multiple lines, and the model would be cut off mid-`</function_calls>`.

## NOTES
- **`autopilot` data bootstrap:** If `data_train/` empty, downloads text/SFT/vision in sequence before profiling
- **Profile gate:** `subprocess.run(profiler)` MUST return 0 or `autopilot` exits with code 1
- **`chat` CLI command:** Defined in `cli.py` directly (subprocesses `tools/inference.py`)
- **17 CLI commands** registered in `cli.py`: `download-all`, `download-vision`, `download-text`, `download-sft`, `download-preset`, `list-presets`, `download-data`, `download-multimodal`, `label-vision`, `autopilot`, `train`, `train-all`, `pipeline`, `escalate`, `stop`, `profile`, `chat`.
- **Data path:** `data_train/` is both source and processed output; PDFs auto-converted by `data/pipeline.py` via Docling
- **Plotter file:** See `tools/plotter.py` for exact metrics.jsonl schema (loss, lr, step, grad_norm, moe_aux)
- **Typer `help`:** All commands have emoji-prefixed `help="..."` for rich output
- **REPL tool flow:** The `interactive_loop` is multi-pass — when the model emits `<function_calls>...</function_calls>`, the REPL asks the user, executes or denies, injects `<function_results>...</function_results>`, and re-prompts. Up to 5 tool-call rounds per user turn. The user can `/auto on` to skip the prompt (not recommended). Special tokens (TOOL_CALLS_START etc.) are MASKED in the sampling logits (see `apply_sampling` in `inference.py`), so the model emits the LITERAL byte sequence for the envelope.
- **Inference checkpoint load:** `tools/inference.py` calls `load_state_dict_safely(model, ckpt["model_state_dict"], strict=False)` and prints a 2-line banner reporting `keys_loaded=N, missing=K, unexpected=U`. If `K > 0` or `U > 0`, the user sees a warning — common case is loading a pre-v5.4 259-vocab checkpoint into a 326-vocab model (the patcher `embed_weight` shape is `(326, d_byte)` vs `(259, d_byte)`).
