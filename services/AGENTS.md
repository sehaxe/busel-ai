# services/ — FastAPI Inference Server

**Scope:** HTTP API for serving trained model checkpoints. `uvicorn` entrypoint.

## STRUCTURE
```
services/
└── inference_api.py   # FastAPI app, InferenceConfig, checkpoint resolution, generation (336 LOC)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add API endpoint | `inference_api.py` → `@app.post("/...")` | FastAPI w/ Pydantic models |
| Change checkpoint resolution | `inference_api.py:resolve_config` | Checkpoint-first, profile-fallback |
| Add generation strategy | `inference_api.py` | Currently has sampling params; top-k/top-p logic |
| Run server | `cli.py serve` (preferred) or `uvicorn services.inference_api:app` | See root AGENTS.md commands |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `app` | FastAPI | inference_api.py | Top-level uvicorn target |
| `InferenceConfig` | class | inference_api.py | Pydantic-ish config (d_model, n_layers, n_heads, expert_hidden, num_experts, top_k, vocab_size) |
| `resolve_config` | function | inference_api.py | Loads from checkpoint `cfg["config"]` OR `configs/default.yaml` profile |
| `find_latest_checkpoint` | function | inference_api.py | Globs `checkpoints/busel_*.pt`, returns newest mtime |
| `detect_device` | function | inference_api.py | `cuda → mps → cpu` |

## CONVENTIONS
- **Framework:** FastAPI + uvicorn (NOT Starlette directly, NOT Flask)
- **Port default:** 8000; host default: 127.0.0.1 (override via `cli.py serve --host --port`)
- **Checkpoint size guard:** Reject `.pt` < 10MB as corrupted (`os.path.getsize < 10_000_000`)
- **Checkpoint pattern:** `checkpoints/busel_*.pt` (prefix `busel_`)
- **Config resolution order:** `checkpoint["config"]` → `configs/default.yaml:profile` → fallback to `shpak`
- **Device resolution:** CUDA > MPS > CPU
- **Inference call pattern:** Reuse `model.patching.StridedFastBLTPatcher` + `model.backbone.buselModel`
- **Tokenization:** Raw bytes (UTF-8) → `StridedFastBLTPatcher` (vocab=259)
- **Min prompt length:** 128 bytes (left-pad with 0x00 if shorter)

## ANTI-PATTERNS
- **NEVER** load checkpoint without size guard — files < 10MB are corrupt, refuse with sys.exit(1)
- **NEVER** skip `weights_only=False` in `torch.load` for inference (config dict needs to deserialize)
- **NEVER** start the FastAPI app via `__main__` — call via `uvicorn services.inference_api:app`
- **NEVER** hardcode `d_model=1536` — always load from `cfg.d_model` (varies per profile)
- **NEVER** return token IDs in API response — return decoded UTF-8 strings only
- **NEVER** serve without `reload=False` in production — set in `cli.py serve`
- **NEVER** cache the model in a global without lock — concurrent requests will race the GPU
- **NEVER** trust user-supplied profile names — validate against `full_cfg["profiles"]` keys
- **NEVER** use `weights_only=True` — checkpoint contains Python objects (config dict, profile name)

## NOTES
- **Standalone inference:** `tools/inference.py` mirrors `resolve_config` — keep them in sync if config shape changes
- **Checkpoint format:** `{model_state_dict, patcher_state_dict, step, file_idx, byte_offset, reason, config, profile}` (see `train.py:save_bot_stopped_checkpoint` for stop case; normal saves in `train.py:main`)
- **CORS:** Not configured (assumed same-origin / behind reverse proxy)
- **Auth:** None (designed for localhost / trusted network)
- **Streaming:** Not implemented (returns full response after generation completes)
- **GPU lock:** Currently no lock around model inference — add if concurrent requests cause OOM
- **Error response shape:** `{"error": "..."}` for client errors; FastAPI default 500 for unhandled
- **Health check:** No `/health` endpoint yet (add `@app.get("/health")` for k8s probes)
- **Configurable sampling:** Top-k, top-p, temperature typically exposed as request body params
