# tests/ — Test Suite & Step Profiler

**Scope:** `unittest`-based smoke tests + custom stable step profiler (no `torch.profiler` on macOS). **175 tests** total (was 171 pre-v8.5; +4 for v8.5 additions: `test_rust_ternary_inference`, `test_fused_glu_and_expert_ffn`, `test_muon_transpose_trick`, `test_complete_backbone_and_gradients`), plus 2 profile scripts (v58_profile.py dual-mode + v63_profile.py 12-experiment kruk A/B).

## STRUCTURE
```
tests/
├── test_suite.py            # TestbuselFramework — 177 def test_* methods (175 unique) (~2.6k LOC)
├── profiler_run.py          # StablebuselProfiler — manual step timing w/ memory stats (~492 LOC)
├── v58_profile.py           # v6.x — 2-mode profile suite (--mode shpak-v60 | shpak-disp)
├── v63_profile.py           # v8.5 — 12-experiment kruk A/B suite (--mode kruk-v85)
├── scaling_laws.py          # Busel scaling law reproduction
└── quick_imu1_profile.py    # Quick IMU-1 vs baseline comparison (~2M params)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add unit test | `test_suite.py` → new `def test_...(self)` | unittest (not pytest) |
| Profile step time | `profiler_run.py` | Uses `time.perf_counter()`, no `torch.profiler` |
| Compare 5 cumulative v6.0 configs on shpak 52.8M | `v58_profile.py --mode shpak-v60` | Baseline / +DA / +DA+Cautious / +DA+Cautious+LCSB / +DA+Cautious+SF+LCSB |
| Compare 4 configs with Dispersion Loss (v6.1) | `v58_profile.py --mode shpak-disp` | Baseline / +Dispersion / +DA+Cautious+LCSB / +DA+Cautious+LCSB+Dispersion |
| Run kruk 65M A/B profiler (12 experiments) | `v63_profile.py --mode kruk-v85` | 12 experiments comparing NS, Muon+, SF, cautious_wd |
| Add memory metric | `profiler_run.py` → `get_memory_stats` | CUDA / MPS / RSS-by-platform |
| Skip test on CUDA-only | use `cls.device` from `setUpClass` | `mps → cuda → cpu` priority |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `TestbuselFramework` | TestCase | test_suite.py | 177 `def test_*` methods (175 unique): Rust IO, BitLinear, GDN-2, MoE, mAR, Muon/NS, patching, optimizer, loss, stages, multimodal, tool executor, sampling, checkpoint, SFT, DPO, eval, research features |
| `StablebuselProfiler` | class | profiler_run.py | Per-step timing (forward/backward/opt/noise) |
| `get_memory_stats` | method | profiler_run.py | `cuda: allocated+peak` / `mps: current` / `cpu: ru_maxrss` |
| `_compiled_newton_schulz` (imported) | function | test_suite.py | Tests Muon NS orthogonalization correctness |

## CONVENTIONS
- **Test framework:** `unittest` (NOT pytest). Discoverable via `python -m unittest tests.test_suite`
- **Device priority in tests:** `mps → cuda → cpu` (Apple Silicon first for dev)
- **Profiler timing:** `time.perf_counter()` for high-resolution wall-clock
- **Profiler avoids `torch.profiler`:** Known to hang on MPS/macOS; manual timing is stable
- **Temp test files:** `temp_test_rust_io.txt` etc.; cleaned up in `finally` block
- **Test data:** Inline strings (e.g. `"Hello from busel Rust IO! " * 350`)
- **Test imports:** `sys.path.insert(0, project_root)` at module top
- **Profiler memory:** Reports `allocated_mb` + `peak_mb` (CUDA), `current` only (MPS), `max_rss_mb` (CPU)
- **`ru_maxrss` units:** MB on Darwin, KB on Linux — `profiler_run.py` handles both

## ANTI-PATTERNS
- **NEVER** use `torch.profiler` in this codebase — known to hang on macOS
- **NEVER** switch to pytest — `test_suite.py` uses `unittest.TestCase` patterns
- **NEVER** leave temp test files behind — always `os.remove` in `finally`
- **NEVER** import `train.py` in tests — heavy dep, prefer testing components in isolation
- **NEVER** assume CUDA in tests — `cls.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"`
- **NEVER** write to `data_train/` from tests — gitignored but pollutes dataset
- **NEVER** test against `targets` > 5K tokens in unit tests — slow; use small synthetic
- **NEVER** add `assertTrue(x == y)` — use `assertEqual` (better failure messages)
- **NEVER** push code with fewer than 175 tests passing — `uv run python -m unittest tests.test_suite` must report `OK` with `Ran 175 tests` (177 `def test_*`, 175 unique — 2 duplicate method names from pre-v8.5 that silently overwrite)

## NOTES
- **175 total unique tests** across 177 `def test_*` methods (2 duplicate names from pre-v8.5 — last definition wins, no runtime impact). Current test coverage:
  - Rust IO: streamer, binary packer, ternary matmul inference
  - BitLinear: quantization, ternary distribution, H_BitLinear, SwishGLU, clamp gradients
  - GDN-2: attention forward, JIT, decoupled erase/write gates, sigmoid bounding, log decay, SeRoPE, latent d_c=128, KV compression dims
  - MoE: load balance loss, fused GLU+expert FFN
  - mAR: Sinkhorn doubly-stochastic (rows, cols, non-neg, overflow, convergence), identity init, shape, gradients, temperature, buselModel integration, residual connection, stream flow
  - Muon/NS: orthogonalization, quintic coefficients, 5-step default, tall/wide matrices, scale formula, momentum=0.95, hybrid routing, MoE+MLA coverage, router/embed exclusion, no-NaN step, transpose trick
  - ByteFlow patching: dynamic vocab, stride=4, no BPE, byte→patch, learned embeddings, GLU gate, causal padding
  - Optimizer: Schedule-Free wrapper, Differential Attn MLA, LCSB selective backward
  - Loss: pretrain+MTP, Dispersion Loss
  - Training: schedule guard, inject noise
  - E2E: complete backbone+gradients
  - Registry: decorator, collision, override, registration completeness
  - Logging: JSONL write/read
  - UI: teto frames, CLI helpers, animated header
  - Multimodal: all 6 encoders, special tokens vocab=326, dynamic vocab, undersized vocab rejection, marker emission, cv2 fast path timing, disable/enable roundtrip, runtime register, legacy marker decode
  - Stages: pretrain registration, lifecycle, config from_profile, bad dmodel rejection, YAML loading, state/spec/config dataclasses, orchestrator command
  - SFT: chat format, mask correctness, DPO pair, dataloader, config from_profile
  - DPO: loss, sequence logprob, dataloader, config from_profile
  - Eval: perplexity, format compliance, stage registration
  - Pipeline: full/quick YAML, CLI commands
  - Tool executor: parse single/multiple invokes, format, strip_ansi, denied_result, wrappers, default registry, execute bash/unknown, interactive_confirm, max constants
  - Checkpoint: strip_compile_prefix, load_state_dict_safely
  - Inference sampling: normal, greedy, special token masking, NaN fallback, repetition penalty, top-p
  - CLI: simpler commands, data presets
  - Research features: Tequila deadzone, Hestia softmax/temperature, SCT spectral/swishglu
- **Profiler runs `tests/profiler_run.py` standalone:** Called by `cli.py profile` and `autopilot`
- **Memory in profiler:** Different stats per device — not a single unified schema
- **Step phases measured:** `forward`, `backward`, `optimizer.step`, `autopilot.update_parameters`, `autopilot.inject_noise`
- **Wall-clock budget per test:** 30s default (unittest); profiler has `steps=10` default
- **HF datasets NOT mocked:** Tests use synthetic data (no HF API calls)
