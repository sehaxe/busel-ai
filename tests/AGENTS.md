# tests/ — Test Suite & Step Profiler

**Scope:** `unittest`-based smoke tests + custom stable step profiler (no `torch.profiler` on macOS). **171 tests** total (was 166 pre-v5.8; +2 in v5.8 for GradLite, LCSB; −1 in v6.0 cleanup for GradLite removal; +3 in v6.0 for Schedule-Free, Cautious, Differential Attn; +1 in v6.1 for Dispersion Loss; −1 in v6.2 cleanup for Sparse-BitNet 6:8 removal), plus a 2-mode shpak comparison script (v6.0 cumulative + v6.1 dispersion).

## STRUCTURE
```
tests/
├── test_suite.py            # TestbuselFramework — 13 unittest cases (171 total via subTest loops) (~205 LOC)
├── profiler_run.py          # StablebuselProfiler — manual step timing w/ memory stats (~340 LOC)
└── v58_profile.py           # v6.x — 2-mode profile suite (--mode shpak-v60 | shpak-disp)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add unit test | `test_suite.py` → new `def test_...(self)` | unittest (not pytest) |
| Profile step time | `profiler_run.py` | Uses `time.perf_counter()`, no `torch.profiler` |
| Compare 5 cumulative v6.0 configs on shpak 52.8M (baseline / +DA / +DA+Cautious / +DA+Cautious+LCSB / +DA+Cautious+SF+LCSB) | `v58_profile.py --mode shpak-v60` | **🆕 v6.0** — One sweep for the best v6.0 config. Final +DA+Cautious+SF+LCSB is the full stack. |
| Compare 4 configs with Dispersion Loss (baseline / +Dispersion / +DA+Cautious+LCSB / +DA+Cautious+LCSB+Dispersion) | `v58_profile.py --mode shpak-disp` | **🆕 v6.1** — Validates Wang 2026 Dispersion Loss on token embeddings. Final config is the v6.1 winner. |
| Add memory metric | `profiler_run.py` → `get_memory_stats` | CUDA / MPS / RSS-by-platform |
| Skip test on CUDA-only | use `cls.device` from `setUpClass` | `mps → cuda → cpu` priority |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `TestbuselFramework` | TestCase | test_suite.py | 13 tests (166→171 cumulative: +2 v5.8, −1 v6.0 cleanup, +3 v6.0 research, +1 v6.1, −1 v6.2 cleanup): Rust IO, binary packer, BitLinear, attention, MoE, optimizer, loss, e2e, **LCSB**, **Schedule-Free**, **Cautious**, **Differential Attn**, **Dispersion Loss** |
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
- **NEVER** push code with fewer than 171 tests passing — `uv run python -m unittest tests.test_suite` must report `OK` with `Ran 171 tests` (was 166 pre-v5.8; +2 in v5.8 for GradLite, LCSB; −1 in v6.0 cleanup for GradLite removal; +3 in v6.0 for SF, Cautious, DA; +1 in v6.1 for Dispersion; −1 in v6.2 cleanup for Sparse-BitNet 6:8)

## NOTES
- **171 total tests** across 13 named test methods (the 13 methods are parameterized into 171 sub-tests via `subTest` and inner loops). The named methods are:
  1. `test_rust_io_streamer` — `ByteStreamer` mmap correctness
  2. `test_rust_binary_packer` — `append_to_binary_file`
  3. `test_bitlinear_quantization` — forward pass on random input
  4. `test_gdn2_jit_compiles` — JIT script warmup
  5. `test_moe_load_balance_loss` — aux loss computation
  6. `test_muon_orthogonalization` — NS step produces orthogonal output
  7. `test_pretrain_loss_with_mtp` — multi-head loss sums correctly
  8. `test_end_to_end_step` — full model + optimizer + loss step
  9. `test_lcsb_selective_backward` — **🆕 v5.8** — `buselModel(selective_backward=True, backward_ratio=0.5)` on n_layers=6 selects 3 layers, gradients non-NaN, `_selected_layers` set correctly
  10. `test_schedule_free_wrapper` — **🆕 v6.0** — 5-step SF sanity check: state['x'/'z'/'t'] present, no NaN, loss decreased, state_dict round-trip works
  11. `test_cautious_wrapper` — **🆕 v6.0** — 5-step Cautious sanity check: no NaN, loss decreased, state_dict round-trip works
  12. `test_differential_attention_mla` — **🆕 v6.0** — DA inside MLA: param count > std MLA, forward+backward non-NaN, gradients flow
  13. `test_dispersion_loss` — **🆕 v6.1** — uniformity loss on L2-normalised embeddings: spread embeddings give lower loss than collapsed, gradients flow (no NaN), non-zero grads
- **Profiler runs `tests/profiler_run.py` standalone:** Called by `cli.py profile` and `autopilot`
- **Memory in profiler:** Different stats per device — not a single unified schema
- **Step phases measured:** `forward`, `backward`, `optimizer.step`, `autopilot.update_parameters`, `autopilot.inject_noise`
- **Wall-clock budget per test:** 30s default (unittest); profiler has `steps=10` default
- **HF datasets NOT mocked:** Tests use synthetic data (no HF API calls)
