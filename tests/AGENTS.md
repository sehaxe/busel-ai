# tests/ — Test Suite & Step Profiler

**Scope:** `unittest`-based smoke tests + custom stable step profiler (no `torch.profiler` on macOS).

## STRUCTURE
```
tests/
├── test_suite.py     # TestbuselFramework — 8 unittest cases covering all subsystems (205 LOC)
└── profiler_run.py   # StablebuselProfiler — manual step timing w/ memory stats (340 LOC)
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add unit test | `test_suite.py` → new `def test_...(self)` | unittest (not pytest) |
| Profile step time | `profiler_run.py` | Uses `time.perf_counter()`, no `torch.profiler` |
| Add memory metric | `profiler_run.py` → `get_memory_stats` | CUDA / MPS / RSS-by-platform |
| Skip test on CUDA-only | use `cls.device` from `setUpClass` | `mps → cuda → cpu` priority |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `TestbuselFramework` | TestCase | test_suite.py | 8 tests: Rust IO, binary packer, BitLinear, attention, MoE, optimizer, loss, e2e |
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

## NOTES
- **8 test cases (current):**
  1. `test_rust_io_streamer` — `ByteStreamer` mmap correctness
  2. `test_rust_binary_packer` — `append_to_binary_file`
  3. `test_bitlinear_quantization` — forward pass on random input
  4. `test_gdn2_jit_compiles` — JIT script warmup
  5. `test_moe_load_balance_loss` — aux loss computation
  6. `test_muon_orthogonalization` — NS step produces orthogonal output
  7. `test_pretrain_loss_with_mtp` — multi-head loss sums correctly
  8. `test_end_to_end_step` — full model + optimizer + loss step
- **Profiler runs `tests/profiler_run.py` standalone:** Called by `cli.py profile` and `autopilot`
- **Memory in profiler:** Different stats per device — not a single unified schema
- **Step phases measured:** `forward`, `backward`, `optimizer.step`, `autopilot.update_parameters`, `autopilot.inject_noise`
- **Wall-clock budget per test:** 30s default (unittest); profiler has `steps=10` default
- **HF datasets NOT mocked:** Tests use synthetic data (no HF API calls)
