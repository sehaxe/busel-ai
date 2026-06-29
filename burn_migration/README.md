# busel-burn — Rust/Burn 0.21 1.58-bit LLM

## Progressive Slowdown (documented 2026-06-28)

### Symptom
Backward pass grows from ~14ms to ~360ms over 500 steps. Optimizer step stays flat at ~13ms. GPU util 3-4% (CPU-bound).

```
step   100: 81ms avg  (backward ~60ms)
step   500: 178ms avg (backward ~150ms)
step  1000: 506ms avg (backward ~360ms)
```

Super-linear growth (not linear).

### What was ruled out
- **Burn autograd tape accumulation** — confirmed clean after deep source code audit. `detach().require_grad()` properly strips graphs (`burn-autodiff-0.21.0/src/tensor.rs:96-118`). Backward BFS traversal (`graph/traversal.rs:20-63`) consumes all reachable nodes from loss root via `steps.remove()`. `GraphMemoryManagement::free_unavailable_nodes()` (`memory_management.rs:56-90`) removes all nodes with `Arc::strong_count == 1`. Only param nodes persist. No tape leak.
- **CPU memory leak** — RSS stable at 2.2 MB.
- **nvidia-smi subprocess** — removed, not the cause (slowdown persisted).
- **GPU clocks** — jumps 855-2032 MHz (not stuck at idle), power ~10W. GPU just isn't busy enough (model too small).
- **Backend::sync()** — added 682ms/step overhead, made everything worse.
- **memory_cleanup()** — no effect.

### Suspected cause (not confirmed)
CUDA/Vulkan driver state accumulation. Burn uses cubecl → Vulkan → CUDA. Without a caching memory allocator (unlike PyTorch), kernel launch overhead, command buffer size, or memory allocation fragmentation could increase super-linearly. But not proven.

### To reproduce
```bash
cargo run -- vorobey --bs 256 --cs 256 --steps 200
```
Watch `step_inner bwd` in per-step timing output grow over time.

### Possible workarounds (untested)
1. Periodic device reset (re-create cubecl device every N steps)
2. Use `Backend::sync()` sparingly (only every 100 steps, not every step)
3. Profile with NVIDIA Nsight Systems to count kernel launches
4. File issue against `burn-cubecl` with minimal reproducer

### When to fix
After vorobey completes (~27 min for 2594 steps). The slowdown adds ~2-3× to training time but doesn't cause incorrectness.
