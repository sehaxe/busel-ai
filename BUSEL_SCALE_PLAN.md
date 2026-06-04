# Busel-Scale Plan: Phase 3 + Phase 4

> **Goal:** build the "perfect architecture" for training models significantly larger than the current 50-150M parameters (target range: 100M-1B+) while preserving the quality of the Muon optimizer and not making anything worse.

## Context: Current State

### What already works (verified)

| Component | Size / Value | Status |
|---|---|---|
| BitLinear_a4_8 (1.58-bit weights) | Shpak ~55M, Zubr ~150M params | in main |
| MoE (2 shared + N routed, Top-2) | 8 layers, experts in each block | in main |
| mAR (Birkhoff polytope) | Sinkhorn-Knopp x 3 | in main |
| GDN-2/MLA hybrid attention | 3:1 ratio | in main |
| MTP-4 (decaying weights) | 4 heads | in main |
| Muon+AdamW hybrid | Newton-Schulz 5 iter, momentum=0.95 | in main |
| torch.compile in train.py | 3.05x speedup (per README) | in main |
| Migration #5 (inference compile) | 1.62x speedup, byte-identical output | in working tree (uncommitted) |

### What blocks scaling

| Problem | Current value | Effect |
|---|---|---|
| VRAM optimizer state | 1.5 GB for 50M params | grows linearly with model size |
| Compute on long ctx from step 1 | full ctx from start | 20-30% wasted compute |
| Compute on deep model from step 1 | full depth from start | 30-40% wasted compute |
| 6-GPU topology | absent | blocks >500M params |

### Target size (open question for user)

| Variant | What it means for the plan |
|---|---|
| 100M-300M | current x2-3, Phases 3+4 useful |
| 1B-3B | current x10-20, Phases 3+4 mandatory |
| 7B+ | needs Phase 5 (multi-GPU), 1xRTX 5060 insufficient |

---

## Phase 3: LOTUS+Muon

**Idea:** store the momentum as two low-rank factors P (d1 x r) and Q (d2 x r) instead of a full (d1 x d2) matrix. On every step, the gradient is projected into this subspace via a randomized sketch, then reconstructed and orthogonalized with Newton-Schulz.

### Why Muon specifically (not GaLore)

| | Muon | GaLore | LOTUS+Muon |
|---|---|---|---|
| Convergence vs Adam (paper claims) | +30-40% | +0-10% | +25-35% |
| Memory (50M params, r=8) | 1.5 GB | 100 MB | 100 MB |
| Implementation complexity | medium | low | medium |

**Decision:** GaLore is excluded by user choice. Focus on Muon quality, no low-rank Adam. The combination "Muon quality + GaLore-class memory" is precisely what LOTUS+Muon delivers.

### Algorithm (pseudocode)

```python
# Init
buf_p = torch.zeros(d1, r)  # left factor
buf_q = torch.zeros(d2, r)  # right factor

# Per step
buf_p = momentum * buf_p + grad @ buf_q    # (d1, r)
buf_q = momentum * buf_q + grad.T @ buf_p  # (d2, r)
m_approx = buf_p @ buf_q.T                  # reconstruction (d1, d2)
O = Newton_Schulz(m_approx)                 # orthogonalize
p -= lr * scale * O
```

### What we implement

| File | Change | LOC |
|---|---|---|
| `training/optimizer.py` | new `LotusMuon` class + `@register("optimizer", "lotus_muon")` | ~120 |
| `cli.py` | flag `--optimizer {muon\|adamw\|lotus_muon}` | ~20 |
| `/tmp/lotus_test.py` | benchmark: 100 steps on validation profile | ~100 |

**Total: ~240 LOC, 1 day**

### Pass / fail criteria (100 steps, validation profile)

| Metric | Baseline (Muon) | Pass (LOTUS) | Fail |
|---|---|---|---|
| Peak VRAM optimizer | 1.5 GB | < 400 MB | > 1.2 GB |
| Loss @ step 100 | L* | L* +/- 5% | > 1.10 x L* |
| Step time (ms) | T | T x (1.0 +/- 0.10) | > 1.15 x T |
| Compile time (one-time) | 84s | < 300s | > 600s |

**If any fail metric triggers, drop the phase and proceed to P4 without LOTUS.**

### Hyperparameters to test

- `rank=8` (default); also test r=4 and r=16
- `lr_scale=0.5` (paper recommendation); also 0.25 and 1.0
- `momentum=0.95` (preserve current)
- `ns_steps=5` (preserve current)

### Risk: GUM (unbiased sampler) deliberately excluded

GUM adds layerwise importance sampling to compensate for low-rank truncation. It costs ~200 MB of sampling weights per parameter set, which on our scale cancels out the LOTUS memory win. It becomes useful again above ~500M params where the relative overhead shrinks.

---

## Phase 4a: GrowLength

**Idea:** progressively increase the max context. Start at 1024 ctx, expand to 2048, 4096, 8192 on a schedule. At each step the model learns to use the current ctx, then we expand. Proven curriculum technique (GrowLength paper, arXiv 2310.00576).

### What we implement

| File | Change | LOC |
|---|---|---|
| `training/stages/pretrain.py` | parameter `ctx_schedule: list[tuple[int, int]]` | ~30 |
| `cli.py` | flag `--ctx-schedule "0:1024,2500:2048,5000:4096,7500:8192"` | ~15 |
| `configs/default.yaml` | three presets (aggressive / moderate / gentle) | ~10 |

**Total: ~55 LOC, 0.5 day**

### Algorithm (pseudocode)

```python
def current_max_ctx(self, step: int) -> int:
    target = self.initial_ctx
    for boundary, ctx in sorted(self.ctx_schedule, key=lambda x: x[0]):
        if step >= boundary:
            target = ctx
    return target

# In training loop:
chunk = min(self.chunk_size, current_max_ctx(step))
```

### Pass / fail criteria

| Metric | Baseline (fixed 4096) | Pass (GrowLength) |
|---|---|---|
| Final loss (same wall time) | L* | L* +/- 2% |
| Compute (FLOPs to L*) | X | < 0.80 x X |

**No fail criteria** — proven technique, low risk. Even a regression of 2% would still be acceptable given the compute savings.

### Presets

```yaml
gentle:     { 0: 1024, 1500: 2048, 3000: 4096 }
moderate:   { 0: 1024, 1000: 2048, 2000: 4096, 3000: 6144 }
aggressive: { 0: 1024,  500: 2048, 1500: 4096, 3000: 8192 }
```

---

## Phase 4b: G_stack

**Idea:** start with 3 layers, grow to 5 mid-training, grow to 8 near the end. New layers are copied from the last existing layer with rescaled init, then warmup-only (AdamW for 200 steps, no Muon momentum). NeurIPS 2024 Spotlight paper (arXiv 2405.15319).

### What we implement

| File | Change | LOC |
|---|---|---|
| `model/backbone.py` | `buselModel.grow(num_new_layers: int, position: str = "middle")` | ~80 |
| `model/layers.py` | copy-and-rescale init for new layers | ~40 |
| `train.py` | handler for `--grow-schedule "0:3,5000:5,10000:8"` | ~50 |
| `training/optimizer.py` | warmup-only mode for new params (AdamW first 200 steps) | ~50 |
| `cli.py` | flag `--grow-schedule` | ~15 |

**Total: ~235 LOC, 2 days**

### Algorithm (pseudocode)

```python
def buselModel.grow(self, num_new_layers: int, position: str = "middle"):
    # 1. Create N new layers (copy of last, scale=0.1)
    new_layers = [copy_layer(self.layers[-1], scale=0.1)
                  for _ in range(num_new_layers)]

    # 2. Insert in the middle (paper: best convergence vs append)
    if position == "middle":
        mid = len(self.layers) // 2
        self.layers = (self.layers[:mid] + new_layers
                       + self.layers[mid:])

    # 3. Mark new params for warmup-only
    for layer in new_layers:
        for p in layer.parameters():
            self._new_params.add(id(p))
```

### Pass / fail criteria

| Metric | Baseline (fixed 8 layers) | Pass (G_stack 3->8) | Fail |
|---|---|---|---|
| Final loss | L* | L* +/- 3% | > 1.10 x L* |
| Spike magnitude at grow | — | < 1.5 x current loss | > 2.0 x current loss |
| Recovery (steps to back-to-pre-spike) | — | < 500 steps | > 2000 steps |
| Compute (FLOPs to L*) | X | < 0.70 x X | > 0.95 x X |

**If fail, drop the phase.**

### Main risk: loss spike at grow event

New layers start with arbitrary initialization. Mitigations:

- **Copy-and-rescale init** — initialize from the last existing layer with scale 0.1
- **Warmup-only optimizer** — AdamW only for the first 200 steps, no Muon momentum
- **Layer-wise LR multiplier** — new layers get 0.5x LR for the first 500 steps
- **Stricter gradient clipping** — 0.5 instead of 1.0 for the first 200 steps after grow

---

## Combined Plan (5 days, ~530 LOC)

| Day | Phase | Effort | What we get |
|---|---|---|---|
| 1 | P3 LOTUS+Muon | 1 day, 240 LOC | -1.2 GB VRAM, headroom for everything |
| 2 | P4a GrowLength | 0.5 day, 55 LOC | -25% compute, same final model |
| 3-4 | P4b G_stack | 2 days, 235 LOC | -35% compute, can grow 3->8 layers |
| 5 | Integration + tests + cleanup | 1 day, 100 LOC | feature ready in main |

---

## Combined Effect (after all 3 phases)

| Metric | Baseline | After Phases 3+4 | Delta |
|---|---|---|---|
| Optimizer state (50M params) | 1.5 GB | ~100 MB | 15x down |
| Peak ctx achievable | 4096 | 8192 | 2x up |
| Peak layers achievable | 8 (fixed) | 8 (grown from 3) | new capability |
| Compute to L* (FLOPs) | X | ~0.6 X | 40% down |
| Peak VRAM | 14-15 GB | 8-10 GB | 33% down |
| Final loss | L* | L* +/- 5% (to be measured) | paper claim: no regression |

---

## What we are NOT doing (and why)

| Technique | Reason |
|---|---|
| **GaLore** | User decision: Muon focus. GaLore is ~Adam quality, loses to LOTUS+Muon on both axes (memory and quality) at comparable complexity. |
| **GUM (unbiased sampler)** | +200 MB sampling overhead, eats the LOTUS win on our scale. Returns to the table above ~500M params. |
| **Unbalanced OT (UOT)** | Needs extra hyperparameters (mass regularizer), complicates mAR. Current balanced mAR works. |
| **Horizon-LM** | Paper WITHDRAWN by author (TFLOPS calculation error). Do not implement. |
| **Phase 5 (multi-GPU NCCL)** | 1x RTX 5060 Ti, no hardware. `AbstractDeviceMesh` abstraction (~50 LOC) is deferred until GPUs arrive. |

---

## When we bail

- **LOTUS:** if loss > 1.10 x baseline OR step time > 1.15 x baseline
- **GrowLength:** do not bail (proven technique, low risk)
- **G_stack:** if loss spike > 2.0 x on 3 consecutive grows, OR recovery > 2000 steps

---

## Open questions for the user

1. **Target model size:** 100M-300M? 1B-3B? 7B+? Needed for accurate VRAM numbers and for picking the right ctx target.
2. **Target ctx:** 4096? 8192? 16k? Affects the GrowLength schedule and the gain.
3. **Migration #5 (inference compile):** commit it separately as its own PR, or include it in the Phase 3 PR? Migration #5 is verified (1.62x speedup, byte-identical), uncommitted, sitting in working tree.
4. **Uncommitted working tree:** `tools/inference.py` (Migration #5) and `uv.lock` — what should happen to them?

---

## Validated paper references

| Paper | arXiv | Status |
|---|---|---|
| Lotus | 2602.01233 | real, Feb 2026 |
| GUM | 2510.17802 | real, Oct 2025 |
| GrowLength | 2310.00576 | real, Oct 2023 |
| G_stack (NeurIPS 2024 Spotlight) | 2405.15319 | real, May 2024 |
| Horizon-LM | 2602.04816 | WITHDRAWN — not implemented |
