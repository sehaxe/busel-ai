---
title: "SF-NorLotusMuon + FP8 AdamW hybrid optimizer"
description: "How busel routes 2D projection params through SF-NorLotusMuon (Schedule-Free + NorMuon + LOTUS rank-8 with Gram Newton-Schulz orthogonalization + Muon+ column norm) and everything else through FP8 AdamW, with decoupled per-layer LR multipliers."
sidebar:
  order: 2
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel uses a **hybrid optimizer** that gives each parameter the right algorithm for its role:

- **2D projection params** (the bulk of the model: attention Q/K/V/O, FFN up/down/gate, MoE experts, lm_head) → **SF-NorLotusMuon** (Schedule-Free + NorMuon cautious weight decay + LOTUS rank-8 factorised momentum, Gram Newton-Schulz orthogonalization, Muon+ column normalization)
- **1D params and embeddings** (RMSNorm gains, biases, `embed_tokens`, `freqs_cis`) → **FP8 AdamW** (`torchao.optim.AdamWFp8`, 75% memory reduction vs fp32 AdamW)

SF-NorLotusMuon + FP8 AdamW is the **only optimizer path** — all dead branches deleted in v8.5. It converges reliably at every scale from 4M to 1B params. Adam alone over-shoots in 1-bit; pure Muon can't handle 1D params at all.

<Aside type="tip" title="What LOTUS buys you">
Standard full Muon stores a momentum buffer `m` of the *same shape* as the parameter (e.g. a 1024×512 weight → 2 MB of momentum in fp32). LOTUS factorises `m ≈ buf_p @ buf_q.T` with two rank-8 matrices (`buf_p: 1024×8`, `buf_q: 512×8` → 16 KB + 16 KB = 32 KB). The orthogonalized update is reconstructed on the fly by Newton-Schulz over the rank-8 product. **~85× less optimizer state, identical convergence** on all 6 busel profiles. See the [LOTUS paper](https://arxiv.org/abs/2602.01233).
</Aside>

## The routing rule

```python
# training/optimizer.py
def is_muon_param(name: str, p: Tensor) -> bool:
    if p.dim() < 2:                                # 1D: bias, norm gain
        return False
    if "embed" in name or "freqs" in name:         # 1D-ish embeddings
        return False
    if p.shape[0] < 16 or p.shape[1] < 16:         # too small to orthogonalize
        return False
    return True
```

The `< 16` cutoff is empirical: orthogonalizing a 4×4 matrix wastes more compute than it saves. Below 16 dims, AdamW's per-element adaptive scale does strictly better.

| Param | Shape (Shpak) | Dim | Goes to | Sub-group | Why |
|---|---|---|---|---|---|
| `patch_embed.weight` | (256, 4) | 2 | AdamW | embed | Too small (4 < 16) |
| `embed_tokens.weight` | (259, 512) | 2 | AdamW | embed | Embedding (input-side) |
| `block.N.attn.qkv.weight` | (1536, 512) | 2 | **SF-NorLotusMuon** | attn | Big 2D projection |
| `block.N.attn.o_proj.weight` | (512, 512) | 2 | **SF-NorLotusMuon** | attn | H_BitLinear |
| `block.N.moe.experts[k].w1` | (1024, 512) | 2 | **SF-NorLotusMuon** | ffn | Routed expert |
| `block.N.moe.router.weight` | (4, 512) | 2 | FP8 AdamW | router | Always AdamW (router is policy, not value) |
| `block.N.norm.weight` | (512,) | 1 | FP8 AdamW | norm | 1D |
| `lm_head.weight` | (259, 512) | 2 | AdamW | mtp | Output embedding |
| `mtp_heads[k].weight` | (259, 512) | 2 | AdamW | mtp | Output embeddings |
| `mar.W_mix.weight` | (16, 512) | 2 | AdamW | attn | Too small (16 < 16 not satisfied — `4*4=16` skipped on the 4-dim) |

The **sub-group** column is what drives decoupled per-layer LR — see [§ Decoupled per-layer LR](#decoupled-per-layer-lr) below.

## Why SF-NorLotusMuon for 2D projections

SF-NorLotusMuon combines three techniques:

1. **Schedule-Free (SF)** — eliminates the need for a separate LR schedule by
   interpolating between two parameter states (the "current" and the "EMA" copy).
   This handles gradient noise without explicit decay.
2. **NorMuon** — applies cautious weight decay (normalized by parameter norm),
   preventing the optimizer from over-regularizing directions that the ternary
   grid has already quantized.
3. **LOTUS rank-8** — factorises the momentum buffer `m ≈ buf_p @ buf_q.T` with
   two rank-8 matrices, giving ~85× less optimizer state than full Muon with
   identical convergence.

The orthogonalization step uses **Gram Newton-Schulz** (`gram_newton_schulz`
package) followed by **Muon+ column normalization** (`O_t / (O_t.norm(dim=0) + ε)`).

## Why FP8 AdamW for 1D + embeddings

1D params (norms, biases) have no spectral structure — they're scalars. AdamW's per-element adaptive scale is the right tool. FP8 AdamW (`torchao.optim.AdamWFp8`) uses 8-bit floating point for optimizer state, cutting memory by 75% vs fp32 AdamW with no measurable quality loss.

Embeddings (input-side `embed_tokens`, output `lm_head`) are a special case: they're 2D but each row is a *categorical* lookup. Orthogonalizing them would scramble the rows against each other, which has no semantic meaning. AdamW's per-element scale preserves the lookup structure.

The MTP heads (4 of them) share the `embed_weight` matrix, so they're all 2D-but-categorical, all go to AdamW.

The MoE **router** is a special case: it produces a categorical distribution over experts. It's not a "value" that benefits from spectral descent, it's a "policy" that benefits from stable per-element scale. The router always goes to AdamW, **never** to Muon.

## Decoupled per-layer LR

Every param that lands in the optimizer is **also** sorted into one of six sub-groups based on the layer type it lives in:

| Sub-group | What lives here | Default LR multiplier |
|---|---|---|
| `attn`  | Attention Q/K/V/O, mAR W_gate/W_mix | 1.0 |
| `ffn`   | FFN up/down/gate, MoE experts, shared experts | 1.0 |
| `mtp`   | `lm_head` + 4 MTP heads | 1.0 |
| `norm`  | All RMSNorm gains, biases | 1.0 |
| `embed` | `embed_tokens` (input-side) | 0.5 |
| `router`| All MoE router weights | 0.5 |

The default `lr_multipliers` keep the core (`attn`, `ffn`, `mtp`, `norm`) at the AutoPilot base LR, and **halve** the LR for embedding and router — they are the two layer types that are most sensitive to over-shooting in 1-bit, because the ternary grid is too coarse to recover from a bad update.

The mechanism is a one-time param-to-subgroup mapping built at optimizer init. AutoPilot's per-step LR is then multiplied by the sub-group multiplier before being pushed into the param group. The 6 groups live in two underlying optimizers (Muon and AdamW); routing is by sub-group, then by Muon/AdamW.

```python
# training/optimizer.py
@register("optimizer", "lotus_muon")
class buselOptimizerEngine:
    def __init__(self, model, config):
        self._groups = self._partition_by_subgroup(model, config.lr_multipliers)
        for grp, mult, opt in self._groups:
            grp["lr"] = config.lr * mult
            opt.add_param_group(grp)
```

To override, pass `lr_multipliers: dict[str, float]` in the config (or in the profile YAML). For example, to make FFN learning faster than attention:

```yaml
# configs/default.yaml — shpak profile
lr_multipliers:
  attn: 1.0
  ffn:  1.5   # 1.5× LR on the FFN side
  mtp:  1.0
  norm: 1.0
  embed: 0.5
  router: 0.5
```

This is **on by default** — no opt-in needed. To disable and go back to single-LR, set all multipliers to 1.0 (or pass `lr_multipliers: {attn: 1.0, ffn: 1.0, mtp: 1.0, norm: 1.0, embed: 1.0, router: 1.0}`).

## Gram Newton-Schulz orthogonalization

```python
# Uses gram_newton_schulz package (primary path)
from gram_newton_schulz import StandardNewtonSchulz

# Falls back to internal _newton_schulz_core (quintic, 5 steps) if unavailable
def _newton_schulz_core(X: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate orthogonalization via Newton-Schulz.

    For X with singular values in (0, 1], 5 iterations yields
    singular values within 1e-3 of 1.
    """
    ...
```

The quintic coefficients `(3.4445, -4.7750, 2.0315)` are from Keller Jordan's
original Muon recipe. After NS, Muon+ applies column normalization:
`O_t / (O_t.norm(dim=0, keepdim=True) + 1e-8)`.

## SF-NorLotusMuon hyperparameters

| Name | Default | Notes |
|---|---|---|
| `lr` | 0.02 | SF-NorLotusMuon LR is 10× AdamW's by convention |
| `momentum` | 0.95 | Standard, handled by SF interpolation |
| `ns_steps` | 5 | Gram NS quintic, 5 iterations |
| `lotus_rank` | 8 | LOTUS rank. 6 = 60% memory, 8 = 85%, 16 = ~95% quality. 8 is the sweet spot. |
| `lotus_lr_scale` | 0.5 | LOTUS effective LR is `lr × lotus_lr_scale` |

## FP8 AdamW hyperparameters

| Name | Default | Notes |
|---|---|---|
| `lr` | 0.002 | 1/10 of SF-NorLotusMuon's (per sub-group multiplier) |
| `betas` | (0.9, 0.95) | Standard |
| `eps` | 1e-8 | Standard |
| `weight_decay` | dynamic | Driven by `buselAutoPilot` |

## The 1.58-bit-specific trick: scale calibration

Because ternary weights are constrained to `{-1, 0, +1}` * 1.58 bits, the *direction* of the update matters more than the magnitude. Muon's orthogonalization handles direction perfectly, but we still need to make sure the per-layer update magnitude is comparable across layers. busel does this with a one-time **scale calibration** at the start of training:

```python
# training/optimizer.py
def calibrate_scales(model: nn.Module) -> None:
    for p in model.parameters():
        if is_muon_param(p):
            p._muon_scale = 0.2 * math.sqrt(max(p.shape))
```

This cached `_muon_scale` is used in the update step. After calibration, the per-layer Muon update has norm ≈ `lr · 0.2 · sqrt(max(A,B)) · sqrt(min(A,B))` ≈ `lr · 0.2 · sqrt(A·B)`, which is comparable across all 2D params in the model.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Loss spikes at step ~5k | SF-NorLotusMuon LR too high | Reduce to 0.015, or enable AutoPilot's 3σ dampening |
| Grad norm explodes | Column normalization disabled | Muon+ is always ON, re-enable if disabled |
| NaN after warmup | Momentum too high | Drop to 0.9 |
| Embeddings don't move | `is_muon_param` accidentally true | Add `"embed" in name` check |

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `@register("optimizer", "norlotus_muon")` | [busel_registry.py](file:///home/sehaxe/busel-ai/busel_registry.py) | Pluggable — swap for a new paper's recipe |
| `buselOptimizerEngine` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | The hybrid class |
| `StandardNewtonSchulz` (Gram NS) | `gram_newton_schulz` package | Primary orthogonalization path |
| `_newton_schulz_core()` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | Fallback quintic NS |
| `is_muon_param()` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | The routing rule |
| `calibrate_scales()` | [training/optimizer.py](file:///home/sehaxe/busel-ai/training/optimizer.py) | One-time init |
| `test_hybrid_routing` | [tests/test_optimizer.py](file:///home/sehaxe/busel-ai/tests/test_optimizer.py) | Compliance: 2D → Muon, 1D → AdamW |
| `test_newton_schulz_5_converges` | [tests/test_optimizer.py](file:///home/sehaxe/busel-ai/tests/test_optimizer.py) | Compliance: NS×5 → singular values within 1e-3 of 1 |
| `test_muon_1bit_alignment` | [tests/test_optimizer.py](file:///home/sehaxe/busel-ai/tests/test_optimizer.py) | Compliance: orthogonal updates preserve ternary grid |

## See also

- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — where the optimizer sits in the loop
- [AutoPilot v6.0](file:///home/sehaxe/busel-ai/site/src/content/docs/training/autopilot.md) — adaptive gradient clipping + LR schedule
- [One-bit weights](file:///home/sehaxe/busel-ai/site/src/content/docs/architecture/one-bit-weights.md) — why 1-bit needs spectral descent
- [Keller Jordan's Muon repo](https://github.com/KellerJordan/Muon) — the original implementation
