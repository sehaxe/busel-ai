---
title: "Checkpointing & resume"
description: "Checkpoint format, the 10MB corruption guard, SIGINT-safe deferred save, resume semantics, and how _strip_compile_prefix keeps torch.compile reloading fast."
sidebar:
  order: 5
---

import { Aside, Tabs, TabItem } from '@astrojs/starlight/components';

busel checkpoints are `.pt` files with a strict format and a **10MB corruption guard**. They're SIGINT-safe (Ctrl-C during compile/recompile no longer crashes), and resume is automatic when you point `train.py` at a checkpoint path.

## What's in a checkpoint

```python
# train.py::save_checkpoint
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "auto_pilot": auto_pilot.state_dict(),
    "step": step,
    "cfg": config,
    "cfg.profile": config.profile,    # ← denormalized for quick identification
    "rng_state": torch.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
}, path)
```

| Field | Purpose |
|---|---|
| `model` | `buselModel.state_dict()` — all parameters + buffers |
| `optimizer` | `buselOptimizerEngine.state_dict()` — Muon + AdamW momentums, scales |
| `auto_pilot` | `buselAutoPilot.state_dict()` — EMA, sigma, dampening counter |
| `step` | int — the global step number |
| `cfg` | `buselConfig` — full config so resume picks up all settings |
| `cfg.profile` | str — denormalized top-level key for quick "what profile is this?" |
| `rng_state` | `torch.get_rng_state()` — Python + CPU RNG |
| `cuda_rng_state` | `torch.cuda.get_rng_state()` if CUDA — for exact reproduction |

For verabey-27m (~70M params) the full checkpoint is ~15-16 MB. For sokal-60m (~170M) it's ~38 MB. For kruk-120m (~350M) it's ~75 MB.

## The 10MB corruption guard

`tools/inference.py` refuses to load any checkpoint under 10MB:

```python
# tools/inference.py
MIN_CHECKPOINT_BYTES = 10 * 1024 * 1024
if path.stat().st_size < MIN_CHECKPOINT_BYTES:
    raise ValueError(
        f"Checkpoint {path} is {path.stat().st_size} bytes, "
        f"below the {MIN_CHECKPOINT_BYTES} byte minimum. "
        f"This usually means a partial write — do not trust this file."
    )
```

The 10MB threshold catches:

- **Partial writes** — Ctrl-C during save (now fixed, see below) used to leave 1-2MB fragments
- **Optimizer-only saves** — accidentally saving just the model dict
- **Download corruption** — `wget` partial files

The threshold is calibrated so the smallest *valid* checkpoint (verabey-27m model + optimizer) is 15MB; a 10MB threshold catches all three failure modes with zero false positives.

## SIGINT-safe save (`load_state_dict_safely`)

`torch.compile` adds an `_orig_mod.` prefix to every parameter name in the state dict (for the `nn.Module` → `OptimizedModule` wrap). If you save a checkpoint from inside the compiled module and try to reload it into a non-compiled model, the keys don't match.

The busel solution: **defer the save to a step boundary, then use `load_state_dict_safely`**:

```python
# training/stages/pretrain.py
_emergency_save_pending = False

def _sigint_handler(signum, frame):
    global _emergency_save_pending
    _emergency_save_pending = True
    # ↑ do NOT save here. FakeTensor state during compile can't be saved safely.

signal.signal(signal.SIGINT, _sigint_handler)

# Inside the training loop:
if _emergency_save_pending and not torch._dynamo.is_compiling():
    save_checkpoint(...)
    _emergency_save_pending = False
```

The save check happens at the **top of every step**, outside any `torch.compile` region. This is the fix for the "Ctrl-C during compile crashes with FakeTensor" bug.

The prefix-stripping helper in `model/checkpoint.py`:

```python
# model/checkpoint.py
def load_state_dict_safely(model, state_dict):
    """Strips _orig_mod. prefix, unwraps OptimizedModule. The single source of
    truth for all checkpoint loads."""
    ...
```

Used both on save and on resume — a checkpoint saved from a compiled model loads safely into an eager model, and vice versa.

<Aside type="caution" title="Never save from inside torch.compile">
The compiled graph holds `FakeTensor` proxies for some intermediates. Calling `torch.save()` on a `FakeTensor` raises `RuntimeError: Cannot save tensors with fake tensor dtype`. The SIGINT handler must defer, not act.
</Aside>

## Resume semantics

```bash
# Auto-detect the latest checkpoint
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --resume auto

# Explicit checkpoint
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --resume checkpoints/ckpt_5000.pt
```

On resume, `train.py`:

1. Loads the checkpoint
2. Verifies `cfg.profile` matches the current `--profile` (refuses to resume Shpak as Zubr)
3. Restores `model`, `optimizer`, `auto_pilot` states
4. Restores RNG state (Python + CUDA) for exact-step reproduction
5. Continues from `step + 1`

If the profile doesn't match, you get a clear error:

```
ValueError: Checkpoint was saved with profile='verabey-27m' but you're trying to resume as 'sokal-60m'.
Either change --profile to match, or use --force-resume to override (WARNING: this will
not load optimizer state correctly).
```

## Checkpoint naming

| Pattern | Created at | Notes |
|---|---|---|
| `ckpt_0.pt` | Before any training | Sanity checkpoint, model with random init |
| `ckpt_{N}.pt` | Every `save_every` steps | The main periodic checkpoints |
| `ckpt_latest.pt` | After every save | Symlink/copy of the most recent periodic |
| `ckpt_emergency.pt` | After a SIGINT-triggered save | If you Ctrl-C, this is what you resume from |
| `best_val.pt` | When val loss improves | For early stopping (currently informational, not auto-triggered) |

`ckpt_latest.pt` is updated atomically (write to temp + rename) so you never get a half-written "latest" file.

## What gets auto-saved

By default, only `ckpt_{N}.pt` and `ckpt_latest.pt`. Emergency and best checkpoints are opt-in:

```bash
# Enable emergency save on SIGINT
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --save-on-sigint

# Enable best-val tracking
uv run python cli.py pipeline --name pretrain-only --profile verabey-27m --track-best-val
```

The defaults avoid wasting disk on long runs where you only care about the periodic checkpoints.

## Disk space planning

A verabey-27m run with 25k steps, saving every 2.5k steps, gives you 10 periodic checkpoints × 15MB = 150MB. sokal-60m is 10 × 40MB = 400MB. kruk-120m is 10 × 75MB = 750MB.

busel auto-prunes by default: keep the most recent 5 checkpoints plus the best-val, delete the rest. This keeps disk usage at ~75MB for verabey-27m indefinitely.

```python
# train.py
config = buselConfig(profile="shpak", keep_checkpoints=5)
```

Set `keep_checkpoints=-1` to disable pruning.

## Resume sanity check

After any resume, `train.py` runs 3 dry steps to verify the loss is in the same ballpark as the pre-save value. If it's wildly off, you get a warning:

```
WARNING: Resumed at step 5001, expected loss ~2.41, got 8.77.
This usually means the checkpoint is corrupt or the profile mismatch wasn't caught.
Continuing anyway — press Ctrl-C if this looks wrong.
```

This is a *warning*, not an error. The rare case is a partial save that survived the 10MB guard; the warning gives you a chance to abort.

## Where to look in the code

| Component | File | Notes |
|---|---|---|
| `save_checkpoint()` | [training/stages/pretrain.py](file:///home/sehaxe/busel-ai/training/stages/pretrain.py) | The atomic writer |
| `load_state_dict_safely()` | [model/checkpoint.py](file:///home/sehaxe/busel-ai/model/checkpoint.py) | The `_orig_mod.` stripper |
| SIGINT handler | [training/stages/pretrain.py](file:///home/sehaxe/busel-ai/training/stages/pretrain.py) | Flag-setter, not action-taker |
| `MIN_CHECKPOINT_BYTES` | [tools/inference.py](file:///home/sehaxe/busel-ai/tools/inference.py) | The 10MB guard |
| `resume_latest()` | [train.py](file:///home/sehaxe/busel-ai/train.py) | Auto-detect latest periodic |
| `test_checkpoint_10mb_guard` | [tests/test_checkpoint.py](file:///home/sehaxe/busel-ai/tests/test_checkpoint.py) | Compliance: rejects <10MB |
| `test_strip_compile_prefix` | [tests/test_checkpoint.py](file:///home/sehaxe/busel-ai/tests/test_checkpoint.py) | Compliance: strips `_orig_mod.` |
| `test_sigint_defers_save` | [tests/test_checkpoint.py](file:///home/sehaxe/busel-ai/tests/test_checkpoint.py) | Compliance: no save during compile |

## See also

- [Training guide](file:///home/sehaxe/busel-ai/site/src/content/docs/training/training-guide.md) — the loop that calls `save_checkpoint`
- [Operations → Inference](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/inference.md) — how the inference CLI loads these
- [Operations → Troubleshooting](file:///home/sehaxe/busel-ai/site/src/content/docs/operations/troubleshooting.md) — common "checkpoint won't load" errors
