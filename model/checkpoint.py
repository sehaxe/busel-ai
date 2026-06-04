"""
🛸 busel CHECKPOINT I/O — torch.compile-safe state_dict loaders

Centralises the prefix-strip + compile-wrapper dance that 5 stage files
+ train.py + tools/inference.py all need. The bug class is:

    OptimizedModule (torch.compile) wraps the model in a `_orig_mod.` proxy.
    `model.state_dict()` on the wrapper returns keys prefixed `_orig_mod.`.
    Loading that same dict into a *fresh, uncompiled* model needs the prefix
    stripped. Loading into a compiled model needs the wrapper bypassed
    (`model._orig_mod.load_state_dict(...)`) so the key naming matches.

Two cross-config cases are non-obvious:
  - saved UNCOMPILED + loaded COMPILED  → load into `obj._orig_mod` (raw keys)
  - saved COMPILED   + loaded UNCOMPILED → strip prefix, direct load

Both are handled by `load_state_dict_safely` below.
"""
from __future__ import annotations
from typing import Any, Mapping

# Old torch._dynamo used `compiled_model.` and `_dynamo.` prefixes in some
# versions. List in priority order; first match wins.
_COMPILE_PREFIXES = ("_orig_mod.", "compiled_model.", "_dynamo.")


def strip_compile_prefix(sd: Mapping[str, Any]) -> dict:
    """Remove torch.compile state_dict prefixes for portable checkpoint loading.

    `torch.compile(model)` wraps the model so that `model.state_dict()` returns
    keys prefixed with `_orig_mod.` (or `compiled_model.` / `_dynamo.` in older
    versions). When loading such a dict into a fresh, uncompiled model, the
    keys must be stripped or `load_state_dict` raises a mismatch error.

    Returns a NEW dict — the input is not mutated.
    """
    if not sd:
        return dict(sd)
    out: dict = {}
    for k, v in sd.items():
        new_k = k
        for prefix in _COMPILE_PREFIXES:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
                break
        out[new_k] = v
    return out


def load_state_dict_safely(obj: Any, sd: Mapping[str, Any], *, strict: bool = True) -> Any:
    """Load `sd` into `obj`, handling torch.compile wrapper transparently.

    Handles all 4 cross-config cases:

    | saved ↓ | loaded → | what happens                                 |
    | -------- | -------- | -------------------------------------------- |
    | uncompiled | uncompiled | strip (no-op) + direct load                |
    | uncompiled | compiled   | load into `obj._orig_mod` (raw keys)        |
    | compiled   | uncompiled | strip prefix + direct load                 |
    | compiled   | compiled   | strip prefix + load into `obj._orig_mod`    |

    Args:
        obj: `nn.Module` (or `OptimizedModule` wrapper from `torch.compile`).
        sd: state_dict to load.
        strict: passed through to `load_state_dict`. `True` (default) raises
            on missing/unexpected keys. `False` returns the
            `_IncompatibleKeys` namedtuple for partial-load diagnostics.

    Returns:
        Whatever the underlying `load_state_dict` returns: the success
        string `"<All keys matched successfully>"` when `strict=True` and
        every key matches; the `_IncompatibleKeys` namedtuple when
        `strict=False`. Callers generally should not rely on the return
        value — use `strict=False` to surface diagnostics via
        `.missing_keys` / `.unexpected_keys`.
    """
    if any(k.startswith(_COMPILE_PREFIXES) for k in sd.keys()):
        sd = strip_compile_prefix(sd)
    target = obj._orig_mod if hasattr(obj, "_orig_mod") else obj
    return target.load_state_dict(sd, strict=strict)
