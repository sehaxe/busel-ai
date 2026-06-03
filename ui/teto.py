"""
🎵 busel TETO v2.0 — Kasane Teto Emoticon-Style Avatar
Single-line kawaii emoticon frames (Teto's signature twin-drill pigtails are
the ξ marks on each side, when present). Designed to be tiny (1 line each)
so it tucks neatly into a terminal corner while the rest of the UI does the
heavy lifting. Animates by cycling through 'idle' frames at ~2 FPS.
"""
from __future__ import annotations

_IDLE_FRAMES: tuple[str, ...] = (
    "ξ(｡•̀ᴗ-)✧ξ",
    "ξ(｡•̀ᴗ•́｡)ξ",
    "ξ(≧◡≦)ξ",
    "ξ(^ω^)ξ",
    "ξ(￣▽￣)ξ",
    "ξ(≧ω≦)ξ",
    "▼ᗜˬᗜ▼",
    "ξ(ᗒᗣᗕ)ξ",
    "ξ(◕‿◕)ξ",
    "ξ(•ω•)ξ",
    "ξ(｡◕‿◕｡)ξ",
    "ξ(ᗒᴥᗕ)ξ",
)

_THINK_FRAME: str = "(ᗜˬᗜ)…"

_WAVE_FRAME: str = "ξ(ᗜ‿ᗜ)ξ"

_TRAINING_FRAMES: tuple[str, ...] = (
    "(ᗜˬᗜ)▶",
    "(ᗜˬᗜ)▶▶",
    "(ᗜˬᗜ)▶▶▶",
)

_DONE_FRAME: str = "★(ᗜ‿ᗜ)★"

_FRAME_MAP: dict[str, str | tuple[str, ...]] = {
    "idle": _IDLE_FRAMES,
    "blink": "(-ˬ- )",
    "smile": "(ᗜ‿ᗜ)",
    "think": _THINK_FRAME,
    "wave": _WAVE_FRAME,
    "training": _TRAINING_FRAMES,
    "done": _DONE_FRAME,
}


def frames() -> list[str]:
    """Return the 12 idle animation frames (kawaii emoticon cycle)."""
    return list(_IDLE_FRAMES)


def frame(state: str = "idle", tick: int = 0) -> str:
    """Return a single frame for the given state at the given tick.

    Args:
        state: One of 'idle', 'blink', 'smile', 'think', 'wave', 'training', 'done'.
        tick: Animation tick. Only used for multi-frame states (idle, training);
            single-frame states return the same string regardless of tick.

    Returns:
        A short emoticon string (1 line). Empty string if state is unknown.
    """
    entry = _FRAME_MAP.get(state, "")
    if isinstance(entry, tuple):
        return entry[tick % len(entry)]
    return entry


def states() -> list[str]:
    """Return the list of all available state names."""
    return list(_FRAME_MAP.keys())
