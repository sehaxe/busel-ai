"""
✨ busel — Teto Emoticon + Live Animation
"""
from __future__ import annotations

import os
import shutil
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

# ── Teto emoticon data ───────────────────────────────────────────────────────

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


def teto_frames() -> list[str]:
    """Return the 12 idle animation frames (kawaii emoticon cycle)."""
    return list(_IDLE_FRAMES)


def teto_frame(state: str = "idle", tick: int = 0) -> str:
    """Return a single frame for the given state at the given tick."""
    entry = _FRAME_MAP.get(state, "")
    if isinstance(entry, tuple):
        return entry[tick % len(entry)]
    return entry


def teto_states() -> list[str]:
    """Return the list of all available state names."""
    return list(_FRAME_MAP.keys())


# ── Animation ─────────────────────────────────────────────────────────────────

try:
    from rich.console import Console as _RichConsole
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

_DEFAULT_FPS = 4.0
_DEFAULT_STATE = "idle"

_STATE_STYLES: dict[str, str] = {
    "idle": "green",
    "blink": "green",
    "smile": "bold green",
    "think": "cyan",
    "wave": "bold magenta",
    "training": "yellow",
    "done": "bold gold1",
}


def _is_tty() -> bool:
    return bool(shutil.is_terminal(None)) and not _is_ci()


def _is_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _style_for(state: str) -> str:
    return _STATE_STYLES.get(state, "green")


class _Animator:
    __slots__ = ("tick", "state", "console")

    def __init__(self, state: str, console):
        self.tick = 0
        self.state = state
        self.console = console

    def set_state(self, name: str) -> None:
        self.state = name

    def render(self, right_panel_renderer=None) -> "Text | Panel":
        teto = teto_frame(self.state, self.tick)
        style = _style_for(self.state)
        left = Text(teto, style=style)
        if right_panel_renderer is None:
            return Panel(
                left,
                title=f"busel · {self.state}",
                title_align="left",
                border_style=style,
                padding=(0, 1),
            )
        right = Text(right_panel_renderer(), style="white")
        body = Text.assemble(
            ("busel ", f"bold {style}"),
            (teto, style),
            ("\n", ""),
            right,
        )
        return Panel(body, title=f"busel · {self.state}", border_style=style, padding=(0, 1))


@contextmanager
def teto_animate(
    right_panel_renderer=None,
    *,
    fps: float = _DEFAULT_FPS,
    state: str = _DEFAULT_STATE,
) -> Iterator[Optional[_Animator]]:
    """Context manager that animates a Teto emoticon on the left.

    Args:
        right_panel_renderer: Optional callable() -> str. Called every frame
            to fetch the latest right-hand content (e.g. status panel).
        fps: Animation refresh rate. Defaults to 4 FPS.
        state: Initial Teto state name (see ui.teto.frame). Use the yielded
            animator's .set_state() to change on the fly.

    Yields:
        _Animator handle exposing .set_state(name). None if rich is
        unavailable or stdout is not a TTY (the manager becomes a no-op).

    Usage:
        with teto_animate(lambda: f"step {step}") as ani:
            ani.set_state("training")
            ...
            ani.set_state("done")
    """
    if not HAS_RICH or not _is_tty():
        yield None
        return

    console = _RichConsole()
    animator = _Animator(state=state, console=console)
    stop_event = threading.Event()
    interval = 1.0 / max(fps, 0.1)

    def _loop() -> None:
        tick = 0
        with Live(
            animator.render(right_panel_renderer),
            console=console,
            refresh_per_second=fps * 2,
            transient=False,
        ) as live:
            while not stop_event.is_set():
                animator.tick = tick
                live.update(animator.render(right_panel_renderer))
                tick += 1
                stop_event.wait(interval)
            animator.tick = tick
            live.update(animator.render(right_panel_renderer))

    thread = threading.Thread(target=_loop, daemon=True, name="busel-teto-anim")
    thread.start()
    try:
        yield animator
    finally:
        stop_event.set()
        thread.join(timeout=interval * 2)

