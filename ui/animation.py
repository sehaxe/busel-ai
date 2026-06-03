"""
✨ busel ANIMATION v2.0 — Teto Live Context Manager (emoticon edition)
Wraps rich.live.Live to display a tiny cycling Teto emoticon alongside
arbitrary right-side content (status panel, progress bar). Teto changes
colour per state (green idle → cyan think → yellow training → gold done).
Auto-falls back to plain print() if rich is unavailable or stdout is not a
TTY (so CI logs stay clean).
"""
from __future__ import annotations

import os
import shutil
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    from rich.console import Console as _RichConsole
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from ui.teto import frame as _teto_frame

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
        teto = _teto_frame(self.state, self.tick)
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

