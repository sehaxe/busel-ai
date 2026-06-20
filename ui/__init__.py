"""
🎯 busel — Teto Emoticon Avatar & Max-Beauty Rich Terminal Helpers
Teto Kasane (重音テト) is a green twin-tail Vocaloid — the busel mascot.
Rendered as a single-line emoticon (12-frame kawaii cycle including
ξ(｡•̀ᴗ-)✧ξ, ξ(≧◡≦)ξ, ▼ᗜˬᗜ▼ and more) that animates beside rich panels,
spinners, progress bars, tables, and trees. State changes colour (green
idle → cyan think → yellow training → gold done).

Public API:
    ui.teto.frames()                       -> list[str]   (12 idle frames)
    ui.teto.frame(state, tick=0)           -> str         (one emoticon)
    ui.teto.states()                       -> list[str]
    ui.animation.teto_animate(renderer)    -> context manager yielding _Animator | None
    ui.cli.gradient_text(text, palette)    -> rich.Text | str
    ui.cli.animated_header(title, ...)     -> None
    ui.cli.header(title, subtitle)         -> None
    ui.cli.status_panel(title, **fields)   -> None
    ui.cli.stats_table(title, rows)        -> None
    ui.cli.project_tree()                  -> None
    ui.cli.spinner(message)                -> context manager
    ui.cli.progress_bar(total, desc)       -> context manager yielding handle
    ui.cli.log(message, level="info")      -> None
    ui.cli.step_line(...)                  -> str
    ui.cli.console()                       -> rich.console.Console | None
"""
from __future__ import annotations

__all__ = [
    "animation",
    "cli",
]
