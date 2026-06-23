"""
💡 busel — Max-Beauty Rich Terminal Helpers
Drop-in helpers for training/inference. Auto-falls back to plain print()
when rich is not installed (no hard dep on rich for tests or CI).

Beauty layer:
    • gradient_text()      — smooth RGB gradient across an ASCII art banner
    • animated_header()    — banner that cycles colors frame by frame
    • spinner()            — rich.status context manager for any blocking op
    • progress_bar()       — context manager with bar/eta/sparkle columns
    • stats_table()        — render a key/value table inside a rich Panel
    • safe_print()         — print that survives broken pipes (head | tail)
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

try:
    from rich.box import ROUNDED
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.spinner import Spinner
    from rich.status import Status
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

_console: Optional["_RichConsole"] = None

_GRADIENT_PALETTES: dict[str, tuple[str, ...]] = {
    "teto": ("#0a3d0a", "#22ee22", "#aaffaa", "#22ee22", "#0a3d0a"),
    "fire": ("#330000", "#ff5500", "#ffaa00", "#ffff66", "#ffaa00"),
    "ocean": ("#001033", "#0055ff", "#00aaff", "#aaeeff", "#00aaff"),
    "rainbow": ("#ff0000", "#ffaa00", "#ffff00", "#00ff00", "#00ffff", "#0055ff", "#aa00ff"),
}


def console() -> Optional["_RichConsole"]:
    """Lazily construct a singleton rich.Console. Returns None if rich missing."""
    global _console
    if not HAS_RICH:
        return None
    if _console is None:
        _console = _RichConsole()
    return _console


def gradient_text(text: str, palette: str = "teto") -> "Text | str":
    """Render text with a smooth colour gradient. Falls back to plain text."""
    if not HAS_RICH:
        return text
    colors = _GRADIENT_PALETTES.get(palette, _GRADIENT_PALETTES["teto"])
    out = Text()
    n = max(1, len(text) - 1)
    for i, ch in enumerate(text):
        idx = int((i / n) * (len(colors) - 1))
        out.append(ch, style=colors[idx])
    return out


def animated_header(title: str, subtitle: str = "", palette: str = "teto", cycles: int = 1) -> None:
    """Print a banner that animates by cycling palettes `cycles` times."""
    if not HAS_RICH:
        header(title, subtitle)
        return
    c = console()
    if c is None:
        header(title, subtitle)
        return
    palettes = list(_GRADIENT_PALETTES.keys()) if palette == "cycle" else [palette]
    total_frames = max(1, cycles)
    for f in range(total_frames):
        pal = palettes[f % len(palettes)]
        body = gradient_text(title, pal)
        if subtitle:
            body.append("\n")
            body.append(subtitle, style="dim")
        c.print(_RichPanel(body, border_style="green", box=ROUNDED, title="busel", title_align="left"))
        if total_frames > 1:
            time.sleep(0.05)


def header(title: str, subtitle: str = "") -> None:
    """Print a rich Panel header. Falls back to ASCII banner if rich absent."""
    if HAS_RICH:
        c = console()
        if c is not None:
            body = title if not subtitle else f"{title}\n[dim]{subtitle}[/dim]"
            c.print(_RichPanel(body, border_style="green", box=ROUNDED, title="busel", title_align="left"))
            return
    bar = "═" * max(len(title) + 4, 40)
    print(bar)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(bar)


def status_panel(title: str, **fields: Any) -> None:
    """Print a key/value status panel (rich Table inside a Panel)."""
    if HAS_RICH:
        c = console()
        if c is not None:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold cyan", justify="right")
            table.add_column(style="white")
            for k, v in fields.items():
                table.add_row(str(k), str(v))
            c.print(_RichPanel(table, title=title, border_style="cyan", box=ROUNDED))
            return
    print(f"--- {title} ---")
    for k, v in fields.items():
        print(f"  {k}: {v}")
    print("---")


def stats_table(title: str, rows: list[tuple[str, str, str]]) -> None:
    """Print a 3-column table (key, value, delta) inside a Panel.

    Args:
        title: Panel title.
        rows: List of (key, value, delta) tuples. delta can include
            a leading +/- for color hinting.
    """
    if HAS_RICH:
        c = console()
        if c is not None:
            t = Table(title=title, box=ROUNDED, border_style="green", show_header=True)
            t.add_column("metric", style="cyan", justify="right")
            t.add_column("value", style="white", justify="left")
            t.add_column("Δ", justify="right")
            for k, v, d in rows:
                style = "green" if d.startswith("+") else ("red" if d.startswith("-") else "dim")
                t.add_row(k, v, Text(d, style=style))
            c.print(t)
            return
    print(f"--- {title} ---")
    for k, v, d in rows:
        print(f"  {k:20s} {v:14s}  {d}")
    print("---")


@contextmanager
def spinner(message: str = "Working…", style: str = "dots") -> Iterator[Optional["Status"]]:
    """rich.status context manager that shows a spinner next to a message.

    Falls back to a plain print if rich is unavailable.
    """
    if not HAS_RICH:
        print(message)
        yield None
        return
    c = console()
    if c is None:
        print(message)
        yield None
        return
    with c.status(message, spinner=style, spinner_style="green"):
        yield c


@contextmanager
def progress_bar(total: int, description: str = "Training") -> Iterator[Any]:
    """rich.progress context manager with bar + ETA + elapsed time."""
    if not HAS_RICH:
        print(f"{description}: 0/{total}")
        yield None
        return
    c = console()
    if c is None:
        print(f"{description}: 0/{total}")
        yield None
        return
    prog = Progress(
        SpinnerColumn("dots", style="green"),
        TextColumn(f"[bold cyan]{description}[/bold cyan]"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bold green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=c,
        transient=False,
    )
    with prog:
        task_id = prog.add_task(description, total=total)
        yield _ProgressHandle(prog, task_id)


class _ProgressHandle:
    __slots__ = ("prog", "task_id")

    def __init__(self, prog, task_id):
        self.prog = prog
        self.task_id = task_id

    def update(self, advance: int = 1, **fields: Any) -> None:
        self.prog.update(self.task_id, advance=advance, **fields)


def log(message: str, *, level: str = "info") -> None:
    """Print a coloured log line. Falls back to plain print."""
    if HAS_RICH:
        c = console()
        if c is not None:
            style = {"info": "blue", "warn": "yellow", "error": "red", "ok": "green", "debug": "dim"}.get(level, "white")
            c.print(f"[{style}]\\[{level.upper()}][/{style}] {message}")
            return
    print(f"[{level.upper()}] {message}")


def step_line(
    step: int,
    total: int,
    loss: float,
    aux_loss: float,
    lr: float,
    clip: float,
    batch: int,
    tokens_per_s: float,
    vram_mb: float = 0.0,
) -> str:
    """Format a single training step line (works as both rich-rendered and plain text)."""
    vram = f" | VRAM: {vram_mb:.0f}MB" if vram_mb > 0 else ""
    return (
        f"Step {step:05d}/{total:05d} | "
        f"Total: {loss:.2f} | "
        f"Aux: {aux_loss:.2f} | "
        f"LR: {lr:.5f} | "
        f"Clip: {clip:.2f} | "
        f"Batch: {batch} | "
        f"{tokens_per_s:.0f} tokens/s{vram}"
    )


def safe_print(msg: str) -> None:
    """print() that survives broken pipes (e.g. when piped to `head`)."""
    try:
        print(msg, flush=True)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            os._exit(0)
        except Exception:
            pass
