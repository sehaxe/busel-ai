# ui/ — Teto Emoticon & Rich Terminal Helpers

**Scope:** Teto Kasane (重音テト) Vocaloid mascot emoticon + beautiful terminal output via `rich` with graceful fallback to plain `print()` when rich is unavailable.

## STRUCTURE
```
ui/
├── __init__.py    # Public API: animation, cli
├── animation.py   # Teto emoticon frames + live animation context manager
└── cli.py         # Max-beauty terminal helpers: gradient_text, spinner, progress_bar, stats_table, project_tree
```

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Change Teto emoticon frames | `animation.py` → `_IDLE_FRAMES` | 12-frame tuple, cycled by tick |
| Add new Teto state | `animation.py` → `_FRAME_MAP` | Must also add to `_STATE_STYLES` for colour |
| Animate a training loop | `animation.py` → `teto_animate()` | Context manager yielding `_Animator` with `set_state()` |
| Render a styled header | `cli.py` → `gradient_text()` | RGB gradient across text |
| Show a spinner | `cli.py` → `spinner()` | Context manager for blocking ops |
| Show a progress bar | `cli.py` → `progress_bar()` | Context manager yielding `handle.update(n)` |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `teto_frames()` | function | animation.py | Returns `list[str]` of 12 idle frames |
| `teto_frame(state, tick)` | function | animation.py | One emoticon for state at tick |
| `teto_states()` | function | animation.py | Returns list of all state names |
| `_Animator` | class | animation.py | Teto animation handle: `set_state(name)`, `render()` |
| `teto_animate(renderer, fps, state)` | contextmanager | animation.py | Live Teto animation; yields `_Animator | None` |
| `gradient_text(text, palette)` | function | cli.py | Smooth RGB gradient across text |
| `animated_header(title, ...)` | function | cli.py | Banner cycling colors |
| `header(title, subtitle)` | function | cli.py | Rich Panel header |
| `status_panel(title, **fields)` | function | cli.py | Key-value panel |
| `stats_table(title, rows)` | function | cli.py | Rich Table inside Panel |
| `project_tree()` | function | cli.py | Module hierarchy tree |
| `spinner(message)` | contextmanager | cli.py | Rich.Status spinner |
| `progress_bar(total, desc)` | contextmanager | cli.py | Progress with ETA/sparkles |
| `log(message, level)` | function | cli.py | Formatted log line |
| `step_line(...)` | function | cli.py | Formatted step output |
| `console()` | function | cli.py | Returns `rich.console.Console | None` |
| `safe_print(text, ...)` | function | cli.py | Print that survives broken pipes |

## CONVENTIONS
- **Teto states:** `idle` (12-frame cycle), `blink`, `smile`, `think`, `wave`, `training` (3-frame cycle), `done`.
- **State colours:** green (idle/blink) → cyan (think) → magenta (wave) → yellow (training) → gold (done).
- **Rich auto-fallback:** All functions check `HAS_RICH` and gracefully degrade to plain print.
- **TTY detection:** `_is_tty()` checks `shutil.is_terminal(None)` and `CI` env var. No animation in CI.
- **No hard dep on rich:** `rich` is imported in function bodies, not at module top.
- **`teto_frame()` vs `teto_frames()`:** Singular returns one frame by state+tick; plural returns all idle frames.
- **Animation threading:** `teto_animate` runs a daemon thread with `Live` refresh. Stop via threading.Event.

## ANTI-PATTERNS
- **NEVER** hardcode frame lists in multiple places — `_IDLE_FRAMES` is the single source of truth.
- **NEVER** call `teto_animate` without checking the return value — it yields `None` when rich is unavailable.
- **NEVER** set `fps < 1.0` in `teto_animate` — clamped to 0.1 minimum by `max(fps, 0.1)`.
- **NEVER** import `ui/teto.py` — it was deleted in v8.5 (moved to `ui/animation.py`).
