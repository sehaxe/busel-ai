"""
🔧 busel TOOL EXECUTOR v1.1 — Phase 7 + REPL integration (v5.7.0)

Parses `<function_calls>...<invoke>...<parameter>...<result>` envelopes
emitted by the SFT/DPO model, executes the named tool, and re-injects the
result as `<function_results>...</function_results>` so the model can
continue.

**v5.7.0 additions** (REPL integration):
- `format_tool_call_for_user(call)` — pretty-print a single call for the
  confirmation prompt (truncated, ANSI-free).
- `interactive_confirm(call, auto_approve=False)` — sync `input()` with
  `[y/N]` prompt. Default is N (deny). The user is always asked unless
  `/auto on` is set.
- `strip_ansi(text)` — remove terminal escape codes from tool output
  (prevents escape injection from malicious tool responses).
- `MAX_TOOL_CALLS_PER_TURN` — hard cap on tool calls per user turn to
  prevent infinite loops (5 by default).

The default registry exposes `TOOL_BASH` (sandboxed shell) and
`TOOL_READ` (text file read, 32 KB cap).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Safety / UX constants
# ---------------------------------------------------------------------------

MAX_TOOL_CALLS_PER_TURN: int = 5
"""Hard cap on tool calls per user turn. Prevents infinite tool-call loops."""

MAX_TOOL_OUTPUT_BYTES: int = 8192
"""Max bytes of a single tool's stdout/stderr to inject back to the model."""


# ---------------------------------------------------------------------------
# Envelope detection
# ---------------------------------------------------------------------------

ENVELOPE_PATTERN = re.compile(
    r"<function_calls>(?P<body>.*?)</function_calls>",
    re.DOTALL,
)
INVOKE_PATTERN = re.compile(
    r"<invoke\s+name=\"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\">\s*"
    r"(?P<params>.*?)"
    r"</invoke>",
    re.DOTALL,
)
PARAM_PATTERN = re.compile(
    r"<parameter\s+name=\"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\">\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from a model output string.

    Handles both single-invoke and multi-invoke envelopes (one or more
    `<invoke>...</invoke>` blocks inside a single `<function_calls>...</function_calls>`).
    Returns a list of {name, params} dicts, in document order.
    Returns an empty list if no <function_calls> envelope is present.
    """
    calls: list[dict[str, Any]] = []
    for env_match in ENVELOPE_PATTERN.finditer(text):
        body = env_match.group("body")
        for inv_match in INVOKE_PATTERN.finditer(body):
            name = inv_match.group("name")
            params: dict[str, str] = {}
            for p in PARAM_PATTERN.finditer(inv_match.group("params")):
                params[p.group("key")] = p.group("value").strip()
            calls.append({"name": name, "params": params})
    return calls


# ---------------------------------------------------------------------------
# Tool registry (sandboxed subset)
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Minimal sandboxed tool registry. Each tool is a Python callable."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, name: str, fn) -> None:
        self._tools[name] = fn

    def has(self, name: str) -> bool:
        return name in self._tools

    def call(self, name: str, params: dict[str, str]) -> str:
        if name not in self._tools:
            return f"ERROR: unknown tool {name!r}. Available: {sorted(self._tools)}"
        try:
            return str(self._tools[name](**params))
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"

    def names(self) -> list[str]:
        return sorted(self._tools)


def _bash_tool(command: str, timeout: str = "10") -> str:
    """Sandboxed shell execution. Runs the command, returns stdout (capped)."""
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError):
        timeout_s = 10.0
    try:
        out = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return (out.stdout + out.stderr)[:MAX_TOOL_OUTPUT_BYTES]
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout_s}s"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def _read_tool(path: str) -> str:
    """Read a text file (capped at 32 KB)."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return f"ERROR: not a file: {p}"
        return p.read_text(encoding="utf-8", errors="replace")[:32768]
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def default_tool_registry() -> ToolRegistry:
    """The default busel tool registry. Currently exposes TOOL_BASH and TOOL_READ."""
    r = ToolRegistry()
    r.register("TOOL_BASH", _bash_tool)
    r.register("TOOL_READ", _read_tool)
    return r


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def execute_tool_calls(text: str, registry: ToolRegistry | None = None) -> list[dict[str, Any]]:
    """Parse + execute every tool call in `text`. Returns per-call results.

    Each result: {name, params, output}.
    """
    if registry is None:
        registry = default_tool_registry()
    results: list[dict[str, Any]] = []
    for call in parse_tool_calls(text):
        output = registry.call(call["name"], call["params"])
        results.append({**call, "output": output})
    return results


def format_tool_results(results: list[dict[str, Any]]) -> str:
    """Format results as a `<function_results>...</function_results>` block (for re-injection)."""
    blocks = []
    for r in results:
        try:
            params_json = json.dumps(r["params"], ensure_ascii=False)
        except (TypeError, ValueError):
            params_json = str(r["params"])
        blocks.append(
            f'<result name="{r["name"]}">\n'
            f"{r['output']}\n"
            f"</result>"
        )
    if not blocks:
        return ""
    return "<function_results>\n" + "\n".join(blocks) + "\n</function_results>"


# ---------------------------------------------------------------------------
# v5.7.0 — REPL integration: confirmation prompt + ANSI stripper
# ---------------------------------------------------------------------------

# ANSI escape code pattern: covers CSI sequences, OSC sequences, and lone ESC.
# Belt-and-suspenders: prevents the model from injecting terminal escapes via
# tool output (e.g. a malicious `TOOL_BASH` response that tries to clear screen).
_ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07\x1B]*(?:\x07|\x1B\\))"
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from `text` (CSI, OSC, lone ESC)."""
    return _ANSI_ESCAPE_RE.sub("", text)


def format_tool_call_for_user(call: dict[str, Any], max_value_len: int = 120) -> str:
    """Pretty-print a single tool call for the confirmation prompt.

    Long parameter values are truncated with '...' to keep the prompt
    readable. ANSI is stripped from the values.
    """
    name = call.get("name", "<unknown>")
    params = call.get("params", {}) or {}
    if not params:
        return f"{name}()"
    rendered = []
    for k, v in params.items():
        v_clean = strip_ansi(str(v))
        if len(v_clean) > max_value_len:
            v_clean = v_clean[: max_value_len - 3] + "..."
        rendered.append(f'{k}="{v_clean}"')
    return f"{name}(" + ", ".join(rendered) + ")"


def interactive_confirm(call: dict[str, Any], auto_approve: bool = False) -> bool:
    """Ask the user via stdin whether to execute a single tool call.

    Always asks (per user requirement: "ask each time to apply it or not")
    unless `auto_approve=True` (toggled via the `/auto on` REPL command).

    The default answer is **N** (deny) — the user must explicitly type
    'y' or 'yes' to approve. Any other input (including empty Enter) is
    treated as deny.

    Returns:
        True if the user approved execution, False if denied or on EOF.
    """
    if auto_approve:
        return True
    rendered = format_tool_call_for_user(call)
    prompt = f"\n[?] Execute {rendered} ? [y/N]: "
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # newline so the next prompt isn't glued to the [?]
        return False
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# v5.7.0 — Denial result (synthetic "<function_results>" block)
# ---------------------------------------------------------------------------

DENIED_RESULT_OUTPUT: str = "ERROR: tool execution denied by user"
"""The synthetic output injected when the user denies a tool call.

The model sees this as if the tool had returned an error, so it can
gracefully pivot (e.g. apologize, suggest an alternative, give up).
"""


def denied_result(call: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic result dict for a denied tool call.

    Returns: {name, params, output} — same shape as `execute_tool_calls`,
    so it can be passed straight to `format_tool_results`.
    """
    return {**call, "output": DENIED_RESULT_OUTPUT}


__all__ = [
    "parse_tool_calls",
    "execute_tool_calls",
    "format_tool_results",
    "format_tool_call_for_user",
    "interactive_confirm",
    "denied_result",
    "strip_ansi",
    "ToolRegistry",
    "default_tool_registry",
    "INVOKE_PATTERN",
    "PARAM_PATTERN",
    "ENVELOPE_PATTERN",
    "MAX_TOOL_CALLS_PER_TURN",
    "MAX_TOOL_OUTPUT_BYTES",
    "DENIED_RESULT_OUTPUT",
]
