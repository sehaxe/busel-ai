"""
🛰️ busel — Plug-in Vocabulary (18 tokens, 5 layers)

Sovereign byte-level vocabulary (vocab_size=277): 256 raw bytes + 3 legacy
reserved tokens (256-258) + 18 plug-in special tokens across 5 layers.
Tool calls, code blocks, file paths etc. are emitted as raw UTF-8 bytes —
the model learns byte sequences, not single-token shortcuts.

See `multimodal/AGENTS.md` for the full layer inventory.
See `list_special_tokens()` and `layer_summary()` for the live registry.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterator, Optional


# === Foundation: byte range + 3 legacy reserved specials ===
BYTE_COUNT: int = 256
"""Number of raw UTF-8 byte values [0, 256)."""

# Legacy reserved tokens — IMMUTABLE, present from v5.0.
# These were hardcoded across the multimodal pipeline; removing them would
# break existing checkpoints. The plug-in tokens start at SPECIAL_VOCAB_BASE.
LEGACY_TOKENS: dict[str, int] = {
    "media_start": 256,
    "media_end":   257,
    "doc_sep":     258,
}
"""Legacy reserved token IDs (v5.0-5.3). Kept for backward compat."""

SPECIAL_VOCAB_BASE: int = BYTE_COUNT + len(LEGACY_TOKENS)
"""First plug-in special token ID. All new tokens start here."""


# === Token dataclass ===
@dataclass(frozen=True, slots=True)
class SpecialToken:
    """An immutable special-token descriptor.

    `id` is allocated once at registration time and never changes. Disabling
    a token sets `enabled=False` (so the encoder won't emit it and the model
    won't be trained to produce it) but preserves `id` (so a future re-enable
    keeps the same row in the embedding table — no checkpoint re-mapping).

    Int-coercible: `int(TOK_BASH) == 296`, `TOK_BASH == 296` → True.
    """
    name:        str
    id:          int
    layer:       str
    description: str
    enabled:     bool = True

    def __int__(self) -> int:
        return self.id

    def __index__(self) -> int:
        return self.id

    def __eq__(self, other) -> bool:
        if isinstance(other, SpecialToken):
            return self.id == other.id
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.id, self.name, self.enabled))


# === Internal registry state ===
_SPECIAL_TOKENS:    dict[str, SpecialToken]      = {}
_LAYER_TOKENS:      dict[str, list[str]]         = {}
_ID_LOCK            = threading.Lock()
_NEXT_ID:           int                          = SPECIAL_VOCAB_BASE


# === Public API ===

def register_special_token(
    name:        str,
    layer:       str,
    description: str = "",
    *,
    override:    bool = False,
) -> SpecialToken:
    """Register a new plug-in special token. ID is auto-allocated.

    The token is enabled by default. To disable at runtime (without removing
    the registration), use `disable_special_token(name)`.

    Args:
        name: Unique identifier within the special-token registry. Lowercase
            snake_case recommended. Must not collide with legacy reserved
            tokens ("media_start", "media_end", "doc_sep").
        layer: Functional layer ("sequence", "modality", "tool", ...). Used
            for grouping in `list_special_tokens(layer=...)` and for docs.
        description: One-line human description (used in `print(tok)` and
            the auto-generated AGENTS.md token table).
        override: If True, allow replacing a previously-disabled token (re-enable
            with new metadata). Default False: raise on collision with an
            already-enabled token.

    Returns:
        The new `SpecialToken` instance.

    Raises:
        ValueError: If `name` is empty, collides with a legacy token, or
            collides with an already-enabled plug-in token (and override=False).
    """
    global _NEXT_ID
    if not name or not isinstance(name, str):
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if name in LEGACY_TOKENS:
        raise ValueError(
            f"{name!r} is a legacy reserved token (ids 256-258); these are "
            f"immutable. Pick a different name."
        )
    if not layer or not isinstance(layer, str):
        raise ValueError(f"layer must be a non-empty string, got {layer!r}")
    with _ID_LOCK:
        existing = _SPECIAL_TOKENS.get(name)
        if existing is not None:
            if existing.enabled and not override:
                raise ValueError(
                    f"special_token {name!r} already registered at id={existing.id} "
                    f"(enabled). Use override=True to replace it, or call "
                    f"disable_special_token({name!r}) first."
                )
            if not existing.enabled:
                # Re-enable disabled token, keeping its original ID
                tok = SpecialToken(name, existing.id, layer, description, enabled=True)
                _SPECIAL_TOKENS[name] = tok
                return tok
            # override=True on enabled token: allocate a new ID at the END
            tok_id = _NEXT_ID
            _NEXT_ID += 1
            tok = SpecialToken(name, tok_id, layer, description, enabled=True)
            _SPECIAL_TOKENS[name] = tok
            _LAYER_TOKENS.setdefault(layer, [])
            if name not in _LAYER_TOKENS[layer]:
                _LAYER_TOKENS[layer].append(name)
            return tok
        # Fresh registration
        tok_id = _NEXT_ID
        _NEXT_ID += 1
        tok = SpecialToken(name, tok_id, layer, description, enabled=True)
        _SPECIAL_TOKENS[name] = tok
        _LAYER_TOKENS.setdefault(layer, []).append(name)
        return tok


def _set_enabled(name: str, enabled: bool) -> SpecialToken:
    with _ID_LOCK:
        if name not in _SPECIAL_TOKENS:
            raise KeyError(f"unknown special token: {name!r}")
        tok = _SPECIAL_TOKENS[name]
        if tok.enabled == enabled:
            return tok
        new_tok = SpecialToken(tok.name, tok.id, tok.layer, tok.description, enabled)
        _SPECIAL_TOKENS[name] = new_tok
        return new_tok


def disable_special_token(name: str) -> SpecialToken:
    return _set_enabled(name, False)


def enable_special_token(name: str) -> SpecialToken:
    return _set_enabled(name, True)


def get_special_token(name: str) -> SpecialToken:
    """Look up a token by name. Raises KeyError if unknown."""
    with _ID_LOCK:
        if name not in _SPECIAL_TOKENS:
            raise KeyError(f"unknown special token: {name!r}")
        return _SPECIAL_TOKENS[name]


def is_enabled(name: str) -> bool:
    """True if the token is registered AND enabled. False otherwise."""
    with _ID_LOCK:
        tok = _SPECIAL_TOKENS.get(name)
        return tok is not None and tok.enabled


def list_special_tokens(
    layer:        Optional[str]    = None,
    *,
    enabled_only: bool            = True,
) -> list[SpecialToken]:
    """List all special tokens, optionally filtered by layer.

    Args:
        layer: If given, return only tokens in this layer. If None, return
            all tokens in registration order.
        enabled_only: If True (default), skip disabled tokens. Set False to
            include disabled tokens (useful for diagnostics).

    Returns:
        Sorted list (by ID, ascending).
    """
    with _ID_LOCK:
        if layer is None:
            tokens = list(_SPECIAL_TOKENS.values())
        else:
            names = _LAYER_TOKENS.get(layer, [])
            tokens = [_SPECIAL_TOKENS[n] for n in names if n in _SPECIAL_TOKENS]
        if enabled_only:
            tokens = [t for t in tokens if t.enabled]
        return sorted(tokens, key=lambda t: t.id)


def enabled_ids() -> list[int]:
    """Return all enabled special-token IDs in ascending order.

    Includes the 3 legacy reserved tokens (256-258) plus all enabled
    plug-in tokens. Total = 3 + N_enabled_plug_ins = 70 by default.
    """
    with _ID_LOCK:
        plug_in = sorted(t.id for t in _SPECIAL_TOKENS.values() if t.enabled)
        return sorted(list(LEGACY_TOKENS.values()) + plug_in)


def vocab_size() -> int:
    """Total vocabulary size: BYTE_COUNT (256) + enabled plug-ins.

    Includes the 3 legacy tokens unconditionally (they are always enabled).
    Re-computed on every call (no caching — token disable/enable is dynamic).
    Cheap: O(N) where N <= 70.
    """
    with _ID_LOCK:
        n = sum(1 for t in _SPECIAL_TOKENS.values() if t.enabled)
        return BYTE_COUNT + len(LEGACY_TOKENS) + n


# === Auto-define the 18 plug-in tokens ===
# To ADD a new token: append one line below. Vocab grows by 1.
# To REMOVE a token: delete the line. Vocab shrinks by 1.
# To DISABLE temporarily: call disable_special_token("name") at runtime.

# Layer 1: Sequence control (3)
BOS = register_special_token("bos", "sequence", "Beginning of sequence")
EOS = register_special_token("eos", "sequence", "End of sequence / generation stop")
PAD = register_special_token("pad", "sequence", "Padding (no-op attention)")

# Layer 2: Modality (6)
MOD_IMAGE = register_special_token("mod_image", "modality", "Image payload header")
MOD_VIDEO = register_special_token("mod_video", "modality", "Video payload header")
MOD_AUDIO = register_special_token("mod_audio", "modality", "Audio payload header")
MOD_PDF   = register_special_token("mod_pdf",   "modality", "PDF payload header (Docling)")
MOD_DOCX  = register_special_token("mod_docx",  "modality", "DOCX payload header (python-docx)")
MOD_TEXT  = register_special_token("mod_text",  "modality", "Text payload header (UTF-8 bytes)")

# Layer 3: Multimodal structure (3)
FRAME_SEP       = register_special_token("frame_sep",       "mm_struct", "End of video frame")
AUDIO_CHUNK_SEP = register_special_token("audio_chunk_sep", "mm_struct", "End of audio chunk (16-bit PCM segment)")
CHANNEL_SEP     = register_special_token("channel_sep",     "mm_struct", "RGB / stereo channel separator")

# Layer 4: Chat roles (4) — Anthropic API style
ROLE_SYSTEM    = register_special_token("role_system",    "role", "<system> turn")
ROLE_USER      = register_special_token("role_user",      "role", "<user> turn")
ROLE_ASSISTANT = register_special_token("role_assistant", "role", "<assistant> turn")
ROLE_TOOL      = register_special_token("role_tool",      "role", "<tool> result turn")

# Layer 5: Reasoning (2) — Claude extended thinking
THINK_START = register_special_token("think_start", "reasoning", "Open extended-thinking block")
THINK_END   = register_special_token("think_end",   "reasoning", "Close extended-thinking block")

# Tool calls, code blocks, file paths, etc. are emitted as raw UTF-8 bytes —
# the model learns the byte sequences, not single-token shortcuts. This keeps
# the vocabulary compact and avoids dead embedding rows that receive gradients
# but are never targeted during training (masked at inference).

# === Layer descriptions (for AGENTS.md / __str__ / docs) ===
LAYER_DESCRIPTIONS: dict[str, str] = {
    "sequence":  "BOS / EOS / PAD",
    "modality":  "Image / Video / Audio / PDF / DOCX / Text payload headers",
    "mm_struct": "Frame / chunk / channel separators",
    "role":      "System / User / Assistant / Tool",
    "reasoning": "Think open/close (Claude extended thinking)",
}


# === Backward-compat aliases (for code that imports the old constant names) ===
MEDIA_START = LEGACY_TOKENS["media_start"]
MEDIA_END   = LEGACY_TOKENS["media_end"]
DOC_SEP     = LEGACY_TOKENS["doc_sep"]


# === Self-test (only when run directly) ===
if __name__ == "__main__":
    print(f"🛰️  busel special tokens — vocabulary expansion report")
    print(f"   Byte range         : [0, {BYTE_COUNT})")
    print(f"   Legacy reserved    : {sorted(LEGACY_TOKENS.values())} ({len(LEGACY_TOKENS)} tokens)")
    print(f"   Plug-in registered : {len(_SPECIAL_TOKENS)} (across {len(_LAYER_TOKENS)} layers)")
    print(f"   Vocab size         : {vocab_size()}")
    print()
    print(f"   Per-layer summary:")
    for layer, desc in LAYER_DESCRIPTIONS.items():
        toks = list_special_tokens(layer)
        n = len(toks)
        print(f"     [{layer:>9}] ×{n:>2}: {desc}")
        for t in toks:
            mark = "✓" if t.enabled else "✗"
            print(f"         {mark} {t.id:>3} = {t.name:<24} ({t.description})")
    print()
    pre = vocab_size()
    disable_special_token("think_start")
    mid = vocab_size()
    enable_special_token("think_start")
    post = vocab_size()
    assert pre == mid + 1 == post, f"vocab toggle invariant: {pre} -> {mid} -> {post}"
    print(f"     ✓ Toggle invariant: {pre} → {mid} → {post}")
