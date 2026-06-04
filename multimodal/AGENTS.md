# multimodal/ — Any-to-Token Encoders

**Scope:** Encoders that turn image, video, audio, PDF, and docx files into the same `list[int]` token stream the byte-level model already consumes (vocab=326). Plus the **special-token registry** that defines the 70-token sovereign vocabulary.

## STRUCTURE
```
multimodal/
├── __init__.py        # public API: build_encoder_for, auto_encode, list_encoders, special_tokens
├── encoders.py        # 6 encoder classes + registry + dispatch (335 LOC)
└── special_tokens.py  # SpecialToken dataclass + 70-token plug-in registry (490 LOC)
```

## VOCABULARY (v5.4.0 — 326 tokens)
- **0-255** — raw UTF-8 bytes
- **256** `MEDIA_START` — legacy payload start
- **257** `MEDIA_END` — legacy payload end
- **258** `DOC_SEP` — cross-document boundary
- **259-325** — 67 plug-in special tokens across 12 functional layers
- See `special_tokens.py` for the full breakdown; `list_special_tokens()` returns the live registry; `enabled_ids()` is the 70 ints that must be masked in inference logits.

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add a new modality | `encoders.py` → new `@register("encoder", "...")` class | Must return `list[int]` with values in `[0, 326)` |
| Add a new special token | `special_tokens.py` → `register_special_token(name, layer, description)` | Plug-in: auto-allocates next available ID, grows vocab |
| Disable a special token | `special_tokens.py` → `disable_special_token(name)` | Vocab stays, ID becomes unused; inference mask updates |
| Change image size | `encoders.py` → `ImageEncoder.size` | Default `(32, 32)`, fixed at 3072 payload tokens |
| Change video frame cap | `encoders.py` → `VideoEncoder.max_frames` | Default 8 frames, evenly subsampled |
| Change audio length cap | `encoders.py` → `AudioEncoder.max_seconds` | Default 8.0 s; no resampling (header stores `sr`) |
| Route by extension | `encoders.py` → `build_encoder_for` | Falls back to `TextEncoder` on unknown |
| Look up by name | `busel_registry.get("encoder", name)` | `list_registered("encoder")` enumerates all |
| Swap to a faster codec | `encoders.py` → set `HAS_CV2` first; `cv2` is the fast path | PIL/imageio are the slow fallbacks |

## KEY CLASSES / FUNCTIONS
| Symbol | Type | Location | Role |
|---|---|---|---|
| `SpecialToken` | dataclass | special_tokens.py | Frozen dataclass: `name`, `id`, `layer`, `description`, `enabled`. Int-coercible. |
| `vocab_size()` | function | special_tokens.py | Dynamic vocab size (256 + 3 legacy + enabled plug-ins). Currently 326. |
| `enabled_ids()` | function | special_tokens.py | Sorted 70-int list for inference logits mask. |
| `get_special_token(name)` | function | special_tokens.py | Lookup by name. |
| `register_special_token(name, layer, description)` | function | special_tokens.py | Add new token at runtime. Auto-allocates next ID. |
| `disable_special_token(name)` / `enable_special_token(name)` | function | special_tokens.py | Toggle active state. |
| `list_special_tokens()` | function | special_tokens.py | Flat list of all registered tokens. |
| `layer_summary()` | function | special_tokens.py | `dict[layer, count]`. |
| `MEDIA_START`, `MEDIA_END`, `DOC_SEP` | constants | encoders.py | Legacy token IDs `256`, `257`, `258` |
| `MOD_IMAGE`, `MOD_VIDEO`, `MOD_AUDIO`, `MOD_PDF`, `MOD_DOCX`, `MOD_TEXT` | constants | special_tokens.py | Modality markers (263-268). Encoder payload prefix. |
| `BOS`, `EOS`, `PAD`, `UNK` | constants | special_tokens.py | Sequence control (259-262) |
| `FRAME_SEP`, `AUDIO_CHUNK_SEP`, `CHANNEL_SEP` | constants | special_tokens.py | Multimodal structure (269-271) |
| `ROLE_*`, `THINK_*`, `PLAN_*`, `CODE_BLOCK_*`, `DIFF_*` | constants | special_tokens.py | Chat/coder semantics (272-283) |
| `TOOL_CALLS_*`, `TOOL_INVOKE_*`, `TOOL_PARAM_*`, `TOOL_RESULTS_*`, `TOOL_RESULT_*`, `TOOL_ERROR_*` | constants | special_tokens.py | Anthropic-style XML tool format (284-295) |
| `TOOL_BASH`, `TOOL_READ`, `TOOL_WRITE`, `TOOL_EDIT`, `TOOL_GREP`, `TOOL_GLOB`, `TOOL_FETCH`, `TOOL_SEARCH`, `TOOL_TASK`, `TOOL_TODO`, `TOOL_LSP`, `TOOL_ASK` | constants | special_tokens.py | opencode 12-tool vocabulary (296-307) |
| `TODO_*`, `TASK_*`, `FILE_PATH_*`, `URL_*`, `CITE_*` | constants | special_tokens.py | Task tracking + references (308-317) |
| `SUBAGENT_*`, `STATUS_*` | constants | special_tokens.py | Subagent delegation + result status (318-325) |
| `IMAGE_BYTES` | constant | encoders.py | `32 * 32 * 3 = 3072` payload tokens per image |
| `HAS_CV2`, `HAS_PIL`, `HAS_IMAGEIO`, `HAS_SOUNDFILE`, `HAS_DOCX`, `HAS_DOCLING` | flags | encoders.py | Lazy import guards (`True` if dep installed) |
| `ImageEncoder` | class | encoders.py | **cv2 (fast) → PIL (fallback)** → 32×32 RGB → `[MOD_IMAGE, *3072 pixels*, MEDIA_END]` |
| `VideoEncoder` | class | encoders.py | **cv2.VideoCapture (fast) → imageio (fallback)** → max_frames subsampled → `[MOD_VIDEO, count, FRAME_SEP, *frames*, MEDIA_END]` |
| `AudioEncoder` | class | encoders.py | soundfile → 16-bit PCM → `[MOD_AUDIO, sr, n, sw, AUDIO_CHUNK_SEP, *pcm*, MEDIA_END]` |
| `PDFEncoder` | class | encoders.py | Docling → markdown → `[MOD_PDF, *utf8*, MEDIA_END]` |
| `DocxEncoder` | class | encoders.py | python-docx → plain text → `[MOD_DOCX, *utf8*, MEDIA_END]` |
| `TextEncoder` | class | encoders.py | UTF-8 → `[MOD_TEXT, *bytes, MEDIA_END]` (modality prefix; legacy bare bytes still decodable) |
| `build_encoder_for(path)` | function | encoders.py | Dispatch by file extension; falls back to `TextEncoder` |
| `auto_encode(path)` | function | encoders.py | `build_encoder_for(path).encode_file(path)` |
| `list_encoders()` | function | encoders.py | `busel_registry.list_registered("encoder")` |

## CONVENTIONS
- **Output type:** `list[int]` (NOT `bytes`). Python `bytes` cannot represent values ≥ 256, but the model vocab is 326. The collate function in `data/pipeline.py:collate_busel_batch` handles `list` input via its `else` branch (produces `int32` tensor).
- **Marker tokens:** `MEDIA_START=256`, `MEDIA_END=257`, `DOC_SEP=258` (legacy), plus `MOD_*` prefixes (263-268) and 60+ chat/tool/reference/status tokens. All are integer token IDs in the model's embedding table, not bytes.
- **Payload range:** Real bytes 0-255; markers 256-325. Every encoder must respect this and never produce values outside `[0, 326)`.
- **Modality prefix contract:** every payload-bearing encoder prepends a `MOD_*` token so the model knows what's coming. Legacy bare `[256, ..., 257]` is still accepted by the decoder for backward compat.
- **Registry pattern (special tokens):** plug-in via `register_special_token(name, layer, description)`. Auto-allocates the next available ID (256+legacy_count, then sequential plug-ins). The full 70-token vocabulary is auto-defined at import time in `special_tokens.py`.
- **Registry pattern (encoders):** every encoder class is decorated with `@register("encoder", name)`. The `name` attribute MUST match the registry key. Use `override=True` to replace a registered encoder.
- **Round-trip lossless:** Each `encode()` is followed by a `decode()` that returns the original artifact (for inspection / debugging). Lossy transforms (e.g. video subsampling, audio truncation) are documented in the docstring.
- **Fast path priority:** `cv2` is the default for image/video. `PIL` and `imageio` are fallbacks (3-5× slower). The class falls back silently when `HAS_CV2` is False.
- **Graceful fallback:** `build_encoder_for` tries each encoder in order; if a heavy dep is missing, it silently falls through to `TextEncoder`.
- **Dispatch by extension:** case-insensitive; the extension is matched against `cls.extensions`. Unknown extensions → `TextEncoder`.

## ANTI-PATTERNS
- **NEVER** return `bytes` from `encode()` — Python's `bytes` cannot represent marker tokens ≥ 256. This will raise `ValueError: bytes must be in range(0, 256)`.
- **NEVER** use `bytearray.append(256)` — same reason. The fix is `list.append(256)`.
- **NEVER** hardcode a token ID — always use `MOD_IMAGE`, `BOS`, etc. from `special_tokens`. The IDs are auto-allocated and may shift when tokens are disabled or added.
- **NEVER** mix `np.uint8` arrays into a token stream without casting to `int` first. The `collate_busel_batch` function expects Python `int`.
- **NEVER** register two encoders with the same `name` attribute without `override=True` — `busel_registry.register` raises `KeyError` on collision.
- **NEVER** register two special tokens with the same `name` — `register_special_token` raises `ValueError` on collision.
- **NEVER** bypass the special-token registry and hardcode `256/257/258` in encoders — use the `MOD_*` constants. (Legacy 256/257/258 are kept for backward-compat decode only.)
- **NEVER** encode an unbounded file (e.g. a 4K video) without subsampling. Use `VideoEncoder.max_frames` and `AudioEncoder.max_seconds`.
- **NEVER** write a custom collate function — use `data.pipeline.collate_busel_batch`, which already handles `list` input.
- **NEVER** add a new modality without first adding the corresponding `try: import X / except ImportError: HAS_X = False` block + extension tuple.
- **NEVER** import `multimodal.encoders` at module top of `train.py` — the multimodal stack is only required when the data path contains non-text files. Use the `HAS_MULTIMODAL_DEPS` pattern from the test suite.
- **NEVER** depend on the order of `cls.extensions` matching — use `os.path.splitext(path)[1].lower()` and a set lookup.
- **NEVER** use PIL for hot-path image resize when cv2 is available — cv2 is ~3× faster on 1024² images and ~6× faster on 256².
- **NEVER** use `imageio.imiter` to count video frames — it forces a full decode pass. Use `cv2.CAP_PROP_FRAME_COUNT` for O(1) metadata lookup.

## NOTES
- **Why `list[int]` and not `bytes`:** The model's `vocab_size = 326` (256 real bytes + 3 reserved legacy tokens + 67 plug-in special tokens). The reserved tokens are integer token IDs in the embedding table — they are NOT representable in Python's `bytes` type. Returning a `list[int]` is the only way to express the multimodal stream in Python.
- **Special-token design (12 layers, 70 tokens):**
  1. **sequence** (4): `BOS EOS PAD UNK` — control
  2. **modality** (6): `MOD_IMAGE MOD_VIDEO MOD_AUDIO MOD_PDF MOD_DOCX MOD_TEXT` — what kind of payload
  3. **mm_struct** (3): `FRAME_SEP AUDIO_CHUNK_SEP CHANNEL_SEP` — payload structure
  4. **role** (4): `ROLE_SYSTEM ROLE_USER ROLE_ASSISTANT ROLE_TOOL` — chat turn ownership
  5. **reasoning** (4): `THINK_START THINK_END PLAN_START PLAN_END` — chain-of-thought / planning
  6. **code** (4): `CODE_BLOCK_START CODE_BLOCK_END DIFF_START DIFF_END` — code/diff regions
  7. **tool_xml** (12): Anthropic-style `<function_calls>...</function_calls>` envelope (start/end per tag × 6 tags)
  8. **tool** (12): 12 opencode tools — `TOOL_BASH TOOL_READ TOOL_WRITE TOOL_EDIT TOOL_GREP TOOL_GLOB TOOL_FETCH TOOL_SEARCH TOOL_TASK TOOL_TODO TOOL_LSP TOOL_ASK`
  9. **task** (4): `TODO_START TODO_END TASK_DONE TASK_PENDING` — todo list state
  10. **reference** (6): `FILE_PATH_START END` + `URL_START END` + `CITE_START END` — references
  11. **subagent** (4): `SUBAGENT_START END` + `SUBAGENT_RESULT_START END` — delegate_task format
  12. **status** (4): `STATUS_SUCCESS ERROR TIMEOUT CANCELLED` — result status
- **Image dimensions are fixed at 32×32.** The model expects exactly 3072 payload tokens per image. Changing the image size requires retraining.
- **Video subsampling** uses `step = max(1, n_total // max_frames)` — videos with fewer than `max_frames` frames yield all frames.
- **Audio header** stores the *source* sample rate (no resampling). The 16-bit PCM payload is `int16` little-endian.
- **PDF support requires `uv add docling`** — heavyweight dep; lazy-imported inside the encoder.
- **Cross-document boundary** is `DOC_SEP = 258` (= `b"\n\n"`). The data loader can insert this between concatenated documents to let the model learn document boundaries.
- **Round-trip property:** every encoder is designed to be lossless for the data it can carry. The only lossy step is *input pre-processing* (image resize, video subsampling, audio truncation), not the encoding itself.
- **Integration point:** `data/pipeline.py:buselOmnivoreTextExtractor.__init__` uses `list` (not `bytearray`) for `self.raw_bytes`. This fixes a latent bug where `bytearray.append(256)` would have raised `ValueError`. The collate function already supported `list` input.
- **Performance (RTX 5060 Ti, validation profile, batch=256 ctx=256, cv2 4.13):**
  - Image encoding: **0.44 ms/image** (256² → 32×32, 100 imgs in 44 ms)
  - Video encoding: **4.5 ms for 60 frames @ 128×128** (extracts 8 evenly-spaced frames)
  - PIL fallback: ~2.5 ms/image (5.7× slower)
- **Tests:** 13 tests in `tests/test_suite.py` (prefix `MM-1` … `MM-13`); cover registry, round-trips, marker validation, layout losslessness, end-to-end pipeline collate, and cv2 fast-path throughput. Plus **13 vocab tests (MM-14 … MM-26)** for the 70-token registry.
