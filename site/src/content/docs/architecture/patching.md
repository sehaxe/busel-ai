---
title: Byte-level patching (ByteFlow)
description: How 4 raw bytes become one model patch — adaptive pooling, boundary detection, and why we don't use BPE.
sidebar:
  order: 3
---

busel has no tokenizer. The model consumes raw UTF-8 bytes and the
`ByteFlowPatcher` compresses 4 bytes into one patch via adaptive
pooling and boundary detection. This
page explains the patcher, the multimodal marker scheme, and why
the project refuses to add BPE.

## The tokenisation budget

A traditional BPE model has 32 000–200 000 tokens in its vocabulary.
For a 50 M-param model, **30–40 % of the parameters are in the
embedding matrix alone** — used once at the input, never again. The
BitNet team call this the *"embedding tax"*. Byte-level models
have a 277-token vocabulary and pay no such tax.

The cost is a 4× longer input sequence: an English sentence that
BPE encodes as 30 tokens becomes 120 bytes. The patcher compresses
those 120 bytes back into 30 patches, restoring the *effective*
context density while keeping the *vocabulary* tiny.

## The patcher

`ByteFlowPatcher` in `model/patching.py`:

```python
ByteFlowPatcher(stride=4, d_model=384)
```

The forward pass:

```text
  raw bytes  (B, T) uint8
      │
      │  nn.Embedding(277, d_byte)  ← learnable byte embedding
      ▼
  embedded  (B, T, d_byte)
      │
      │  Conv1d(d_byte, d_byte, kernel=5, stride=4, padding=3, groups=d_byte)
      │  ← causal-left padding:  F.pad(x, (3, 0))
      ▼
  conv out  (B, d_byte, T/4)
      │
      │  GLU gate:  sigmoid(gate_proj(x)) * up_proj(x)
      │  where gate_proj, up_proj : BitLinear_a4_8(d_byte, d_byte)
      ▼
  patches  (B, T/4, d_byte)
      │
      │  Linear projection: BitLinear_a4_8(d_byte, d_model)
      ▼
  patches  (B, T/4, d_model)
```

### Why adaptive pooling + boundary detection?

`stride=4` with adaptive pooling means the patcher dynamically pools
bytes based on local entropy — low-entropy regions (repeated bytes,
whitespace runs) are pooled more aggressively; high-entropy regions
(boundaries between tokens, punctuation) are preserved. Boundary
detection identifies byte transitions that likely mark word or
token edges, giving the model explicit positional cues.

The stride `4` is the byte-to-patch compression ratio. It is
**hard-coded**; changing it requires re-deriving the MTP-4 target
alignment in `train.py:build_targets`.

### Why the GLU gate?

A vanilla 1D conv would treat every byte equally. The gate
(sigmoid on a `BitLinear_a4_8(d_byte, d_byte)` projection, multiplied
element-wise with a parallel `BitLinear_a4_8(d_byte, d_byte)`)
learns to *suppress* bytes that don't carry information — typically
whitespace, markup, and repeated punctuation. This is busel's
response to the "spelling tax" of BPE: at the byte level we
*can* distinguish " " from "a" in 1.58 bits because the model
learns to ignore " ".

## The multimodal markers

Twenty-one "special" byte values (tokens 256-276) mark non-text content in the byte
stream. These include media start/end markers, modality identifiers, and
content metadata. See [Multimodal encoding](/busel-ai/data/multimodal/)
for the full 21-token special vocab breakdown.

## Why no BPE

Three reasons:

1. **Parameter efficiency.** A 277-token embedding matrix for
   50 M params is 0.2 % of the model. A 32 000-token embedding
   would be 25 %.
2. **Robustness to noise.** BPE on out-of-vocabulary words falls
   back to character-level tokens, breaking consistency. Bytes
   never have an OOV.
3. **Multimodality for free.** Any file format is a byte stream;
   the model never has to know about JPEG vs PNG vs raw.

The trade-off is **sequence length** — a 4 096-byte input is only
1 024 patches. The patcher compresses back to 1 024 patches, but
you can never have more than 1 024 patches of context (≈ 1 024
* 4 = 4 096 bytes ≈ 1 024 English tokens). The MLA latent cache
helps (4 096 patches fits in ~98 MB), but busel is not the right
tool for million-token contexts.

## The hard constraint

`vocab_size` is exactly `277` everywhere it appears. The
embedding, the MTP head, the loss engine, the data loader — all
of them assume 256 byte values + 21 multimodal specials. Changing
this number is an anti-pattern; the model will silently misbehave
if you do.

## Where to look in the code

| Symbol                         | File                  | Role                          |
|--------------------------------|-----------------------|-------------------------------|
| `ByteFlowPatcher`              | `model/patching.py`   | The whole patcher             |
| `build_targets`                | `train.py`            | Aligns MTP targets to stride  |
| `buselOmnivoreTextExtractor`   | `data/pipeline.py`    | Image + PDF + JSON + parquet  |
| `RustByteStreamDataset`        | `data/pipeline.py`    | Mmap'd byte stream iterator   |

## See also

- [Architecture overview](/busel-ai/architecture/overview/) —
  where the patcher sits in the full pipeline.
- [Data → Multimodal encoding](/busel-ai/data/multimodal/) — how
  images and PDFs enter the byte stream.
- [Data → Pipeline](/busel-ai/data/pipeline/) — the loader side.
