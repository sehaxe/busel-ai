"""🛰️ busel MULTIMODAL — encoders for image, video, audio, PDF, docx.

Example:
    from multimodal import build_encoder_for
    enc = build_encoder_for("photo.jpg")
    blob = enc.encode_file("photo.jpg")

Special tokens (v5.4, 67 plug-in + 3 legacy, 12 layers):
    from multimodal import special_tokens
    print(special_tokens.vocab_size())         # 326
    disable_special_token = special_tokens.disable_special_token
    disable_special_token("think_start")        # token ID 276, vocab -> 325
"""
from multimodal.encoders import (
    ImageEncoder,
    VideoEncoder,
    AudioEncoder,
    PDFEncoder,
    DocxEncoder,
    TextEncoder,
    auto_encode,
    build_encoder_for,
    list_encoders,
)
from multimodal import special_tokens

__all__ = [
    "ImageEncoder",
    "VideoEncoder",
    "AudioEncoder",
    "PDFEncoder",
    "DocxEncoder",
    "TextEncoder",
    "auto_encode",
    "build_encoder_for",
    "list_encoders",
    "special_tokens",
]
