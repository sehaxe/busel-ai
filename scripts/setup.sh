#!/usr/bin/env bash
# 🛸 busel auto-setup — detects GPU and runs `uv sync --extra <match>` + maturin.
set -euo pipefail

EXTRA="${1:-}"

detect() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "cu130"
        return
    fi
    if command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ]; then
        echo "rocm63"
        return
    fi
    if [ "$(uname -m)" = "Darwin" ] || [ "$(uname -s)" = "MINGW"* ] || [ "$(uname -s)" = "CYGWIN"* ]; then
        echo "cpu"
        return
    fi
    echo "cpu"
}

if [ -z "$EXTRA" ]; then
    EXTRA=$(detect)
fi

case "$EXTRA" in
    cpu|cu118|cu126|cu128|cu130|rocm63) ;;
    *)
        echo "Unknown extra: $EXTRA"
        echo "Valid extras: cpu cu118 cu126 cu128 cu130 rocm63"
        exit 1
        ;;
esac

echo "🛸 busel setup → using extra: $EXTRA"
uv sync --extra "$EXTRA"
uv run maturin develop --release
echo "✅ done. Run: uv run python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"

