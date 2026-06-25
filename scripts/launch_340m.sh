#!/bin/bash
# Launch busel 340M training with compile
# kill any existing
pkill -f "cli.py pipeline" 2>/dev/null
sleep 1
cd /home/sehaxe/busel-ai
rm -f checkpoints/training_340m.log
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=2
export NO_COLOR=1
uv run python cli.py pipeline --name sovereign-340m --config-dir configs/pipelines > checkpoints/training_340m.log 2>&1
