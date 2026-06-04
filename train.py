"""🛸 busel training — DEPRECATED SHIM (v5.8.0+).

The legacy cybernetic orchestrator (650 LOC) was removed; training now goes
through the multi-stage pipeline. Use either:

    uv run python cli.py pipeline --name pretrain-only   # canonical
    uv run train.py --profile shpak                       # this shim

The shim supports --profile, --resume, --max-steps, --warmup-steps.
Other legacy flags (--no-compile / --compile-mode / --no-checkpointing / --seed)
are dropped — configure them in configs/pipelines/<name>.yaml.
"""
import sys
from tools.orchestrator import train_single_profile
sys.exit(train_single_profile(sys.argv[1:]))
