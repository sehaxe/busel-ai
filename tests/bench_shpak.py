"""
⏱️  Быстрый замер скорости shpak.
Создаёт модель 55.8M, батч 512, чанк 4096, compile, grad_ckpt.
Печатает время шага и tok/s — как training loop.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from training.stages.pretrain import buselPretrainStage, buselPretrainConfig
from training.stages.base import StageState
from training.stages.pretrain import _setup_inductor_speed_config

# Грузим shpak из YAML
with open("configs/default.yaml") as f:
    full = yaml.safe_load(f)
profile_dict = full["profiles"]["shpak"]

# === Сетап как в training loop ===
stage = buselPretrainStage()
stage.setup(
    profile=profile_dict,
    profile_name="shpak",
    no_compile=False,
)
# форсируем chunk_size = 4096 (без curriculum)
stage.cfg.chunk_size = 4096
# пересоздаём даталоадер с chunk=4096
from data.pipeline import get_busel_dataloader
stage.dataloader = get_busel_dataloader(
    stage.cfg.data_path,
    chunk_size=4096,
    batch_size=stage.cfg.batch_size,
    start_file_idx=stage.start_file_idx,
    start_byte_offset=stage.start_byte_offset,
    num_workers=0,
)
stage.dataloader_iter = iter(stage.dataloader)
stage.cfg.max_steps = 5
stage.cfg.warmup_steps = 1
stage.start_step = 0

# run() сам сделает compile warmup, потом 5 шагов с замерами
state = StageState()
t0 = time.time()
state = stage.run(state)
total = time.time() - t0

print(f"\n{'='*60}")
params = sum(p.numel() for p in stage.model.parameters())
print(f"Модель: {params/1e6:.1f}M params")
print(f"batch={stage.cfg.batch_size} chunk=4096")
if hasattr(stage, '_spd_window') and stage._spd_window:
    avg_sps = sum(stage._spd_window) / len(stage._spd_window)
    print(f"Среднее: {avg_sps:.2f} шаг/с ({1/avg_sps*1000:.0f}ms/шаг)")
    print(f"tok/s: {stage.cfg.batch_size * 4096 * avg_sps:.0f}")
print(f"Всего {stage.cfg.max_steps} шагов за {total:.1f}s")
print(f"{'='*60}")
