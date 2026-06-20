"""
🤖 busel — Base pretraining (next-byte CE on raw bytes)
The first stage of the pipeline. Trains a buselModel from scratch (or
resumes from a checkpoint) on byte-level data via MTP-4 + MoE + AutoPilot.

Extracted from train.py:main() so it can be invoked by the pipeline
orchestrator (tools/orchestrator.py:pipeline) in addition to the legacy
CLI mode. Behavior is preserved 1:1 with train.py.
"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from busel_logging import log_event, setup_logging
from training.stages.base import (
    StageState,
    _apply_model_profile,
    register_stage,
)


_STOP_FILE = os.environ.get("BUSEL_STOP_FILE", "/tmp/busel_stop")


def _setup_inductor_cache(cache_dir: str, clean: bool, max_gb: float = 0.0) -> str:
    import shutil

    path = os.path.abspath(os.path.expanduser(cache_dir))
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    if clean and os.path.isdir(path):
        for entry in os.listdir(path):
            try:
                p = os.path.join(path, entry)
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            except OSError:
                pass
    os.makedirs(path, exist_ok=True)

    if max_gb > 0:
        entries = []
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    sz = os.path.getsize(fp)
                    entries.append((os.path.getmtime(fp), sz, fp))
                    total += sz
                except OSError:
                    pass
        cap_bytes = int(max_gb * 1024**3)
        if total > cap_bytes:
            entries.sort()
            for _mtime, sz, fp in entries:
                if total <= cap_bytes:
                    break
                try:
                    os.remove(fp)
                    total -= sz
                except OSError:
                    pass

    return path


from .pretrain_config import buselPretrainConfig  # noqa: E402



def _enforce_stability(seed: int = 42) -> None:
    """Set TF32, cuDNN benchmark, seed (mirrors train.py:enforce_stability)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def _detect_device() -> str:
    assert torch.cuda.is_available(), "busel v9: CUDA required"
    return "cuda"


def _build_targets(byte_batch: torch.Tensor, input_length: int, stride: int = 4):
    """Compute MTP-4 targets (mirrors train.py:build_targets)."""
    targets = byte_batch[:, 1::stride][:, :input_length]
    if targets.shape[1] < input_length:
        pad = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad), value=0)
    mtp_targets = []
    for shift in (2, 3, 4):
        mt = byte_batch[:, shift::stride][:, :input_length]
        if mt.shape[1] < input_length:
            pad = input_length - mt.shape[1]
            mt = torch.nn.functional.pad(mt, (0, pad), value=0)
        mtp_targets.append(mt)
    return targets, mtp_targets


@register_stage("pretrain")
class buselPretrainStage:
    """Base pretraining stage.

    Lifecycle (per BaseStage Protocol):
        setup(cfg, profile_name, ...) → builds model, optimizer, dataloader
        run(state)                    → executes the training loop
        finalize(state)               → saves final checkpoint + log
    """

    name: str = "pretrain"

    def __init__(self) -> None:
        self.cfg: buselPretrainConfig | None = None
        self.profile_name: str = "shpak"
        self.device: str = "cpu"
        self.model: Any = None
        self.patcher: Any = None
        self.opt_engine: Any = None
        self.autopilot: Any = None
        self.loss_engine: Any = None
        self.dataloader: Any = None
        self.dataloader_iter: Any = None
        self.global_current_file_idx: int = 0
        self.global_current_byte_offset: int = 0
        self._compile_in_progress: dict = {"value": False}
        self._emergency_save_requested: dict = {"value": False}
        self.start_step: int = 0
        self.start_file_idx: int = 0
        self.start_byte_offset: int = 0
        self.no_compile: bool = False
        self.compile_mode: str = "default"
        self._oom_reductions: int = 0
        self._max_oom_reductions: int = 6
        self._last_chunk_block_step: int = -1000

    def _vram_used_mb(self) -> float:
        """Peak GPU VRAM in MB (0 on CPU). Uses max_memory_allocated for accurate peak."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def _vram_total_mb(self) -> float:
        """Total GPU VRAM in MB (0 on CPU). Cached after first call."""
        if not hasattr(self, "_vram_total_mb_cache"):
            if self.device == "cuda" and torch.cuda.is_available():
                self._vram_total_mb_cache = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            else:
                self._vram_total_mb_cache = 0.0
        return self._vram_total_mb_cache

    def _vram_used_mb(self) -> float:
        """Peak GPU VRAM in MB (0 on CPU). Uses max_memory_allocated for accurate peak."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def _ram_total_mb(self) -> float:
        """Total system RAM in MB (Linux, 0 on other platforms). Cached after first call."""
        if not hasattr(self, "_ram_total_mb_cache"):
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            self._ram_total_mb_cache = int(line.split()[1]) / 1024  # KB→MB
                            break
            except Exception:
                self._ram_total_mb_cache = 0.0
        return self._ram_total_mb_cache

    def _ram_used_mb(self) -> float:
        """Current process RSS in MB (Linux, 0 on other platforms)."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0

    def _ram_total_mb(self) -> float:
        """Total system RAM in MB (Linux, 0 on other platforms)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024  # KB→MB
        except Exception:
            pass
        return 0.0

    def _ram_available_mb(self) -> float:
        """Available system RAM in MB (Linux, 0 on other platforms)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / 1024  # KB→MB
        except Exception:
            pass
        return 0.0

    def _compute_auto_preset(self, total_params: int) -> dict:
        """v8.5 AUTO-MODE decision matrix.

        Picks a preset of optimization flags based on param count and device.
        Hierarchical: this preset is applied first, then `manual_overrides`
        from the YAML profile can override any of these on top.

        Buckets (tuned for RTX 5060 Ti 16GB / 2x 3090):
          <50M       — small sandbox (chyzh/scale_*); keep it simple
          50M-120M   — kruk/shpak/zubr; free wins only (Cautious, CLA, FlexAttn)
          120M-1B    — production; add SF + SCT rank=64 + Hestia + QuEST
          1B-7B      — target for 5060 Ti 16GB; MuonQ + SCT rank=32 + full stack
          >7B        — 2x 3090 territory; same as 1-7B but more aggressive LCSB
        """
        has_gpu = True

        if total_params < 50_000_000:
            return {
                "use_cautious": True,
                "use_schedule_free": False,
                "sct_rank": 0,
                "use_flex_attention": False,
                "use_cla": False,
                "use_hestia": False,
                "use_quest": False,
            }
        if total_params < 120_000_000:
            return {
                "use_cautious": True,
                "use_schedule_free": False,
                "sct_rank": 0,
                "use_flex_attention": False,
                "use_cla": False,
                "use_hestia": False,
                "use_quest": False,
                "use_dispersion_loss": False,
                "use_tequila": False,
            }
        if total_params < 1_000_000_000:
            return {
                "use_cautious": True,
                "use_schedule_free": True,
                "min_lr_ratio": 1.0,
                "sct_rank": 64,
                "use_flex_attention": has_gpu,
                "use_cla": False,
                "use_hestia": True,
                "use_quest": True,
            }
        if total_params < 7_000_000_000:
            return {
                "use_cautious": True,
                "use_schedule_free": True,
                "min_lr_ratio": 1.0,
                "sct_rank": 32,
                "use_flex_attention": has_gpu,
                "use_cla": False,
                "use_hestia": True,
                "use_quest": True,
            }
        return {
            "use_cautious": True,
            "use_schedule_free": True,
            "min_lr_ratio": 1.0,
            "sct_rank": 32,
            "use_flex_attention": has_gpu,
            "use_cla": False,
            "use_hestia": True,
            "use_quest": True,
            "backward_ratio": 0.4,
        }

    def _apply_auto_mode(self) -> bool:
        """Apply auto-mode preset + manual overrides.

        Must be called AFTER `self.model` is built (we need total_params) but
        BEFORE the scale gate. If any architectural field (sct_rank, use_cla,
        use_flex_attention) changed, the model is rebuilt with the new cfg.

        Returns True if the model was rebuilt.
        """
        if self.cfg.optimization_mode != "auto" or self.model is None:
            return False

        total_params = sum(p.numel() for p in self.model.parameters())
        preset = self._compute_auto_preset(total_params)

        applied: list[tuple] = []
        for k, v in preset.items():
            if not hasattr(self.cfg, k):
                continue
            if getattr(self.cfg, k) != v:
                setattr(self.cfg, k, v)
                applied.append((k, v, "preset"))

        if self.cfg.manual_overrides:
            for k, v in self.cfg.manual_overrides.items():
                if not hasattr(self.cfg, k):
                    print(f"⚠️  [AUTO-OVERRIDE]: cfg has no field {k!r}, skipping")
                    continue
                if getattr(self.cfg, k) != v:
                    setattr(self.cfg, k, v)
                    applied.append((k, v, "override"))

        arch_fields = {"sct_rank", "use_cla", "use_flex_attention"}
        arch_changed = any(k in arch_fields for k, _, _ in applied)
        rebuilt = False
        if arch_changed:
            del self.model
            del self.patcher
            if self.device == "cuda":
                torch.cuda.empty_cache()
            from model.backbone import buselModel as _Model
            from model.patching import StridedFastBLTPatcher as _Patcher
            self.patcher = _Patcher(d_model=self.cfg.d_model).to(self.device)
            self.model = _Model(self.cfg).to(self.device)
            rebuilt = True

        if applied:
            new_params = sum(p.numel() for p in self.model.parameters())
            print(f"🤖 [AUTO-MODE]: preset applied for {new_params:,} params "
                  f"(device={self.device})")
            for k, v, src in applied:
                tag = "🎛️  OVERRIDE" if src == "override" else "   preset "
                print(f"  {tag}: {k}={v}")
        return rebuilt

    def _rebuild_dataloader(self, new_batch_size: int, chunk_size: int | None = None) -> None:
        """Recreate dataloader with a new batch size (OOM recovery or auto-batcher)."""
        from data.pipeline import get_busel_dataloader
        if chunk_size is None:
            chunk_size = self.cfg.chunk_size // 4
        current_workers = 0
        if hasattr(self, 'dataloader') and self.dataloader is not None:
            current_workers = getattr(self.dataloader, 'num_workers', 0) or 0
            try:
                del self.dataloader
            except Exception:
                pass
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        self.dataloader = get_busel_dataloader(
            self.cfg.data_path,
            chunk_size=chunk_size,
            batch_size=new_batch_size,
            start_file_idx=self.global_current_file_idx,
            start_byte_offset=self.global_current_byte_offset,
            num_workers=current_workers,
        )
        self.dataloader_iter = iter(self.dataloader)

    def _load_teacher_model(self) -> None:
        """Load teacher model for SALT Knowledge Distillation."""
        import yaml as _yaml
        with open("configs/default.yaml", encoding="utf-8") as f:
            full = _yaml.safe_load(f)

        teacher_profile_name = self.cfg.salt_teacher_profile
        if teacher_profile_name not in full["profiles"]:
            print(f"⚠️ [SALT]: Teacher profile {teacher_profile_name!r} not found, skipping KD")
            return

        teacher_profile_dict = full["profiles"][teacher_profile_name]
        teacher_cfg = buselPretrainConfig.from_profile(teacher_profile_dict)

        from model.backbone import buselModel
        from model.patching import StridedFastBLTPatcher

        self.teacher_patcher = StridedFastBLTPatcher(d_model=teacher_cfg.d_model).to(self.device)
        self.teacher_model = buselModel(teacher_cfg).to(self.device)

        checkpoint_path = f"checkpoints/busel_{teacher_profile_name}_FINAL.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            from model.checkpoint import load_state_dict_safely
            load_state_dict_safely(self.teacher_model, checkpoint["model_state_dict"])
            load_state_dict_safely(self.teacher_patcher, checkpoint["patcher_state_dict"])
            print(f"🎓 [SALT]: Loaded teacher model from {checkpoint_path}")
        else:
            print(f"⚠️ [SALT]: Teacher checkpoint {checkpoint_path!r} not found, training from scratch")

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def setup(
        self,
        profile: dict | str,
        profile_name: str = "shpak",
        *,
        resume: str | None = None,
        no_compile: bool = False,
        compile_mode: str = "default",
        no_checkpointing: bool = False,
        use_ema: bool | None = None,
        ema_decay: float | None = None,
        lotus_rank: int | None = None,
        lr_multipliers: Any = None,
        **kwargs,
    ) -> None:
        """Initialize model + optimizer + dataloader for pretraining.

        Args:
            profile: Either a profile dict (from configs/default.yaml) or a
                profile NAME to look up. If a name is passed, configs/default.yaml
                is loaded and the matching profile is used.
            profile_name: Profile name to remember (used for logging + checkpoints).
            resume: Optional path to a checkpoint to resume from.
            no_compile: Disable torch.compile entirely.
            compile_mode: torch.compile mode (default|reduce-overhead|max-autotune).
            no_checkpointing: Disable gradient checkpointing.
        """
        stage_params = kwargs.pop("stage_params", None) or {}
        if use_ema is None:
            use_ema = stage_params.get("use_ema")
        if ema_decay is None:
            ema_decay = stage_params.get("ema_decay")
        if lotus_rank is None:
            lotus_rank = stage_params.get("lotus_rank")
        if lr_multipliers is None:
            lr_multipliers = stage_params.get("lr_multipliers")
        override_batch_size = stage_params.get("batch_size")
        override_chunk_size = stage_params.get("chunk_size")
        override_max_steps = stage_params.get("max_steps")
        override_warmup_steps = stage_params.get("warmup_steps")
        self._checkpoint_out = stage_params.get("checkpoint_out")
        if stage_params.get("no_compile") and not no_compile:
            no_compile = True
        if "inductor_cache_clean" in stage_params:
            self._override_cache_clean = bool(stage_params["inductor_cache_clean"])
        if isinstance(profile, str):
            with open("configs/default.yaml", encoding="utf-8") as f:
                full = yaml.safe_load(f)
            if profile not in full["profiles"]:
                raise ValueError(f"Profile {profile!r} not in configs/default.yaml")
            profile_dict = full["profiles"][profile]
            self.profile_name = profile
        else:
            profile_dict = profile

        self.cfg = buselPretrainConfig.from_profile(profile_dict)
        self.profile_name = profile_name
        self.no_compile = no_compile
        self.compile_mode = compile_mode
        self.no_checkpointing = no_checkpointing
        if use_ema is not None:
            self.cfg.use_ema = bool(use_ema)
        if ema_decay is not None:
            self.cfg.ema_decay = float(ema_decay)
        if lotus_rank is not None:
            self.cfg.lotus_rank = int(lotus_rank)
        if lr_multipliers is not None:
            self.cfg.lr_multipliers = dict(lr_multipliers)
        if override_batch_size is not None:
            self.cfg.batch_size = int(override_batch_size)
        if override_chunk_size is not None:
            self.cfg.chunk_size = int(override_chunk_size)
        if override_max_steps is not None:
            self.cfg.max_steps = int(override_max_steps)
        if override_warmup_steps is not None:
            self.cfg.warmup_steps = int(override_warmup_steps)
        if hasattr(self, "_override_cache_clean"):
            self.cfg.inductor_cache_clean = self._override_cache_clean

        _enforce_stability()
        self._logger = setup_logging()
        log_event("training_start", profile=profile_name)

        cache_path = _setup_inductor_cache(
            self.cfg.inductor_cache_dir,
            self.cfg.inductor_cache_clean,
            self.cfg.inductor_cache_max_gb,
        )
        log_event("inductor_cache_ready", path=cache_path, clean=self.cfg.inductor_cache_clean, max_gb=self.cfg.inductor_cache_max_gb)
        print(f"🗂️  Inductor cache: {cache_path} (clean={self.cfg.inductor_cache_clean}, max_gb={self.cfg.inductor_cache_max_gb})")

        self.device = _detect_device()

        if not os.path.exists(self.cfg.data_path):
            raise FileNotFoundError(f"Path {self.cfg.data_path!r} does not exist")

        from model.backbone import buselModel
        from model.layers import configure_bitlinear
        from model.patching import StridedFastBLTPatcher
        from training.autopilot import buselAutoPilot
        from training.optimizer import buselOptimizerEngine
        from training.recipe import buselLossEngine, validate_training_schedule

        self._hestia_temp = None
        if getattr(self.cfg, "use_hestia", False):
            self._hestia_temp = torch.tensor(self.cfg.hestia_init_temp, device=self.device, dtype=torch.float32)
        configure_bitlinear(
            use_tequila=self.cfg.use_tequila,
            tequila_lambda=self.cfg.tequila_lambda,
            hestia_temperature=self._hestia_temp,
        )

        self.patcher = StridedFastBLTPatcher(
            d_model=self.cfg.d_model,
            use_byteflow=getattr(self.cfg, "use_byteflow", False),
            byteflow_patches=getattr(self.cfg, "byteflow_patches", 0),
        ).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)

        # FP8: always ON for Ampere+. Skip in test mode.
        if self.device == "cuda" and not getattr(self.cfg, "_test_mode", False):
            from torchao.float8 import convert_to_float8_training
            self.model = convert_to_float8_training(self.model)
            self.patcher = convert_to_float8_training(self.patcher)
            print("⚡ [FP8]: Float8Linear enabled — 40% memory + 34% speedup")

        # Per-layer gradient offload for >3B models
        if getattr(self.cfg, "per_layer", False):
            from training.per_layer import enable_per_layer_gradient_offload
            self._per_layer_cleanup = enable_per_layer_gradient_offload(self.model, self.device)

        if self._apply_auto_mode():
            total_params = sum(p.numel() for p in self.model.parameters())
        else:
            total_params = sum(p.numel() for p in self.model.parameters())

        if self.cfg.use_hestia and self._hestia_temp is None:
            self._hestia_temp = torch.tensor(self.cfg.hestia_init_temp, device=self.device, dtype=torch.float32)
        configure_bitlinear(
            use_tequila=self.cfg.use_tequila,
            tequila_lambda=1e-3,
            hestia_temperature=None,
        )

        log_event(
            "model_initialized",
            profile=profile_name,
            device=self.device,
            total_params=total_params,
            model_size_mb=round(total_params * 2 / 1024**2, 2),
        )

        if self.cfg.max_steps == "auto" or self.cfg.max_steps is None:
            global_batch = self.cfg.batch_size * self.cfg.grad_accum_steps
            tokens_per_step = global_batch * (self.cfg.chunk_size // 4)
            # Busel Scaling Laws: two-tier model
            #   - Small models (<3B params): 37 tok/param (empirical, from 2.68M-param benchmark)
            #   - Large models (≥3B params): 80 tok/param (BitNet scaling, matches Chinchilla for fp16)
            # See README "Busel Scaling Laws" for full derivation.
            _BUSEL_THRESHOLD = 3_000_000_000  # 3B params
            _SMALL_TOKENS_PER_PARAM = 37
            _LARGE_TOKENS_PER_PARAM = 80
            if total_params >= _BUSEL_THRESHOLD:
                tokens_per_param = _LARGE_TOKENS_PER_PARAM
            else:
                tokens_per_param = _SMALL_TOKENS_PER_PARAM
            busel_tokens = tokens_per_param * total_params
            self.cfg.max_steps = math.ceil(busel_tokens / tokens_per_step)
            log_event(
                "busel_scaling_planned",
                target_tokens=busel_tokens,
                planned_steps=self.cfg.max_steps,
                tokens_per_step=tokens_per_step,
                global_batch_size=global_batch,
                tokens_per_param=tokens_per_param,
                total_params=total_params,
            )
        else:
            self.cfg.max_steps = int(self.cfg.max_steps)

        if self.cfg.warmup_steps == "auto" or self.cfg.warmup_steps is None:
            self.cfg.warmup_steps = max(50, int(0.05 * self.cfg.max_steps))

        self.cfg.max_steps, self.cfg.warmup_steps = validate_training_schedule(
            self.cfg.max_steps, self.cfg.warmup_steps
        )
        print(f"📊 Training: {self.cfg.max_steps} steps, warmup {self.cfg.warmup_steps}, batch {self.cfg.batch_size}×{self.cfg.grad_accum_steps}")

        if self.device == "cuda" and not self.no_checkpointing:
            self.model.enable_gradient_checkpointing(every=2)

        if self.device == "cuda" and not self.no_compile:
            if total_params < 10_000_000:
                print(f"⏭️  Skipping torch.compile: {total_params:,} params < 10M threshold")
            else:
                import torch._dynamo
                torch._dynamo.config.force_parameter_static_shapes = False
                torch._dynamo.config.capture_scalar_outputs = True
                print(f"⚡ torch.compile enabled (dynamic): 2-3× speedup")
                self._compile_in_progress["value"] = True
                try:
                    self.model = torch.compile(
                        self.model, fullgraph=False, dynamic=True, mode=self.compile_mode
                    )
                    self.patcher = torch.compile(
                        self.patcher, fullgraph=False, dynamic=self.cfg.dynamic_compile, mode=self.compile_mode
                    )
                except Exception as e:
                    err_str = str(e)
                    if "CUDAGraphs" in err_str or "FakeTensor" in err_str or "overwritten" in err_str:
                        try:
                            self.model = torch.compile(self.model, fullgraph=False, dynamic=self.cfg.dynamic_compile)
                            self.patcher = torch.compile(self.patcher, fullgraph=False, dynamic=self.cfg.dynamic_compile)
                        except Exception:
                            pass
                finally:
                    self._compile_in_progress["value"] = False

        self.opt_engine = buselOptimizerEngine(
            self.model, self.patcher,
            lr_muon=self.cfg.learning_rate_muon,
            lr_adamw=self.cfg.learning_rate_adamw,
            lotus_rank=self.cfg.lotus_rank,
        )
        self.autopilot = buselAutoPilot(
            self.opt_engine,
            max_lr_muon=self.cfg.learning_rate_muon,
            max_lr_adamw=self.cfg.learning_rate_adamw,
            target_wd=self.cfg.weight_decay,
            warmup_steps=self.cfg.warmup_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
            lr_schedule=self.cfg.lr_schedule,
            wsd_decay_fraction=self.cfg.wsd_decay_fraction,
            wsd_s_enabled=self.cfg.wsd_s_enabled,
            wsd_s_interval=self.cfg.wsd_s_interval,
            wsd_s_decay_steps=self.cfg.wsd_s_decay_steps,
        )
        self.loss_engine = buselLossEngine(self.cfg.vocab_size)

        self.teacher_model = None
        self.teacher_patcher = None
        if self.cfg.use_salt and self.cfg.salt_kd_steps > 0:
            self._load_teacher_model()

        if self.cfg.use_ema:
            from training.optimizer import EMA
            self.ema = EMA(self.model, decay=self.cfg.ema_decay)
            print(f"📈 EMA enabled: decay={self.cfg.ema_decay}")

        if resume and os.path.exists(resume):
            checkpoint = torch.load(resume, map_location=self.device)
            from model.checkpoint import load_state_dict_safely
            load_state_dict_safely(self.model, checkpoint["model_state_dict"])
            load_state_dict_safely(self.patcher, checkpoint["patcher_state_dict"])
            if self.ema is not None and "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"], model=self.model)
            if checkpoint.get("step") != "emergency_backup":
                self.start_step = checkpoint["step"]
                self.start_file_idx = checkpoint.get("file_idx", 0)
                self.start_byte_offset = checkpoint.get("byte_offset", 0)

        from data.pipeline import get_busel_dataloader

        # ProTrain: auto-configure batch, accum, ckpt, defrag from VRAM profile
        if getattr(self.cfg, "memory_mode", "") == "auto":
            from training.auto_memory import ProTrainMemory
            pt = ProTrainMemory(self.model, self.patcher, self.cfg, self.device)
            config = pt.configure()
            self.cfg.batch_size = config["batch_size"]
            self.cfg.grad_accum_steps = config["grad_accum"]
            self.cfg._protrain_ckpt_every = config["ckpt_every"]
            self.cfg._protrain_defrag = config["defrag"]

        current_chunk_size = self.cfg.chunk_size // 4
        self.dataloader = get_busel_dataloader(
            self.cfg.data_path,
            chunk_size=current_chunk_size,
            batch_size=self.cfg.batch_size,
            start_file_idx=self.start_file_idx,
            start_byte_offset=self.start_byte_offset,
            num_workers=0,
        )
        self.dataloader_iter = iter(self.dataloader)

        self.global_current_file_idx = self.start_file_idx
        self.global_current_byte_offset = self.start_byte_offset

        def _save_emergency_checkpoint(signum, frame):
            if self._compile_in_progress["value"] or self._emergency_save_requested["value"]:
                return
            self._emergency_save_requested["value"] = True
            try:
                log_event("emergency_save_requested", step=self.start_step, signal=signum)
            except Exception:
                pass

        signal.signal(signal.SIGINT, _save_emergency_checkpoint)
        signal.signal(signal.SIGTERM, _save_emergency_checkpoint)

    def _init_training_state(self):
        """Initialise run-level state (autocast, prefetch stream, first batch, accumulators)."""
        self._autocast_dtype = torch.bfloat16
        self._autocast_enabled = True
        self._use_cuda_stream = self.device == "cuda"
        self._prefetch_stream = torch.cuda.Stream() if self._use_cuda_stream else None
        self._spd_window: list[float] = []
        self._stop_check_interval = 50   # check graceful-stop file every N steps (syscall avoidance)
        self._watchdog_interval = 10     # check memory/ram every N steps

    def _fetch_initial_batch(self, state: StageState):
        """Return the first batch from the dataloader, or None if empty."""
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            return None

    def _check_early_stop(self, step: int, state: StageState) -> StageState | None:
        """If stop file exists, log and return a completed state. Otherwise None.
        Checks only every _stop_check_interval steps to avoid O(1) syscall per step.
        """
        if step % self._stop_check_interval != 0:
            return None
        if not os.path.exists(_STOP_FILE):
            return None
        print(f"\n🛑 Graceful stop requested (file {_STOP_FILE} present) at step {step}.")
        log_event("stop_requested", step=step, reason="stop_file_present", profile=self.profile_name)
        try:
            os.remove(_STOP_FILE)
        except OSError:
            pass
        state.step = step
        return state

    def _update_hestia_temp(self, progress: float):
        """Decay hestia temperature from init_temp → end_temp via cosine."""
        if self.cfg.use_hestia and self._hestia_temp is not None:
            import math as _math
            cosine_decay = 0.5 * (1.0 + _math.cos(_math.pi * progress))
            hestia_val = self.cfg.hestia_end_temp + (
                self.cfg.hestia_init_temp - self.cfg.hestia_end_temp
            ) * cosine_decay
            self._hestia_temp.fill_(hestia_val)

    def _compute_chunk_target(self, progress: float) -> int | None:
        """Return the target chunk_size for this progress point, or None for no change."""
        if getattr(self.cfg, "optimization_mode", "manual") == "auto":
            return None
        if getattr(self.cfg, "use_chunk_curriculum", True) is False:
            return None
        if progress < 0.15:
            return self.cfg.chunk_size // 4
        elif progress < 0.35:
            return self.cfg.chunk_size // 2
        else:
            return self.cfg.chunk_size

    def _maybe_block_chunk_growth(
        self, step: int, old_chunk: int, new_chunk: int, new_batch: int
    ) -> tuple[int, int]:
        """If VRAM/RAM is too tight, block chunk growth and return the old sizes."""
        if not (new_chunk > old_chunk and self.device == "cuda"):
            return new_chunk, new_batch
        vram_now = self._vram_used_mb()
        vram_total = self._vram_total_mb()
        ram_now = self._ram_used_mb()
        ram_total = self._ram_total_mb()
        vram_high = vram_total > 0 and vram_now / vram_total > 0.85
        ram_high = ram_total > 0 and ram_now / ram_total > 0.90
        if not (vram_high or ram_high):
            return new_chunk, new_batch

        if step - self._last_chunk_block_step >= 100:
            reason = []
            if vram_high:
                reason.append(f"VRAM {vram_now:.0f}/{vram_total:.0f}MB")
            if ram_high:
                reason.append(f"RAM {ram_now:.0f}/{ram_total:.0f}MB")
            print(f"⏸️  Chunk {old_chunk}→{new_chunk} blocked: {', '.join(reason)} too high for growth")
            self._last_chunk_block_step = step
        log_event("chunk_growth_blocked", step=step, chunk=old_chunk,
                  vram_mb=round(vram_now, 1), vram_total_mb=round(vram_total, 1),
                  ram_mb=round(ram_now, 1), ram_total_mb=round(ram_total, 1))
        return old_chunk, old_chunk  # chunk = batch inverse-proportional, so both stay

    def _rebuild_dataloader_at_size(self, chunk_size: int, batch_size: int):
        """Rebuild the dataloader with new chunk/batch sizes and return the first batch."""
        from data.pipeline import get_busel_dataloader
        self.dataloader = get_busel_dataloader(
            self.cfg.data_path,
            chunk_size=chunk_size,
            batch_size=batch_size,
            start_file_idx=self.global_current_file_idx,
            start_byte_offset=self.global_current_byte_offset,
            num_workers=0,
        )
        self.dataloader_iter = iter(self.dataloader)
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            return None

    def _handle_emergency_save(self, step: int):
        """If SIGINT/SIGTERM was caught, save emergency checkpoint and exit."""
        if not self._emergency_save_requested["value"]:
            return
        os.makedirs("checkpoints", exist_ok=True)
        try:
            from model.checkpoint import strip_compile_prefix
            torch.save(
                {
                    "model_state_dict": strip_compile_prefix(self.model.state_dict()),
                    "patcher_state_dict": strip_compile_prefix(self.patcher.state_dict()),
                    "step": step,
                    "file_idx": self.global_current_file_idx,
                    "byte_offset": self.global_current_byte_offset,
                },
                "checkpoints/latest_crash_backup.pt",
            )
            log_event("emergency_checkpoint", step=step, path="checkpoints/latest_crash_backup.pt")
        except Exception as save_err:
            print(f"❌ Emergency save failed: {type(save_err).__name__}: {save_err}")
        finally:
            self._emergency_save_requested["value"] = False
        sys.exit(0)

    def _memory_watchdog(self, step: int, current_batch_size: int, current_chunk_size: int):
        """If VRAM/RAM exceeds threshold, auto-reduce batch size.
        Only performs the check every 10 steps to avoid per-step syscall overhead.
        """
        if self._oom_reductions >= self._max_oom_reductions:
            return current_batch_size
        if step % self._watchdog_interval != 0:
            return current_batch_size
        if self.device == "cuda":
            vram_now = torch.cuda.max_memory_allocated() / 1024 ** 2
            torch.cuda.reset_peak_memory_stats()
        else:
            vram_now = 0.0
        vram_total = self._vram_total_mb()
        ram_now = self._ram_used_mb()
        ram_total = self._ram_total_mb()

        reduce_reason = None
        if vram_total > 0:
            vram_pct = vram_now / vram_total
            if vram_pct > 0.93 and current_batch_size > 1:
                reduce_reason = f"VRAM {vram_now:.0f}/{vram_total:.0f}MB ({vram_pct * 100:.0f}%)"
        if ram_total > 0:
            ram_pct = ram_now / ram_total
            if ram_pct > 0.95 and current_batch_size > 1:
                reason = f"RAM {ram_now:.0f}/{ram_total:.0f}MB ({ram_pct * 100:.0f}%)"
                reduce_reason = f"{reduce_reason} + {reason}" if reduce_reason else reason

        if not reduce_reason:
            return current_batch_size

        new_bs = max(1, current_batch_size - 2)
        self.cfg.batch_size = new_bs
        self._rebuild_dataloader(new_bs, current_chunk_size)
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"📉 {reduce_reason} → batch↓{new_bs}")
        log_event("memory_auto_reduce", step=step, new_batch=new_bs,
                  vram_mb=round(vram_now, 1), ram_mb=round(ram_now, 1))
        return new_bs

    def _log_progress(
        self, step: int, step_offset: int,
        start_time: float, last_log_time: float, last_log_tokens: int,
        cumulative_processed_tokens: int,
        accumulated_loss: float, accumulated_aux_loss: float,
        current_lr: float, dynamic_clip: float,
        current_batch_size: int, current_chunk_size: int, tokens_this_step: int,
    ) -> tuple[float, int, float]:
        """Print progress line, write metrics.jsonl, emit log_event every 10 steps.

        Returns (last_log_time, last_log_tokens, speed) for the next iteration.
        """
        speed = 0.0
        if step % 10 != 0:
            return last_log_time, last_log_tokens, speed

        current_time = time.time()
        if step_offset == 0:
            elapsed_interval = current_time - start_time
            tokens_interval = tokens_this_step * self.cfg.grad_accum_steps
        else:
            elapsed_interval = current_time - last_log_time
            tokens_interval = cumulative_processed_tokens - last_log_tokens
        speed = tokens_interval / elapsed_interval if elapsed_interval > 0 else 0.0
        last_log_time = current_time
        last_log_tokens = cumulative_processed_tokens

        steps_per_s = 10.0 / elapsed_interval if elapsed_interval > 0 else 0.0
        self._spd_window.append(steps_per_s)
        if len(self._spd_window) > 30:
            self._spd_window = self._spd_window[-30:]
        avg_steps_per_s = sum(self._spd_window) / len(self._spd_window) if self._spd_window else steps_per_s
        remaining = max(0, self.cfg.max_steps - step)
        eta_s = remaining / avg_steps_per_s if avg_steps_per_s > 0 else 0.0

        vram_mb = 0.0
        if self.device == "cuda":
            vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2

        if eta_s >= 3600:
            eta_str = f"{int(eta_s // 3600)}h {int((eta_s % 3600) // 60):02d}m"
        elif eta_s >= 60:
            eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"
        else:
            eta_str = f"{int(eta_s)}s"

        print(
            f"step {step:5d}/{self.cfg.max_steps:<5d} | "
            f"loss {accumulated_loss:7.2f}  aux {accumulated_aux_loss / max(1, self.cfg.grad_accum_steps):5.2f} | "
            f"lr {current_lr:.5f}  clip {dynamic_clip:<5.2f} | "
            f"{speed:.0f} tok/s"
            + (f"  vram {vram_mb:.0f}MB" if self.device == "cuda" else "")
            + f"  ETA {eta_str}"
        )

        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/metrics.jsonl", "a", encoding="utf-8") as log_f:
            log_f.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": accumulated_loss / max(1, self.cfg.grad_accum_steps),
                        "aux_loss": accumulated_aux_loss / max(1, self.cfg.grad_accum_steps),
                        "lr": current_lr,
                        "speed": speed,
                        "vram": vram_mb,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        log_event(
            "step_complete",
            step=step,
            loss=round(accumulated_loss / max(1, self.cfg.grad_accum_steps), 4),
            aux_loss=round(accumulated_aux_loss / max(1, self.cfg.grad_accum_steps), 4),
            lr=round(current_lr, 7),
            tokens_per_s=round(speed, 1),
            vram_mb=round(vram_mb, 1),
            batch=current_batch_size,
            chunk=current_chunk_size,
        )
        return last_log_time, last_log_tokens, speed

    def run(self, state: StageState) -> StageState:
        """Execute the pretrain training loop for cfg.max_steps."""
        if self.cfg is None:
            raise RuntimeError("setup() must be called before run()")

        self._init_training_state()
        current_batch = self._fetch_initial_batch(state)
        if current_batch is None:
            return state

        start_time = time.time()
        last_log_time = start_time
        last_log_tokens = 0
        current_chunk_size = self.cfg.chunk_size // 4
        current_batch_size = self.cfg.batch_size
        cumulative_processed_tokens = (
            self.start_step * current_batch_size * self.cfg.grad_accum_steps * current_chunk_size
        )

        for step_offset in range(self.cfg.max_steps):
            step = self.start_step + step_offset
            progress = float(step) / float(self.cfg.max_steps) if self.cfg.max_steps else 0.0

            self._update_hestia_temp(progress)

            stopped = self._check_early_stop(step, state)
            if stopped is not None:
                return stopped

            # --- chunk curriculum ---
            target_chunk = self._compute_chunk_target(progress)
            if target_chunk is not None and target_chunk != current_chunk_size:
                new_batch_size = max(1, (current_batch_size * current_chunk_size) // target_chunk)
                new_chunk, new_batch = self._maybe_block_chunk_growth(
                    step, current_chunk_size, target_chunk, new_batch_size
                )
                current_chunk_size = new_chunk
                current_batch_size = new_batch
                new_batch = self._rebuild_dataloader_at_size(current_chunk_size, current_batch_size)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if new_batch is not None:
                    current_batch = new_batch
                else:
                    break

            # --- forward/backward ---
            self.opt_engine.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_aux_loss = 0.0
            tokens_this_step = 0

            _fwd_bwd_ok = False
            for _fwd_bwd_attempt in range(3):
                try:
                    self.opt_engine.zero_grad(set_to_none=True)
                    accumulated_loss = 0.0
                    accumulated_aux_loss = 0.0
                    tokens_this_step = 0

                    _iter_batch = current_batch
                    for _ in range(self.cfg.grad_accum_steps):
                        if _iter_batch is None:
                            break
                        byte_batch, last_file_idx, last_byte_offset = _iter_batch
                        byte_batch = byte_batch.to(self.device, non_blocking=True)
                        self.global_current_file_idx = last_file_idx
                        self.global_current_byte_offset = last_byte_offset
                        input_bytes = (
                            byte_batch[:, :-self.patcher.stride]
                            if byte_batch.shape[1] > self.patcher.stride
                            else byte_batch
                        )

                        with torch.autocast(
                            device_type=self.device,
                            dtype=self._autocast_dtype,
                            enabled=self._autocast_enabled,
                        ):
                            if self.cfg.use_dispersion_loss:
                                patches, embed_for_dispersion = self.patcher(input_bytes, return_embedding=True)
                            else:
                                patches = self.patcher(input_bytes)
                            T_patches = patches.shape[1]
                            targets, mtp_targets = _build_targets(
                                byte_batch, T_patches, stride=self.patcher.stride
                            )
                            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = self.model(
                                patches, [targets] + mtp_targets[:-1], progress=progress
                            )
                            # RHO-Loss: gradient only for hard tokens
                            kr = max(0.3, min(0.7, 1.0 - progress))
                            t1_rho = self.loss_engine.compute_rho_loss(
                                logits_t1, targets, keep_ratio=kr,
                            )
                            t1_ce = self.loss_engine.compute_pretrain_loss(
                                logits_t1, targets, [], [],
                            )
                            loss = self.loss_engine.compute_pretrain_loss(
                                logits_t1, targets,
                                [logits_t2, logits_t3, logits_t4],
                                mtp_targets,
                            )
                            loss = t1_rho + (loss - t1_ce)
                            loss = loss + aux_loss.float()
                            if self.cfg.use_dispersion_loss:
                                loss = loss + self.loss_engine.compute_dispersion_loss(
                                    embed_for_dispersion,
                                    weight=self.cfg.dispersion_weight,
                                    temperature=self.cfg.dispersion_temperature,
                                )
                            if self.cfg.use_salt and self.teacher_model is not None and step < self.cfg.salt_kd_steps:
                                with torch.no_grad():
                                    teacher_patches = self.teacher_patcher(input_bytes)
                                    (teacher_logits_t1, _, _, _), _ = self.teacher_model(
                                        teacher_patches, [targets], progress=progress
                                    )
                                kd_loss = self.loss_engine.compute_kd_loss(
                                    logits_t1, teacher_logits_t1, targets,
                                    temperature=self.cfg.salt_kd_temperature,
                                    alpha=self.cfg.salt_kd_alpha,
                                )
                                loss = loss + kd_loss

                        loss = loss / self.cfg.grad_accum_steps
                        loss.backward()
                        for layer in self.model.layers:
                            if hasattr(layer, 'moe') and hasattr(layer.moe, 'update_bias'):
                                layer.moe.update_bias()
                        accumulated_loss += loss.item() * self.cfg.grad_accum_steps
                        accumulated_aux_loss += aux_loss.item()
                        tokens_this_step = current_batch_size * current_chunk_size
                        cumulative_processed_tokens += tokens_this_step

                        next_batch = None
                        if self._use_cuda_stream:
                            with torch.cuda.stream(self._prefetch_stream):
                                try:
                                    next_batch = next(self.dataloader_iter)
                                except StopIteration:
                                    next_batch = None
                        else:
                            try:
                                next_batch = next(self.dataloader_iter)
                            except StopIteration:
                                next_batch = None
                        if self._use_cuda_stream:
                            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
                        _iter_batch = next_batch

                    current_batch = _iter_batch
                    _fwd_bwd_ok = True
                    break

                except torch.cuda.OutOfMemoryError:
                    if self.device != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    self._oom_reductions += 1
                    old_bs = current_batch_size
                    current_batch_size = max(1, current_batch_size - 2)
                    self.cfg.batch_size = current_batch_size
                    self._rebuild_dataloader(current_batch_size, current_chunk_size)
                    print(
                        f"⚠️  OOM at step {step}! batch {old_bs}→{current_batch_size} "
                        f"(attempt {self._oom_reductions}/{self._max_oom_reductions})"
                    )
                    log_event(
                        "oom_recovery",
                        step=step,
                        old_batch=old_bs,
                        new_batch=current_batch_size,
                        vram_mb=round(self._vram_used_mb(), 1),
                    )
                    cumulative_processed_tokens -= tokens_this_step
                    continue

            if not _fwd_bwd_ok:
                print(f"❌ OOM: gave up after {self._oom_reductions} reductions at step {step}")
                log_event("oom_gave_up", step=step, batch=current_batch_size)
                break

            # --- optimiser step ---
            dynamic_clip = self.autopilot.before_step(self.model, step, self.cfg.max_steps)
            if self.device == "cuda":
                self.autopilot.inject_noise(self.model)
            current_lr, _ = self.autopilot.update_parameters(step, accumulated_loss, self.cfg.max_steps)
            
            if getattr(self.cfg, "per_layer", False):
                from training.per_layer import per_layer_optimizer_step
                per_layer_optimizer_step(self.model, self.opt_engine, self.device)
            else:
                self.opt_engine.step()
                
            if self.ema is not None:
                self.ema.update(self.model)

            # --- memory watchdog ---
            current_batch_size = self._memory_watchdog(step, current_batch_size, current_chunk_size)

            # --- emergency checkpoint ---
            self._handle_emergency_save(step)

            # --- logging ---
            last_log_time, last_log_tokens, speed = self._log_progress(
                step, step_offset, start_time, last_log_time, last_log_tokens,
                cumulative_processed_tokens, accumulated_loss, accumulated_aux_loss,
                current_lr, dynamic_clip, current_batch_size, current_chunk_size,
                tokens_this_step,
            )

            # --- scheduled checkpoint ---
            if step % 100 == 0 and step > 0:
                self._save_scheduled_checkpoint(
                    step, last_file_idx, last_byte_offset, accumulated_loss, current_lr
                )

            # --- state update ---
            state.step = step
            state.best_loss = (
                min(state.best_loss, accumulated_loss) if accumulated_loss > 0 else state.best_loss
            )
            state.metrics = {
                "loss": accumulated_loss,
                "lr": current_lr,
                "tokens_per_s": speed if step % 10 == 0 else state.metrics.get("tokens_per_s", 0.0),
            }

            # Validation
            val_every = getattr(self.cfg, "val_every", 0)
            if val_every > 0 and step > 0 and step % val_every == 0:
                self._run_validation(step, state)

        state.last_checkpoint_path = None
        return state

    def _run_validation(self, step, state):
        """Quick perplexity check on current data."""
        self.model.eval()
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.model.train()
            return
        byte_batch = batch.to(self.device)
        input_bytes = byte_batch[:, :-self.patcher.stride] if byte_batch.shape[1] > self.patcher.stride else byte_batch
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            patches = self.patcher(input_bytes)
            T = patches.shape[1]
            targets = byte_batch[:, self.patcher.stride::self.patcher.stride][:, :T]
            if targets.shape[1] < T:
                targets = F.pad(targets, (0, T - targets.shape[1]))
            (logits_t1, _, _, _), _ = self.model(patches)
            val_loss = self.loss_engine.compute_pretrain_loss(logits_t1, targets, [], [])
        ppl = torch.exp(val_loss).item()
        state.metrics["val_loss"] = val_loss.item()
        state.metrics["val_ppl"] = ppl
        print(f"  [val] step {step}: loss {val_loss.item():.4f}, ppl {ppl:.1f}")
        self.model.train()

    def _save_scheduled_checkpoint(self, step, last_file_idx, last_byte_offset, accumulated_loss, current_lr) -> None:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/busel_{self.profile_name}_step_{step}.pt"
        temp_path = checkpoint_path + ".tmp"
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "patcher_state_dict": self.patcher.state_dict(),
            "step": step,
            "file_idx": last_file_idx,
            "byte_offset": last_byte_offset,
            "loss": accumulated_loss,
            "lr_muon": current_lr,
            "profile": self.profile_name,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        try:
            # Async: clone to CPU, save in background thread
            import threading
            ckpt_cpu = {k: v.cpu().clone() if hasattr(v, 'cpu') else v for k, v in ckpt.items()}
            def _bg_save():
                torch.save(ckpt_cpu, temp_path)
                sz = os.path.getsize(temp_path)
                if sz >= 2_000_000:
                    os.rename(temp_path, checkpoint_path)
                    print(f"[save] Checkpoint: {checkpoint_path} ({sz/1024/1024:.1f} MB)")
            threading.Thread(target=_bg_save, daemon=True).start()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            log_event("checkpoint_failed", step=step, error=str(e))

    def finalize(self, state: StageState) -> StageState:
        """Save the final checkpoint + emit stage_complete event."""
        if self.cfg is None or self.model is None:
            return state

        os.makedirs("checkpoints", exist_ok=True)
        final_path = self._checkpoint_out or f"checkpoints/busel_{self.profile_name}_FINAL.pt"
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "patcher_state_dict": self.patcher.state_dict(),
            "step": state.step,
            "file_idx": self.global_current_file_idx,
            "byte_offset": self.global_current_byte_offset,
            "profile": self.profile_name,
            "config": self.cfg.__dict__,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        try:
            torch.save(ckpt, final_path)
            print(f"[save] Final checkpoint: {final_path}")
            log_event(
                "stage_complete",
                stage=self.name,
                profile=self.profile_name,
                total_steps=state.step,
                final_path=final_path,
            )
            state.last_checkpoint_path = final_path
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

        return state


def _cleanup_old_checkpoints(profile_name: str, keep_last_n: int) -> int:
    """Keep only the most recent `keep_last_n` scheduled checkpoints for this profile.

    Scheduled checkpoints follow the pattern `busel_<profile>_step_<N>.pt`.
    The FINAL checkpoint (`busel_<profile>_FINAL.pt`) and emergency backups
    (`latest_crash_backup.pt`) are NEVER touched.

    Returns the number of files deleted. Returns 0 if keep_last_n <= 0
    (caller asked to keep everything) or if there is nothing to clean up.
    """
    if keep_last_n <= 0:
        return 0
    pattern = f"checkpoints/busel_{profile_name}_step_*.pt"
    files = glob.glob(pattern)
    if len(files) <= keep_last_n:
        return 0

    files.sort(key=lambda p: os.path.getmtime(p))
    to_delete = files[:-keep_last_n] if len(files) > keep_last_n else []
    deleted = 0
    bytes_freed = 0
    for f in to_delete:
        try:
            size = os.path.getsize(f)
            os.remove(f)
            deleted += 1
            bytes_freed += size
        except OSError:
            pass
    if deleted > 0:
        log_event(
            "checkpoint_cleanup",
            deleted=deleted,
            kept=keep_last_n,
            freed_mb=round(bytes_freed / 1024 / 1024, 2),
        )
    return deleted
