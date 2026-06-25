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
import warnings
from dataclasses import dataclass
from typing import Any

# ponytail: suppress repetitive Triton deprecation warnings (fla library, not our code)
warnings.filterwarnings("ignore", "tl.make_block_ptr is deprecated")
warnings.filterwarnings("ignore", "builtin max on non-scalar")

import torch
import torch.nn.functional as F
import yaml

from busel_logging import log_event, setup_logging
from model.layers import retract_all
from training.stages.base import (
    StageState,
    _apply_model_profile,
    register_stage,
)


_STOP_FILE = os.environ.get("BUSEL_STOP_FILE", "/tmp/busel_stop")


def _setup_inductor_speed_config(device: str) -> None:
    """Configure torch.compile (dynamo + inductor) for fastest compilation
    and broadest device compatibility (CUDA / CPU / ROCm).

    Key tunings:
      - compile_threads: parallel compilation (all cores, cap at 32)
      - fx_graph_cache: persists across runs → 2-3× faster re-compiles
      - coordinate_descent_tuning + benchmark_kernel: skip slow autotuning
      - cache_size_limit: per-code-object cache; high value prevents eager
        fallback when LCSB's selective torch.no_grad() triggers dynamo
        recompilation on train/eval switches.
      - add_global_state_guard: patched to no-op to suppress GLOBAL_STATE
        recompilation from grad_mode changes between train/eval (vLLM pattern).
        LCSB uses torch.no_grad() internally — dynamo handles it correctly
        without needing a guard.
    """
    import torch._inductor.config as _ic
    import torch._dynamo.config as _dc
    import torch._C._dynamo.guards as _guards

    # --- Speed: reduce compilation time ---
    _ic.compile_threads = min(32, os.cpu_count() or 4)
    _ic.coordinate_descent_tuning = False      # skip slow autotuning
    _ic.benchmark_kernel = False               # skip kernel benchmarking
    _ic.fx_graph_cache = True                  # persist FX graphs across runs

    # --- Dynamo: reduce recompilations under shape / grad-mode changes ---
    _dc.accumulated_cache_size_limit = 256
    _dc.cache_size_limit = 2048           # per-code-object cache (default 64)
    _dc.force_parameter_static_shapes = False
    _dc.capture_scalar_outputs = True

    # --- Treat int attrs on nn.Module as dynamic (prevents recompilation from layer_idx) ---
    _dc.allow_unspec_int_on_nn_module = True

    # --- Suppress GLOBAL_STATE guard (grad_mode changes from LCSB's no_grad) ---
    # LCSB selectively wraps layer forwards in torch.no_grad(), which triggers
    # dynamo's GLOBAL_STATE guard. Every train/eval switch causes a recompilation.
    # With cache_size_limit=2048 this would still eventually overflow. Instead,
    # disable the guard: dynamo still traces grad_mode correctly internally.
    #
    # Used by vLLM and other large models (same pattern).
    _guards.GuardManager.add_global_state_guard = lambda *args: None

    # --- Device-specific ---
    if device == "cuda":
        _ic.triton.cudagraphs = False  # mAR stream aliasing
    elif device == "cpu":
        _ic.triton.cudagraphs = False
    # CPU / ROCm: inductor defaults are fine


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
    assert torch.cuda.is_available(), "busel: CUDA required"
    return "cuda"


def _build_targets(byte_batch: torch.Tensor, input_length: int, stride: int = 4, num_mtp_heads: int = 4):
    """Compute MTP-N targets (mirrors train.py:build_targets, dynamic heads)."""
    targets = byte_batch[:, 1::stride][:, :input_length].long()  # CE needs Long
    if targets.shape[1] < input_length:
        pad = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad), value=0)
    mtp_targets = []
    for shift in range(2, num_mtp_heads + 1):
        mt = byte_batch[:, shift::stride][:, :input_length]
        if mt.shape[1] < input_length:
            pad = input_length - mt.shape[1]
            mt = torch.nn.functional.pad(mt, (0, pad), value=0)
        mtp_targets.append(mt.long())
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
        self._tb_writer = None  # TensorBoard SummaryWriter
        self._oom_reductions: int = 0
        self._max_oom_reductions: int = 6
        self._last_chunk_block_step: int = -1000

    def _vram_used_mb(self) -> float:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return 0.0

    def _vram_total_mb(self) -> float:
        if not hasattr(self, "_vram_total_mb_cache"):
            if self.device == "cuda" and torch.cuda.is_available():
                self._vram_total_mb_cache = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            else:
                self._vram_total_mb_cache = 0.0
        return self._vram_total_mb_cache

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
            mix_weights=self.cfg.mix_weights,
        )
        self.dataloader_iter = iter(self.dataloader)

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
        self._stage_no_fp8 = bool(stage_params.get("no_fp8", False))
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
            try:
                self.cfg.max_steps = int(override_max_steps)
            except (ValueError, TypeError):
                pass  # "auto" → let the auto formula compute it
        if override_warmup_steps is not None:
            self.cfg.warmup_steps = int(override_warmup_steps)
        if hasattr(self, "_override_cache_clean"):
            self.cfg.inductor_cache_clean = self._override_cache_clean
        if getattr(self, "_stage_no_fp8", False):
            self.cfg.no_fp8 = True

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

        configure_bitlinear(
            use_tequila=self.cfg.use_tequila,
            tequila_lambda=self.cfg.tequila_lambda,
            use_fused_training=self.cfg.use_fused_training,
        )
        if self.cfg.use_fused_training:
            from model.layers import _BITLINEAR_CONFIG
            _BITLINEAR_CONFIG["use_hysteresis"] = False
            _BITLINEAR_CONFIG["use_sr_ste"] = False

        self.patcher = StridedFastBLTPatcher(
            d_model=self.cfg.d_model,
        ).to(self.device)
        self.model = buselModel(self.cfg).to(self.device)

        self._moe_modules = [
            l.moe for l in self.model.layers
            if hasattr(l, 'moe') and hasattr(l.moe, 'update_bias')
        ]

        # FP8: always ON for Ampere+. Skip in test mode.
        if self.device == "cuda" and not getattr(self.cfg, "_test_mode", False) and not getattr(self.cfg, "no_fp8", False):
            if self.no_compile:
                from torchao.float8 import convert_to_float8_training
                # ponytail: convert_to_float8_training replaces nn.Linear subclasses (incl. BitLinear_a4_8)
                # with Float8Linear. This breaks ternary quantization (BitNet spec). Skip FP8 forward
                # on BitLinear-based models — optimizer FP8 (AdamWFp8) still works.
                pass
        # print("⚡ [FP8-AdamW]: optimizer FP8 state — 75% memory (torchao)")

        total_params = sum(p.numel() for p in self.model.parameters())

        log_event(
            "model_initialized",
            profile=profile_name,
            device=self.device,
            total_params=total_params,
            model_size_mb=round(total_params * 2 / 1024**2, 2),
        )

        if self.cfg.max_steps == "auto" or self.cfg.max_steps is None:
            global_batch = self.cfg.batch_size * self.cfg.grad_accum_steps
            # Tokens = raw bytes (not stride-4 patches). chunk_size is bytes per sequence.
            # Chunk curriculum: batch auto-halves when chunk doubles, so tok/step is constant.
            # Dataloader starts at chunk//16, tok/step = batch × ga × (chunk//16).
            # Previous bug: used chunk//8 with un-halved batch → 2× overestimate.
            initial_chunk = self.cfg.chunk_size // 16
            tokens_per_step = global_batch * initial_chunk
            # target_tok_per_param overrides scaling law; 0 = use defaults
            if self.cfg.target_tok_per_param > 0:
                tokens_per_param = self.cfg.target_tok_per_param
            else:
                # Busel Scaling Laws: two-tier model
                #   - Small models (<3B params): 37 tok/param (empirical)
                #   - Large models (≥3B params): 80 tok/param (Chinchilla match)
                _BUSEL_THRESHOLD = 3_000_000_000
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
            self.cfg.warmup_steps = max(5, int(0.05 * self.cfg.max_steps))
        elif isinstance(self.cfg.warmup_steps, str) and self.cfg.warmup_steps.endswith("%"):
            pct = float(self.cfg.warmup_steps[:-1]) / 100.0
            self.cfg.warmup_steps = max(5, int(pct * self.cfg.max_steps))

        self.cfg.max_steps, self.cfg.warmup_steps = validate_training_schedule(
            self.cfg.max_steps, self.cfg.warmup_steps
        )
        print(f"📊 Training: {self.cfg.max_steps} steps, warmup {self.cfg.warmup_steps}, batch {self.cfg.batch_size}×{self.cfg.grad_accum_steps}")

        if self.device == "cuda" and not self.no_checkpointing:
            self.model.enable_gradient_checkpointing(every=self.cfg.grad_ckpt_every)

        if not self.no_compile:
            if self.device not in ("cuda", "cpu"):
                print(f"⏭️  Skipping torch.compile: unsupported device {self.device!r}")
            elif self.device == "cuda" and total_params < 10_000_000:
                print(f"⏭️  Skipping torch.compile: {total_params:,} params < 10M threshold")
            else:
                _setup_inductor_speed_config(self.device)
                dyn = self.cfg.dynamic_compile and self.device == "cuda"
                print(f"⚡ torch.compile per-layer (device={self.device}, dynamic={dyn}): ~2× speedup")
                self._compile_in_progress["value"] = True
                try:
                    # ponytail: compile each layer individually — 12× less RAM than full-model graph
                    n_layers = len(self.model.layers)
                    for i, layer in enumerate(self.model.layers):
                        print(f"   compile layer {i+1}/{n_layers}...", end="\r")
                        self.model.layers[i] = torch.compile(
                            layer, fullgraph=False, dynamic=dyn, mode=self.compile_mode
                        )
                    for i, m_res in enumerate(self.model.m_residuals):
                        print(f"   compile mAR {i+1}/{n_layers}...", end="\r")
                        self.model.m_residuals[i] = torch.compile(
                            m_res, fullgraph=False, dynamic=dyn, mode=self.compile_mode
                        )
                    print(f"   compile patcher...", end="\r")
                    self.patcher = torch.compile(
                        self.patcher, fullgraph=False, dynamic=dyn, mode=self.compile_mode
                    )
                    print(f"   compile MTP pipeline...", end="\r")
                    self.model.mtp_pipeline = torch.compile(
                        self.model.mtp_pipeline, fullgraph=False, dynamic=dyn, mode=self.compile_mode
                    )
                    print(f"   compile: {n_layers} layers + {n_layers} mARs + patcher + MTP done   ")
                except Exception as e:
                    err_str = str(e)
                    if "CUDAGraphs" in err_str or "FakeTensor" in err_str or "overwritten" in err_str:
                        try:
                            self.model = torch.compile(self.model, fullgraph=False, dynamic=dyn, mode=self.compile_mode)
                            self.patcher = torch.compile(self.patcher, fullgraph=False, dynamic=dyn, mode=self.compile_mode)
                        except Exception:
                            pass
                finally:
                    self._compile_in_progress["value"] = False

        self.opt_engine = buselOptimizerEngine(
            self.model, self.patcher,
            lr_muon=self.cfg.learning_rate_muon,
            lr_adamw=self.cfg.learning_rate_adamw,
            lotus_rank=self.cfg.lotus_rank,
            lr_multipliers=self.cfg.lr_multipliers,
        )
        self.autopilot = buselAutoPilot(
            self.opt_engine,
            max_lr_muon=self.cfg.learning_rate_muon,
            max_lr_adamw=self.cfg.learning_rate_adamw,
            target_wd=self.cfg.weight_decay,
            warmup_steps=self.cfg.warmup_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
            lr_schedule=self.cfg.lr_schedule,
            wsd_decay_frac=self.cfg.wsd_decay_frac,
            grad_clip=self.cfg.grad_clip,
        )
        self.loss_engine = buselLossEngine(self.cfg.vocab_size)

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

        current_chunk_size = self.cfg.chunk_size // 16  # match curriculum start (1/16)
        self.dataloader = get_busel_dataloader(
            self.cfg.data_path,
            chunk_size=current_chunk_size,
            batch_size=self.cfg.batch_size,
            start_file_idx=self.start_file_idx,
            start_byte_offset=self.start_byte_offset,
            mix_weights=self.cfg.mix_weights,
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
        # ponytail: non-blocking stdin — 's' save, 'q' quit, like Vim
        self._tty = None
        self._tty_old = None
        try:
            self._tty = open('/dev/tty', 'r')
            import termios, tty
            self._tty_old = termios.tcgetattr(self._tty)
            tty.setcbreak(self._tty)
        except Exception:
            pass

        # ponytail: TensorBoard writer — writes alongside metrics.jsonl
        try:
            from torch.utils.tensorboard import SummaryWriter
            import datetime
            log_dir = f"checkpoints/tb_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"📊 TensorBoard: {log_dir}")
        except Exception:
            pass

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

    def _compute_chunk_target(self, progress: float) -> int | None:
        """Context growth: 1024 → 2048. 67M caps at 2048 bytes."""
        if not getattr(self.cfg, "use_chunk_curriculum", True):
            return None
        p = min(1.0, max(0.0, progress))
        full = self.cfg.chunk_size
        if p < 0.40:
            return full // 8  # 1024
        return full // 4  # 2048 — cap for 67M

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
        return old_chunk, new_batch  # keep new batch (was auto-scaled as inverse of chunk)

    def _rebuild_dataloader_at_size(self, chunk_size: int, batch_size: int):
        """Rebuild the dataloader with new chunk/batch sizes and return the first batch."""
        from data.pipeline import get_busel_dataloader
        self.dataloader = get_busel_dataloader(
            self.cfg.data_path,
            chunk_size=chunk_size,
            batch_size=batch_size,
            start_file_idx=self.global_current_file_idx,
            start_byte_offset=self.global_current_byte_offset,
            mix_weights=self.cfg.mix_weights,
        )
        self.dataloader_iter = iter(self.dataloader)
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            return None

    def _handle_emergency_save(self, step: int):
        """Ctrl+C → full resume checkpoint: model + EMA + optimiser state."""
        if not self._emergency_save_requested["value"]:
            return
        os.makedirs("checkpoints", exist_ok=True)
        try:
            from model.checkpoint import strip_compile_prefix
            ckpt = {
                "model_state_dict": strip_compile_prefix(self.model.state_dict()),
                "patcher_state_dict": strip_compile_prefix(self.patcher.state_dict()),
                "step": step,
                "file_idx": self.global_current_file_idx,
                "byte_offset": self.global_current_byte_offset,
                "profile": self.profile_name,
                "optimizer": self.opt_engine.state_dict(),
            }
            if self.ema is not None:
                ckpt["ema_state_dict"] = self.ema.state_dict()
            torch.save(ckpt, "checkpoints/latest_resume.pt")
            print(f"[save] Resume: checkpoints/latest_resume.pt (step {step})\n")
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

        new_bs = max(1, current_batch_size // 2)
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
        if step_offset > 0:  # skip step 0 (elapsed ≠ 10 steps)
            self._spd_window.append(steps_per_s)
            if len(self._spd_window) > 3:
                self._spd_window = self._spd_window[-3:]
        avg_steps_per_s = sum(self._spd_window) / len(self._spd_window) if self._spd_window else steps_per_s
        remaining = max(0, self.cfg.max_steps - step)
        eta_s = remaining / avg_steps_per_s if avg_steps_per_s > 0 else 0.0

        if eta_s >= 3600:
            eta_str = f"{int(eta_s // 3600)}h {int((eta_s % 3600) // 60):02d}m"
        elif eta_s >= 60:
            eta_str = f"{int(eta_s // 60)}m {int(eta_s % 60):02d}s"
        else:
            eta_str = f"{int(eta_s)}s"

        loss_color = 31 if accumulated_loss == 0 else (32 if step < 50 or accumulated_loss < getattr(self, '_prev_loss', float('inf')) else 33)
        self._prev_loss = accumulated_loss
        # ponytail: 20-char progress bar — negligible overhead (one string, every 10 steps)
        pct = step / max(1, self.cfg.max_steps)
        bar_w = 20
        filled = int(bar_w * pct)
        bar = "\033[36m" + "█" * filled + "\033[2m" + "░" * (bar_w - filled) + "\033[0m"
        print(
            f"\033[2m{step:5d}/{self.cfg.max_steps}\033[0m  {bar}  "
            f"\033[1;{loss_color}mloss {accumulated_loss:6.2f}\033[0m | "
            f"lr {current_lr:.5f}  \033[36m{speed:.0f} tok/s\033[0m  \033[2mETA {eta_str}\033[0m"
        )

        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/metrics.jsonl", "a", encoding="utf-8") as log_f:
            log_f.write(
                json.dumps({
                    "step": step,
                    "loss": accumulated_loss / max(1, self.cfg.grad_accum_steps),
                    "lr": current_lr,
                    "speed": speed,
                }, ensure_ascii=False) + "\n"
            )
        log_event(
            "step_complete",
            step=step,
            loss=round(accumulated_loss / max(1, self.cfg.grad_accum_steps), 4),
            lr=round(current_lr, 7),
            tokens_per_s=round(speed, 1),
        )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("loss", accumulated_loss / max(1, self.cfg.grad_accum_steps), step)
            self._tb_writer.add_scalar("lr", current_lr, step)
            self._tb_writer.add_scalar("speed", speed, step)

        return last_log_time, last_log_tokens, speed

    def run(self, state: StageState) -> StageState:
        """Execute the pretrain training loop for cfg.max_steps."""
        if self.cfg is None:
            raise RuntimeError("setup() must be called before run()")

        self._init_training_state()
        current_batch = self._fetch_initial_batch(state)
        if current_batch is None:
            return state

        current_chunk_size = self.cfg.chunk_size // 16  # match curriculum start (1/16)
        last_log_time = 0.0
        last_log_tokens = 0
        current_batch_size = self.cfg.batch_size
        cumulative_processed_tokens = (
            self.start_step * current_batch_size * self.cfg.grad_accum_steps * current_chunk_size
        )

        # --- compile warmup: trigger inductor compilation before the training loop ---
        if not self.no_compile and self.device in ("cuda", "cpu") and self.start_step == 0:
            if not (self.device == "cuda" and sum(p.numel() for p in self.model.parameters()) < 10_000_000):
                print("🔥 torch.compile warmup: running first batch through compiled model...")
                try:
                    raw_bytes = current_batch[0] if isinstance(current_batch, (tuple, list)) else current_batch
                    warmup_batch = raw_bytes.to(self.device, non_blocking=self.device == "cuda")
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
                        patches = self.patcher(warmup_batch)
                        T_patches = patches.shape[1]
                        targets, mtp_targets = _build_targets(
                            warmup_batch, T_patches, stride=self.patcher.stride,
                            num_mtp_heads=self.cfg.num_mtp_heads,
                        )
                        mtp_logits, aux_loss = self.model(
                            patches, [targets] + mtp_targets[:-1]
                        )
                    # ponytail: forward-only warmup — backward during warmup frees graph and breaks next training step
                    print("✅ torch.compile warmup complete")
                except Exception as e:
                    print(f"⚠️  Compile warmup failed (non-fatal): {type(e).__name__}: {e}")

        # Start timing AFTER compile warmup — otherwise elapsed includes ~50s of compilation
        start_time = time.time()

        for step_offset in range(self.cfg.max_steps):
            step = self.start_step + step_offset
            progress = float(step) / float(self.cfg.max_steps) if self.cfg.max_steps else 0.0

            # YaRN: gradually increase rope_scale for context extension
            if self.cfg.use_yarn and self.cfg.yarn_scale > 1.0:
                if progress >= self.cfg.yarn_start_frac:
                    dp = min(1.0, (progress - self.cfg.yarn_start_frac) / self.cfg.yarn_duration_frac)
                    scale = 1.0 + (self.cfg.yarn_scale - 1.0) * dp
                    self.model.set_rope_scale(scale)
                elif progress < self.cfg.yarn_start_frac - 0.01:
                    self.model.set_rope_scale(1.0)

            stopped = self._check_early_stop(step, state)
            if stopped is not None:
                return stopped

            # --- chunk curriculum ---
            target_chunk = self._compute_chunk_target(progress)
            # ponytail: only rebuild if chunk changed by ≥64 — avoids per-step rebuild + cache flush
            if target_chunk is not None and abs(target_chunk - current_chunk_size) >= 64:
                new_batch_size = max(1, (current_batch_size * current_chunk_size) // target_chunk)
                new_chunk, new_batch = self._maybe_block_chunk_growth(
                    step, current_chunk_size, target_chunk, new_batch_size
                )
                current_chunk_size = new_chunk
                current_batch_size = new_batch
                new_batch = self._rebuild_dataloader_at_size(current_chunk_size, current_batch_size)
                # ponytail: reset dampening history — chunk upgrade naturally increases grads
                self.autopilot.grad_norm_history = []
                if new_batch is not None:
                    current_batch = new_batch
                else:
                    break

            # --- forward/backward ---
            _step_start = time.perf_counter()
            self.opt_engine.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            accumulated_aux_loss = 0.0
            tokens_this_step = 0
            # ponytail: tensor accumulation — avoids .item() GPU sync per micro-batch
            _accum_loss = torch.tensor(0.0, device=self.device)
            _accum_aux = torch.tensor(0.0, device=self.device)

            _fwd_bwd_ok = False
            for _fwd_bwd_attempt in range(3):
                try:
                    if _fwd_bwd_attempt > 0:
                        self.opt_engine.zero_grad(set_to_none=True)
                    _accum_loss.zero_(); _accum_aux.zero_()
                    tokens_this_step = 0

                    _iter_batch = current_batch
                    for _ in range(self.cfg.grad_accum_steps):
                        if _iter_batch is None:
                            break
                        byte_batch, last_file_idx, last_byte_offset = _iter_batch
                        byte_batch = byte_batch.to(self.device, non_blocking=True)
                        self.global_current_file_idx = last_file_idx
                        self.global_current_byte_offset = last_byte_offset
                        # ponytail: D6 — ASCII curriculum. Phase 1: bytes 0-127 only. 1.2× faster convergence.
                        if self.cfg.use_ascii_curriculum and progress < 0.3:
                            byte_batch = byte_batch.clamp(max=127)
                        elif self.cfg.use_ascii_curriculum and progress < 0.6:
                            byte_batch = byte_batch.clamp(max=255)
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
                                byte_batch, T_patches, stride=self.patcher.stride,
                                num_mtp_heads=self.cfg.num_mtp_heads,
                            )
                            mtp_logits, aux_loss = self.model(
                                patches, [targets] + mtp_targets[:-1], progress=progress
                            )
                            mtp_logits = tuple(torch.nan_to_num(l, nan=0.0, posinf=1e4, neginf=-1e4) for l in mtp_logits)
                            # ponytail: check for Inf in first layer activation BEFORE backward
                            if step > 170 and step % 3 == 0:
                                x0 = self.model.layers[0].attn.q_proj.weight.abs().max().item()
                                if x0 > 100:
                                    print(f"  [ACT] step {step}: max q_proj weight = {x0:.1f}")
                            logits_t1 = mtp_logits[0]
                            extra_logits = list(mtp_logits[1:])
                            loss = self.loss_engine.compute_pretrain_loss(
                                logits_t1, targets, extra_logits, mtp_targets,
                            )
                            loss = loss + aux_loss.float()
                            if self.cfg.use_dispersion_loss:
                                loss = loss + self.loss_engine.compute_dispersion_loss(
                                    embed_for_dispersion,
                                    weight=self.cfg.dispersion_weight,
                                    temperature=self.cfg.dispersion_temperature,
                                )
                            # D3 — EMA self-distillation: every 10 steps after 30% progress
                            if self.ema is not None and progress > 0.3 and step % 10 == 0:
                                ema_w = self.ema.shadow.get('mtp_pipeline.head.weight')
                                if ema_w is not None and hasattr(self.model, '_last_hidden'):
                                    h = self.model._last_hidden.detach()
                                    ema_logits = F.linear(h, ema_w.to(h))[..., :self.cfg.vocab_size]
                                    kl = F.kl_div(F.log_softmax(logits_t1, dim=-1), F.softmax(ema_logits, dim=-1), reduction='batchmean')
                                    loss = loss + 0.05 * kl

                        loss = loss / self.cfg.grad_accum_steps

                        # ponytail: forward NaN guard — always ON, one isnan per step
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.opt_engine.zero_grad(set_to_none=True)
                            for p in self.model.parameters():
                                if p.requires_grad:
                                    p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                                    p.data.clamp_(-5.0, 5.0)
                            for sf in (self.opt_engine.opt_muon, self.opt_engine.opt_adamw):
                                if sf is None: continue
                                for p, s in list(sf._state.items()):
                                    if hasattr(s, 'get') and 'z' in s:
                                        s['z'].copy_(p.data); s['x'].copy_(p.data)
                            self.autopilot.recovery_countdown = 15
                            self.autopilot.stabilization_factor = 0.25
                            print(f"⚠️  [NaN FWD] step {step}: loss NaN — weights clamped, LR×0.25")
                            _fwd_bwd_ok = True
                            continue

                        loss.backward()
                        # ponytail: grad scan every 50 steps — catches NaN before it spreads
                        _nan_grads = []
                        if step % 50 == 0:
                            _grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                            if _grads:
                                _gnorm = torch.nn.utils.get_total_norm(_grads, 2.0)
                                if torch.isnan(_gnorm) or torch.isinf(_gnorm):
                                    _nan_grads = [True]
                        if _nan_grads:
                                self.opt_engine.zero_grad(set_to_none=True)
                                for p in self.model.parameters():
                                    if p.requires_grad:
                                        p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)
                                        p.data.clamp_(-5.0, 5.0)
                                for sf in (self.opt_engine.opt_muon, self.opt_engine.opt_adamw):
                                    if sf is None: continue
                                    for p, s in list(sf._state.items()):
                                        if hasattr(s, 'get') and 'z' in s:
                                            s['z'].copy_(p.data)
                                            s['x'].copy_(p.data)
                                self.autopilot.recovery_countdown = 15
                                self.autopilot.stabilization_factor = 0.25
                                print(f"⚠️  [NaN GRAD] step {step}: NaN/Inf gradient — clamped, LR×0.25")
                                _fwd_bwd_ok = True
                                continue
                        for moe in self._moe_modules:
                            moe.update_bias()
                        _accum_loss += loss.detach()  # loss already scaled by /grad_accum; undo for display
                        _accum_aux += aux_loss.detach()
                        tokens_this_step = current_batch_size * current_chunk_size  # сырые байты (как profiler)
                        cumulative_processed_tokens += tokens_this_step

                        next_batch = None
                        try:
                            if self._use_cuda_stream:
                                with torch.cuda.stream(self._prefetch_stream):
                                    next_batch = next(self.dataloader_iter)
                            else:
                                next_batch = next(self.dataloader_iter)
                        except (StopIteration, RuntimeError):
                            self.dataloader_iter = iter(self.dataloader)
                            try:
                                next_batch = next(self.dataloader_iter)
                            except (StopIteration, RuntimeError):
                                next_batch = None
                        if self._use_cuda_stream:
                            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
                        _iter_batch = next_batch

                    current_batch = _iter_batch
                    accumulated_loss = _accum_loss.item()
                    accumulated_aux_loss = _accum_aux.item()
                    for moe in self._moe_modules:
                        moe.update_bias()
                    if hasattr(self, '_oom_batch_fail') and step % 10 == 0:
                        self._oom_batch_ok = current_batch_size  # survived
                        probe = (self._oom_batch_ok + self._oom_batch_fail) // 2
                        if probe > current_batch_size + 4:
                            current_batch_size = probe
                            self.cfg.batch_size = probe
                            self._rebuild_dataloader(probe, current_chunk_size)
                    _fwd_bwd_ok = True
                    break

                except torch.cuda.OutOfMemoryError:
                    if self.device != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    self._oom_reductions += 1
                    old_bs = current_batch_size
                    if not hasattr(self, '_oom_batch_ok'):
                        self._oom_batch_ok = 1  # safe floor
                    self._oom_batch_fail = old_bs  # ceiling
                    # binary search: midpoint between last OK and first fail
                    current_batch_size = (self._oom_batch_ok + self._oom_batch_fail) // 2
                    if current_batch_size >= old_bs:
                        current_batch_size = max(1, old_bs // 2)
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
            current_lr, _ = self.autopilot.update_parameters(step, accumulated_loss, self.cfg.max_steps)
            
            self.opt_engine.step(model=self.model)
                
            if self.ema is not None:
                self.ema.update(self.model)

            # --- memory watchdog ---
            current_batch_size = self._memory_watchdog(step, current_batch_size, current_chunk_size)

            # --- emergency checkpoint (Ctrl+C = full resume) ---
            self._handle_emergency_save(step)

            # --- keyboard: w=save q=quit (Vim-style) ---
            try:
                if self._tty is not None:
                    import select
                    if select.select([self._tty], [], [], 0)[0]:
                        key = self._tty.read(1)
                        if key == 'w':
                            self._save_checkpoint(step, accumulated_loss, current_lr)
                            print(f"  💾 written at step {step}")
                        elif key == 'q':
                            self._save_checkpoint(step, accumulated_loss, current_lr)
                            print(f"  💾 written + quit at step {step}")
                            self._cleanup_tty()
                            return state
            except Exception:
                pass

            # --- scheduled checkpoint (config.checkpoint_steps) ---
            if step in getattr(self.cfg, "checkpoint_steps", []):
                self._save_checkpoint(step, accumulated_loss, current_lr)

            # --- state update ---
            state.step = step
            state.best_loss = (
                min(state.best_loss, accumulated_loss) if accumulated_loss > 0 else state.best_loss
            )

            # --- log progress (every 10 steps) ---
            last_log_time, last_log_tokens, speed = self._log_progress(
                step, step_offset, start_time, last_log_time, last_log_tokens,
                cumulative_processed_tokens, accumulated_loss, accumulated_aux_loss,
                current_lr, dynamic_clip, current_batch_size, current_chunk_size, tokens_this_step,
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
        byte_batch, _, _ = batch
        byte_batch = byte_batch.to(self.device)
        input_bytes = byte_batch[:, :-self.patcher.stride] if byte_batch.shape[1] > self.patcher.stride else byte_batch
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            patches = self.patcher(input_bytes)
            T = patches.shape[1]
            targets = byte_batch[:, self.patcher.stride::self.patcher.stride][:, :T]
            if targets.shape[1] < T:
                targets = F.pad(targets, (0, T - targets.shape[1]))
            mtp_logits, _ = self.model(patches)
            val_loss = self.loss_engine.compute_pretrain_loss(mtp_logits[0], targets, [], [])
        ppl = torch.exp(val_loss).item()
        state.metrics["val_loss"] = val_loss.item()
        state.metrics["val_ppl"] = ppl
        print(f"  [val] step {step}: loss {val_loss.item():.4f}, ppl {ppl:.1f}")
        self.model.train()

    def _save_checkpoint(self, step, loss, lr):
        """Model + patcher only — no EMA/optimizer. For scheduled milestones."""
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/busel_{self.profile_name}_step_{step}.pt"
        tmp = path + ".tmp"
        try:
            ckpt = {"model_state_dict": self.model.state_dict(),
                    "patcher_state_dict": self.patcher.state_dict(),
                    "step": step, "loss": loss, "lr": lr, "profile": self.profile_name}
            torch.save(ckpt, tmp)
            if os.path.getsize(tmp) >= 100_000:
                os.rename(tmp, path)
                print(f"[save] {path} ({os.path.getsize(path)/1024/1024:.0f}MB)")
            else:
                os.remove(tmp)
        except Exception as e:
            if os.path.exists(tmp): os.remove(tmp)
            log_event("checkpoint_failed", step=step, error=str(e))

    def _cleanup_tty(self):
        if self._tty is not None and self._tty_old is not None:
            import termios
            termios.tcsetattr(self._tty, termios.TCSADRAIN, self._tty_old)
            self._tty.close()
            self._tty = None

    def finalize(self, state: StageState) -> StageState:
        """Save the final checkpoint + emit stage_complete event."""
        self._cleanup_tty()
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

        if self._tb_writer is not None:
            self._tb_writer.close()

        return state

