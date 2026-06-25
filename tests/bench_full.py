"""
📊 busel FULL BENCHMARK v1.0 — Gorlyshki, Utilizatsiya, Skorost
Comprehensive profiling: GPU/CPU util, compile vs eager, per-phase breakdown,
data throughput, memory, step timing distribution.

Usage:
  uv run python tests/bench_full.py                     # default: 30 steps, both modes
  uv run python tests/bench_full.py --steps 50          # more steps
  uv run python tests/bench_full.py --compile-only      # only compiled mode
  uv run python tests/bench_full.py --eager-only        # only eager (no-compile)
  uv run python tests/bench_full.py --output bench.json # export raw data
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data.pipeline import get_busel_dataloader
from model.backbone import buselModel
from model.patching import StridedFastBLTPatcher
from training.autopilot import buselAutoPilot
from training.optimizer import buselOptimizerEngine
from training.recipe import buselLossEngine

# ─────────────────────────────────────────────────────────────
# 1. Monitoring helpers
# ─────────────────────────────────────────────────────────────


class GPUMonitor:
    """Background GPU monitoring via nvidia-smi dmon (default table format).
    Columns: gpu_idx, pwr(W), gtemp(C), mtemp(C), sm(%), mem(%), ...
    Parsed by splitting on whitespace (fixed-width table format)."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._samples: list[dict[str, float]] = []

    def start(self):
        cmd = ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", "1"]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def stop(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            self._proc.kill()
        stdout, _ = self._proc.communicate()
        self._parse_samples(stdout)
        self._proc = None

    def _parse_samples(self, raw: str):
        """Parse dmon table output (fixed-width columns, whitespace-delimited).
        Data rows: idx power temp_mem temp_gpu sm_util mem_util ..."""
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6 or parts[0] == "gpu":
                continue
            try:
                self._samples.append({
                    "gpu_util": float(parts[4]),   # sm = streaming multiprocessor util %
                    "mem_util": float(parts[5]),   # mem = memory util %
                    "power_w": float(parts[1]),    # pwr = power draw W
                    "temp_c": float(parts[2]),     # gtemp = GPU temp C
                })
            except (ValueError, IndexError):
                pass

    @property
    def avg_gpu_util(self) -> float:
        return float(np.mean([s["gpu_util"] for s in self._samples])) if self._samples else 0.0

    @property
    def avg_mem_util(self) -> float:
        return float(np.mean([s["mem_util"] for s in self._samples])) if self._samples else 0.0

    @property
    def avg_power(self) -> float:
        return float(np.mean([s["power_w"] for s in self._samples])) if self._samples else 0.0

    @property
    def max_temp(self) -> float:
        return float(max(s["temp_c"] for s in self._samples)) if self._samples else 0.0

    def report(self) -> dict:
        if not self._samples:
            return {"error": "no samples"}
        return {
            "gpu_util_avg_pct": round(self.avg_gpu_util, 1),
            "gpu_util_max_pct": round(float(max(s["gpu_util"] for s in self._samples)), 1),
            "mem_util_avg_pct": round(self.avg_mem_util, 1),
            "power_avg_w": round(self.avg_power, 1),
            "power_max_w": round(float(max(s["power_w"] for s in self._samples)), 1),
            "temp_max_c": round(self.max_temp, 0),
            "samples": len(self._samples),
        }


class CPUMonitor:
    """Background CPU monitoring via psutil."""

    def __init__(self):
        self._samples: list[float] = []
        self._per_core: list[list[float]] = []
        self._running = False

    def start(self):
        import psutil
        self._running = True
        self._samples = []
        self._per_core = []
        # Take initial sample
        psutil.cpu_percent(interval=None, percpu=True)

    def sample(self):
        if not self._running:
            return
        import psutil
        self._samples.append(psutil.cpu_percent(interval=None))
        self._per_core.append(psutil.cpu_percent(interval=None, percpu=True))

    def stop(self):
        self._running = False

    @property
    def avg_cpu_util(self) -> float:
        return float(np.mean(self._samples)) if self._samples else 0.0

    def report(self) -> dict:
        if not self._samples:
            return {"error": "no samples"}
        cores = np.array(self._per_core)
        avg_per_core = cores.mean(axis=0) if cores.size > 0 else []
        return {
            "cpu_avg_pct": round(self.avg_cpu_util, 1),
            "cpu_max_pct": round(float(max(self._samples)), 1),
            "cpu_per_core_avg": [round(float(v), 1) for v in avg_per_core],
            "samples": len(self._samples),
        }


# ─────────────────────────────────────────────────────────────
# 2. Config
# ─────────────────────────────────────────────────────────────


@dataclass
class BenchConfig:
    """Shpak profile config for benchmarking."""
    vocab_size: int = 326
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_hyper: int = 2
    expert_hidden: int = 768
    num_experts: int = 4
    top_k: int = 1
    mod_capacity: float = 1.0
    sct_rank: int = 0
    selective_backward: bool = True
    backward_ratio: float = 0.5
    use_differential_attention: bool = True
    use_dispersion_loss: bool = True
    use_tequila: bool = True
    use_hestia: bool = False
    lcsb_deterministic: bool = True
    # Data
    data_path: str = "data_train"
    chunk_size: int = 4096
    batch_size: int = 64
    # Optimizer
    learning_rate_muon: float = 0.0005
    learning_rate_adamw: float = 5e-05
    weight_decay: float = 0.1
    lotus_rank: int = 8
    grad_accum_steps: int = 1
    #
    max_steps: int = 30
    warmup_steps: int = 0


# ─────────────────────────────────────────────────────────────
# 3. Benchmark runner
# ─────────────────────────────────────────────────────────────


@dataclass
class StepMetrics:
    """Timing for one training step."""
    data_load_ms: float = 0.0
    patcher_ms: float = 0.0
    forward_ms: float = 0.0
    loss_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    total_ms: float = 0.0
    vram_mb: float = 0.0


@dataclass
class BenchResult:
    """Result of one benchmark run (eager or compile)."""
    mode: str = ""
    steps: int = 0
    compile_time_s: float = 0.0
    step_metrics: list = field(default_factory=list)
    gpu_report: dict = field(default_factory=dict)
    cpu_report: dict = field(default_factory=dict)
    total_tokens: int = 0
    total_time_s: float = 0.0
    avg_tok_per_s: float = 0.0
    avg_step_ms: float = 0.0
    vram_peak_mb: float = 0.0


def _build_targets(byte_batch, input_length, stride=4):
    """MTP-4 targets (same as in pretrain.py)."""
    targets = byte_batch[:, 1::stride][:, :input_length]
    if targets.shape[1] < input_length:
        targets = torch.nn.functional.pad(targets, (0, input_length - targets.shape[1]), value=0)
    mtp_targets = []
    for shift in (2, 3, 4):
        mt = byte_batch[:, shift::stride][:, :input_length]
        if mt.shape[1] < input_length:
            mt = torch.nn.functional.pad(mt, (0, input_length - mt.shape[1]), value=0)
        mtp_targets.append(mt)
    return targets, mtp_targets


def run_bench(
    mode: str,
    steps: int,
    cfg: BenchConfig,
    data_path: str,
    output_dir: str,
) -> BenchResult:
    """Run benchmark in given mode ('eager' or 'compile')."""
    device = "cuda"
    result = BenchResult(mode=mode, steps=steps)

    print(f"\n{'='*70}")
    print(f"  🔬 MODE: {mode.upper()}")
    print(f"{'='*70}")

    # ── Build model ──
    print("  Building model...", end=" ", flush=True)
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    # RMSNorm fix
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data = module.weight.data.to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} params")

    # ── FP8: disabled for benchmark ──
    # FP8 _scaled_mm has dim%16 constraints that conflict with FWHT transforms
    # in H_BitLinear. Both modes run without FP8 for fair comparison.

    # ── Gradient checkpointing ──
    model.enable_gradient_checkpointing(every=2)

    # ── torch.compile ──
    if mode == "compile":
        from training.stages.pretrain import _setup_inductor_speed_config
        _setup_inductor_speed_config(device)
        print("  Compiling model...", end=" ", flush=True)
        t0 = time.perf_counter()
        model = torch.compile(model, fullgraph=False, dynamic=True)
        patcher = torch.compile(patcher, fullgraph=False, dynamic=True)
        torch.cuda.synchronize()
        result.compile_time_s = time.perf_counter() - t0
        print(f"{result.compile_time_s:.1f}s")

    # ── Optimizer + Autopilot + Loss ──
    # Build AFTER compile so param groups are on the compiled model
    opt_engine = buselOptimizerEngine(
        model, patcher,
        lr_muon=cfg.learning_rate_muon,
        lr_adamw=cfg.learning_rate_adamw,
        lotus_rank=cfg.lotus_rank,
    )
    autopilot = buselAutoPilot(
        opt_engine,
        max_lr_muon=cfg.learning_rate_muon,
        max_lr_adamw=cfg.learning_rate_adamw,
        target_wd=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
    )
    loss_engine = buselLossEngine(cfg.vocab_size)
    # EMA omitted in benchmark (adds ~0.1ms, not relevant for throughput)

    # ── DataLoader ──
    torch.cuda.reset_peak_memory_stats()
    print("  Loading data...", end=" ", flush=True)
    dataloader = get_busel_dataloader(
        data_path,
        chunk_size=cfg.chunk_size,
        batch_size=cfg.batch_size,
    )
    dataloader_iter = iter(dataloader)
    print("✅")

    # ── Warmup ──
    print("  Warmup (2 steps)...", end=" ", flush=True)
    for _ in range(2):
        batch = next(dataloader_iter)
        byte_batch = batch[0].to(device, non_blocking=True)
        opt_engine.zero_grad(set_to_none=True)
        input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = patcher(input_bytes)
            T_patches = patches.shape[1]
            targets, mtp_targets = _build_targets(byte_batch, T_patches, stride=patcher.stride)
            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(
                patches, [targets] + mtp_targets[:-1]
            )
            t1_ce = loss_engine.compute_pretrain_loss(logits_t1, targets, [], [])
            loss = loss_engine.compute_pretrain_loss(
                logits_t1, targets, [logits_t2, logits_t3, logits_t4], mtp_targets
            )
            loss = loss + aux_loss.float()
        loss.backward()
        opt_engine.step()
    torch.cuda.synchronize()
    print("✅")

    # ── Monitors ──
    gpu_mon = GPUMonitor()
    cpu_mon = CPUMonitor()

    # ── Training loop (timed) ──
    print(f"  Measuring {steps} steps...")
    gpu_mon.start()
    cpu_mon.start()

    step_metrics: list[StepMetrics] = []
    batch_times: list[float] = []
    data_times: list[float] = []
    t0_run = time.perf_counter()

    for step_idx in range(steps):
        sm = StepMetrics()
        t_step_start = time.perf_counter()

        # 1. Data loading
        t0 = time.perf_counter()
        batch = next(dataloader_iter)
        byte_batch = batch[0].to(device, non_blocking=True)
        torch.cuda.synchronize()
        t_load = time.perf_counter() - t0

        opt_engine.zero_grad(set_to_none=True)
        input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch

        # 2. Patcher
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = patcher(input_bytes)
        torch.cuda.synchronize()
        t_patcher = time.perf_counter() - t0

        T_patches = patches.shape[1]
        targets, mtp_targets = _build_targets(byte_batch, T_patches, stride=patcher.stride)

        # 3. Model forward
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(
                patches, [targets] + mtp_targets[:-1]
            )
        torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0

        # 4. Loss
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t1_ce = loss_engine.compute_pretrain_loss(logits_t1, targets, [], [])
            loss = loss_engine.compute_pretrain_loss(
                logits_t1, targets, [logits_t2, logits_t3, logits_t4], mtp_targets
            )
            loss = loss + aux_loss.float()
        torch.cuda.synchronize()
        t_loss = time.perf_counter() - t0

        # 5. Backward
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_backward = time.perf_counter() - t0

        # 6. Optimizer (clip + noise + step)
        t0 = time.perf_counter()
        autopilot.before_step(model, step_idx, steps)
        autopilot.update_parameters(step_idx, loss.item(), steps)
        opt_engine.step()
        torch.cuda.synchronize()
        t_opt = time.perf_counter() - t0

        sm.data_load_ms = t_load * 1000
        sm.patcher_ms = t_patcher * 1000
        sm.forward_ms = t_forward * 1000
        sm.loss_ms = t_loss * 1000
        sm.backward_ms = t_backward * 1000
        sm.optimizer_ms = t_opt * 1000
        sm.total_ms = (time.perf_counter() - t_step_start) * 1000
        sm.vram_mb = torch.cuda.memory_allocated() / 1024**2

        step_metrics.append(sm)

        # CPU sample every step
        cpu_mon.sample()

        if (step_idx + 1) % 10 == 0:
            tok_s = cfg.batch_size * cfg.chunk_size / (sm.total_ms / 1000)
            print(f"    step {step_idx + 1:3d}/{steps} "
                  f"| {sm.total_ms:7.1f} ms/step "
                  f"| {tok_s:8.0f} tok/s "
                  f"| VRAM {sm.vram_mb:6.0f} MB"
                  f"| loss {loss.item():.3f}")

    t_run = time.perf_counter() - t0_run

    # Stop monitors
    gpu_mon.stop()
    cpu_mon.stop()

    # ── Collect results ──
    result.step_metrics = step_metrics
    result.gpu_report = gpu_mon.report()
    result.cpu_report = cpu_mon.report()
    result.total_tokens = steps * cfg.batch_size * cfg.chunk_size
    result.total_time_s = t_run
    result.avg_tok_per_s = result.total_tokens / t_run if t_run > 0 else 0
    result.avg_step_ms = float(np.mean([sm.total_ms for sm in step_metrics]))
    result.vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    return result


# ─────────────────────────────────────────────────────────────
# 4. Report
# ─────────────────────────────────────────────────────────────


def print_table(title: str, rows: list[tuple], header: tuple[str, ...]):
    """Pretty-print a table."""
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    sep = "  " + "-" * (sum(w) + 3 * (len(w) - 1))
    print(f"\n  {title}")
    print(sep)
    hdr = " | ".join(f"{str(h):<{w[i]}}" for i, h in enumerate(header))
    print(f"  {hdr}")
    print(sep)
    for row in rows:
        line = " | ".join(f"{str(r):<{w[i]}}" for i, r in enumerate(row))
        print(f"  {line}")
    print(sep)


def print_phase_breakdown(results: list[BenchResult]):
    """Print per-phase timing breakdown."""
    for r in results:
        if not r.step_metrics:
            continue
        metrics = r.step_metrics
        phases = [
            ("Data Load", [sm.data_load_ms for sm in metrics]),
            ("Patcher", [sm.patcher_ms for sm in metrics]),
            ("Forward", [sm.forward_ms for sm in metrics]),
            ("Loss", [sm.loss_ms for sm in metrics]),
            ("Backward", [sm.backward_ms for sm in metrics]),
            ("Optimizer", [sm.optimizer_ms for sm in metrics]),
        ]
        avg_total = float(np.mean([sm.total_ms for sm in metrics]))
        print(f"\n  ⏱  PHASE BREAKDOWN — {r.mode.upper()}")
        print(f"  {'Phase':<18} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'% of step':>10}")
        print(f"  " + "-" * 75)
        for name, vals in phases:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            pct = (mean_v / avg_total) * 100 if avg_total > 0 else 0
            print(f"  {name:<18} | {mean_v:>7.2f}ms | {std_v:>7.2f} | {min_v:>7.2f} | {max_v:>7.2f} | {pct:>8.1f}%")
        print(f"  {'Total':<18} | {avg_total:>7.2f}ms | {'':>8} | {'':>8} | {'':>8} | {'100.0%':>10}")


def print_comparison(results: list[BenchResult]):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"  📊 EAGER vs COMPILE — COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Metric':<30}", end="")
    for r in results:
        print(f" | {r.mode:<12}", end="")
    print()
    print(f"  " + "-" * (30 + 16 * len(results)))

    rows = [
        ("Steps", [str(r.steps) for r in results]),
        ("Total time", [f"{r.total_time_s:.1f}s" for r in results]),
        ("Tokens processed", [f"{r.total_tokens:,}" for r in results]),
        ("Avg tok/s", [f"{r.avg_tok_per_s:,.0f}" for r in results]),
        ("Avg step time", [f"{r.avg_step_ms:.1f}ms" for r in results]),
        ("GPU util (avg)", [f"{r.gpu_report.get('gpu_util_avg_pct', '?'):>5}%" for r in results]),
        ("GPU util (max)", [f"{r.gpu_report.get('gpu_util_max_pct', '?'):>5}%" for r in results]),
        ("GPU mem util", [f"{r.gpu_report.get('mem_util_avg_pct', '?'):>5}%" for r in results]),
        ("GPU power (avg)", [f"{r.gpu_report.get('power_avg_w', '?'):>4}W" for r in results]),
        ("GPU temp (max)", [f"{r.gpu_report.get('temp_max_c', '?'):>3}°C" for r in results]),
        ("CPU util (avg)", [f"{r.cpu_report.get('cpu_avg_pct', '?'):>5}%" for r in results]),
        ("VRAM peak", [f"{r.vram_peak_mb:,.0f} MB" for r in results]),
        ("Compile time", [f"{r.compile_time_s:.1f}s" if hasattr(r, 'compile_time_s') and r.compile_time_s > 0 else "N/A" for r in results]),
    ]

    for name, vals in rows:
        print(f"  {name:<30}", end="")
        for v in vals:
            print(f" | {v:>12}", end="")
        print()


def print_bottleneck_analysis(results: list[BenchResult]):
    """Identify bottlenecks."""
    print(f"\n{'='*70}")
    print(f"  🔍 BOTTLENECK ANALYSIS")
    print(f"{'='*70}")

    for r in results:
        if not r.step_metrics:
            continue
        metrics = r.step_metrics
        avg_total = float(np.mean([sm.total_ms for sm in metrics]))
        avg_fwd = float(np.mean([sm.forward_ms for sm in metrics]))
        avg_bwd = float(np.mean([sm.backward_ms for sm in metrics]))
        avg_opt = float(np.mean([sm.optimizer_ms for sm in metrics]))
        avg_data = float(np.mean([sm.data_load_ms for sm in metrics]))
        avg_patcher = float(np.mean([sm.patcher_ms for sm in metrics]))
        avg_loss = float(np.mean([sm.loss_ms for sm in metrics]))

        print(f"\n  🎯 {r.mode.upper()} — bottlenecks:")

        issues = []

        # Check forward/backward ratio (healthy: ~1:2)
        if avg_bwd > 0:
            fwd_bwd_ratio = avg_fwd / avg_bwd if avg_bwd > 0 else 0
            if fwd_bwd_ratio > 2.0:
                issues.append(f"  ⚠️  Forward ({avg_fwd:.1f}ms) is {fwd_bwd_ratio:.1f}× backward ({avg_bwd:.1f}ms)")
            elif fwd_bwd_ratio < 0.2:
                issues.append(f"  ⚠️  Suspicious: Backward ({avg_bwd:.1f}ms) >> Forward ({avg_fwd:.1f}ms)")
            else:
                issues.append(f"  ✅ Fwd/Bwd ratio = {fwd_bwd_ratio:.2f} (healthy ~0.3-1.0)")

        # Data loading overhead
        data_pct = (avg_data / avg_total) * 100
        if data_pct > 15:
            issues.append(f"  ⚠️  Data loading: {avg_data:.1f}ms ({data_pct:.1f}% of step) — dataloader may be bottleneck")
        else:
            issues.append(f"  ✅ Data loading: {avg_data:.1f}ms ({data_pct:.1f}% of step) — fast enough")

        # Optimizer overhead
        opt_pct = (avg_opt / avg_total) * 100
        if opt_pct > 25:
            issues.append(f"  ⚠️  Optimizer: {avg_opt:.1f}ms ({opt_pct:.1f}% of step) — high overhead")
        elif opt_pct > 15:
            issues.append(f"  📊 Optimizer: {avg_opt:.1f}ms ({opt_pct:.1f}% of step) — moderate")
        else:
            issues.append(f"  ✅ Optimizer: {avg_opt:.1f}ms ({opt_pct:.1f}% of step) — fast")

        # GPU utilization
        gpu_util = r.gpu_report.get("gpu_util_avg_pct", 0)
        if gpu_util < 50:
            issues.append(f"  ⚠️  GPU util: {gpu_util}% — GPU is underutilized! Check CPU/IO bottlenecks")
        elif gpu_util < 80:
            issues.append(f"  📊 GPU util: {gpu_util}% — moderate, room for improvement")
        else:
            issues.append(f"  ✅ GPU util: {gpu_util}% — GPU is well utilized")

        # Patcher overhead
        patcher_pct = (avg_patcher / avg_total) * 100
        if patcher_pct > 5:
            issues.append(f"  📊 Patcher: {avg_patcher:.1f}ms ({patcher_pct:.1f}% of step)")

        # Loss overhead
        loss_pct = (avg_loss / avg_total) * 100
        if loss_pct > 5:
            issues.append(f"  📊 Loss computation: {avg_loss:.1f}ms ({loss_pct:.1f}% of step)")

        for issue in issues:
            print(issue)

    # Cross-mode comparison
    if len(results) >= 2:
        r_eager = results[0] if results[0].mode == "eager" else results[1]
        r_compile = results[1] if results[1].mode == "compile" else results[0]
        if r_eager.avg_tok_per_s > 0 and r_compile.avg_tok_per_s > 0:
            speedup = r_compile.avg_tok_per_s / r_eager.avg_tok_per_s
            print(f"\n  📈 COMPILE SPEEDUP: {speedup:.2f}×")
            if speedup < 1.0:
                print(f"  ❌ Compile is SLOWER than eager (recompilation or overhead)")
            elif speedup < 1.5:
                print(f"  📊 Compile gives modest gain ({speedup:.2f}×)")
            elif speedup < 2.5:
                print(f"  ✅ Compile gives good gain ({speedup:.2f}×)")
            else:
                print(f"  🚀 Compile gives EXCELLENT gain ({speedup:.2f}×)")
            print(f"     Eager: {r_eager.avg_tok_per_s:,.0f} tok/s")
            print(f"     Compile: {r_compile.avg_tok_per_s:,.0f} tok/s")


# ─────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="busel Full Benchmark")
    parser.add_argument("--steps", type=int, default=30, help="Steps per mode (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Chunk size (default: 4096)")
    parser.add_argument("--compile-only", action="store_true", help="Run compile mode only")
    parser.add_argument("--eager-only", action="store_true", help="Run eager mode only")
    parser.add_argument("--output", type=str, default="", help="Save raw data as JSON")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA required for benchmarking")
        sys.exit(1)

    os.makedirs("checkpoints", exist_ok=True)

    data_path = "data_train"

    cfg = BenchConfig(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_steps=args.steps,
    )

    # Warm up CUDA
    _ = torch.tensor([1], device="cuda")
    print(f"\n🔬 busel FULL BENCHMARK v1.0")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Config: d_model={cfg.d_model}, layers={cfg.n_layers}, "
          f"batch={cfg.batch_size}, ctx={cfg.chunk_size}")
    print(f"  Steps per mode: {args.steps}")

    results: list[BenchResult] = []

    if not args.compile_only:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        r = run_bench("eager", args.steps, cfg, data_path, "checkpoints")
        results.append(r)

    if not args.eager_only:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        r = run_bench("compile", args.steps, cfg, data_path, "checkpoints")
        results.append(r)

    # ── Final reports ──
    print(f"\n\n{'='*70}")
    print(f"  📊 FINAL REPORT")
    print(f"{'='*70}")

    print_comparison(results)
    for r in results:
        print_phase_breakdown([r])
    print_bottleneck_analysis(results)

    # Export raw data
    if args.output:
        data = {
            "config": asdict(cfg),
            "results": [],
        }
        for r in results:
            rd = asdict(r)
            rd["step_metrics"] = [asdict(sm) for sm in r.step_metrics]
            data["results"].append(rd)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n  💾 Raw data saved to: {args.output}")

    print(f"\n  ✅ Done\n")


if __name__ == "__main__":
    main()
