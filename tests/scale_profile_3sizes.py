"""
🧪 busel 3-SIZE PAIR-INTERACTION PROFILER — v5.8
Compares 5 configurations across 3 model sizes (micro_test / shpak / zubr)
to measure pair/triple interaction overhead at scale.
Per run: 5 warmup + 30 measured steps, batch=16 ctx=4096 (same workload).
zubr falls back to batch=4 if OOM. Reports mean/stddev/min/max step time.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import time
import json
import math
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import get_busel_dataloader
from multimodal.special_tokens import vocab_size as _vocab_size
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine


PROFILE_NAMES = ["micro_test", "shpak", "zubr"]
BATCH_DEFAULT = 16
BATCH_FALLBACK_FOR_ZUBR = 4
CHUNK_SIZE_FORCED = 4096

N_WARMUP = 5
N_MEASURE = 30


def load_profile(name: str) -> dict:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    return full["profiles"][name]


def build_model_and_optim(
    cfg_profile: dict,
    batch_size: int,
    use_error_feedback: bool = False,
    sparse_6_8: bool = False,
    selective_backward: bool = False,
    backward_ratio: float = 1.0,
    device: str = "cuda",
):
    profile = dict(cfg_profile)
    profile["model"] = dict(profile["model"])
    profile["model"]["sparse_6_8"] = sparse_6_8
    profile["model"]["selective_backward"] = selective_backward
    profile["model"]["backward_ratio"] = backward_ratio
    profile["data"] = dict(profile["data"])
    profile["data"]["batch_size"] = batch_size
    profile["data"]["chunk_size"] = CHUNK_SIZE_FORCED
    profile["training"] = dict(profile["training"])
    profile["training"]["use_error_feedback"] = use_error_feedback

    class Cfg:
        pass
    cfg = Cfg()
    cfg.vocab_size = _vocab_size()
    cfg.optimizer_type = "lotus_muon"
    cfg.lotus_rank = 8
    cfg.lotus_lr_scale = 0.5
    m = profile["model"]
    for k, v in m.items():
        setattr(cfg, k, v)
    d = profile["data"]
    cfg.data_path = d["data_path"]
    cfg.chunk_size = d["chunk_size"]
    cfg.batch_size = d["batch_size"]
    t = profile["training"]
    for k, v in t.items():
        setattr(cfg, k, v)

    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)

    target_dtype = torch.bfloat16
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__:
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.data = module.weight.data.to(target_dtype)

    if device == "cuda":
        model.enable_gradient_checkpointing(every=2)

    opt = buselOptimizerEngine(
        model,
        lr_muon=cfg.learning_rate_muon,
        lr_adamw=cfg.learning_rate_adamw,
        optimizer_type=cfg.optimizer_type,
        lotus_rank=cfg.lotus_rank,
        lotus_lr_scale=cfg.lotus_lr_scale,
        use_error_feedback=use_error_feedback,
    )
    autopilot = buselAutoPilot(
        opt,
        max_lr_muon=cfg.learning_rate_muon,
        max_lr_adamw=cfg.learning_rate_adamw,
        target_wd=cfg.weight_decay,
    )
    loss_engine = buselLossEngine(cfg.vocab_size)
    return model, patcher, opt, autopilot, loss_engine, cfg


def run_one(
    profile_name: str,
    cfg_profile: dict,
    flags: dict,
    device: str = "cuda",
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    batch_default: int = BATCH_DEFAULT,
    batch_fallback: int | None = None,
):
    name = f"[{profile_name}] {flags.get('label', '')}"
    print(f"\n{'=' * 90}\n🔬 RUN: {name}\n   flags: {flags}\n{'=' * 90}")

    batch = batch_default
    use_fb = flags.get("use_error_feedback", False)
    s6_8 = flags.get("sparse_6_8", False)
    sb = flags.get("selective_backward", False)
    br = flags.get("backward_ratio", 1.0)

    def _build(b):
        return build_model_and_optim(cfg_profile, batch_size=b, use_error_feedback=use_fb,
                                      sparse_6_8=s6_8, selective_backward=sb,
                                      backward_ratio=br, device=device)

    model, patcher, opt, ap, loss_engine, cfg = _build(batch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params: {n_params:,} ({n_params * 2 / 1024**2:.2f} MB FP16)  |  batch={batch}  |  ctx={cfg.chunk_size}")

    os.makedirs("data_train", exist_ok=True)
    test_file = "profiler_3sizes_test_data.txt"
    test_path = os.path.join("data_train", test_file)
    created = not os.path.exists(test_path)
    if created:
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("busel 3-size profiler. " * 500)
    try:
        dataloader = get_busel_dataloader(
            "data_train", chunk_size=cfg.chunk_size // 4, batch_size=cfg.batch_size
        )
        it = iter(dataloader)

        def _one_step():
            bb, _, _ = next(it)
            bb = bb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ib = bb[:, :-patcher.stride] if bb.shape[1] > patcher.stride else bb
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                patches = patcher(ib)
                T = patches.shape[1]
                tg = bb[:, 1::patcher.stride][:, :T]
                if tg.shape[1] < T:
                    tg = torch.nn.functional.pad(tg, (0, T - tg.shape[1]), value=0)
                (lo, _, _, _), aux = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(lo, tg) + aux.float()
            loss.backward()
            ap.inject_noise(model)
            opt.step()
            torch.cuda.synchronize()
            return loss.item()

        for w in range(n_warmup):
            try:
                _one_step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_fallback is not None and batch != batch_fallback:
                    print(f"   ⚠️  OOM in warmup step {w} at batch={batch} → retry with batch={batch_fallback}")
                    del model, patcher, opt, ap, loss_engine
                    torch.cuda.empty_cache()
                    batch = batch_fallback
                    model, patcher, opt, ap, loss_engine, cfg = _build(batch)
                    n_params = sum(p.numel() for p in model.parameters())
                    print(f"   re-built: params={n_params:,}  |  batch={batch}")
                    dataloader = get_busel_dataloader(
                        "data_train", chunk_size=cfg.chunk_size // 4, batch_size=cfg.batch_size
                    )
                    it = iter(dataloader)
                    _one_step()
                else:
                    raise
        torch.cuda.synchronize()

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        step_times, losses = [], []
        for s in range(n_measure):
            t0 = time.perf_counter()
            bb, _, _ = next(it)
            bb = bb.to(device, non_blocking=True)
            ib = bb[:, :-patcher.stride] if bb.shape[1] > patcher.stride else bb
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                patches = patcher(ib)
                T = patches.shape[1]
                tg = bb[:, 1::patcher.stride][:, :T]
                if tg.shape[1] < T:
                    tg = torch.nn.functional.pad(tg, (0, T - tg.shape[1]), value=0)
                (lo, _, _, _), aux = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(lo, tg) + aux.float()
            loss.backward()
            ap.inject_noise(model)
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            step_times.append(dt)
            losses.append(loss.item())

        arr = np.array(step_times)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        tokens_per_step = cfg.batch_size * cfg.chunk_size
        mean_s = float(arr.mean())
        tps = tokens_per_step / mean_s
        print(
            f"   ✅ step={mean_s * 1000:.1f}ms ±{arr.std() * 1000:.1f}ms  |  "
            f"min={arr.min() * 1000:.1f}  max={arr.max() * 1000:.1f}  |  "
            f"peak={peak_mb:.0f} MB  |  tps={tps:.0f}  |  loss@end={losses[-1]:.3f}"
        )
        return {
            "name": name,
            "profile": profile_name,
            "batch": batch,
            "n_params": n_params,
            "step_ms_mean": mean_s * 1000,
            "step_ms_std": float(arr.std()) * 1000,
            "step_ms_min": float(arr.min()) * 1000,
            "step_ms_max": float(arr.max()) * 1000,
            "peak_mb": peak_mb,
            "tps": tps,
            "final_loss": float(losses[-1]),
            "n_warmup": n_warmup,
            "n_measure": n_measure,
            "flags": flags,
        }
    finally:
        if created:
            if os.path.exists(test_path):
                os.remove(test_path)
            try:
                os.rmdir("data_train")
            except OSError:
                pass


CONFIG_RUNS = [
    ("baseline",            {}),
    ("+LCSB ratio=0.5",     {"selective_backward": True, "backward_ratio": 0.5}),
    ("+Sparse 6:8 + LCSB",  {"sparse_6_8": True, "selective_backward": True, "backward_ratio": 0.5}),
    ("+GradLite + LCSB",    {"use_error_feedback": True, "selective_backward": True, "backward_ratio": 0.5}),
    ("+ALL three",          {"sparse_6_8": True, "use_error_feedback": True,
                             "selective_backward": True, "backward_ratio": 0.5}),
]

def main():
    print("🧪 busel 3-SIZE PAIR-INTERACTION PROFILER (v5.8)")
    print(f"   Sizes: {PROFILE_NAMES}")
    print(f"   Configs: {len(CONFIG_RUNS)}")
    print(f"   Steps per run: {N_WARMUP} warmup + {N_MEASURE} measured")
    print(f"   Workload: batch={BATCH_DEFAULT} ctx=4096 (zubr fallback: batch={BATCH_FALLBACK_FOR_ZUBR})")
    print(f"   Same workload across all sizes for fair overhead % comparison.\n")

    profiles = {n: load_profile(n) for n in PROFILE_NAMES}
    all_results: dict[str, list[dict]] = {n: [] for n in PROFILE_NAMES}

    for pname in PROFILE_NAMES:
        print("\n" + "#" * 90)
        print(f"# 📐 PROFILE: {pname}")
        print("#" * 90)
        cfg = profiles[pname]
        for cname, flags in CONFIG_RUNS:
            flags = {**flags, "label": cname}
            try:
                fb = BATCH_FALLBACK_FOR_ZUBR if pname == "zubr" else None
                r = run_one(pname, cfg, flags, batch_default=BATCH_DEFAULT, batch_fallback=fb)
                all_results[pname].append(r)
            except Exception as e:
                print(f"   ❌ FAILED: {type(e).__name__}: {e}")
                all_results[pname].append({"name": f"[{pname}] {cname}", "profile": pname, "error": str(e)})

    out = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "n_warmup": N_WARMUP,
        "n_measure": N_MEASURE,
        "batch_default": BATCH_DEFAULT,
        "batch_fallback_zubr": BATCH_FALLBACK_FOR_ZUBR,
        "results": all_results,
    }
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/scale_profile_3sizes.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: checkpoints/scale_profile_3sizes.json")

    for pname in PROFILE_NAMES:
        results = all_results[pname]
        base = results[0] if results and "step_ms_mean" in results[0] else None
        print("\n" + "=" * 105)
        title = f"📊 {pname.upper()} — {CONFIG_RUNS[0][0]}  vs  +LCSB  vs  pair interactions"
        print(title.center(105))
        print(f"   {N_MEASURE} measured steps  |  batch={BATCH_DEFAULT} ctx=4096  |  Δstep shown vs baseline".center(105))
        print("=" * 105)
        print(
            f"{'Config':<28} | {'Step mean (ms)':>15} | {'std (ms)':>10} | {'min–max':>14} | "
            f"{'Peak (MB)':>10} | {'tok/s':>8} | {'Δstep %':>8}"
        )
        print("-" * 105)
        for r in results:
            if "error" in r:
                print(f"{r['name']:<28} | {'ERR':>15} | {'ERR':>10} | {'ERR':>14} | {'ERR':>10} | {'ERR':>8} | {'ERR':>8}")
                continue
            ds_pct = ""
            if base is not None and r is not base:
                delta = (r["step_ms_mean"] - base["step_ms_mean"]) / base["step_ms_mean"] * 100
                ds_pct = f"{delta:+.1f}%"
            minmax = f"{r['step_ms_min']:.0f}–{r['step_ms_max']:.0f}"
            print(
                f"{r['flags'].get('label', '?'):<28} | {r['step_ms_mean']:>15.1f} | {r['step_ms_std']:>10.1f} | "
                f"{minmax:>14} | {r['peak_mb']:>10.0f} | {r['tps']:>8.0f} | {ds_pct:>8}"
            )
        print("=" * 105)


if __name__ == "__main__":
    main()
