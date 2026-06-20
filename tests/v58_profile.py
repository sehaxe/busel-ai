"""
🧪 busel PROFILE SUITE — v6.0 cumulative + v6.1 dispersion + v8.5 kruk profiler.
Three modes: shpak-v60 / shpak-disp / kruk-v85 (see --help).
"""
import argparse
import json
import os
import sys
import time
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pipeline import get_busel_dataloader
from multimodal.special_tokens import vocab_size as _vocab_size
from model.patching import StridedFastBLTPatcher
from model.backbone import buselModel
from training.optimizer import buselOptimizerEngine
from training.autopilot import buselAutoPilot
from training.recipe import buselLossEngine

BATCH = 16
CHUNK_SIZE_FORCED = 4096
N_WARMUP_5RUN = 2
N_MEASURE_5RUN = 10
BATCH_FALLBACK_FOR_ZUBR = 4

_MODEL_FLAGS = frozenset((
    "selective_backward", "backward_ratio", "use_differential_attention",
    "use_qknorm_l2", "use_hestia", "use_cla", "cla_share_every",
    "sct_rank", "use_flex_attention", "n_hyper", "top_k", "num_experts",
    "d_model", "n_layers", "n_heads", "expert_hidden",
))


def _load_profile(name: str) -> dict:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    return full["profiles"][name]


def _build(profile_name, batch_size, device, **flags):
    cfg_profile = _load_profile(profile_name)
    profile = dict(cfg_profile)
    profile["model"] = dict(profile["model"])
    profile["data"] = dict(profile["data"])
    profile["data"]["batch_size"] = batch_size
    profile["data"]["chunk_size"] = CHUNK_SIZE_FORCED
    profile["training"] = dict(profile["training"])

    for k, v in flags.items():
        if v is None:
            continue
        if k in _MODEL_FLAGS:
            profile["model"][k] = v
        else:
            profile["training"][k] = v

    class Cfg:
        pass
    cfg = Cfg()
    cfg.vocab_size = _vocab_size()
    cfg.lotus_rank = 8
    cfg.lr_multipliers = None
    cfg.n_hyper = 2
    cfg.use_ema = True
    cfg.ema_decay = 0.999
    cfg.use_schedule_free = False
    cfg.sf_beta = 0.9
    cfg.sf_gamma_factor = 2.0
    cfg.use_cautious = False
    cfg.use_adafactor = False
    cfg.use_quest = False
    cfg.quest_bits = 1.58
    cfg.use_tequila = False
    cfg.tequila_lambda = 1e-3
    cfg.use_hestia = False
    cfg.hestia_init_temp = 6.0
    cfg.hestia_end_temp = 0.0
    cfg.use_differential_attention = False
    cfg.use_qknorm_l2 = False
    cfg.use_cla = False
    cfg.cla_share_every = 2
    cfg.sct_rank = 0
    cfg.sct_scope = "mlp"
    cfg.use_flex_attention = False
    cfg.use_dispersion_loss = False
    cfg.dispersion_weight = 0.1
    cfg.dispersion_temperature = 2.0
    cfg.lr_schedule = "cosine"
    cfg.wsd_decay_fraction = 0.2
    cfg.wsd_s_enabled = False
    cfg.wsd_s_interval = 1000
    cfg.wsd_s_decay_steps = 200
    cfg.optimization_mode = "manual"
    cfg.use_salt = False
    cfg.salt_teacher_profile = "chyzh"
    cfg.salt_kd_temperature = 2.0
    cfg.salt_kd_alpha = 0.5
    cfg.salt_kd_steps = 0
    for src in (profile["model"], profile["data"], profile["training"]):
        for k, v in src.items():
            setattr(cfg, k, v)

    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = buselModel(cfg).to(device)
    target_dtype = torch.bfloat16
    for obj in [model, patcher]:
        for module in obj.modules():
            if "RMSNorm" in module.__class__.__name__ and hasattr(module, "weight") and module.weight is not None:
                module.weight.data = module.weight.data.to(target_dtype)
    if device == "cuda":
        model.enable_gradient_checkpointing(every=2)
    opt = buselOptimizerEngine(
        model,
        lr_muon=cfg.learning_rate_muon,
        lr_adamw=cfg.learning_rate_adamw,
        lotus_rank=cfg.lotus_rank,
    )
    autopilot = buselAutoPilot(opt, max_lr_muon=cfg.learning_rate_muon,
                                 max_lr_adamw=cfg.learning_rate_adamw, target_wd=cfg.weight_decay)
    loss_engine = buselLossEngine(cfg.vocab_size)
    return model, patcher, opt, autopilot, loss_engine, cfg


def _run_one(name, profile_name, batch_size, device, n_warmup, n_measure, **flags):
    flag_str = " ".join(f"{k}={v}" for k, v in flags.items() if v is not None) or "(defaults)"
    print(f"\n{'=' * 80}\n🔬 RUN: {name}\n   profile={profile_name} batch={batch_size} flags={flag_str}\n{'=' * 80}")
    model, patcher, opt, ap, loss_engine, cfg = _build(profile_name, batch_size, device, **flags)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   params: {n_params:,} ({n_params * 2 / 1024**2:.2f} MB FP16)")

    os.makedirs("data_train", exist_ok=True)
    test_file = "profiler_v58_test_data.txt"
    created_dir = len(os.listdir("data_train")) == 0
    if created_dir:
        with open(os.path.join("data_train", test_file), "w", encoding="utf-8") as f:
            f.write("v58 profile busel profiler. " * 300)
    try:
        dataloader = get_busel_dataloader("data_train", chunk_size=cfg.chunk_size // 4, batch_size=cfg.batch_size)
        it = iter(dataloader)

        def _one_step():
            bb, _, _ = next(it)
            bb = bb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ib = bb[:, :-patcher.stride] if bb.shape[1] > patcher.stride else bb
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if cfg.use_dispersion_loss:
                    patches, embed_for_dispersion = patcher(ib, return_embedding=True)
                else:
                    patches = patcher(ib)
                T = patches.shape[1]
                tg = bb[:, 1::patcher.stride][:, :T]
                if tg.shape[1] < T:
                    tg = torch.nn.functional.pad(tg, (0, T - tg.shape[1]), value=0)
                (lo, _, _, _), aux = model(patches, None)
                loss = loss_engine.compute_pretrain_loss(lo, tg) + aux.float()
                if cfg.use_dispersion_loss:
                    loss = loss + loss_engine.compute_dispersion_loss(
                        embed_for_dispersion,
                        weight=cfg.dispersion_weight,
                        temperature=cfg.dispersion_temperature,
                    )
            loss.backward()
            ap.inject_noise(model)
            opt.step()
            torch.cuda.synchronize()
            return loss.item()

        for _ in range(n_warmup):
            _one_step()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        step_times, losses = [], []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            loss = _one_step()
            step_times.append(time.perf_counter() - t0)
            losses.append(loss)

        mean_step = float(np.mean(step_times))
        std_step = float(np.std(step_times))
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        tokens_per_step = cfg.batch_size * cfg.chunk_size
        tps = tokens_per_step / mean_step
        print(f"   ✅ step={mean_step * 1000:.1f}±{std_step * 1000:.1f}ms  |  peak={peak_mb:.0f} MB  |  tps={tps:.0f}  |  loss@end={np.mean(losses[-3:]):.3f}")
        return {
            "name": name,
            "profile": profile_name,
            "batch_size": batch_size,
            "params": n_params,
            "step_ms": mean_step * 1000,
            "step_ms_std": std_step * 1000,
            "peak_mb": peak_mb,
            "tps": tps,
            "final_loss": float(np.mean(losses[-3:])),
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and profile_name == "zubr":
            print(f"   ⚠️  OOM at batch={batch_size}; falling back to batch={BATCH_FALLBACK_FOR_ZUBR}")
            del model, patcher, opt, ap, loss_engine
            torch.cuda.empty_cache()
            return _run_one(name, profile_name, BATCH_FALLBACK_FOR_ZUBR, device, n_warmup, n_measure, **flags)
        raise
    finally:
        if created_dir:
            path = os.path.join("data_train", test_file)
            if os.path.exists(path):
                os.remove(path)
            try:
                os.rmdir("data_train")
            except OSError:
                pass


def _print_table(title, results):
    print("\n" + "=" * 90)
    print(title.center(90))
    print("=" * 90)
    print(f"{'Run':<35} | {'Step (ms)':>10} | {'Peak (MB)':>10} | {'tok/s':>8} | {'Loss':>8}")
    print("-" * 90)
    base = next((r for r in results if "step_ms" in r), None)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<35} | {'ERR':>10} | {'ERR':>10} | {'ERR':>8} | {'ERR':>8}")
            continue
        ds = ""
        if r is not base and base and "step_ms" in base:
            delta_ms = r["step_ms"] - base["step_ms"]
            delta_mb = r["peak_mb"] - base["peak_mb"]
            ds = f"  (Δstep={delta_ms:+.1f}ms, Δmem={delta_mb:+.0f}MB)"
        print(f"{r['name']:<35} | {r['step_ms']:>10.1f} | {r['peak_mb']:>10.0f} | "
              f"{r['tps']:>8.0f} | {r['final_loss']:>8.3f}{ds}")
    print("=" * 90)


def mode_shpak_v60(device):
    print("🚀 busel SHPAK CUMULATIVE PROFILER — builds the best config from validated winners")
    print("   Profile: shpak 52.8M params, batch=16 ctx=4096")
    print(f"   Steps per run: {N_MEASURE_5RUN} ({N_WARMUP_5RUN} warmup + {N_MEASURE_5RUN} measured)\n")
    print("   Each run adds one opt-in feature (all opt-in via buselPretrainConfig).")
    print("   For best SF results set min_lr_ratio=1.0 in profile to disable cosine interference.\n")
    runs = [
        ("1. baseline",                              {}),
        ("2. + DA",                                  {"use_differential_attention": True}),
        ("3. + DA + Cautious",                       {"use_differential_attention": True, "use_cautious": True}),
        ("4. + DA + Cautious + LCSB",                {"use_differential_attention": True, "use_cautious": True, "selective_backward": True, "backward_ratio": 0.5}),
        ("5. + DA + Cautious + SF + LCSB (full)",    {"use_differential_attention": True, "use_cautious": True, "use_schedule_free": True, "selective_backward": True, "backward_ratio": 0.5}),
    ]
    results = []
    for name, flags in runs:
        try:
            results.append(_run_one(name, "shpak", BATCH, n_warmup=N_WARMUP_5RUN, n_measure=N_MEASURE_5RUN, **flags, device=device))
        except Exception as e:
            print(f"   ❌ FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "error": str(e)})
    _print_table("SHPAK CUMULATIVE COMPARISON (52.8M, batch=16 ctx=4096, 10 steps)", results)
    return results


def mode_shpak_disp(device):
    print("🔵 busel SHPAK DISPERSION PROFILER — Wang 2026 on token embeddings")
    print("   Profile: shpak 52.8M params, batch=16 ctx=4096")
    print(f"   Steps per run: {N_MEASURE_5RUN} ({N_WARMUP_5RUN} warmup + {N_MEASURE_5RUN} measured)\n")
    print("   Each run adds Dispersion Loss to the v6.0 winner (DA+Cautious+LCSB).")
    print("   Expected: similar step time, lower loss if Wang 2026 claim holds at scale.\n")
    runs = [
        ("1. baseline",                                           {}),
        ("2. + Dispersion",                                       {"use_dispersion_loss": True}),
        ("3. + DA + Cautious + LCSB (v6.0 winner)",              {"use_differential_attention": True, "use_cautious": True, "selective_backward": True, "backward_ratio": 0.5}),
        ("4. + DA + Cautious + LCSB + Dispersion (v6.1)",        {"use_differential_attention": True, "use_cautious": True, "selective_backward": True, "backward_ratio": 0.5, "use_dispersion_loss": True}),
    ]
    results = []
    for name, flags in runs:
        try:
            results.append(_run_one(name, "shpak", BATCH, n_warmup=N_WARMUP_5RUN, n_measure=N_MEASURE_5RUN, **flags, device=device))
        except Exception as e:
            print(f"   ❌ FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "error": str(e)})
    _print_table("SHPAK DISPERSION COMPARISON (52.8M, batch=16 ctx=4096, 10 steps)", results)
    return results


def mode_kruk_v85(device):
    """v8.5 Singularity profiler on kruk (~104M, 12 layers, 6 experts)."""
    print("🔮 busel KRUK SINGULARITY PROFILER — SCT + CLA + FlexAttention + auto-mode")
    print("   Profile: kruk ~104M params (12 layers, 6 experts), batch=16 ctx=4096")
    print(f"   Steps per run: {N_MEASURE_5RUN} ({N_WARMUP_5RUN} warmup + {N_MEASURE_5RUN} measured)\n")
    print("   Tests v8.5 technologies: SCT, CLA, FlexAttention, Cautious, SF, QuEST, Hestia.\n")
    runs = [
        ("1. baseline (kruk defaults)",            {}),
        ("2. + Cautious",                          {"use_cautious": True}),
        ("3. + Cautious + CLA",                    {"use_cautious": True, "use_cla": True, "cla_share_every": 2}),
        ("4. + Cautious + CLA + FlexAttn",         {"use_cautious": True, "use_cla": True, "cla_share_every": 2, "use_flex_attention": True}),
        ("5. + Cautious + CLA + SCT r=32",         {"use_cautious": True, "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
        ("6. + Cautious + CLA + SCT r=64",         {"use_cautious": True, "use_cla": True, "cla_share_every": 2, "sct_rank": 64}),
        ("7. + Cautious + SF + CLA + SCT r=32",    {"use_cautious": True, "use_schedule_free": True, "min_lr_ratio": 1.0, "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
        ("8. + Cautious + QuEST + CLA + SCT r=32", {"use_cautious": True, "use_quest": True, "quest_bits": 1.58, "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
        ("9. + Cautious + Hestia + CLA + SCT r=32",{"use_cautious": True, "use_hestia": True, "hestia_init_temp": 6.0, "hestia_end_temp": 0.0, "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
        ("10. + DA + Cautious + CLA + SCT r=32",   {"use_differential_attention": True, "use_cautious": True, "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
        ("11. SOAP + CLA + SCT r=32",              {"optimizer_type": "soap", "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
        ("12. MuonQ + CLA + SCT r=32",             {"optimizer_type": "muonq", "use_cla": True, "cla_share_every": 2, "sct_rank": 32}),
    ]
    results = []
    for name, flags in runs:
        try:
            results.append(_run_one(name, "kruk", BATCH, n_warmup=N_WARMUP_5RUN, n_measure=N_MEASURE_5RUN, **flags, device=device))
        except Exception as e:
            print(f"   ❌ FAILED: {type(e).__name__}: {e}")
            results.append({"name": name, "error": str(e)})
    _print_table("KRUK SINGULARITY COMPARISON (~104M, batch=16 ctx=4096, 10 steps)", results)
    return results


def main():
    parser = argparse.ArgumentParser(description="busel profile suite")
    parser.add_argument("--mode", choices=["shpak-v60", "shpak-disp", "kruk-v85"],
                        default="kruk-v85", help="Which comparison to run (default: kruk-v85)")
    parser.add_argument("--out", default="checkpoints/v58_profile.json",
                        help="Output JSON path (default: checkpoints/v58_profile.json)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if args.mode == "shpak-v60":
        results = mode_shpak_v60(device)
    elif args.mode == "shpak-disp":
        results = mode_shpak_disp(device)
    else:
        results = mode_kruk_v85(device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "device": device, "results": results}, f, indent=2)
    print(f"\n📄 Results saved to {args.out}")


if __name__ == "__main__":
    main()
