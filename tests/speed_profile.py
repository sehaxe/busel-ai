"""
⚡ busel SPEED PROFILER — times every phase of one training step with your model.
Usage: uv run python tests/speed_profile.py --profile verabey-27m [--no-compile]
"""
import time, torch, yaml, sys
sys.path.insert(0, ".")

from model.backbone import buselModel
from model.patching import StridedFastBLTPatcher
from model.layers import configure_bitlinear
from training.optimizer import buselOptimizerEngine
from training.recipe import buselLossEngine
from training.stages.pretrain_config import buselPretrainConfig

def profile(profile_name="verabey-27m", no_compile=False):
    with open("configs/default.yaml") as f:
        profiles = yaml.safe_load(f)["profiles"]
    if profile_name not in profiles:
        print(f"Profile {profile_name!r} not found. Available: {list(profiles.keys())}")
        return
    profile = profiles[profile_name]
    cfg = buselPretrainConfig.from_profile(profile)

    class PC:
        pass
    pc = PC()
    for k, v in profile.get("model", {}).items():
        setattr(pc, k, v)
    pc.vocab_size = cfg.vocab_size
    pc.sct_rank = cfg.sct_rank
    pc.use_matmul_free = cfg.use_matmul_free

    print(f"Model: {pc.d_model}x{pc.n_layers} · experts={pc.num_experts} · sct_rank={cfg.sct_rank}")
    print(f"Batch={cfg.batch_size} · chunk={cfg.chunk_size} · accumulate={cfg.grad_accum_steps}")

    configure_bitlinear(use_fused_training=cfg.use_fused_training)
    if cfg.use_fused_training:
        import model.layers as _l
        _l._BITLINEAR_CONFIG["use_hysteresis"] = False
        _l._BITLINEAR_CONFIG["use_sr_ste"] = False

    device = "cuda"
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    patcher = StridedFastBLTPatcher(d_model=pc.d_model).to(device)
    model = buselModel(pc).to(device).train()
    params = sum(p.numel() for p in model.parameters())
    print(f"Model build: {time.perf_counter() - t0:.1f}s · {params/1e6:.1f}M params\n")

    if not no_compile:
        import torch._dynamo.config as _dc
        import torch._inductor.config as _ic
        import torch._C._dynamo.guards as _guards
        _dc.cache_size_limit = 2048
        _dc.accumulated_cache_size_limit = 256
        _dc.force_parameter_static_shapes = False
        _dc.capture_scalar_outputs = True
        _dc.allow_unspec_int_on_nn_module = True
        _ic.compile_threads = 4
        _ic.coordinate_descent_tuning = False
        _ic.benchmark_kernel = False
        _ic.fx_graph_cache = True
        _ic.triton.cudagraphs = False
        _guards.GuardManager.add_global_state_guard = lambda *args: None

        t0 = time.perf_counter()
        for i in range(len(model.layers)):
            model.layers[i] = torch.compile(model.layers[i], fullgraph=False, dynamic=True)
        for i in range(len(model.m_residuals)):
            model.m_residuals[i] = torch.compile(model.m_residuals[i], fullgraph=False, dynamic=True)
        patcher = torch.compile(patcher, fullgraph=False, dynamic=True)
        model.mtp_pipeline = torch.compile(model.mtp_pipeline, fullgraph=False, dynamic=True)
        print(f"Compile: {time.perf_counter() - t0:.1f}s")

    opt = buselOptimizerEngine(model, patcher, lr_muon=cfg.learning_rate_muon,
                                lr_adamw=cfg.learning_rate_adamw, lotus_rank=cfg.lotus_rank)
    loss_engine = buselLossEngine(cfg.vocab_size)

    n_heads = pc.num_mtp_heads
    batch = cfg.batch_size  # full batch for accurate profiling
    chunk = cfg.chunk_size // 16  # match curriculum start
    print(f"Actual: batch={batch} · chunk={chunk}\n")

    # --- WARMUP (1 step to init) ---
    torch.cuda.synchronize()
    x = torch.randint(0, 200, (batch, chunk), device=device)
    patches = patcher(x)
    T = patches.shape[1]
    targets = x[:, 1::4][:, :T].to(device)
    mtp = [x[:, (h+2)::4][:, :T].to(device) for h in range(n_heads)]
    logits, aux = model(patches, [targets] + mtp[:-1], progress=0.0)
    loss = loss_engine.compute_pretrain_loss(logits[0], targets, list(logits[1:]), mtp)
    (loss + aux).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print("Warmup done\n")

    # --- TIMING (10 steps with realistic overhead: EMA, noise, weight guard) ---
    from training.autopilot import buselAutoPilot
    from training.optimizer import EMA
    autopilot = buselAutoPilot(opt, max_lr_muon=cfg.learning_rate_muon,
                                max_lr_adamw=cfg.learning_rate_adamw,
                                target_wd=cfg.weight_decay, warmup_steps=10,
                                min_lr_ratio=cfg.min_lr_ratio, lr_schedule=cfg.lr_schedule,
                                wsd_decay_frac=cfg.wsd_decay_frac, grad_clip=cfg.grad_clip)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    phases = {"fwd_total": [], "bwd_total": [], "opt": [], "autopilot": [], "ema": [], "data": [], "weight_guard": []}

    for step in range(10):
        opt.zero_grad(set_to_none=True)

        # Data loading (simulates mmap read via random + transfer)
        t = time.perf_counter()
        x = torch.randint(0, 200, (batch, chunk), device=device)
        torch.cuda.synchronize()
        phases["data"].append(time.perf_counter() - t)

        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            input_bytes = x[:, :-4]
            patches = patcher(input_bytes)
            T = patches.shape[1]
            targets = x[:, 1::4][:, :T].to(device)
            mtp = [x[:, (h+2)::4][:, :T].to(device) for h in range(n_heads)]
            logits, aux = model(patches, [targets] + mtp[:-1], progress=0.0)
            loss = loss_engine.compute_pretrain_loss(logits[0], targets, list(logits[1:]), mtp)
            total = loss + aux
        torch.cuda.synchronize()
        phases["fwd_total"].append(time.perf_counter() - t)

        t = time.perf_counter()
        total.backward()
        torch.cuda.synchronize()
        phases["bwd_total"].append(time.perf_counter() - t)

        t = time.perf_counter()
        autopilot.before_step(model, step, 1000)
        autopilot.inject_noise(model)
        autopilot.update_parameters(step, loss.item(), 1000)
        phases["autopilot"].append(time.perf_counter() - t)

        t = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        phases["opt"].append(time.perf_counter() - t)

        t = time.perf_counter()
        if ema is not None:
            ema.update(model)
        phases["ema"].append(time.perf_counter() - t)

        # Weight guard (realistic frequency — every 50 steps, but simulate once)
        if step == 1:
            t = time.perf_counter()
            _ = torch.stack([p.data.abs().max() for p in model.parameters() if p.ndim == 2]).max().item()
            torch.cuda.synchronize()
            phases["weight_guard"].append(time.perf_counter() - t)
        else:
            phases["weight_guard"].append(0.0)

    # Results (skip first 2 warmup steps)
    tok_per_step = batch * (T if T else chunk)
    print(f"Tokens/step: {tok_per_step:,} · Patches: ({batch}, {T}, {pc.d_model})")
    print(f"\n{'Phase':<15} {'Time (ms)':>10} {'%':>6} {'raw tok/s':>14}")
    print("-" * 50)
    total_ms = 0
    for phase, times in phases.items():
        avg = sum(times[2:]) / max(1, len(times) - 2) * 1000
        total_ms += avg
    for phase, times in phases.items():
        avg = sum(times[2:]) / max(1, len(times) - 2) * 1000
        pct = avg / total_ms * 100 if total_ms > 0 else 0
        tps = tok_per_step / (avg / 1000) if avg > 0 else 0
        if avg > 0.01:
            print(f"{phase:<15} {avg:>10.1f} {pct:>5.1f}% {tps:>12,.0f} tok/s")

    total_tps = tok_per_step / (total_ms / 1000)
    raw_tps = batch * chunk / (total_ms / 1000)  # raw bytes per second
    print(f"\n{'TOTAL/step':<15} {total_ms:>10.1f}ms → {total_tps:,.0f} patch-tok/s → {raw_tps:,.0f} raw-byte/s")
    print(f"VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
    try:
        import pynvml; pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        print(f"GPU util: {u.gpu}% · mem: {u.memory}%")
    except:
        pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="verabey-27m")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()
    profile(args.profile, args.no_compile)
