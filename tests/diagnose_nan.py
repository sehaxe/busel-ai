"""
🔬 NaN Diagnostic — standalone script, не трогает тренировку.
Загружает чекпоинт, прогоняет forward+backward, ловит источник NaN.

Usage: uv run python -m tests.diagnose_nan [--checkpoint PATH] [--steps 5]
"""

import argparse, sys, gc, math
import torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

def _find_latest_checkpoint():
    import glob
    files = sorted(glob.glob("checkpoints/busel_kruk-210m_step_*.pt"))
    for pat in [170, 160, 150, 140, 130, 120, 110, 100]:
        for f in files:
            if f"step_{pat}" in f:
                return f
    return files[-1] if files else None

def _load_config(profile_name="kruk-210m"):
    from training.stages.pretrain import buselPretrainConfig
    import yaml, pathlib
    cfg_path = pathlib.Path("configs/default.yaml")
    if cfg_path.exists():
        with open(cfg_path) as fh:
            all_cfg = yaml.safe_load(fh)
        profile = all_cfg.get("profiles", {}).get(profile_name, {})
        return buselPretrainConfig.from_profile(profile)
    return buselPretrainConfig()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--profile", default="kruk-210m")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.0002)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 NaN Diagnostic | device={device} | profile={args.profile}")

    # --- Config & Model ---
    cfg = _load_config(args.profile)
    from model.backbone import buselModel
    model = buselModel(cfg).to(device)
    print(f"   Model: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Load checkpoint ---
    ckpt_path = args.checkpoint or _find_latest_checkpoint()
    if ckpt_path is None:
        print("❌ No checkpoint found!"); return 1
    print(f"   Checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    from model.checkpoint import strip_compile_prefix, load_state_dict_safely
    model_sd = strip_compile_prefix(sd.get("model_state_dict", sd))
    load_state_dict_safely(model, model_sd, strict=False)

    # --- Patcher ---
    from model.patching import StridedFastBLTPatcher
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    patcher_sd = sd.get("patcher_state_dict", {})
    if patcher_sd:
        load_state_dict_safely(patcher, strip_compile_prefix(patcher_sd), strict=False)

    # --- Dataloader ---
    vocab = getattr(cfg, "vocab_size", 326)
    stride = patcher.stride
    chunk = ((args.chunk + stride) // stride) * stride
    data = torch.randint(0, min(vocab, 256), (args.batch, chunk), device=device)

    # --- Hooks: forward NaN ---
    _first_nan_fwd = [None]
    def _fwd_hook_factory(name):
        def hook(module, inp, out):
            if _first_nan_fwd[0] is not None: return
            for t in ([inp] if isinstance(inp, torch.Tensor) else (inp if isinstance(inp, (list, tuple)) else [])):
                if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
                    _first_nan_fwd[0] = f"FWD {'INPUT' if t is inp else 'ARG'} {name}"
                    return
            if isinstance(out, (list, tuple)):
                for o in out:
                    if isinstance(o, torch.Tensor) and (torch.isnan(o).any() or torch.isinf(o).any()):
                        _first_nan_fwd[0] = f"FWD OUTPUT {name}"; return
            elif isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
                _first_nan_fwd[0] = f"FWD OUTPUT {name}"
        return hook
    for n, m in model.named_modules():
        if len(list(m.children())) == 0:
            m.register_forward_hook(_fwd_hook_factory(n))

    # --- Hooks: backward NaN ---
    _first_nan_grad = [None]
    for n, p in model.named_parameters():
        if p.requires_grad:
            def _grad_hook_factory(name):
                def hook(grad):
                    if _first_nan_grad[0] is not None: return grad
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        _first_nan_grad[0] = name
                    return grad
                return hook
            p.register_hook(_grad_hook_factory(n))

    # --- SCT diagnostic ---
    def _check_sct(tag=""):
        s_vals = []
        for m in model.modules():
            if m.__class__.__name__ == "SpectralLinear":
                s_vals.append(m.s.data.float().cpu())
        if not s_vals: return
        all_s = torch.cat(s_vals)
        print(f"   [{tag}] SCT s: min={all_s.min().item():.4f} mean={all_s.abs().mean().item():.4f} max={all_s.max().item():.4f} "
              f"n={len(s_vals)} >5={(all_s.abs()>5).sum().item()} >10={(all_s.abs()>10).sum().item()}")

    _check_sct("LOAD")

    # --- Optimizer ---
    from training.optimizer import buselOptimizerEngine
    opt = buselOptimizerEngine(model, lr_muon=args.lr, lr_adamw=args.lr * 0.1,
                               lotus_rank=getattr(cfg, "sct_rank", 8))
    from model.layers import retract_all

    model.train()
    patcher.train()

    for step in range(args.steps):
        opt.zero_grad()
        _first_nan_fwd[0] = None
        _first_nan_grad[0] = None

        # Prepare targets
        input_bytes = data[:, :-stride] if data.shape[1] > stride else data
        targets = data[:, stride::stride]
        T_patches = min(input_bytes.shape[1] // stride, targets.shape[1])

        # Forward
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches, embed = patcher(input_bytes, return_embedding=True)
            mtp_outputs, aux = model(patches, next_token_ids=None, progress=step / max(1, args.steps))
            logits_t1 = mtp_outputs[0]  # (B, T_patches, vocab)
            T_p = logits_t1.size(1)
            loss = nn.functional.cross_entropy(
                logits_t1[:, :T_p-1].reshape(-1, logits_t1.size(-1)),
                targets[:, :T_p-1].reshape(-1)
            ) + aux.float()

        if _first_nan_fwd[0]:
            print(f"\n❌ step {step}: FORWARD NaN/Inf at '{_first_nan_fwd[0]}'")
            break
        if torch.isnan(loss) or torch.isinf(loss):
            if _first_nan_fwd[0]:
                print(f"\n❌ step {step}: NaN/Inf loss, first NaN op: {_first_nan_fwd[0]}")
            else:
                print(f"\n❌ step {step}: NaN/Inf loss (no forward hook caught it)")
            break

        loss.backward()

        if _first_nan_grad[0]:
            print(f"\n❌ step {step}: BACKWARD NaN in '{_first_nan_grad[0]}' (loss={loss.item():.4f})")
            if _first_nan_fwd[0]:
                print(f"   First NaN op (fwd): {_first_nan_fwd[0]}")
            _check_sct(f"step{step}")
            gnorms = {}
            for n, p in model.named_parameters():
                if p.grad is not None and not torch.isnan(p.grad).any():
                    gn = p.grad.norm().item()
                    if gn > 0: gnorms[n] = gn
            top = sorted(gnorms.items(), key=lambda x: -x[1])[:10]
            print(f"   Top-10 finite grad norms:")
            for n, g in top:
                print(f"      {n}: {g:.4f}")
            break

        opt.step(model=model)
        retract_all(model)

        _loss_val = loss.item()
        gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None and not torch.isnan(p.grad).any()]
        gnorm = math.sqrt(sum(g**2 for g in gnorms))
        print(f"   step {step:3d}: loss={_loss_val:.4f} grad_norm={gnorm:.4f} "
              f"VRAM={torch.cuda.max_memory_allocated()//1024//1024}MB", flush=True)

    else:
        print(f"\n✅ All {args.steps} steps clean — no NaN detected")
        _check_sct("FINAL")

    torch.cuda.empty_cache()
    return 0

if __name__ == "__main__":
    sys.exit(main())
