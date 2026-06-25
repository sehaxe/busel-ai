"""
⚡ From-scratch NaN reproducer — trains 200 steps with random data.
Tells us if NaN is training-dynamics or data-dependent.

Usage: uv run python -m tests.repro_nan
"""
import sys, math, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="kruk-210m")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"⚡ NaN Reproducer | profile={args.profile} batch={args.batch} chunk={args.chunk} steps={args.steps}")

    # --- Config ---
    from training.stages.pretrain import buselPretrainConfig
    import yaml
    with open("configs/default.yaml") as fh:
        all_cfg = yaml.safe_load(fh)
    profile = all_cfg.get("profiles", {}).get(args.profile, {})
    cfg = buselPretrainConfig.from_profile(profile)

    # --- Model + Patcher ---
    from model.backbone import buselModel
    from model.patching import StridedFastBLTPatcher
    model = buselModel(cfg).to(device)
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model.train()
    patcher.train()

    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer (same as real training) ---
    from training.optimizer import buselOptimizerEngine
    opt = buselOptimizerEngine(
        model, lr_muon=args.lr, lr_adamw=args.lr * 0.1,
        lotus_rank=getattr(cfg, "sct_rank", 8)
    )

    # --- Data (random, same shape as real training) ---
    vocab = getattr(cfg, "vocab_size", 326)
    stride = patcher.stride
    chunk = ((args.chunk + stride) // stride) * stride
    data = torch.randint(0, min(vocab, 256), (args.batch, chunk), device=device)

    # --- Hooks: per-param NaN grad ---
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

    # --- Forward hooks: first NaN in activations ---
    _first_nan_fwd = [None]
    for n, m in model.named_modules():
        if len(list(m.children())) == 0:
            def _fwd_hook_factory(name):
                def hook(mod, inp, out):
                    if _first_nan_fwd[0] is not None: return
                    if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
                        _first_nan_fwd[0] = f"FWD {name}"
                return hook
            m.register_forward_hook(_fwd_hook_factory(n))

    for step in range(args.steps):
        opt.zero_grad()
        _first_nan_fwd[0] = None
        _first_nan_grad[0] = None

        input_bytes = data[:, :-stride] if data.shape[1] > stride else data
        targets = data[:, stride::stride]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches = patcher(input_bytes)
            mtp_out, aux = model(patches, next_token_ids=None, progress=step / args.steps)
            logits = mtp_out[0]
            T = logits.size(1)
            loss = nn.functional.cross_entropy(
                logits[:, :T-1].reshape(-1, logits.size(-1)),
                targets[:, :T-1].reshape(-1)
            ) + aux.float()

        if _first_nan_fwd[0]:
            print(f"\n❌ step {step}: FORWARD NaN at '{_first_nan_fwd[0]}'")
            break
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n❌ step {step}: NaN/Inf loss")
            break

        loss.backward()

        if _first_nan_grad[0]:
            grad_norm_val = total_norm_function(model)
            print(f"\n❌ step {step}: BACKWARD NaN in '{_first_nan_grad[0]}' loss={loss.item():.4f} grad_norm={grad_norm_val:.4f}")
            break

        opt.step(model=model)

        if step % 20 == 0 or step < 10:
            grad_norm_val = total_norm_function(model)
            print(f"   step {step:3d}: loss={loss.item():.4f} grad_norm={grad_norm_val:.4f} VRAM={torch.cuda.max_memory_allocated()//1024//1024}MB", flush=True)

    else:
        print(f"\n✅ All {args.steps} steps clean")

    # SCT check
    s_vals = []
    for m in model.modules():
        if m.__class__.__name__ == "SpectralLinear":
            s_vals.append(m.s.data.float().cpu())
    if s_vals:
        all_s = torch.cat(s_vals)
        print(f"   SCT s: min={all_s.min():.4f} mean={all_s.abs().mean():.4f} max={all_s.max():.4f} >5={(all_s.abs()>5).sum().item()}")

    torch.cuda.empty_cache()
    return 0

def total_norm_function(model):
    gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None and not torch.isnan(p.grad).any()]
    return math.sqrt(sum(g**2 for g in gnorms)) if gnorms else 0.0

if __name__ == "__main__":
    sys.exit(main())
