"""
🔍 LR Finder v2 — binary search max stable LR for busel scaling laws.
All technologies ON (SCT, MoD, MoE, GDN-2, MTP-6, DropBP, SF-NorLotusMuon).
Usage: uv run python tests/find_lr.py --profile kruk-210m --steps 100
"""
import sys, math, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

def _build_mtp_targets(data, T_patches, stride, num_mtp_heads):
    """Match pretrain stage: stride-based target windows."""
    targets = data[:, stride:stride + T_patches * stride:stride][:, :T_patches].contiguous()
    mtp_targets = []
    for i in range(1, num_mtp_heads):
        t = data[:, stride + i:stride + i + T_patches * stride:stride][:, :T_patches].contiguous()
        mtp_targets.append(t)
    return targets, mtp_targets

def try_lr(lr, cfg, args):
    device = torch.device("cuda")
    from model.backbone import buselModel
    from model.patching import StridedFastBLTPatcher
    from training.optimizer import buselOptimizerEngine

    model = buselModel(cfg).to(device)
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model.train(); patcher.train()

    lotus_rank = getattr(cfg, "sct_rank", 128)
    opt = buselOptimizerEngine(model, lr_muon=lr, lr_adamw=lr*0.1,
                               lotus_rank=lotus_rank,
                               lr_multipliers=cfg.lr_multipliers if hasattr(cfg, 'lr_multipliers') else None,
                               sf_beta=0.9, sf_gamma_factor=0.5)

    vocab = getattr(cfg, "vocab_size", 326)
    stride = patcher.stride
    chunk = (((args.chunk // stride) + 1) * stride) + args.num_mtp_heads  # MTP headroom
    data = torch.randint(0, min(vocab, 256), (args.batch, chunk), device=device)

    mtp_weights = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    
    for step in range(args.steps):
        opt.zero_grad()
        
        for _acc in range(args.grad_accum):
            input_bytes = data[:, :-stride] if data.shape[1] > stride else data
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                patches = patcher(input_bytes)
                T_p = patches.shape[1]
                targets, mtp_targets = _build_mtp_targets(
                    data, T_p, stride, args.num_mtp_heads
                )
                mtp_logits, aux_loss = model(
                    patches, [targets] + [t for t in mtp_targets[:-1]],
                    progress=step/args.steps
                )
                
                loss = nn.functional.cross_entropy(
                    mtp_logits[0].reshape(-1, vocab), targets.reshape(-1)
                )
                for i, logits in enumerate(mtp_logits[1:]):
                    if logits is not None and i < len(mtp_targets) and i < len(mtp_weights):
                        loss = loss + mtp_weights[i] * nn.functional.cross_entropy(
                            logits.reshape(-1, vocab), mtp_targets[i].reshape(-1)
                        )
                loss = loss + aux_loss.float()
                loss = loss / args.grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                return float('nan'), step + _acc/args.grad_accum, "forward NaN"

            loss.backward()
            
            if step > 0:
                for n, p in model.named_parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        return float('nan'), step + _acc/args.grad_accum, f"grad NaN: {n}"
        
        opt.step(model=model)
        
        if step == args.steps - 1:
            gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            gnorm = math.sqrt(sum(g**2 for g in gnorms))
            if gnorm > 10 and step > 50:
                return loss.item() * args.grad_accum, step, f"grad_norm={gnorm:.1f} (diverging)"

    final_loss = loss.item() * args.grad_accum  # undo /grad_accum for true loss
    torch.cuda.empty_cache()
    return final_loss, args.steps, "stable"

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="kruk-210m")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=2, help="gradient accumulation micro-batches")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr-low", type=float, default=0.0005)
    p.add_argument("--lr-high", type=float, default=0.02)
    p.add_argument("--sweep", type=int, default=10, help="number of LR points to sweep")
    args = p.parse_args()

    import yaml
    with open("configs/default.yaml") as fh:
        all_cfg = yaml.safe_load(fh)
    profile = all_cfg.get("profiles", {}).get(args.profile, {})
    from training.stages.pretrain_config import buselPretrainConfig
    cfg = buselPretrainConfig.from_profile(profile)
    cfg.debug = False
    cfg.grad_accum_steps = args.grad_accum
    
    args.num_mtp_heads = getattr(cfg, "num_mtp_heads", 6)

    print(f"🔍 LR Finder v2 | {args.profile} | batch={args.batch}×{args.grad_accum} chunk={args.chunk} steps={args.steps}")
    print(f"   MTP-{args.num_mtp_heads} | SCT-{cfg.sct_rank} | MoD={cfg.mod_capacity} | MoE top-{cfg.top_k}")
    print(f"{'LR(muon)':>12s}  {'Loss':>10s}  {'Step':>6s}  {'Status':>8s}  Detail")

    lo, hi = args.lr_low, args.lr_high
    best_lr, best_loss = lo, float('inf')
    
    # logarithmically-spaced sweep
    lrs = [lo * (hi/lo) ** (i/(args.sweep-1)) for i in range(args.sweep)]
    for lr in lrs:
        lr = round(lr, 8)
        loss_val, step_died, detail = try_lr(lr, cfg, args)
        status = "✅" if detail == "stable" else "💥" if math.isnan(loss_val) else "⚠️"
        print(f"  {lr:10.6f}  {loss_val:10.4f}  {step_died:6.1f}  {status:>8}  {detail}")
        if detail == "stable":
            best_lr = max(best_lr, lr)
            best_loss = min(best_loss, loss_val)
        torch.cuda.empty_cache()

    print(f"\n🏆 Max stable LR (Muon): {best_lr:.6f}")
    print(f"   Final loss at max LR:  {best_loss:.4f}")
    print(f"   Recommended (80% margin): lr_muon = {best_lr * 0.8:.6f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
