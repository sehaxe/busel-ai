"""
⚡ Quick NaN scanner — one forward+bw at specific byte offsets using real data.
Usage: uv run python -m tests.scan_nan_batch [--checkpoint PATH]
"""
import sys, math, torch, torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/busel_kruk-210m_step_100.pt")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--chunk", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"⚡ NaN Scanner | ckpt={args.checkpoint} batch={args.batch} chunk={args.chunk}")

    # --- Config ---
    from training.stages.pretrain import buselPretrainConfig
    import yaml
    with open("configs/default.yaml") as fh:
        all_cfg = yaml.safe_load(fh)
    profile = all_cfg.get("profiles", {}).get("kruk-210m", {})
    cfg = buselPretrainConfig.from_profile(profile)

    # --- Model + Patcher + Load ---
    from model.backbone import buselModel
    from model.patching import StridedFastBLTPatcher
    from model.checkpoint import strip_compile_prefix, load_state_dict_safely

    model = buselModel(cfg).to(device)
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
    load_state_dict_safely(model, strip_compile_prefix(sd["model_state_dict"]), strict=False)
    load_state_dict_safely(patcher, strip_compile_prefix(sd.get("patcher_state_dict", {})), strict=False)
    model.train()
    patcher.train()

    print(f"   Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # --- DataLoader from real files ---
    from data.pipeline import get_busel_dataloader
    dl = get_busel_dataloader(
        data_dirs=["data_train"],
        batch_size=args.batch,
        block_size=args.chunk,
        num_workers=0,
        shuffle=False,
    )
    print(f"   Dataloader ready.")

    # --- Forward hooks ---
    _first_nan = [None]
    for n, m in model.named_modules():
        if len(list(m.children())) == 0:
            def _hook_factory(name):
                def hook(mod, inp, out):
                    if _first_nan[0] is not None: return
                    for t in ([inp] if isinstance(inp, torch.Tensor) else []):
                        if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
                            _first_nan[0] = f"FWD IN  {name}"; return
                    if isinstance(out, torch.Tensor) and (torch.isnan(out).any() or torch.isinf(out).any()):
                        _first_nan[0] = f"FWD OUT {name}"
                return hook
            m.register_forward_hook(_hook_factory(n))

    stride = patcher.stride
    max_steps = 200
    for step, batch in enumerate(dl):
        if step >= max_steps: break
        _first_nan[0] = None

        byte_batch = batch.to(device)
        byte_batch = byte_batch[:, :(byte_batch.shape[1] // stride) * stride]
        input_bytes = byte_batch[:, :-stride] if byte_batch.shape[1] > stride else byte_batch
        targets = byte_batch[:, stride::stride]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            patches = patcher(input_bytes)
            mtp_out, aux = model(patches, next_token_ids=None, progress=0.3)
            logits = mtp_out[0]
            T = logits.size(1)
            loss = nn.functional.cross_entropy(
                logits[:, :T-1].reshape(-1, logits.size(-1)),
                targets[:, :T-1].reshape(-1)
            ) + aux.float()

        if _first_nan[0]:
            print(f"\n❌ step {step}: FWD NaN at '{_first_nan[0]}' loss={loss.item() if not torch.isnan(loss) else 'NaN'}")
            break

        loss.backward()

        # Check any grad NaN
        nan_params = []
        for n, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                nan_params.append(n)
                if len(nan_params) == 1:
                    print(f"\n❌ step {step}: first NaN grad='{n}' loss={loss.item():.4f}")
                if len(nan_params) >= 5:
                    break
        if nan_params:
            print(f"   Total {len(nan_params)} params with NaN/Inf grad")
            break

        # Simple SGD step (no optimizer)
        for p in model.parameters():
            if p.grad is not None:
                p.data.sub_(p.grad, alpha=0.0001)
        model.zero_grad()

        if step % 20 == 0:
            print(f"   step {step}: loss={loss.item():.4f} VRAM={torch.cuda.max_memory_allocated()//1024//1024}MB", flush=True)

    else:
        print(f"\n✅ All {max_steps} steps clean")

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
