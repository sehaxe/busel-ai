"""ponytail: catch Muon NaN with hook at step level."""
import torch, sys, yaml, math, os
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # no compile — speed up debug
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/inductor_debug"
torch.backends.cudnn.benchmark = True

with open("configs/default.yaml") as f:
    profiles = list(yaml.safe_load_all(f))
cfg_dict = profiles[0]["profiles"]["kruk-210m"]

from training.stages.pretrain_config import buselPretrainConfig
cfg = buselPretrainConfig.from_profile(cfg_dict)
cfg.max_steps = 200
cfg.grad_accum_steps = 1
cfg.chunk_size = min(cfg.chunk_size, 512)

from model.backbone import buselModel
from model.patching import StridedFastBLTPatcher

patcher = StridedFastBLTPatcher(d_model=cfg.d_model).cuda().train()
model = buselModel(cfg).cuda().train()
model = model.to(torch.bfloat16)

batch_size = cfg.batch_size // 4  # reduce VRAM for debug
seq_len = cfg.chunk_size if cfg.chunk_size >= 257 else 512  # patcher stride=64 needs min seq

def make_batch():
    d = torch.randint(0, 256, (batch_size, seq_len), device="cuda")
    if cfg.use_ascii_curriculum:
        d = d.clamp(max=127)
    return d

dummy = make_batch()

from training.optimizer import buselOptimizerEngine
opt = buselOptimizerEngine(model, lr_muon=cfg.learning_rate_muon, lr_adamw=cfg.learning_rate_adamw,
                           lotus_rank=cfg.lotus_rank, lr_multipliers=cfg.lr_multipliers)
from training.autopilot import buselAutoPilot
ap = buselAutoPilot(opt, cfg.learning_rate_muon, cfg.learning_rate_adamw, target_wd=0.1,
                    warmup_steps=cfg.warmup_steps if isinstance(cfg.warmup_steps, int) else cfg.max_steps // 20,
                    min_lr_ratio=cfg.min_lr_ratio, lr_schedule=cfg.lr_schedule)

# ── Patch Muon step to catch buffer NaN ──
_muon_step_orig = opt.opt_muon.base_optimizer.step

def _patched_step():
    lotus = opt.opt_muon.base_optimizer
    for group in lotus.param_groups:
        for p in group['params']:
            s = lotus.state.get(p)
            if s and 'buf_p' in s:
                bpn = s['buf_p'].norm().item()
                bqn = s['buf_q'].norm().item()
                if math.isnan(bpn) or math.isnan(bqn):
                    print(f"  ⚡ Lotus buffer NaN in {p.shape}! bp={bpn:.1e} bq={bqn:.1e}")
                    raise RuntimeError("buffer NaN")
                if bpn > 1e6 or bqn > 1e6:
                    print(f"  ⚡ Lotus buffer EXPLODED {p.shape}: bp={bpn:.1e} bq={bqn:.1e}")
    _muon_step_orig()

opt.opt_muon.base_optimizer.step = _patched_step

# Track first 3 Muon dense params for norm monitoring
track_params = {}
for name, p in model.named_parameters():
    if p.ndim == 2 and p.requires_grad and not any(x in name.lower() for x in ('u', 'v', 's', 'router', 'embed')):
        track_params[name] = p
        if len(track_params) >= 3: break

print(f"Model: {sum(p.numel() for p in model.parameters()):,}")
print(f"Muon:  {sum(p.numel() for g in opt.opt_muon.base_optimizer.param_groups for p in g['params']):,}")
print(f"AdamW: {sum(p.numel() for g in opt.opt_adamw.base_optimizer.param_groups for p in g['params']):,}")
print(f"Tracking: {list(track_params.keys())}")

# ── Build targets inline ──

for step in range(cfg.max_steps):
    opt.zero_grad()

    try:
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            input_bytes = dummy[:, :-patcher.stride] if dummy.shape[1] > patcher.stride else dummy
            patches = patcher(input_bytes)
            T_p = patches.shape[1]
            shift = patcher.stride
            targets = dummy[:, shift:shift + T_p*shift:shift][:, :T_p].contiguous()
            mtp_targets = []
            for i in range(1, cfg.num_mtp_heads):
                t = dummy[:, shift + i:shift + i + T_p*shift:shift][:, :T_p].contiguous() if shift + i + T_p*shift <= seq_len else None
                mtp_targets.append(t)
            mtp_logits, aux = model(patches, [targets] + [t for t in mtp_targets[:-1] if t is not None],
                                    progress=step/cfg.max_steps)
            from torch.nn.functional import cross_entropy
            loss = cross_entropy(mtp_logits[0].reshape(-1, cfg.vocab_size), targets.reshape(-1))
            for i, logits in enumerate(mtp_logits[1:]):
                if logits is not None and i < len(mtp_targets) and mtp_targets[i] is not None:
                    w = 0.5 if i == 0 else 0.25
                    loss = loss + w * cross_entropy(logits.reshape(-1, cfg.vocab_size), mtp_targets[i].reshape(-1))
    except Exception as e:
        print(f"  ❌ Forward crash step {step}: {type(e).__name__}: {e}")
        break

    if torch.isnan(loss):
        print(f"  ⚡ FWD NaN at step {step}")
        break

    loss.backward()

    nan_params = [name for name, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    if nan_params:
        print(f"  ⚡ GRAD NaN step {step}: {nan_params[:10]}")
        for n in nan_params[:5]:
            p = dict(model.named_parameters())[n]
            print(f"     {n}: data_norm={p.data.norm().item():.2e}")
        break

    ap.before_step(model, step, cfg.max_steps)
    ap.update_parameters(step, loss.item(), cfg.max_steps)
    opt.step(model=model)

    if step % 10 == 0:
        gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None) ** 0.5
        pnorms = {n: f"{p.data.norm().item():.2f}" for n, p in track_params.items()}
        print(f"  {step:4d}  loss {loss.item():6.3f}  gnorm {gnorm:.1e}  {pnorms}")

    dummy = make_batch()

print("Done.")
