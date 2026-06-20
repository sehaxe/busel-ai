"""busel optimizer — SF-NorMuon + GramNS + Muon+. Single path, no dead branches."""
import torch
import math
from busel_registry import register

try:
    from gram_newton_schulz.gram_newton_schulz import StandardNewtonSchulz
    _NS = StandardNewtonSchulz(ns_use_kernels=False)
    HAS_GRAM_NS = True
except ImportError:
    _NS = None
    HAS_GRAM_NS = False

# ── Newton-Schulz core ──────────────────────────────────────────────────

_NS_COEFFS = (3.4445, -4.7750, 2.0315)

def _newton_schulz_core(X, steps=5):
    a, b, c = _NS_COEFFS
    scale = X.norm()
    X = X / scale
    for _ in range(steps):
        G = X.mT @ X
        X = a * X + b * X @ G + c * X @ G @ G
    return X

def _compiled_newton_schulz(X, steps=5):
    try: return torch.compile(_newton_schulz_core)(X, steps)
    except Exception: return _newton_schulz_core(X, steps)

# ── Muon base + LOTUS + NorMuon ────────────────────────────────────────

class _MuonBase(torch.optim.Optimizer):
    def __init__(self, params, extra_defaults, lr=1e-3, weight_decay=0.1, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        defaults.update(extra_defaults)
        super().__init__(params, defaults)

    def _has_momentum(self, state): return 'momentum_buffer' in state
    def _init_momentum(self, p, state, group): state['momentum_buffer'] = torch.zeros_like(p)
    def _update_momentum(self, p, state, grad, momentum):
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad.to(buf.dtype))
        return buf
    def _apply_weight_decay(self, p, lr, wd, m_t, group): p.mul_(1.0 - lr * wd)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd, momentum = group['lr'], group['weight_decay'], group['momentum']
            ns_steps = group['ns_steps']
            lr_scale = group.get('lr_scale', 1.0)
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad; state = self.state[p]
                if not self._has_momentum(state): self._init_momentum(p, state, group)
                m_t = self._update_momentum(p, state, grad, momentum)
                self._apply_weight_decay(p, lr, wd, m_t, group)
                if HAS_GRAM_NS and m_t.numel() > 1:
                    O_t = _NS(m_t)
                else:
                    O_t = _compiled_newton_schulz(m_t, steps=ns_steps)
                O_t = O_t / (O_t.norm(dim=0, keepdim=True) + 1e-8)  # Muon+
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))
                p.add_(O_t.to(p.dtype), alpha=-lr * scale * lr_scale)


@register("optimizer", "lotus_muon")
class LotusMuon(_MuonBase):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, ns_steps=5, rank=8, lr_scale=0.5):
        super().__init__(params, {'rank': rank, 'lr_scale': lr_scale}, lr, weight_decay, momentum, ns_steps)
    def _has_momentum(self, state): return 'buf_p' in state
    def _init_momentum(self, p, state, group):
        d1, d2 = p.shape; r = min(group['rank'], d1, d2)
        state['buf_p'] = torch.zeros(d1, r, dtype=torch.bfloat16, device=p.device)
        state['buf_q'] = torch.zeros(d2, r, dtype=torch.bfloat16, device=p.device)
    def _update_momentum(self, p, state, grad, momentum):
        bp, bq = state['buf_p'], state['buf_q']
        bp.mul_(momentum).add_(grad.to(bp.dtype) @ bq)
        bq.mul_(momentum).add_(grad.to(bq.dtype).T @ bp)
        return bp @ bq.T


@register("optimizer", "norlotus_muon")
class NorLotusMuon(LotusMuon):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, ns_steps=5, rank=8, lr_scale=0.5, cautious_wd=True):
        super().__init__(params, lr, weight_decay, momentum, ns_steps, rank, lr_scale)
        for g in self.param_groups: g['cautious_wd'] = cautious_wd
    def _apply_weight_decay(self, p, lr, wd, m_t, group):
        if group.get('cautious_wd', True):
            wd_mask = (m_t * p.data > 0).to(m_t.dtype)
            p.mul_(1.0 - lr * wd * wd_mask)
        else: p.mul_(1.0 - lr * wd)

# ── Schedule-Free wrapper ───────────────────────────────────────────────

class _ScheduleFreeWrapper:
    def __init__(self, base_optimizer, beta=0.9, gamma_factor=2.0):
        self.base_optimizer = base_optimizer; self.beta = beta; self.gamma_factor = gamma_factor; self._state = {}
    @property
    def param_groups(self): return self.base_optimizer.param_groups
    def zero_grad(self, set_to_none=True): self.base_optimizer.zero_grad(set_to_none=set_to_none)
    @torch.no_grad()
    def step(self):
        params = [p for g in self.base_optimizer.param_groups for p in g['params']]
        for g in self.base_optimizer.param_groups: g['lr'] = g['lr'] * self.gamma_factor
        for p in params:
            if p.grad is None: continue
            s = self._state.get(p)
            if s is None: self._state[p] = {'x': p.data.clone(), 'z': p.data.clone(), 't': 0}
            p.data.copy_(self._state[p]['z'])
        self.base_optimizer.step()
        for g in self.base_optimizer.param_groups: g['lr'] = g['lr'] / self.gamma_factor
        for p in params:
            if p.grad is None: continue
            s = self._state[p]
            z_new = p.data.clone(); t_new = s['t'] + 1
            x_new = (1.0 - 1.0 / t_new) * s['x'] + (1.0 / t_new) * z_new
            y_new = (1.0 - self.beta) * x_new + self.beta * z_new
            s['z'] = z_new; s['x'] = x_new; s['t'] = t_new
            p.data.copy_(y_new)
    def state_dict(self): return {'base': self.base_optimizer.state_dict(), 'sf': self._state, 'beta': self.beta, 'gamma_factor': self.gamma_factor}
    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd['base']); self._state = sd['sf']; self.beta = sd['beta']; self.gamma_factor = sd['gamma_factor']

# ── Optimizer Engine (hybrid Muon+AdamW) ────────────────────────────────

@register("optimizer", "hybrid_muon_adamw")
class buselOptimizerEngine:
    _MUON_EXCLUDE = ("router", "embed")
    _LR_GROUPS = ("attn", "ffn", "mtp", "norm", "embed", "router")

    @staticmethod
    def _classify_param(name):
        n = name.lower()
        if "router" in n: return "router"
        if "embed" in n: return "embed"
        if "norm" in n: return "norm"
        if "mtp" in n: return "mtp"
        if "ffn" in n or "blackboard" in n: return "ffn"
        if any(t in n for t in ("q_proj","k_proj","v_proj","o_proj","qkv","wk","wv","wq")): return "attn"
        if "moe" in n: return "ffn"
        return "attn"

    def __init__(self, *modules, lr_muon=0.002, lr_adamw=0.0002, lotus_rank=8, sf_beta=0.9, sf_gamma_factor=2.0):
        muon_params, adamw_params = [], []
        for module in modules:
            for name, param in module.named_parameters():
                if not param.requires_grad: continue
                if param.ndim == 2 and all(t not in name for t in self._MUON_EXCLUDE):
                    muon_params.append((name, param))
                else: adamw_params.append((name, param))

        mults = {k: 1.0 for k in self._LR_GROUPS}
        mults["embed"] = 0.5; mults["router"] = 0.5

        def _build_groups(items):
            groups = {k: [] for k in self._LR_GROUPS}
            for name, p in items: groups[self._classify_param(name)].append(p)
            return [{"params": v, "lr_mult": mults[k], "name": k} for k, v in groups.items() if v]

        print(f"🪷 [SF-NorMuon]: NorMuon + LOTUS rank={lotus_rank} + Schedule-Free")
        self.opt_muon = _ScheduleFreeWrapper(
            NorLotusMuon(_build_groups(muon_params), lr=lr_muon, momentum=0.95, rank=lotus_rank, cautious_wd=True),
            beta=sf_beta, gamma_factor=sf_gamma_factor
        )
        
        # FP8 AdamW: always ON (Ampere+ native). 75% memory reduction.
        from torchao.optim import AdamWFp8
        adamw_opt = AdamWFp8(_build_groups(adamw_params), lr=lr_adamw, weight_decay=0.01)
        print("🧊 [FP8-AdamW]: torchao FP8 optimizer — 75% memory")
        
        self.opt_adamw = _ScheduleFreeWrapper(adamw_opt, beta=sf_beta, gamma_factor=sf_gamma_factor)

        muon_count = sum(p.numel() for g in self.opt_muon.base_optimizer.param_groups for p in g['params'])
        adamw_count = sum(p.numel() for g in self.opt_adamw.base_optimizer.param_groups for p in g['params'])
        total = muon_count + adamw_count
        print(f"⚙️  Hybrid optimiser routing: {muon_count:,} → sf_normuon ({100*muon_count/total:.1f}%), {adamw_count:,} → AdamW ({100*adamw_count/total:.1f}%)")

    def zero_grad(self, set_to_none=True):
        self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt_muon.step()
        self.opt_adamw.step()

# ── EMA ──────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay; self._step_count = 0
        self.shadow = {k: (v.detach().clone().float() if v.dtype.is_floating_point else v.detach().clone()) for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self._step_count += 1
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd = model.state_dict()
        for k in sd:
            if sd[k].dtype.is_floating_point: sd[k].copy_(self.shadow[k].to(sd[k].dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        sd = model.state_dict()
        for k in sd: sd[k].copy_(backup[k].to(sd[k].dtype))
