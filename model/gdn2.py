"""
💡 busel — pure PyTorch GDN-2 linear attention.
Per-token recurrence from NVlabs/GatedDeltaNet-2 (NVIDIA), no Triton.
"""
import torch
import torch.nn.functional as F


def gdn2_recurrent(q, k, v, g, b, w, alpha_a):
    """GDN-2 token-by-token recurrence (NVIDIA formula).
    S_t = (I - k·(b⊙k)^T) · diag(exp(g)) · S_{t-1} + k · (w⊙v)^T
    o_t = S_t^T · q_t

    Shapes:
        q/k/g/b: [B, T, H, d_k]  — q,k L2-normalised inside
        v/w:     [B, T, H, d_v]
        alpha_a: [H, 1]  — log base decay per head (negative)
    Returns:    [B, T, H, d_v]
    """
    B, T, H, d_k = q.shape
    d_v = v.shape[-1]
    q = F.normalize(q, dim=-1, eps=1e-5)
    k = F.normalize(k, dim=-1, eps=1e-5)
    base_decay = torch.exp(alpha_a.clamp(min=-10.0, max=0.0))
    S = torch.zeros(B, H, d_k, d_v, device=q.device, dtype=torch.float32)
    out = torch.empty(B, T, H, d_v, device=q.device, dtype=torch.float32)
    for t in range(T):
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        bt, wt, gt = b[:, t], w[:, t], g[:, t]
        decay = torch.exp(-base_decay * F.softplus(gt))
        S = S * decay.unsqueeze(-1)
        bk = bt * kt
        v_new = wt * vt - (bk.unsqueeze(-2) @ S).squeeze(-2)
        S = S + kt.unsqueeze(-1) * v_new.unsqueeze(-2)
        out[:, t] = (S.transpose(-2, -1) @ qt.unsqueeze(-1)).squeeze(-1)
    return out.to(v.dtype)
