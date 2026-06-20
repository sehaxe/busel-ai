"""ByteFlow patcher — coding-rate boundary detection. Variable-length patches."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import RMSNorm, nvtx_range_push, nvtx_range_pop
from multimodal.special_tokens import vocab_size as _vocab_size

class StridedFastBLTPatcher(nn.Module):
    def __init__(self, d_model=768, d_byte=128):
        super().__init__()
        self.d_model = d_model
        self.d_byte = d_byte
        self.stride = 4  # kept for MTP target alignment
        self.kernel_size = 5
        self.n_patches = 16  # K = stride * 4, keep same output dim

        self.vocab_size = _vocab_size()
        self.embed_weight = nn.Parameter(torch.randn(self.vocab_size, d_byte) * 0.02)
        
        self.gate_proj_down = nn.Linear(d_byte, max(1, d_byte // 4))
        self.gate_proj_up = nn.Linear(max(1, d_byte // 4), d_byte)
        
        self.boundary_conv = nn.Conv1d(d_byte, 1, kernel_size=3, padding=1)
        self.patch_pool = nn.AdaptiveAvgPool1d(self.n_patches)
        self.conv = nn.Conv1d(d_byte, d_model, kernel_size=5, stride=1)
        self.norm = RMSNorm(d_model)

    def forward(self, byte_ids, return_embedding=False):
        nvtx_range_push("busel_Byte_Patching_Forward")
        x = F.embedding(byte_ids.to(self.embed_weight.device), self.embed_weight)

        embed_for_dispersion = x if return_embedding else None
        gate = torch.sigmoid(self.gate_proj_up(F.silu(self.gate_proj_down(x))))
        x = x * gate
        
        x_t = x.transpose(1, 2)
        scores = torch.sigmoid(self.boundary_conv(x_t))
        x_weighted = x_t * scores
        x_pooled = self.patch_pool(x_weighted)
        x_padded = F.pad(x_pooled, (4, 0))
        patches = self.conv(x_padded).transpose(1, 2)
        
        out = self.norm(patches)
        nvtx_range_pop()
        return (out, embed_for_dispersion) if return_embedding else out
