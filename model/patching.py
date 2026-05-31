"""
FastBLT: Byte-Level Tokenizer (безтокенный ввод)
"""

import torch
import torch.nn as nn
from model.layers import RMSNorm, nvtx_range_push, nvtx_range_pop


class StridedFastBLTPatcher(nn.Module):
    def __init__(self, d_model=768, d_byte=128, stride=4, kernel_size=5):
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        self.d_byte = d_byte
        
        # Веса эмбеддингов
        self.embed_weight = nn.Parameter(torch.randn(259, d_byte) * 0.02)
        
        # Conv1d
        self.conv = nn.Conv1d(
            d_byte, d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        
        self.norm = RMSNorm(d_model)

    def forward(self, byte_ids):
        nvtx_range_push("Bysel_Byte_Patching_Forward")
        
        # Перенос на целевое устройство весов
        byte_ids_device = byte_ids.to(self.embed_weight.device)
        
        # 🎯 СВЕРХНАДЕЖНЫЙ ФИКС: F.embedding для обхода багов Advanced Indexing на Mac
        x = torch.nn.functional.embedding(byte_ids_device, self.embed_weight)
        
        # [B, T, d_byte] -> [B, d_byte, T] для Conv1d
        x = x.transpose(1, 2)
        
        # Свертка и нормализация
        patches = self.conv(x)
        patches = patches.transpose(1, 2)
        out = self.norm(patches)
        
        nvtx_range_pop()
        return out