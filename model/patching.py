"""
FastBLT: Byte-Level Tokenizer (безтокенный ввод с причинной сверткой и гейтированием)
Оптимизирован для снижения орфографической нагрузки на слои внимания.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import RMSNorm, nvtx_range_push, nvtx_range_pop


class StridedFastBLTPatcher(nn.Module):
    def __init__(self, d_model=768, d_byte=128, stride=4, kernel_size=5):
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        self.d_byte = d_byte
        self.kernel_size = kernel_size
        
        # Веса эмбеддингов для байтового словаря (256 байт + спецсимволы)
        self.embed_weight = nn.Parameter(torch.randn(259, d_byte) * 0.02)
        
        # 🎯 СИСТЕМНАЯ ОПТИМИЗАЦИЯ GATED BLT:
        # Маленький полносвязный гейт перед сверткой.
        # Позволяет модели на уровне патчера «отфильтровывать» шумные байты (избыточный синтаксис, пробелы)
        # и выделять семантически важные символы, снижая "налог на правописание" (spelling tax).
        self.gate_proj = nn.Linear(d_byte, d_byte)
        
        # Conv1d с нулевым паддингом (паддинг рассчитывается вручную в forward)
        self.conv = nn.Conv1d(
            d_byte, d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=0
        )
        
        self.norm = RMSNorm(d_model)

    def forward(self, byte_ids):
        nvtx_range_push("Bysel_Byte_Patching_Forward")
        
        # Перенос на целевое устройство весов
        byte_ids_device = byte_ids.to(self.embed_weight.device)
        
        # Безопасный F.embedding для предотвращения багов индексации на MPS/CUDA
        x = F.embedding(byte_ids_device, self.embed_weight)
        
        # 🎯 ПРИМЕНЕНИЕ СИГМОИДАЛЬНОГО ГЕЙТА ФИЛЬТРАЦИИ:
        # Автокаст автоматически переведет вычисления гейта в BF16/FP16 на вашей RTX 5060 Ti
        gate = torch.sigmoid(self.gate_proj(x))
        x = x * gate
        
        # [B, T, d_byte] -> [B, d_byte, T] для Conv1d
        x = x.transpose(1, 2)
        
        # Строго причинный паддинг (Causal Left-Side Padding):
        # Добавляем (kernel_size - 1) нулевых значений исключительно слева.
        x_padded = F.pad(x, (self.kernel_size - 1, 0))
        
        # Свертка и приведение к исходному формату
        patches = self.conv(x_padded)
        patches = patches.transpose(1, 2)
        out = self.norm(patches)
        
        nvtx_range_pop()
        return out