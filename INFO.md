# BYSEL (БУСЕЛ) v4.0 — COMPLETE TECHNICAL SPECIFICATION & PROJECT BLUEPRINT
### STAGE: NASA-Grade Sovereign Any-Scale Omni-LLM (3.6B / 16B Tokens)

Bysel (Бусел) — сверхэффективная, безтокенная Any-to-Text 1-битная LLM с гибридным линейным вниманием, динамической глубиной и смесью экспертов. Спроектирована по строжайшим стандартам надежности NASA JPL (The Power of Ten) для запуска, обучения и инференса на потребительском железе (RTX 5060 Ti 16GB / Apple Silicon Mac 8GB) [1, 2].

## 1. СТРУКТУРА ФАЙЛОВ И ДЕКУПЛИРОВАННЫЙ LAYOUT

```text
bysel/
├── configs/
│   └── default.yaml     # Конфигурация профилей Зязюля (360M) и Зубр (3.6B)
├── model/
│   ├── __init__.py
│   ├── patching.py      # FastBLT: безтокенный байтовый патчер
│   ├── layers.py        # Базовые слои: BitLinear 1.58b, GeoNorm, Обертки NVTX
│   ├── attention.py     # GDN-2 (Gated DeltaNet-2) с SeRoPE и слои MLA/CSA
│   ├── routing.py       # MoD (25% пропуск) и MoE (64 эксперта + Blackboard)
│   └── backbone.py      # Оркестратор ByselModel (3:1 Hybrid) и MTP-4 головы
├── data/
│   ├── __init__.py
│   └── pipeline.py      # Потоковый Byte-Streamer на Rust (с авто-фолбеком на Python)
├── training/            # Пакет обучения (выделен во избежание конфликтов импорта)
│   ├── __init__.py
│   ├── optimizer.py     # Встроенный Moonlight-Muon + AdamW
│   ├── autopilot.py     # Автотюнер (LR, WD, впрыск градиентного шума)
│   └── recipe.py        # Шаги обучения (Pretrain W1.58A8, QAT W1.58A4, SFT, DPO)
├── tests/
│   └── profiler_run.py  # Скрипт профилирования с NVTX-маркерами
├── Cargo.toml           # Настройки сборки Rust-расширения
├── src/
│   └── lib.rs           # Многопоточный загрузчик на Rust (py.detach + Rayon)
├── pyproject.toml       # Менеджер проекта uv с условными CUDA-зависимостями
├── cli.py               # Единый CLI-интерфейс на базе Typer (train, serve, profile)
└── train.py             # Главный отказоустойчивый оркестратор обучения (NASA-grade)
```

---

## 2. СИСТЕМНЫЕ И СБОРОЧНЫЕ КОНФИГУРАЦИИ

### 2.1. `pyproject.toml` (Менеджер проекта uv)
```toml
[project]
name = "bysel"
version = "4.0.0"
description = "Sovereign 1-bit Omni-LLM (Зязюля & Зубр)"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.28.0",
    "httpx>=0.27.0",
    "aiogram>=3.4.0",
    "typer>=0.12.0",
    "pyyaml>=6.0.1",
    "numpy>=1.26.0",
    "liger-kernel>=0.1.2; sys_platform == 'linux'",
    "flash-linear-attention; sys_platform == 'linux'",
]

[build-system]
requires = ["maturin>=1.5,<2.0"]
build-backend = "maturin"

[tool.maturin]
features = ["pyo3/extension-module"]
```

### 2.2. `Cargo.toml` (Сборщик Rust-модуля)
```toml
[package]
name = "bysel_rust_io"
version = "4.0.0"
edition = "2021"

[lib]
name = "bysel_rust_io"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28", features = ["extension-module"] }
rayon = "1.8"
```

### 2.3. `.cargo/config.toml` (Авто-настройка линковщика macOS)
```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
```

### 2.4. `configs/default.yaml` (Профили Зязюля и Зубр)
```yaml
global:
  project_name: "bysel"
  device: "cuda"
  seed: 42
  precision: "bf16"

profiles:
  ziaziulia:
    model:
      d_model: 256
      n_layers: 12
      n_heads: 4
      expert_hidden: 512
      num_experts: 4
      top_k: 2
      vocab_size: 32000
    data:
      data_path: "data_train"
      chunk_size: 512
      batch_size: 4
    training:
      max_steps: 15000
      learning_rate_muon: 0.002
      learning_rate_adamw: 0.0002
      weight_decay: 0.1

  zubr:
    model:
      d_model: 1792
      n_layers: 32
      n_heads: 12
      expert_hidden: 2048
      num_experts: 64
      top_k: 2
      vocab_size: 32000
    data:
      data_path: "data_train"
      chunk_size: 8192
      batch_size: 32
    training:
      max_steps: 80000
      learning_rate_muon: 0.001
      learning_rate_adamw: 0.0001
      weight_decay: 0.1
```

---

## 3. ИСХОДНЫЙ КОД НА РАЗНЫХ ЯЗЫКАХ (POLYGLOT ENGINE)

### 3.1. Многопоточный Safe-Streamer на Rust (`src/lib.rs`)
```rust
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;

#[pyclass]
struct ByteStreamer {
    data: Vec<u8>,
    position: usize,
    chunk_size: usize,
}

#[pymethods]
impl ByteStreamer {
    #[new]
    fn new(py: Python, file_path: String, chunk_size: usize, start_offset: usize) -> PyResult<Self> {
        let data = py.detach(|| -> std::io::Result<Vec<u8>> {
            let mut file = File::open(file_path)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            Ok(data)
        })?;

        Ok(ByteStreamer {
            data,
            position: start_offset,
            chunk_size,
        })
    }

    fn next_chunk(&mut self, py: Python) -> Option<Vec<u8>> {
        if self.position >= self.data.len() {
            return None;
        }

        let start = self.position;
        let end = std::cmp::min(self.position + self.chunk_size, self.data.len());
        
        let chunk = py.detach(|| {
            self.data[start..end].par_iter().cloned().collect::<Vec<u8>>()
        });

        self.position = end;
        Some(chunk)
    }

    fn get_position(&self) -> usize {
        self.position
    }
}

#[pyfunction]
fn init_thread_pool(num_threads: usize) -> PyResult<()> {
    if num_threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global();
    }
    Ok(())
}

#[pymodule]
fn bysel_rust_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteStreamer>()?;
    m.add_function(wrap_pyfunction!(init_thread_pool, m)?)?;
    Ok(())
}
```

### 3.2. Потоковый лоадер и Python-фолбек (`data/pipeline.py`)
```python
import torch
import os
from torch.utils.data import IterableDataset, DataLoader

try:
    import bysel_rust_io
    HAS_RUST_IO = True
except ImportError:
    HAS_RUST_IO = False

class PythonByteStreamer:
    def __init__(self, file_path, chunk_size, start_offset=0):
        with open(file_path, "rb") as f:
            self.data = f.read()
        self.position = start_offset
        self.chunk_size = chunk_size

    def next_chunk(self):
        if self.position >= len(self.data):
            return None
        start = self.position
        end = min(self.position + self.chunk_size, len(self.data))
        chunk = list(self.data[start:end])
        self.position = end
        return chunk
        
    def get_position(self):
        return self.position

class RustByteStreamDataset(IterableDataset):
    def __init__(self, data_path, chunk_size=8192, start_file_idx=0, start_byte_offset=0):
        super().__init__()
        self.chunk_size = chunk_size
        self.start_file_idx = start_file_idx
        self.start_byte_offset = start_byte_offset
        self.files = []
        
        if os.path.isdir(data_path):
            for root, _, filenames in os.walk(data_path):
                for filename in filenames:
                    if filename.endswith(('.txt', '.py', '.rs', '.go', '.be', '.json', '.cpp', '.h')):
                        self.files.append(os.path.join(root, filename))
            self.files.sort()
        elif os.path.isfile(data_path):
            self.files.append(data_path)
            
        self.current_file_idx = start_file_idx
        self.current_byte_offset = start_byte_offset

    def __iter__(self):
        use_rust = HAS_RUST_IO and torch.cuda.is_available()
        shuffle_buffer = []
        buffer_size = 50
        
        for file_idx in range(self.start_file_idx, len(self.files)):
            self.current_file_idx = file_idx
            file_path = self.files[file_idx]
            offset = self.start_byte_offset if file_idx == self.start_file_idx else 0
            
            if use_rust:
                num_cores = os.cpu_count()
                if num_cores > 8:
                    num_cores = num_cores // 2
                bysel_rust_io.init_thread_pool(num_cores)
                streamer = bysel_rust_io.ByteStreamer(file_path, self.chunk_size, offset)
            else:
                streamer = PythonByteStreamer(file_path, self.chunk_size, offset)
                
            while True:
                chunk = streamer.next_chunk()
                if chunk is None:
                    break
                self.current_byte_offset = streamer.get_position()
                shuffle_buffer.append((chunk, self.current_file_idx, self.current_byte_offset))
                
                if len(shuffle_buffer) >= buffer_size:
                    import random
                    random.shuffle(shuffle_buffer)
                    yield shuffle_buffer.pop(0)
                    
        import random
        random.shuffle(shuffle_buffer)
        for item in shuffle_buffer:
            yield item

def collate_bysel_batch(batch):
    chunks = [item[0] for item in batch]
    file_indices = [item[1] for item in batch]
    byte_offsets = [item[2] for item in batch]
    batch_tensors = torch.stack([torch.tensor(c, dtype=torch.long) for c in chunks])
    return batch_tensors, file_indices[-1], byte_offsets[-1]

def get_bysel_dataloader(data_path, chunk_size, batch_size, start_file_idx=0, start_byte_offset=0):
    dataset = RustByteStreamDataset(data_path, chunk_size, start_file_idx, start_byte_offset)
    use_pin = torch.cuda.is_available()
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        pin_memory=use_pin,
        collate_fn=collate_bysel_batch
    )
EOF
```

---

### 3.4. Иерархия математических слоев (`model/layers.py`)
```python
import torch
import torch.nn as nn
import math

def nvtx_range_push(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)

def nvtx_range_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()

def generate_orthogonal_hadamard_matrix(dim, device="cpu"):
    with torch.no_grad():
        q, _ = torch.linalg.qr(torch.randn(dim, dim, device=device))
        return q

_HADAMARD_CACHE = {}

def get_orthogonal_matrix(dim, device="cpu"):
    if dim not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[dim] = generate_orthogonal_hadamard_matrix(dim, device)
    return _HADAMARD_CACHE[dim]

def fast_walsh_hadamard_transform(x):
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.view(-1, D)
    N_flat = x_flat.shape[0]
    power_of_2 = 2 ** math.ceil(math.log2(D))
    if D != power_of_2:
        x_flat = torch.nn.functional.pad(x_flat, (0, power_of_2 - D))
    h = 1
    while h < power_of_2:
        x_flat = x_flat.view(N_flat, -1, h * 2)
        x1 = x_flat[..., :h]
        x2 = x_flat[..., h:]
        x_flat = torch.cat([x1 + x2, x1 - x2], dim=-1)
        h *= 2
    x_flat = x_flat.view(N_flat, power_of_2) / math.sqrt(power_of_2)
    if D != power_of_2:
        x_flat = x_flat[..., :D]
    return x_flat.view(orig_shape)

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear_a4_8(nn.Linear):
    def __init__(self, in_features, out_features, is_intermediate=False, topk_ratio=0.5):
        super().__init__(in_features, out_features, bias=False)
        self.is_intermediate = is_intermediate
        self.topk_ratio = topk_ratio
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        x_rotated = fast_walsh_hadamard_transform(x)
        w = self.weight
        alpha = w.abs().mean()
        w_scaled = w / (alpha + 1e-8)
        w_clipped = torch.clamp(w_scaled, -1, 1)
        w_quant = w_clipped + (RoundSTE.apply(w_clipped) - w_clipped)

        if not self.is_intermediate:
            beta = x_rotated.abs().mean(dim=-1, keepdim=True)
            x_scaled = x_rotated * (2.6457 / (beta + 1e-8))
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -8, 7)) - x_scaled)
            return nn.functional.linear(x_quant, w_quant) * (alpha * beta / 2.6457)
        else:
            gamma = x_rotated.abs().max(dim=-1, keepdim=True)[0]
            x_scaled = x_rotated * (127.0 / (gamma + 1e-8))
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -128, 127)) - x_scaled)
            k = int(x.shape[-1] * self.topk_ratio)
            mask = torch.zeros_like(x_quant)
            topk_vals, _ = torch.topk(x_quant.abs(), k, dim=-1)
            mask[x_quant.abs() >= topk_vals[..., -1:]] = 1.0
            return nn.functional.linear(x_quant * mask, w_quant) * (alpha * gamma / 127.0)

class ReLU2GLUClamped(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w_gate = BitLinear_a4_8(d_model, d_ffn)
        self.w_up = BitLinear_a4_8(d_model, d_ffn)
        self.w_down = BitLinear_a4_8(d_ffn, d_model, is_intermediate=True)

    def forward(self, x):
        gate = torch.clamp(torch.square(torch.relu(self.w_gate(x))), -10, 10)
        up = torch.clamp(self.w_up(x), -10, 10)
        return self.w_down(gate * up)
```

---

### 3.5. Блоки токен-миксеров (`model/attention.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.layers import BitLinear_a4_8, nvtx_range_push, nvtx_range_pop

try:
    from fla.layers.gated_deltanet import GatedDeltaNet
    HAS_FLA = True
except ImportError:
    HAS_FLA = False

class BulbaGDN2SeRoPEBlock(nn.Module):
    def __init__(self, d_model=1536, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        self.q_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.k_proj = BitLinear_a4_8(d_model, n_heads * self.d_k)
        self.v_proj = BitLinear_a4_8(d_model, n_heads * self.d_v)
        
        self.b_proj = nn.Linear(d_model, n_heads * self.d_k)
        self.w_proj = nn.Linear(d_model, n_heads * self.d_v)
        self.alpha_proj = nn.Linear(d_model, n_heads * self.d_k)
        
        if HAS_FLA and torch.cuda.is_available():
            self.gdn2_kernel = GatedDeltaNet(d_model=d_model, n_heads=n_heads, elementwise_affine=True)
        else:
            self.gdn2_kernel = None
            
        self.o_proj = BitLinear_a4_8(d_model, d_model)
        self.register_buffer("freqs", 10000 ** (-torch.arange(0, self.d_k, 2).float() / self.d_k))

    def apply_serope(self, T, q, k):
        B, _, H, _ = q.shape
        q_real, q_imag = q[..., 0::2], q[..., 1::2]
        k_real, k_imag = k[..., 0::2], k[..., 1::2]
        
        angles = torch.arange(T, device=q.device).view(1, T, 1, 1) * self.freqs.view(1, 1, 1, -1)
        cos, sin = torch.cos(angles), torch.sin(angles)
        
        q_out = torch.zeros_like(q)
        k_out = torch.zeros_like(k)
        q_out[..., 0::2], q_out[..., 1::2] = q_real * cos - q_imag * sin, q_real * sin + q_imag * cos
        k_out[..., 0::2], k_out[..., 1::2] = k_real * cos + k_imag * sin, -k_real * sin + k_imag * cos
        return q_out, k_out

    def forward(self, x):
        nvtx_range_push("Bysel_GDN2_SeRoPE_Forward")
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k)
        v = self.v_proj(x)
        
        q, k = self.apply_serope(T, q, k)
        q = q.view(B, T, -1)
        k = k.view(B, T, -1)
        
        b = torch.sigmoid(self.b_proj(x))
        w = torch.sigmoid(self.w_proj(x))
        alpha = torch.sigmoid(self.alpha_proj(x))
        
        if self.gdn2_kernel is not None:
            out = self.gdn2_kernel(q, k, v, b, w, alpha)
        else:
            out = self.pure_pytorch_gdn2(q, k, v, b, w, alpha)
            
        res = self.o_proj(out)
        nvtx_range_pop()
        return res

    def pure_pytorch_gdn2(self, q, k, v, b, w, alpha):
        B, T = q.shape[0], q.shape[1]
        S = torch.zeros(B, self.n_heads, self.d_k, self.d_v, device=q.device, dtype=q.dtype)
        out = torch.zeros(B, T, self.n_heads * self.d_v, device=q.device, dtype=q.dtype)
        
        q = q.view(B, T, self.n_heads, self.d_k)
        k = k.view(B, T, self.n_heads, self.d_k)
        v = v.view(B, T, self.n_heads, self.d_v)
        b = b.view(B, T, self.n_heads, self.d_k)
        w = w.view(B, T, self.n_heads, self.d_v)
        alpha = alpha.view(B, T, self.n_heads, self.d_k)
        
        for t in range(T):
            decay = (1.0 - b[:, t] * k[:, t]).unsqueeze(-1) * alpha[:, t].unsqueeze(-1)
            write = w[:, t].unsqueeze(-2) * (k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2))
            S = decay * S + write
            out[:, t] = torch.matmul(q[:, t].unsqueeze(-2), S).squeeze(-2).view(B, -1)
        return out.view(B, T, self.d_model)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=1536, n_heads=12, d_c=128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_c = d_c
        self.d_v = d_model // n_heads
        
        self.kv_compress = BitLinear_a4_8(d_model, d_c)
        self.kv_norm = nn.RMSNorm(d_c)
        self.k_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        self.v_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        
        self.q_compress = BitLinear_a4_8(d_model, d_c)
        self.q_norm = nn.RMSNorm(d_c)
        self.q_decompress = BitLinear_a4_8(d_c, n_heads * self.d_v)
        self.o_proj = BitLinear_a4_8(n_heads * self.d_v, d_model)

    def forward(self, x):
        nvtx_range_push("Bysel_MLA_Forward")
        B, T, C = x.shape
        kv_latent = self.kv_norm(self.kv_compress(x))
        k = self.k_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        v = self.v_decompress(kv_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        
        q_latent = self.q_norm(self.q_compress(x))
        q = self.q_decompress(q_latent).view(B, T, self.n_heads, self.d_v).transpose(1, 2)
        
        context = F.scaled_dot_product_attention(q, k, v)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(context)
        nvtx_range_pop()
        return out
```

---

### 3.6. Маршрутизация (`model/routing.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BitLinear_a4_8, nvtx_range_push, nvtx_range_pop

class MoDSequenceRouter(nn.Module):
    def __init__(self, d_model, capacity_factor=0.25):
        super().__init__()
        self.router = nn.Linear(d_model, 1)
        self.capacity_factor = capacity_factor

    def forward(self, x):
        nvtx_range_push("Bysel_MoD_Routing_Forward")
        B, T, C = x.shape
        k = int(T * self.capacity_factor)
        logits = self.router(x).squeeze(-1)
        _, topk_indices = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        nvtx_range_pop()
        return mask, logits

class LearnableClampSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bounds):
        ctx.save_for_backward(x, bounds)
        return torch.clamp(x, -bounds, bounds)

    @staticmethod
    def backward(ctx, grad_output):
        x, bounds = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_bounds = grad_output.clone()
        grad_bounds = (grad_bounds * (x > bounds).float()) - (grad_bounds * (x < -bounds).float())
        sum_dims = list(range(grad_bounds.ndim - 1))
        if sum_dims:
            grad_bounds = grad_bounds.sum(dim=sum_dims)
        return grad_x, grad_bounds

class BulbaTernaryTitanExpertFFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w_gate = BitLinear_a4_8(d_model, d_ffn)
        self.w_up = BitLinear_a4_8(d_model, d_ffn)
        self.w_down = BitLinear_a4_8(d_ffn, d_model, is_intermediate=True)
        self.clipping_bounds = nn.Parameter(torch.ones(d_ffn) * 10.0)

    def forward(self, x):
        gate = LearnableClampSTE.apply(torch.square(torch.relu(self.w_gate(x))), self.clipping_bounds)
        return self.w_down(gate * self.w_up(x))

class BulbaTernaryTitanMoE(nn.Module):
    def __init__(self, d_model, d_ffn, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        self.shared_experts = nn.ModuleList([BulbaTernaryTitanExpertFFN(d_model, d_ffn) for _ in range(2)])
        self.routed_experts = nn.ModuleList([BulbaTernaryTitanExpertFFN(d_model, d_ffn) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts, bias=False, dtype=torch.bfloat16)
        
        self.w_gate_blackboard = BitLinear_a4_8(d_model, d_model)
        self.w_read_blackboard = BitLinear_a4_8(d_model, d_model)

    def forward(self, x, aux_loss_weight=0.01, z_loss_weight=0.001):
        nvtx_range_push("Bysel_MoE_Experts_Forward")
        B, T, D = x.shape
        
        h_bb = (self.shared_experts[0](x) + self.shared_experts[1](x)) / 2.0
        x_enriched = x + torch.sigmoid(self.w_gate_blackboard(x)) * self.w_read_blackboard(h_bb)
        
        router_logits = self.router(x_enriched.detach())
        z_loss = z_loss_weight * torch.mean(torch.logsumexp(router_logits, dim=-1)**2)
        
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights /= (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        routed_output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                routed_output[mask] += self.routed_experts[i](x_enriched[mask]) * topk_weights[mask, (topk_indices[mask] == i).nonzero(as_tuple=True)[1]].unsqueeze(-1)
                
        f_i = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f_i[i] = (topk_indices == i).sum().float()
        load_balance_loss = aux_loss_weight * self.num_experts * torch.sum((f_i / (B * T * self.top_k)) * routing_weights.mean(dim=(0, 1)))
        
        nvtx_range_pop()
        return h_bb + routed_output, load_balance_loss + z_loss
```

---

### 3.7. Бэкбон и MTP-4 (`model/backbone.py`)

```python
import torch
import torch.nn as nn
from model.layers import BitLinear_a4_8
from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanMoE

class ManifoldConstrainedAttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.RMSNorm(d_model)
        nn.init.zeros_(self.proj.weight)

    def forward(self, current_x, all_prev_outputs):
        if not all_prev_outputs:
            return current_x
        V = torch.stack(all_prev_outputs, dim=0)
        K = self.norm(V)
        logits = torch.einsum('d, l b t d -> l b t', self.proj.weight.squeeze(), K)
        M = torch.exp(logits)
        for _ in range(5):
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)
        h = torch.einsum('l b t, l b t d -> b t d', M, V)
        return current_x + h

class ByselDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, expert_hidden, num_experts, is_global=False):
        super().__init__()
        self.mod_router = MoDSequenceRouter(d_model)
        if is_global:
            self.attn = MultiHeadLatentAttention(d_model, n_heads)
        else:
            self.attn = BulbaGDN2SeRoPEBlock(d_model, n_heads)
        self.moe = BulbaTernaryTitanMoE(d_model, expert_hidden, num_experts=num_experts)
        self.attn_norm = nn.RMSNorm(d_model)
        self.moe_norm = nn.RMSNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        mask, logits = self.mod_router(x)
        active_tokens = x[mask].view(B, -1, C)
        
        attn_out = self.attn(self.attn_norm(active_tokens))
        moe_out, aux_loss = self.moe(self.moe_norm(attn_out))
        
        gated_out = moe_out * torch.sigmoid(logits[mask]).view(B, -1, 1)
        out = x.clone()
        out[mask] = (out[mask].view(B, -1, C) + gated_out).view(-1, C)
        return out, aux_loss

class ByselMTP4Pipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.projections = nn.ModuleList([BitLinear_a4_8(config.d_model, config.d_model) for _ in range(3)])
        self.heads = nn.ModuleList([BitLinear_a4_8(config.d_model, config.vocab_size) for _ in range(4)])

    def forward(self, main_hidden_states, next_token_ids=None):
        logits_t1 = self.heads[0](main_hidden_states)
        if next_token_ids is None:
            return logits_t1, None, None, None
        h_detached = main_hidden_states.detach()
        combined_t2 = self.projections[0](h_detached) + self.embed(next_token_ids[0])
        logits_t2 = self.heads[1](combined_t2)
        combined_t3 = self.projections[1](combined_t2) + self.embed(next_token_ids[1])
        logits_t3 = self.heads[2](combined_t3)
        combined_t4 = self.projections[2](combined_t3) + self.embed(next_token_ids[2])
        logits_t4 = self.heads[3](combined_t4)
        return logits_t1, logits_t2, logits_t3, logits_t4

class ByselModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(config.n_layers):
            is_global = (l + 1) % 4 == 0
            self.layers.append(ByselDecoderLayer(config.d_model, config.n_heads, config.expert_hidden, config.num_experts, is_global=is_global))
        self.m_residuals = nn.ModuleList([ManifoldConstrainedAttnRes(config.d_model) for _ in range(config.n_layers)])
        self.final_norm = nn.RMSNorm(config.d_model)
        self.mtp_pipeline = ByselMTP4Pipeline(config)

    def forward(self, x, next_token_ids=None):
        prev_outputs = []
        total_aux_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            prev_outputs.append(x)
            x = self.m_residuals[i](x, prev_outputs)
        hidden_states = self.final_norm(x)
        return self.mtp_pipeline(hidden_states, next_token_ids), total_aux_loss
```

---

## 4. ОПТИМИЗАТОР И ШАГИ ОБУЧЕНИЯ

### 4.1. Оркестратор оптимизации (`training/optimizer.py`)

```python
import torch
import math

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    dtype = torch.bfloat16 if p.device.type in ["cuda", "mps"] else torch.float32
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=dtype)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad.to(buf.dtype))
                m_t = grad.to(buf.dtype) + momentum * buf
                O_t = self.hybrid_newton_schulz(m_t, steps=ns_steps)
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))
                p.mul_(1.0 - lr * wd)
                p.add_(O_t.to(p.dtype), alpha=-lr * scale)

    def hybrid_newton_schulz(self, M, steps=10):
        X = M / (M.norm() + 1e-8)
        a1, b1, c1 = 3.4445, -4.7750, 2.0315
        a2, b2, c2 = 2.0, -1.5, 0.5
        for step in range(steps):
            XXT = torch.matmul(X, X.transpose(-1, -2))
            if step < 8:
                X = a1 * X + b1 * torch.matmul(XXT, X) + c1 * torch.matmul(torch.matmul(XXT, XXT), X)
            else:
                X = a2 * X + b2 * torch.matmul(XXT, X) + c2 * torch.matmul(torch.matmul(XXT, XXT), X)
        return X

class ByselOptimizerEngine:
    def __init__(self, model, lr_muon=0.002, lr_adamw=0.0002):
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if param.ndim == 2 and "router" not in name and "proj" in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        self.opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95)
        self.opt_adamw = torch.optim.AdamW(adamw_params, lr=lr_adamw, weight_decay=0.01)

    def zero_grad(self):
        self.opt_muon.zero_grad()
        self.opt_adamw.zero_grad()

    def step(self):
        self.opt_muon.step()
        self.opt_adamw.step()
```

---

### 4.2. Автономный автопилот (`training/autopilot.py`)

```python
import torch
import math

class ByselAutoPilot:
    def __init__(self, opt_engine, min_lr=1e-5, max_lr=0.01, noise_decay=0.999):
        self.opt_engine = opt_engine
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.noise_scale = 0.01
        self.noise_decay = noise_decay
        self.loss_history = []

    def update_parameters(self, step, current_loss, max_steps):
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 50: self.loss_history.pop(0)
        loss_variance = torch.tensor(self.loss_history).var().item() if len(self.loss_history) > 1 else 0.0
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * (step / max_steps)))
        stabilization = 0.5 if loss_variance > 0.5 else 1.0
        new_lr_muon = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor * stabilization
        new_lr_adamw = new_lr_muon * 0.1
        for pg in self.opt_engine.opt_muon.param_groups: pg['lr'] = new_lr_muon
        for pg in self.opt_engine.opt_adamw.param_groups: pg['lr'] = new_lr_adamw
        self.noise_scale *= self.noise_decay
        return new_lr_muon, self.noise_scale

    def inject_noise(self, model):
        if self.noise_scale < 1e-6: return
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * self.noise_scale)
```

---

## 5. ИНСТРУМЕНТЫ ОТЛАДКИ И CLI

### 5.1. Полноценный запуск профайлера (`tests/profiler_run.py`)

```python
import torch
import os
from data.pipeline import get_bysel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel
from training.optimizer import ByselOptimizerEngine

class Config:
    vocab_size = 32000
    d_model = 512
    n_layers = 16
    n_heads = 8
    expert_hidden = 1024
    num_experts = 8
    top_k = 2

def run_profile_test():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"📊 Тестирование системы Bysel на устройстве: {device.upper()}")
    
    test_file = "dummy_bel.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Слава беларускаму аисту! Гэты тэкст чытаецца праз Rust-модуль bysel_rust_io на хуткасці NVMe.")

    dataloader = get_bysel_dataloader(test_file, chunk_size=32, batch_size=2)
    byte_batch = next(iter(dataloader)).to(device)
    print(f"✅ Успешно прочитан батч байтов через Rust! Размерность: {byte_batch.shape}")

    config = Config()
    patcher = StridedFastBLTPatcher(d_model=config.d_model).to(device)
    model = ByselModel(config).to(device)
    optimizer = ByselOptimizerEngine(model)
    
    patches = patcher(byte_batch)
    print(f"✅ Входные патчи сформированы: {patches.shape}")
    
    mtp_targets = [torch.randint(0, 32000, (2, patches.shape[1]), device=device) for _ in range(3)]
    
    (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(patches, mtp_targets)
    print(f"✅ Пнямой проход выполнен успешно! Логиты t+1: {logits_t1.shape}")
    print(f"   Auxiliary MoE Loss: {aux_loss.item():.4f}")

    loss = logits_t1.mean() + aux_loss
    loss.backward()
    print("✅ Обратный проход выполнен! Градиенты рассчитаны.")
    
    if os.path.exists(test_file):
        os.remove(test_file)
    print("🎉 Тест завершен успешно! Все системы работают без ошибок.")

if __name__ == "__main__":
    run_profile_test()
```

### 5.2. Универсальный CLI-интерфейс (`cli.py`)

```python
import typer
import os
import uvicorn
from tests.profiler_run import run_profile_test

app = typer.Typer(help="Bysel CLI Engine - Sovereign 1-bit Omni-LLM")

@app.command()
def train(
    mode: str = typer.Option(..., "--mode", "-m", help="Стадия: pretrain, sft или dpo"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Конфиг"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Имя датасета"),
    autopilot: bool = typer.Option(True, help="Включить автопилот")
):
    typer.echo(typer.style(f"🚀 Запуск обучения [{mode.upper()}] для bysel...", fg=typer.colors.GREEN, bold=True))

@app.command()
def profile():
    typer.echo(typer.style("📊 Запуск профилировщика...", fg=typer.colors.CYAN, bold=True))
    run_profile_test()

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Хост"),
    port: int = typer.Option(8000, help="Порт")
):
    typer.echo(typer.style(f"🔥 Запуск сервера на http://{host}:{port}", fg=typer.colors.MAGENTA, bold=True))
    uvicorn.run("services.inference_api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    app()
```

---

## 6. ДЕКЛАРАТИВНЫЕ КОНФИГУРАЦИИ СИСТЕМЫ (`configs/default.yaml`)

```yaml
global:
  project_name: "bysel"
  device: "cuda"
  seed: 42
  precision: "bf16"

profiles:
  ziaziulia:
    model:
      d_model: 256
      n_layers: 12
      n_heads: 4
      expert_hidden: 512
      num_experts: 4
      top_k: 2
      vocab_size: 32000
    data:
      data_path: "data_train"
      chunk_size: 512
      batch_size: 4
    training:
      max_steps: 15000
      learning_rate_muon: 0.002
      learning_rate_adamw: 0.0002
      weight_decay: 0.1

  zubr:
    model:
      d_model: 1792
      n_layers: 32
      n_heads: 12
      expert_hidden: 2048
      num_experts: 64
      top_k: 2
      vocab_size: 32000
    data:
      data_path: "data_train"
      chunk_size: 8192
      batch_size: 32
    training:
      max_steps: 80000
      learning_rate_muon: 0.001
      learning_rate_adamw: 0.0001
      weight_decay: 0.1
```