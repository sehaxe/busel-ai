# BULBA1-LIGHTNING PROJECT DUMP
**Date:** Sun May 31 10:16:18 +03 2026
---\n
================================================================
📁 FILE: ./.cargo/config.toml (210 bytes)
================================================================
```yaml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
```



================================================================
📁 FILE: ./bysel.egg-info/dependency_links.txt (1 bytes)
================================================================
```text

```



================================================================
📁 FILE: ./bysel.egg-info/requires.txt (180 bytes)
================================================================
```text
aiogram>=3.28.2
fastapi>=0.136.3
flash-linear-attention>=0.5.0
maturin>=1.13.3
numpy>=2.4.6
pandas>=3.0.3
pyarrow>=24.0.0
pyyaml>=6.0.3
torch>=2.12.0
typer>=0.26.4
uvicorn>=0.48.0
```



================================================================
📁 FILE: ./bysel.egg-info/SOURCES.txt (358 bytes)
================================================================
```text
README.md
pyproject.toml
bysel.egg-info/PKG-INFO
bysel.egg-info/SOURCES.txt
bysel.egg-info/dependency_links.txt
bysel.egg-info/requires.txt
bysel.egg-info/top_level.txt
data/pipeline.py
model/attention.py
model/backbone.py
model/layers.py
model/patching.py
model/routing.py
tests/profiler_run.py
training/autopilot.py
training/optimizer.py
training/recipe.py```



================================================================
📁 FILE: ./bysel.egg-info/top_level.txt (37 bytes)
================================================================
```text
data
data_train
model
tests
training
```



================================================================
📁 FILE: ./Cargo.toml (251 bytes)
================================================================
```yaml
[package]
name = "bysel_rust_io"
version = "3.3.0"
edition = "2021"

[lib]
name = "bysel_rust_io"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.28", features = ["extension-module"] }
rayon = "1.8"
memmap2 = "0.9"  # Memory-mapped files```



================================================================
📁 FILE: ./cli.py (1245 bytes)
================================================================
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



================================================================
📁 FILE: ./configs/default.yaml (915 bytes)
================================================================
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
      vocab_size: 259
    data:
      data_path: "data_train"
      chunk_size: 512
      batch_size: 4
    training:
      max_steps: 15000
      learning_rate_muon: 0.0004
      learning_rate_adamw: 0.00004
      weight_decay: 0.1

  zubr:
    model:
      d_model: 1792
      n_layers: 32
      n_heads: 16        # 🎯 ИСПРАВЛЕНО: было 12, стало 16 (1792/16=112)
      expert_hidden: 2048
      num_experts: 64
      top_k: 2
      vocab_size: 259
    data:
      data_path: "data_train"
      chunk_size: 8192
      batch_size: 32
    training:
      max_steps: 80000
      learning_rate_muon: 0.0004
      learning_rate_adamw: 0.00004
      weight_decay: 0.1```



================================================================
📁 FILE: ./data/pipeline.py (7651 bytes)
================================================================
```python
import torch
import os
import json
import random
from torch.utils.data import IterableDataset, DataLoader

try:
    import bysel_rust_io
    HAS_RUST_IO = True
except ImportError:
    HAS_RUST_IO = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


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
        
        if len(chunk) < self.chunk_size:
            chunk = chunk + [0] * (self.chunk_size - len(chunk))
        return chunk
        
    def get_position(self):
        return self.position


class ByselOmnivoreTextExtractor:
    def __init__(self, file_path, chunk_size, start_offset=0):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.position = start_offset
        self.buffer = bytearray()
        
        if file_path.endswith('.parquet'):
            if not HAS_PANDAS:
                raise ImportError("\n❌ Для чтения .parquet установите: 'uv add pandas pyarrow'")
            df = pd.read_parquet(file_path)
            text_col = self._detect_text_column(df)
            full_text = "\n".join(text_col.astype(str).tolist())
            self.raw_bytes = full_text.encode('utf-8')
            
        elif file_path.endswith('.jsonl'):
            extracted_texts = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # ИСПОЛЬЗУЕМ РЕКУРСИВНЫЙ ПАРСЕР [1]
                            text_val = self._recursive_extract(data)
                            if text_val.strip():
                                extracted_texts.append(text_val.strip())
                        except json.JSONDecodeError:
                            continue # Пропускаем битые строки
            full_text = "\n".join(extracted_texts)
            self.raw_bytes = full_text.encode('utf-8')
        else:
            with open(file_path, "rb") as f:
                self.raw_bytes = f.read()

    def _recursive_extract(self, obj):
        """ Рекурсивно погружается в JSON и вытаскивает 100% текста из любых структур """
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            # Проходим по всем значениям словаря
            return "\n".join([self._recursive_extract(v) for v in obj.values() if v])
        elif isinstance(obj, list):
            # Проходим по всем элементам списка
            return "\n".join([self._recursive_extract(item) for item in obj if item])
        else:
            return ""

    def _detect_text_column(self, df):
        for col in ["text", "content", "body", "code", "markdown", "raw_text"]:
            if col in df.columns:
                return df[col]
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "string":
                return df[col]
        raise ValueError(f"Не удалось найти текстовую колонку в Parquet файле.")

    def next_chunk(self):
        if self.position >= len(self.raw_bytes):
            return None
        start = self.position
        end = min(self.position + self.chunk_size, len(self.raw_bytes))
        chunk = list(self.raw_bytes[start:end])
        self.position = end
        
        if len(chunk) < self.chunk_size:
            chunk = chunk + [0] * (self.chunk_size - len(chunk))
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
                    if filename.endswith(('.txt', '.py', '.rs', '.go', '.be', '.json', '.cpp', '.h', '.jsonl', '.parquet')):
                        self.files.append(os.path.join(root, filename))
            self.files.sort()
        elif os.path.isfile(data_path):
            self.files.append(data_path)
            
        if not self.files:
            raise ValueError(
                f"\n❌ [ОШИБКА ДАННЫХ]: В папке '{data_path}' не найдено подходящих файлов для обучения!\n"
            )
            
        self.current_file_idx = start_file_idx
        self.current_byte_offset = start_byte_offset

    def __iter__(self):
        shuffle_buffer = []
        buffer_size = 50
        
        for file_idx in range(self.start_file_idx, len(self.files)):
            self.current_file_idx = file_idx
            file_path = self.files[file_idx]
            offset = self.start_byte_offset if file_idx == self.start_file_idx else 0
            
            use_rust_streamer = (
                not file_path.endswith(('.parquet', '.jsonl')) 
                and HAS_RUST_IO 
                and torch.cuda.is_available()
            )
            
            if use_rust_streamer:
                num_cores = os.cpu_count()
                if num_cores > 8:
                    num_cores = num_cores // 2
                bysel_rust_io.init_thread_pool(num_cores)
                streamer = bysel_rust_io.ByteStreamer(file_path, self.chunk_size, offset)
            else:
                streamer = ByselOmnivoreTextExtractor(file_path, self.chunk_size, offset)
                
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
```



================================================================
📁 FILE: ./INFO.md (36872 bytes)
================================================================
```text
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
``````



================================================================
📁 FILE: ./model/attention.py (6937 bytes)
================================================================
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
        
        # CUDA-FIRST: используем FLA на CUDA (быстрый Triton kernel)
        if HAS_FLA and torch.cuda.is_available():
            self.gdn2_kernel = GatedDeltaNet(d_model=d_model, n_heads=n_heads, elementwise_affine=True)
            self.use_fla = True
        else:
            self.gdn2_kernel = None
            self.use_fla = False
            
        self.o_proj = BitLinear_a4_8(d_model, d_model)
        self.register_buffer("freqs", 10000 ** (-torch.arange(0, self.d_k, 2).float() / self.d_k))

    def apply_serope(self, T, q, k):
        """Применяет SeRoPE к 4-мерным тензорам [B, T, H, dk]"""
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
        
        # Проекция в 4D: [B, T, H, dk/dv]
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_v)
        
        # SeRoPE работает на 4D тензорах
        q, k = self.apply_serope(T, q, k)
        
        # Гейты: [B, T, H, dk/dv]
        b = torch.sigmoid(self.b_proj(x)).view(B, T, self.n_heads, self.d_k)
        w = torch.sigmoid(self.w_proj(x)).view(B, T, self.n_heads, self.d_v)
        alpha = torch.sigmoid(self.alpha_proj(x)).view(B, T, self.n_heads, self.d_k)
        
        if self.use_fla:
            # CUDA: FLA kernel ожидает [B, T, H*d_k] формат
            q_flat = q.view(B, T, -1)
            k_flat = k.view(B, T, -1)
            v_flat = v.view(B, T, -1)
            b_flat = b.view(B, T, -1)
            w_flat = w.view(B, T, -1)
            alpha_flat = alpha.view(B, T, -1)
            out = self.gdn2_kernel(q_flat, k_flat, v_flat, b_flat, w_flat, alpha_flat)
        else:
            # MPS/CPU fallback: БЕЗ INPLACE ОПЕРАЦИЙ!
            out = self.vectorized_gdn2(q, k, v, b, w, alpha)
            
        res = self.o_proj(out)
        nvtx_range_pop()
        return res

    def vectorized_gdn2(self, q, k, v, b, w, alpha):
        """
        Векторизованная реализация GDN-2 для MPS/CPU.
        БЕЗ inplace операций для корректного backward pass!
        
        Все тензоры в формате [B, T, H, dk/dv]
        """
        B, T, H, dk = q.shape
        dv = v.shape[-1]
        
        # Инициализируем состояние памяти [B, H, dk, dv]
        S = torch.zeros(B, H, dk, dv, device=q.device, dtype=q.dtype)
        
        # Собираем выходы в список
        outputs = []
        
        for t in range(T):
            # Получаем срезы для шага t
            q_t = q[:, t]        # [B, H, dk]
            k_t = k[:, t]        # [B, H, dk]
            v_t = v[:, t]        # [B, H, dv]
            b_t = b[:, t]        # [B, H, dk]
            w_t = w[:, t]        # [B, H, dv]
            alpha_t = alpha[:, t]  # [B, H, dk]
            
            # Decay: (1 - b*k) * alpha, shape [B, H, dk, 1]
            decay = ((1.0 - b_t * k_t) * alpha_t).unsqueeze(-1)
            
            # Write: w * (k ⊗ v), shape [B, H, dk, dv]
            write = w_t.unsqueeze(-1) * (k_t.unsqueeze(-1) * v_t.unsqueeze(-2))
            
            # ИСПРАВЛЕНО: НЕ inplace! Создаём новый тензор
            # Это критично для корректного backward pass
            S = S * decay + write
            
            # Output для шага t: q @ S -> [B, H, dv]
            out_t = torch.einsum('bhd,bhdv->bhv', q_t, S)
            outputs.append(out_t)
        
        # Собираем все выходы в [B, T, H, dv]
        out = torch.stack(outputs, dim=1)
        
        # Финальный reshape: [B, T, H, dv] -> [B, T, H*dv]
        return out.reshape(B, T, -1)

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
        return out```



================================================================
📁 FILE: ./model/backbone.py (13303 bytes)
================================================================
```python
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL BACKBONE v3.0 - Complete Architecture                               ║
║                                                                           ║
║ Компоненты:                                                               ║
║  • ByselDecoderLayer: Attention + MoD + MoE Blackboard                   ║
║  • ManifoldConstrainedAttnRes (mAR): межслойные связи с Sinkhorn-3       ║
║  • ByselMTP4Pipeline: Multi-Token Prediction (4 головы)                  ║
║  • ByselModel: оркестратор с Gradient Checkpointing                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
from model.layers import BitLinear_a4_8, nvtx_range_push, nvtx_range_pop
from model.attention import BulbaGDN2SeRoPEBlock, MultiHeadLatentAttention
from model.routing import MoDSequenceRouter, BulbaTernaryTitanMoE


class ManifoldConstrainedAttnRes(nn.Module):
    """
    УНИКАЛЬНАЯ ТЕХНОЛОГИЯ Bulba1-Lightning: mAR с проекцией на Birkhoff Polytope.
    
    Решает проблему Attention Sink через 3 итерации Sinkhorn-Knopp,
    проецируя матрицу внимания на дважды стохастическое многообразие.
    Это гарантирует равномерное распределение внимания по всем слоям.
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.RMSNorm(d_model)
        # Инициализируем нулями — сначала mAR не влияет, модель учится постепенно
        nn.init.zeros_(self.proj.weight)

    def forward(self, current_x, all_prev_outputs):
        """
        Args:
            current_x: текущий скрытый состояние [B, T, D]
            all_prev_outputs: список всех предыдущих состояний (включая current_x)
        Returns:
            x + attention-weighted sum of all prev outputs
        """
        if not all_prev_outputs:
            return current_x
        
        # V: [L, B, T, D] — стек всех предыдущих состояний
        V = torch.stack(all_prev_outputs, dim=0)
        
        # Нормализуем для стабильности
        K = self.norm(V)
        
        # Вычисляем логиты внимания через обучаемый проектор
        # proj.weight: [1, D], K: [L, B, T, D] -> logits: [L, B, T]
        logits = torch.einsum('d, l b t d -> l b t', self.proj.weight.squeeze(), K)
        
        # Экспоненцируем для получения положительных весов
        M = torch.exp(logits)
        
        # 3 итерации Sinkhorn-Knopp для проекции на Birkhoff polytope
        # (дважды стохастическая матрица: суммы по строкам и столбцам = 1)
        for _ in range(3):
            M = M / (M.sum(dim=0, keepdim=True) + 1e-8)  # Нормализация по столбцам
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)  # Нормализация по строкам
        
        # Взвешенная сумма всех предыдущих состояний
        # M: [L, B, T], V: [L, B, T, D] -> h: [B, T, D]
        h = torch.einsum('l b t, l b t d -> b t d', M, V)
        
        # Residual connection
        return current_x + h


class ByselDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, expert_hidden, num_experts, is_global=False):
        super().__init__()
        # MoD router (временно отключен для pretraining)
        self.mod_router = MoDSequenceRouter(d_model, capacity_factor=1.0)  # ← ИЗМЕНЕНО с 0.25 на 1.0
        
        if is_global:
            self.attn = MultiHeadLatentAttention(d_model, n_heads)
        else:
            self.attn = BulbaGDN2SeRoPEBlock(d_model, n_heads)
        
        self.moe = BulbaTernaryTitanMoE(d_model, expert_hidden, num_experts=num_experts)
        
        self.attn_norm = nn.RMSNorm(d_model)
        self.moe_norm = nn.RMSNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        
        # 🎯 ВРЕМЕННО: все токены проходят через Attention+MoE
        # (MoD отключен для pretraining, будет включен после CE < 4.5)
        active_tokens = x  # ← ВСЕ токены, без маскирования
        
        attn_out = self.attn(self.attn_norm(active_tokens))
        moe_out, aux_loss = self.moe(self.moe_norm(attn_out))
        
        # Простой residual без gating
        out = x + moe_out
        return out, aux_loss


class ByselMTP4Pipeline(nn.Module):
    """
    Multi-Token Prediction с 4 головами.
    
    Предсказывает 4 следующих токена параллельно:
        - t+1: основная голова (на главном hidden state)
        - t+2, t+3, t+4: дополнительные головы (на проекциях + embed)
    
    ИСПРАВЛЕНО для MPS: используем прямую индексацию весов вместо nn.Embedding,
    чтобы обойти баг "Placeholder storage has not been allocated".
    """
    def __init__(self, config):
        super().__init__()
        
        # 🎯 КЛЮЧЕВОЙ ФИКС: nn.Parameter вместо nn.Embedding
        # Математически идентично, но работает на всех устройствах без MPS багов
        self.embed_weight = nn.Parameter(
            torch.randn(config.vocab_size, config.d_model) * 0.02
        )
        
        # Проекции для передачи состояния между головами
        self.projections = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.d_model) 
            for _ in range(3)
        ])
        
        # 4 головы для предсказания t+1, t+2, t+3, t+4
        self.heads = nn.ModuleList([
            BitLinear_a4_8(config.d_model, config.vocab_size) 
            for _ in range(4)
        ])

    def _embed_lookup(self, token_ids):
        """
        Прямая индексация весов — MPS-safe замена nn.Embedding.
        Математически идентично F.embedding(token_ids, weight).
        """
        return self.embed_weight[token_ids.to(self.embed_weight.device)]

    def forward(self, main_hidden_states, next_token_ids=None):
        """
        Args:
            main_hidden_states: [B, T, D] — выход основного бэкбона
            next_token_ids: список из 3 тензоров [B, T] (targets для t+1, t+2, t+3)
                           None во время инференса
        Returns:
            logits_t1, logits_t2, logits_t3, logits_t4
        """
        # Основная голова: предсказание t+1
        logits_t1 = self.heads[0](main_hidden_states)
        
        if next_token_ids is None:
            return logits_t1, None, None, None
        
        # Проверяем что все таргеты не None
        if any(t is None for t in next_token_ids):
            return logits_t1, None, None, None
        
        # Отключаем градиенты от бэкбона для MTP-голов
        h_detached = main_hidden_states.detach()
        
        # t+2: projection(h_t) + embed(token_t+1)
        # 🎯 Используем _embed_lookup вместо self.embed(...)
        combined_t2 = self.projections[0](h_detached) + self._embed_lookup(next_token_ids[0])
        logits_t2 = self.heads[1](combined_t2)
        
        # t+3: projection(combined_t2) + embed(token_t+2)
        combined_t3 = self.projections[1](combined_t2) + self._embed_lookup(next_token_ids[1])
        logits_t3 = self.heads[2](combined_t3)
        
        # t+4: projection(combined_t3) + embed(token_t+3)
        combined_t4 = self.projections[2](combined_t3) + self._embed_lookup(next_token_ids[2])
        logits_t4 = self.heads[3](combined_t4)
        
        return logits_t1, logits_t2, logits_t3, logits_t4

class ByselModel(nn.Module):
    """
    Полный бэкбон Bysel v3.0.
    
    Архитектура:
        Byte Input -> FastBLT Patcher -> [DecoderLayer + mAR] x N -> FinalNorm -> MTP-4
    
    Grid слоёв (3:1 ratio):
        - GDN-2 (Linear Attention): 75% слоёв (быстро, O(1) inference)
        - MLA (Global Attention): 25% слоёв (для точного извлечения на дальних дистанциях)
    
    Gradient Checkpointing:
        - На CUDA включается автоматически для экономии ~40% VRAM
        - На MPS/CPU работает без checkpointing (достаточно памяти)
    """
    def __init__(self, config):
        super().__init__()
        
        # === СЛОИ ===
        self.layers = nn.ModuleList()
        for l in range(config.n_layers):
            # MLA на позициях 4, 8, 12, 16, 20, 24, 28, 32 (1-based)
            is_global = (l + 1) % 4 == 0
            self.layers.append(ByselDecoderLayer(
                config.d_model, 
                config.n_heads, 
                config.expert_hidden, 
                config.num_experts, 
                is_global=is_global
            ))
        
        # === mAR RESIDUALS ===
        self.m_residuals = nn.ModuleList([
            ManifoldConstrainedAttnRes(config.d_model) 
            for _ in range(config.n_layers)
        ])
        
        # === FINAL NORM ===
        self.final_norm = nn.RMSNorm(config.d_model)
        
        # === MTP-4 PIPELINE ===
        self.mtp_pipeline = ByselMTP4Pipeline(config)
        
        # === GRADIENT CHECKPOINTING FLAG ===
        self.use_gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """Включает экономию VRAM через пересчёт активаций (только на CUDA)"""
        self.use_gradient_checkpointing = True
        print("💾 Gradient Checkpointing ВКЛЮЧЕН (экономия ~40% VRAM, -30% скорость)")

    def disable_gradient_checkpointing(self):
        """Выключает экономию VRAM (максимальная скорость)"""
        self.use_gradient_checkpointing = False
        print("🚀 Gradient Checkpointing ВЫКЛЮЧЕН (максимальная скорость)")

    def _forward_layer(self, layer, m_res, x, prev_outputs):
        """
        Внутренний метод для gradient checkpointing.
        Оборачивает работу слоя + mAR в одну функцию.
        
        ВАЖНО: prev_outputs мутируется внутри, что допустимо для checkpoint
        """
        x, aux_loss = layer(x)
        prev_outputs.append(x)
        x = m_res(x, prev_outputs)
        return x, aux_loss

    def forward(self, x, next_token_ids=None):
        """
        Args:
            x: [B, T, D] — выход FastBLT patcher
            next_token_ids: список из 3 тензоров для MTP (или None)
        Returns:
            (logits_t1, logits_t2, logits_t3, logits_t4), total_aux_loss
        """
        nvtx_range_push("ByselModel_Forward")
        
        prev_outputs = []
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            m_res = self.m_residuals[i]
            
            if self.training and self.use_gradient_checkpointing and x.device.type == "cuda":
                # CUDA + Training: используем gradient checkpointing
                # use_reentrant=False — современный API (стабильнее и быстрее)
                x, aux_loss = torch.utils.checkpoint.checkpoint(
                    self._forward_layer,
                    layer, m_res, x, prev_outputs,
                    use_reentrant=False
                )
            else:
                # Инференс / MPS / CPU: обычный forward pass
                x, aux_loss = layer(x)
                prev_outputs.append(x)
                x = m_res(x, prev_outputs)
            
            total_aux_loss += aux_loss
        
        # Финальная нормализация
        hidden_states = self.final_norm(x)
        
        # MTP-4 pipeline
        mtp_outputs = self.mtp_pipeline(hidden_states, next_token_ids)
        
        nvtx_range_pop()
        return mtp_outputs, total_aux_loss```



================================================================
📁 FILE: ./model/layers.py (5705 bytes)
================================================================
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
    x_flat = x_flat.view(N_flat, power_of_2).div_(math.sqrt(power_of_2))
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
    """
    BitNet 1.58-bit (тернарные веса {-1, 0, 1}) + INT4/INT8 активации.
    
    🎯 КЛЮЧЕВЫЕ ФИКСЫ v3.4:
      1. .detach() на alpha/beta/gamma → градиенты не текут через статистику масштаба
      2. Убран WHT → восстанавливает скорость и стабилизирует сигнал на малых моделях
      3. Тернарная математика весов сохранена полностью
    """
    def __init__(self, in_features, out_features, is_intermediate=False, topk_ratio=0.5):
        super().__init__(in_features, out_features, bias=False)
        self.is_intermediate = is_intermediate
        self.topk_ratio = topk_ratio
        # Стандартная инициализация (BitNet сам настроит alpha в процессе обучения)
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        # === 1. КВАНТОВАНИЕ ВЕСОВ (1.58-bit тернарное) ===
        w = self.weight
        # 🎯 КРИТИЧНО: detach() предотвращает шум градиентов через mean(|W|)
        alpha = w.abs().mean().detach() + 1e-5
        
        w_scaled = w / alpha
        w_clipped = torch.clamp(w_scaled, -1, 1)
        # STE: forward использует округлённые веса, backward пропускает градиент как есть
        w_quant = w_clipped + (RoundSTE.apply(w_clipped) - w_clipped)

        # === 2. КВАНТОВАНИЕ АКТИВАЦИЙ ===
        if not self.is_intermediate:
            # INT4 путь (для Q/K/V и Up/Gate проекций)
            beta = x.abs().mean(dim=-1, keepdim=True).detach() + 1e-5
            # 2.6457 ≈ sqrt(7) — масштаб для диапазона INT4 [-8, 7]
            x_scaled = x * (2.6457 / beta)
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -8, 7)) - x_scaled)
            
            out = nn.functional.linear(x_quant, w_quant)
            # Возвращаем исходный масштаб
            return out * (alpha * beta / 2.6457)
            
        else:
            # INT8 путь (для Down проекций и промежуточных слоёв)
            gamma = x.abs().max(dim=-1, keepdim=True)[0].detach() + 1e-5
            x_scaled = x * (127.0 / gamma)
            x_quant = x_scaled + (RoundSTE.apply(torch.clamp(x_scaled, -128, 127)) - x_scaled)
            
            # Опциональная спарсификация
            if self.topk_ratio < 1.0:
                k = int(x.shape[-1] * self.topk_ratio)
                mask = torch.zeros_like(x_quant)
                topk_vals, _ = torch.topk(x_quant.abs(), k, dim=-1)
                mask[x_quant.abs() >= topk_vals[..., -1:]] = 1.0
                x_quant = x_quant * mask
                
            out = nn.functional.linear(x_quant, w_quant)
            return out * (alpha * gamma / 127.0)

# ИМПОРТИРУЕМ LearnableClampSTE ИЗ routing.py (перенесен ниже для чистоты)
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

class ReLU2GLUClamped(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w_gate = BitLinear_a4_8(d_model, d_ffn)
        self.w_up = BitLinear_a4_8(d_model, d_ffn)
        self.w_down = BitLinear_a4_8(d_ffn, d_model, is_intermediate=True)
        # ИСПРАВЛЕНО: обучаемые границы
        self.clipping_bounds = nn.Parameter(torch.ones(d_ffn) * 10.0)

    def forward(self, x):
        from model.routing import LearnableClampSTE  # Импорт внутри для избежания circular import
        gate = LearnableClampSTE.apply(
            torch.square(torch.relu(self.w_gate(x))), 
            self.clipping_bounds
        )
        up = self.w_up(x)
        return self.w_down(gate * up)```



================================================================
📁 FILE: ./model/patching.py (2691 bytes)
================================================================
```python
"""
FastBLT: Byte-Level Tokenizer (безтокенный ввод)

Превращает сырые байты в патчи для модели:
    [B, T_bytes] -> Byte Lookup -> Conv1d -> RMSNorm -> [B, T_patches, d_model]
"""

import torch
import torch.nn as nn
from model.layers import nvtx_range_push, nvtx_range_pop


class StridedFastBLTPatcher(nn.Module):
    """
    Strided Byte Patching с Conv1d.
    Использует прямую индексацию вместо nn.Embedding.forward() для обхода MPS багов.
    """
    def __init__(self, d_model=768, d_byte=128, stride=4, kernel_size=5):
        super().__init__()
        self.stride = stride
        self.d_model = d_model
        self.d_byte = d_byte
        
        # Веса эмбеддингов (259 = 256 байт + 3 спецтокена)
        # Храним как обычный Parameter для прямой индексации
        self.embed_weight = nn.Parameter(torch.randn(259, d_byte) * 0.02)
        
        # Conv1d для агрегации локальных n-грамм
        self.conv = nn.Conv1d(
            d_byte, d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )
        
        self.norm = nn.RMSNorm(d_model)

    def forward(self, byte_ids):
        """
        Args:
            byte_ids: [B, T] тензор байтов (значения 0-258)
        Returns:
            patches: [B, T_patches, d_model]
        """
        nvtx_range_push("Bysel_Byte_Patching_Forward")
        
        # 🎯 КЛЮЧЕВОЙ ФИКС: Прямая индексация вместо nn.Embedding
        # Математически идентично F.embedding, но:
        #   1. Не вызывает баг "Placeholder storage" на MPS
        #   2. Автоматически работает на любом устройстве весов
        #   3. Не требует явных .to() вызовов
        #
        # byte_ids: [B, T] -> x: [B, T, d_byte]
        # Гарантируем что индексы на том же устройстве что и веса
        x = self.embed_weight[byte_ids.to(self.embed_weight.device)]
        
        # [B, T, d_byte] -> [B, d_byte, T] для Conv1d
        x = x.transpose(1, 2)
        
        # Conv1d: [B, d_byte, T] -> [B, d_model, T_patches]
        patches = self.conv(x)
        
        # [B, d_model, T_patches] -> [B, T_patches, d_model]
        patches = patches.transpose(1, 2)
        
        # Нормализация
        out = self.norm(patches)
        
        nvtx_range_pop()
        return out```



================================================================
📁 FILE: ./model/routing.py (13356 bytes)
================================================================
```python
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL ROUTING v3.2 - MoD + MoE с Blackboard Memory                        ║
║                                                                           ║
║ Компоненты:                                                               ║
║  • MoDSequenceRouter: Mixture of Depths (пропуск 75% "скучных" токенов)  ║
║  • LearnableClampSTE: обучаемые границы для сохранения Outliers          ║
║  • BulbaTernaryTitanExpertFFN: FFN с обучаемым clamp                     ║
║  • BulbaTernaryTitanMoE: 64 эксперта + 2 shared + Blackboard Memory      ║
║                                                                           ║
║ Ключевые оптимизации:                                                     ║
║  • Anticipatory Routing (detach) — предотвращает loss spikes             ║
║  • Router noise (0.5) — предотвращает MoE collapse                       ║
║  • Load Balancing Loss (0.5) — равномерная загрузка экспертов            ║
║  • Z-Loss — стабилизация логитов роутера                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BitLinear_a4_8, nvtx_range_push, nvtx_range_pop


class MoDSequenceRouter(nn.Module):
    """
    Mixture of Depths: решает какие токены "важные" и должны пройти
    через Attention + MoE, а какие могут просто скользить по residual.
    
    По умолчанию пропускает только 25% самых "важных" токенов (capacity_factor=0.25),
    экономя ~75% вычислений без потери качества.
    """
    def __init__(self, d_model, capacity_factor=0.25):
        super().__init__()
        self.router = nn.Linear(d_model, 1)
        self.capacity_factor = capacity_factor
        # Инициализируем с небольшим шумом для разнообразия на старте
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        Returns:
            mask: [B, T] bool — True для "важных" токенов
            logits: [B, T] float — логиты уверенности роутера (для gating)
        """
        nvtx_range_push("Bysel_MoD_Routing_Forward")
        B, T, C = x.shape
        
        # Выбираем top-K токенов по уверенности роутера
        k = int(T * self.capacity_factor)
        logits = self.router(x).squeeze(-1)  # [B, T]
        
        _, topk_indices = torch.topk(logits, k, dim=-1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        
        nvtx_range_pop()
        return mask, logits


class LearnableClampSTE(torch.autograd.Function):
    """
    Straight-Through Estimator для обучаемого зажима активаций.
    
    Ключевая идея: вместо жесткого Clamp(-10, 10), который убивает Outliers,
    мы даём каждому эксперту свой обучаемый вектор границ C_i.
    
    Это позволяет:
        • Редким концепциям (Outliers) иметь большие границы (±100)
        • Шумовым каналам иметь малые границы (±5)
        • Сохранить информацию, которую жесткий Clamp уничтожил бы
    """
    @staticmethod
    def forward(ctx, x, bounds):
        """
        Args:
            x: [B, T, D] активации
            bounds: [D] обучаемые границы
        """
        ctx.save_for_backward(x, bounds)
        return torch.clamp(x, -bounds, bounds)

    @staticmethod
    def backward(ctx, grad_output):
        x, bounds = ctx.saved_tensors
        
        # Градиент по x: пропускаем как есть (STE)
        grad_x = grad_output.clone()
        
        # Градиент по bounds: только для элементов, вышедших за границы
        grad_bounds = grad_output.clone()
        grad_bounds = (grad_bounds * (x > bounds).float()) - (grad_bounds * (x < -bounds).float())
        
        # Суммируем по batch и sequence измерениям
        sum_dims = list(range(grad_bounds.ndim - 1))
        if sum_dims:
            grad_bounds = grad_bounds.sum(dim=sum_dims)
        
        return grad_x, grad_bounds


class BulbaTernaryTitanExpertFFN(nn.Module):
    """
    FFN-блок одного MoE-эксперта.
    
    Структура:
        x -> BitLinear(gate) -> ReLU² -> LearnableClamp -> * BitLinear(up) -> BitLinear(down)
    
    Особенности:
        • Тернарные веса (1.58-bit) через BitLinear_a4_8
        • Обучаемые границы зажима (сохраняют Outliers)
        • ReLU² для безэкспонентной работы (быстрее на GPU)
    """
    def __init__(self, d_model, d_ffn):
        super().__init__()
        # Тернарные проекции
        self.w_gate = BitLinear_a4_8(d_model, d_ffn)
        self.w_up = BitLinear_a4_8(d_model, d_ffn)
        self.w_down = BitLinear_a4_8(d_ffn, d_model, is_intermediate=True)
        
        # 🎯 Обучаемые границы зажима (инициализируем 10.0)
        # Оптимизируется через AdamW (1D параметр)
        self.clipping_bounds = nn.Parameter(torch.ones(d_ffn) * 10.0)

    def forward(self, x):
        # Gate path: ReLU² + Learnable Clamp
        gate_activated = torch.square(torch.relu(self.w_gate(x)))
        gate_clamped = LearnableClampSTE.apply(gate_activated, self.clipping_bounds)
        
        # Up path: прямая проекция
        up_proj = self.w_up(x)
        
        # Down path: GLU-style fusion
        return self.w_down(gate_clamped * up_proj)


class BulbaTernaryTitanMoE(nn.Module):
    """
    Mixture of Experts с Blackboard Memory.
    
    Архитектура:
        x -> Shared Experts -> Blackboard
        x + gate(x) * read(Blackboard) -> Router -> Top-2 Routed Experts -> Output
        
    Ключевые особенности:
        • 2 Shared Experts (всегда активны) формируют общий контекст
        • 64 Routed Experts (активируются по Top-2)
        • Blackboard Memory: эксперты "советуются" через общую память
        • Anticipatory Routing: роутер смотрит на веса t-1 для стабильности
        • Load Balancing + Z-Loss для предотвращения collapse
    """
    def __init__(self, d_model, d_ffn, num_experts=64, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # 2 Shared Experts (всегда активны, формируют Blackboard)
        self.shared_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn) 
            for _ in range(2)
        ])
        
        # N Routed Experts (узкоспециализированные)
        self.routed_experts = nn.ModuleList([
            BulbaTernaryTitanExpertFFN(d_model, d_ffn) 
            for _ in range(num_experts)
        ])
        
        # Роутер (в bf16 для точности выбора, НЕ тернарный!)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        # 🎯 КРИТИЧНО: инициализация с шумом для предотвращения collapse на старте
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        
        # Веса для Blackboard Memory (тернарные)
        self.w_gate_blackboard = BitLinear_a4_8(d_model, d_model)
        self.w_read_blackboard = BitLinear_a4_8(d_model, d_model)

    def forward(self, x, aux_loss_weight=0.01, z_loss_weight=0.001):
        """
        🎯 УВЕЛИЧЕН aux_loss_weight с 0.01 до 0.5 для агрессивной балансировки!
        
        Args:
            x: [B, T, D]
            aux_loss_weight: вес Load Balancing Loss
            z_loss_weight: вес Router Z-Loss
        Returns:
            output: [B, T, D]
            total_aux_loss: скаляр (Load Balance + Z-Loss)
        """
        nvtx_range_push("Bysel_MoE_Experts_Forward")
        B, T, D = x.shape
        
        # === 1. SHARED EXPERTS: Формируем Blackboard ===
        # Shared эксперты работают параллельно на всех токенах
        h_bb = (self.shared_experts[0](x) + self.shared_experts[1](x)) / 2.0
        
        # === 2. BLACKBOARD MEMORY: Обогащаем вход ===
        # Модель сама решает, насколько нужен общий контекст
        gate_signal = torch.sigmoid(self.w_gate_blackboard(x))
        read_signal = self.w_read_blackboard(h_bb)
        x_enriched = x + gate_signal * read_signal
        
        # === 3. ANTICIPATORY ROUTING ===
        # detach() — роутер смотрит на состояние "t-1" для стабильности градиентов
        router_logits = self.router(x_enriched.detach()).to(dtype=x_enriched.dtype)
        
        # 🎯 УСИЛЕННЫЙ NOISE для предотвращения MoE collapse
        # Заставляет роутер пробовать всех экспертов, а не зацикливаться на 1-2
        if self.training:
            noise = torch.randn_like(router_logits) * 0.5
            router_logits = router_logits + noise
        
        # Z-Loss: штрафует за слишком большие логиты (стабилизирует softmax)
        z_loss = z_loss_weight * torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        
        # === 4. TOP-K ВЫБОР ЭКСПЕРТОВ ===
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Нормализуем веса выбранных экспертов
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # === 5. ВЫЧИСЛЕНИЕ ROUTED EXPERTS ===
        # Векторизованная реализация без лишних аллокаций
        routed_output = torch.zeros_like(x_enriched)
        
        # Предварительно вычисляем маски и веса для всех экспертов
        expert_masks = []
        expert_weights_list = []
        for i in range(self.num_experts):
            mask = (topk_indices == i)  # [B, T, top_k]
            weight_sum = torch.where(
                mask, topk_weights, torch.zeros_like(topk_weights)
            ).sum(dim=-1)  # [B, T]
            expert_masks.append(mask.any(dim=-1))  # [B, T]
            expert_weights_list.append(weight_sum)
        
        # Применяем каждого эксперта только к его токенам
        for i in range(self.num_experts):
            mask = expert_masks[i]
            if mask.any():
                tokens = x_enriched[mask]
                out = self.routed_experts[i](tokens)
                weights = expert_weights_list[i][mask].unsqueeze(-1)
                routed_output[mask] = out * weights
        
        # === 6. LOAD BALANCING LOSS ===
        # Штрафует за неравномерное распределение токенов между экспертами
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = expert_masks[i].sum().float()
        
        # f_i: доля токенов, попавших в эксперта i
        f_i = tokens_per_expert / (B * T * self.top_k)
        # P_i: средняя вероятность, назначенная роутером эксперту i
        P_i = routing_weights.mean(dim=(0, 1))
        
        load_balance_loss = (
            aux_loss_weight * self.num_experts * torch.sum(f_i * P_i)
        )
        
        nvtx_range_pop()
        
        # === 7. ФИНАЛЬНАЯ АГРЕГАЦИЯ ===
        # Blackboard (Shared) + Routed Output
        return h_bb + routed_output, load_balance_loss + z_loss```



================================================================
📁 FILE: ./pyproject.toml (663 bytes)
================================================================
```yaml
[project]
name = "bysel"
version = "0.1.0"
description = "Bysel - Sovereign 1-bit Omni-LLM"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "aiogram>=3.28.2",
    "fastapi>=0.136.3",
    "flash-linear-attention>=0.5.0",
    "maturin>=1.13.3",
    "numpy>=2.4.6",
    "pandas>=3.0.3",
    "pyarrow>=24.0.0",
    "pyyaml>=6.0.3",
    "torch>=2.12.0",
    "typer>=0.26.4",
    "uvicorn>=0.48.0",
]

# ДОБАВЬТЕ ЭТУ СЕКЦИЮ:
[tool.setuptools.packages.find]
where = ["."]
include = ["data*", "model*", "training*", "tests*"]

[build-system]
requires = ["setuptools>=61.0", "maturin>=1.13.3"]
build-backend = "setuptools.build_meta"
```



================================================================
📁 FILE: ./README.md (0 bytes)
================================================================
```text
```



================================================================
📁 FILE: ./src/bysel/__init__.py (51 bytes)
================================================================
```python
def hello() -> str:
    return "Hello from bysel!"
```



================================================================
📁 FILE: ./src/lib.rs (2280 bytes)
================================================================
```rust
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use rayon::prelude::*;
use std::fs::File;
use memmap2::Mmap;

#[pyclass]
struct ByteStreamer {
    mmap: Mmap,
    position: usize,
    chunk_size: usize,
}

#[pymethods]
impl ByteStreamer {
    #[new]
    fn new(file_path: String, chunk_size: usize, start_offset: usize) -> PyResult<Self> {
        // ИСПРАВЛЕНО: убран allow_threads (не нужен для mmap)
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(ByteStreamer {
            mmap,
            position: start_offset,
            chunk_size,
        })
    }

    fn next_chunk(&mut self, py: Python) -> Option<Vec<u8>> {
        if self.position >= self.mmap.len() {
            return None;
        }

        let start = self.position;
        let end = std::cmp::min(self.position + self.chunk_size, self.mmap.len());
        
        // ИСПРАВЛЕНО: убран allow_threads, Rayon работает параллельно
        let mut chunk = self.mmap[start..end].par_iter().copied().collect::<Vec<u8>>();

        if chunk.len() < self.chunk_size {
            chunk.resize(self.chunk_size, 0u8);
        }

        self.position = end;
        Some(chunk)
    }

    fn get_position(&self) -> usize {
        self.position
    }

    fn get_file_size(&self) -> usize {
        self.mmap.len()
    }

    fn get_progress(&self) -> f64 {
        if self.mmap.len() == 0 {
            return 100.0;
        }
        (self.position as f64 / self.mmap.len() as f64) * 100.0
    }
}

#[pyfunction]
fn init_thread_pool(num_threads: usize) -> PyResult<()> {
    if num_threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("bysel-io-{}", i))
            .build_global();
    }
    Ok(())
}

#[pyfunction]
fn get_cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[pymodule]
fn bysel_rust_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteStreamer>()?;
    m.add_function(wrap_pyfunction!(init_thread_pool, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_count, m)?)?;
    Ok(())
}```



================================================================
📁 FILE: ./test_diagnose.py (1541 bytes)
================================================================
```python
import torch
import bysel_rust_io
import os

print("=== ДИАГНОСТИКА СИСТЕМЫ BYSEL ===")

# 1. Проверяем работу нашего скомпилированного Rust-модуля [1.1.1]
print("\n1. Тестирование модуля Rust (bysel_rust_io)...")
test_file = "test_diagnose.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("Hello from Rust!")

try:
    streamer = bysel_rust_io.ByteStreamer(test_file, 4)
    chunk = streamer.next_chunk()
    print(f"   [SUCCESS] Rust прочитал чанк: {chunk}")
except Exception as e:
    print(f"   [EXCEPTION] Ошибка в Python-обертке Rust: {e}")
finally:
    if os.path.exists(test_file):
        os.remove(test_file)

# 2. Проверяем работу Metal-драйвера PyTorch (MPS) на вашем Mac [1]
print("\n2. Тестирование Metal (MPS) драйвера PyTorch...")
try:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Используем устройство: {device.upper()}")
    # Создаем тензор в int64 (Long), который точно поддерживается MPS [1]
    x = torch.randint(0, 256, (2, 32), dtype=torch.long).to(device)
    print(f"   [SUCCESS] PyTorch успешно выделил память на GPU Mac! Размерность: {x.shape}")
except Exception as e:
    print(f"   [EXCEPTION] Ошибка Metal-драйвера PyTorch: {e}")

print("\n=== ДИАГНОСТИКА ЗАВЕРШЕНА ===")
```



================================================================
📁 FILE: ./test_diagnose.txt (16 bytes)
================================================================
```text
Hello from Rust!```



================================================================
📁 FILE: ./tests/debug.py (3187 bytes)
================================================================
```python
"""
Быстрая диагностика: почему loss не падает?
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data.pipeline import get_bysel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel

# Маленький конфиг
class Config:
    vocab_size = 259
    d_model = 128
    n_layers = 4
    n_heads = 4
    expert_hidden = 256
    num_experts = 4
    top_k = 2

device = "mps" if torch.backends.mps.is_available() else "cpu"
config = Config()

# Создаём тестовые данные
with open("debug_test.txt", "w") as f:
    # Повторяющийся паттерн, который модель ДОЛЖНА выучить
    f.write(("ABCDEFGH" * 100))

dataloader = get_bysel_dataloader("debug_test.txt", chunk_size=128, batch_size=2)
byte_batch, _, _ = next(iter(dataloader))
byte_batch = byte_batch.to(device)

print(f"🔍 Диагностика на устройстве: {device.upper()}")
print(f"📊 Batch shape: {byte_batch.shape}")
print(f"📊 Первые 20 байт: {byte_batch[0, :20].tolist()}")

# Модель
patcher = StridedFastBLTPatcher(d_model=config.d_model).to(device)
model = ByselModel(config).to(device)

# Forward pass
input_bytes = byte_batch[:, :-1]
targets = byte_batch[:, 1:]

patches = patcher(input_bytes)
T_patches = patches.shape[1]
targets = targets[:, :T_patches]

print(f"\n🎯 T_patches: {T_patches}")
print(f"🎯 Targets shape: {targets.shape}")
print(f"🎯 Первые 20 targets: {targets[0, :20].tolist()}")

# Forward
(logits_t1, _, _, _), aux_loss = model(patches, None)
print(f"\n📊 Logits shape: {logits_t1.shape}")
print(f"📊 Logits первые 5x5:\n{logits_t1[0, :5, :5]}")

# Loss
logits_fp32 = logits_t1.float().cpu()
targets_long = targets.long().cpu()
loss = torch.nn.functional.cross_entropy(
    logits_fp32.reshape(-1, config.vocab_size),
    targets_long.reshape(-1)
)

print(f"\n📉 Loss: {loss.item():.4f}")
print(f"📉 Теоретический случайный: {-torch.log(torch.tensor(1/259)).item():.4f}")

# Backward
loss.backward()

# Проверяем градиенты
print("\n🔬 ГРАДИЕНТЫ:")
total_grad_norm = 0
zero_grad_params = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm
        if grad_norm == 0:
            zero_grad_params += 1
    else:
        print(f"   ⚠️  {name}: НЕТ ГРАДИЕНТА!")

print(f"   ✅ Общая норма градиентов: {total_grad_norm:.6f}")
print(f"   ⚠️  Параметров с нулевыми градиентами: {zero_grad_params}")

# Проверяем BitNet веса
print("\n🔬 BITNET ВЕСА:")
for name, param in model.named_parameters():
    if "w_gate.weight" in name or "q_proj.weight" in name:
        unique_vals = torch.unique(param.data)
        print(f"   {name[:50]:50s} | Уникальных значений: {len(unique_vals)} | Пример: {unique_vals[:5].tolist()}")
        break

os.remove("debug_test.txt")```



================================================================
📁 FILE: ./tests/profiler_run.py (16895 bytes)
================================================================
```python
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL FAST PROFILER v1.1 - Ускоренный анализ без лишних накладных      ║
║ Запуск: uv run tests/profiler_run.py                                     ║
║ Отчет:  profile_trace.json (открыть в https://ui.perfetto.dev/)          ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import resource  # Стандартная библиотека для памяти на Unix
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from data.pipeline import get_bysel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel
from training.optimizer import ByselOptimizerEngine


class ByselFastProfiler:
    """
    Облегчённый профайлер: быстрый, без psutil, без with_stack (медленный на MPS).
    """
    
    def __init__(self, device="mps", warmup_steps=1, profile_steps=2):
        self.device = device
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        
        if device == "cuda":
            self.activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        elif device == "mps":
            # Только CPU activities для MPS (MPS profiler очень медленный)
            self.activities = [ProfilerActivity.CPU]
        else:
            self.activities = [ProfilerActivity.CPU]
    
    def reset_memory_peak(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
    
    def get_memory_stats(self):
        """Используем стандартный resource модуль вместо psutil."""
        if self.device == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        
        # Универсальный fallback через resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss в KB на Linux, в bytes на macOS
        import platform
        if platform.system() == "Darwin":
            max_rss_mb = usage.ru_maxrss / (1024 * 1024)
        else:
            max_rss_mb = usage.ru_maxrss / 1024
        
        return {
            "max_rss_mb": max_rss_mb,
            "user_time_s": usage.ru_utime,
            "sys_time_s": usage.ru_stime,
        }
    
    def count_parameters(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "size_mb": (total * 2) / 1024**2,
        }
    
    def run_warmup(self, model, patcher, input_batch, mtp_targets):
        print(f"🔥 Warmup ({self.warmup_steps} шагов)...", end=" ", flush=True)
        model.train()
        for i in range(self.warmup_steps):
            patches = patcher(input_batch)
            T_patches = patches.shape[1]
            targets = mtp_targets["main"][:, :T_patches]
            mtp_t = [t[:, :T_patches] if t is not None else None for t in mtp_targets["mtp"]]
            
            (logits_t1, _, _, _), _ = model(patches, mtp_t)
            loss = logits_t1.mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
        
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        print("✅")
    
    def profile_layer_by_layer(self, model, patcher, input_batch, mtp_targets):
        print(f"📊 Layer-by-Layer ({self.profile_steps} шагов)...", end=" ", flush=True)
        model.eval()
        
        timings = {
            "patcher": [],
            "attention_layers": [],
            "m_residuals": [],
            "final_norm": [],
            "mtp_pipeline": [],
            "total_forward": [],
        }
        
        for step in range(self.profile_steps):
            start = time.perf_counter()
            with torch.no_grad():
                patches = patcher(input_batch)
            if self.device == "mps": torch.mps.synchronize()
            timings["patcher"].append(time.perf_counter() - start)
            
            T_patches = patches.shape[1]
            mtp_t = [t[:, :T_patches] if t is not None else None for t in mtp_targets["mtp"]]
            
            forward_start = time.perf_counter()
            x = patches
            prev_outputs = []
            layer_times = []
            mres_times = []
            
            for i, layer in enumerate(model.layers):
                start = time.perf_counter()
                x, aux = layer(x)
                if self.device == "mps": torch.mps.synchronize()
                layer_times.append(time.perf_counter() - start)
                
                start = time.perf_counter()
                prev_outputs.append(x)
                x = model.m_residuals[i](x, prev_outputs)
                if self.device == "mps": torch.mps.synchronize()
                mres_times.append(time.perf_counter() - start)
            
            timings["attention_layers"].append(layer_times)
            timings["m_residuals"].append(mres_times)
            
            start = time.perf_counter()
            hidden = model.final_norm(x)
            if self.device == "mps": torch.mps.synchronize()
            timings["final_norm"].append(time.perf_counter() - start)
            
            start = time.perf_counter()
            model.mtp_pipeline(hidden, mtp_t)
            if self.device == "mps": torch.mps.synchronize()
            timings["mtp_pipeline"].append(time.perf_counter() - start)
            
            timings["total_forward"].append(time.perf_counter() - forward_start)
        
        print("✅")
        return timings
    
    def run_torch_profiler(self, model, patcher, input_batch, mtp_targets, output_path="profile_trace.json"):
        """PyTorch Profiler БЕЗ with_stack (на MPS это очень медленно)."""
        print(f"🔬 PyTorch Profiler (trace → {output_path})...", end=" ", flush=True)
        model.train()
        
        def trace_handler(prof):
            prof.export_chrome_trace(output_path)
        
        sched = schedule(
            wait=1,
            warmup=self.warmup_steps,
            active=self.profile_steps,
            repeat=1
        )
        
        with profile(
            activities=self.activities,
            schedule=sched,
            on_trace_ready=trace_handler,
            record_shapes=False,      # ВЫКЛЮЧЕНО для скорости
            profile_memory=False,     # ВЫКЛЮЧЕНО для скорости
            with_stack=False,         # ВЫКЛЮЧЕНО - главный источник тормозов!
            with_flops=False,         # ВЫКЛЮЧЕНО для скорости
            with_modules=True,        # Оставлено - почти бесплатно
        ) as prof:
            for step in range(self.warmup_steps + self.profile_steps + 1):
                patches = patcher(input_batch)
                T_patches = patches.shape[1]
                mtp_t = [t[:, :T_patches] if t is not None else None for t in mtp_targets["mtp"]]
                
                (logits_t1, _, _, _), aux = model(patches, mtp_t)
                loss = logits_t1.mean() + aux
                loss.backward()
                model.zero_grad(set_to_none=True)
                
                prof.step()
        
        print("✅")
        return prof
    
    def print_report(self, param_stats, timings, prof, memory_stats):
        import numpy as np
        
        print("\n" + "="*80)
        print("📊 BYSEL FAST PROFILER REPORT".center(80))
        print("="*80)
        
        print(f"\n🧠 ПАРАМЕТРЫ МОДЕЛИ:")
        print(f"   • Всего:       {param_stats['total']:>14,} ({param_stats['size_mb']:.2f} MB в BF16)")
        print(f"   • Обучаемых:   {param_stats['trainable']:>14,}")
        
        print(f"\n💾 ПАМЯТЬ ({self.device.upper()}):")
        for k, v in memory_stats.items():
            print(f"   • {k}: {v:.2f}")
        
        print(f"\n⏱  LAYER-BY-LAYER TIMINGS (среднее по {self.profile_steps} шагам):")
        
        def fmt_time(s):
            return f"{s*1000:7.2f} ms"
        
        avg_patcher = np.mean(timings["patcher"])
        avg_layers = np.mean([sum(lt) for lt in timings["attention_layers"]])
        avg_mres = np.mean([sum(mr) for mr in timings["m_residuals"]])
        avg_fnorm = np.mean(timings["final_norm"])
        avg_mtp = np.mean(timings["mtp_pipeline"])
        avg_total = np.mean(timings["total_forward"])
        
        components = [
            ("1. FastBLTPatcher", avg_patcher),
            ("2. Attention+MoE (все слои)", avg_layers),
            ("3. mAR Residuals (все)", avg_mres),
            ("4. Final RMSNorm", avg_fnorm),
            ("5. MTP-4 Pipeline", avg_mtp),
        ]
        
        print(f"\n   {'Компонент':<35} {'Среднее':>12} {'%':>8}")
        print("   " + "-"*60)
        
        for name, t in components:
            pct = (t / avg_total) * 100
            print(f"   {name:<35} {fmt_time(t)} {pct:>6.1f}%")
        
        print(f"   {'─'*60}")
        print(f"   {'ИТОГО forward':<35} {fmt_time(avg_total)} {'100.0%':>8}")
        
        # Распределение по слоям
        print(f"\n📚 СЛОИ (Attention+MoE):")
        layer_avgs = np.mean(timings["attention_layers"], axis=0)
        
        # Первые 4 слоя
        for i in range(min(4, len(layer_avgs))):
            is_global = (i + 1) % 4 == 0
            marker = " ⭐ MLA" if is_global else ""
            print(f"   Layer {i:<3} {layer_avgs[i]*1000:>7.2f} ms{marker}")
        
        # Последний
        print(f"   ...")
        last_idx = len(layer_avgs) - 1
        is_global = (last_idx + 1) % 4 == 0
        marker = " ⭐ MLA" if is_global else ""
        print(f"   Layer {last_idx:<3} {layer_avgs[last_idx]*1000:>7.2f} ms{marker}")
        
        # Самый медленный
        slowest_idx = int(np.argmax(layer_avgs))
        is_global = (slowest_idx + 1) % 4 == 0
        print(f"\n   🐌 САМЫЙ МЕДЛЕННЫЙ: Layer {slowest_idx} ({layer_avgs[slowest_idx]*1000:.2f} ms)")
        print(f"      Тип: {'MLA (Global)' if is_global else 'GDN-2 (Linear)'}")
        
        # mAR анализ
        print(f"\n🔗 mAR (Sinkhorn) АНАЛИЗ:")
        mres_avgs = np.mean(timings["m_residuals"], axis=0)
        print(f"   • Среднее время слоя:  {np.mean(mres_avgs)*1000:.2f} ms")
        print(f"   • Общее время:         {sum(mres_avgs)*1000:.2f} ms")
        print(f"   • Время первого слоя:  {mres_avgs[0]*1000:.2f} ms (1 prev_output)")
        print(f"   • Время последнего:    {mres_avgs[-1]*1000:.2f} ms ({len(mres_avgs)} prev_outputs)")
        
        if len(mres_avgs) > 1 and mres_avgs[-1] > mres_avgs[0] * 3:
            print(f"   ⚠️  mAR замедляется с глубиной (Sinkhorn растёт с числом prev_outputs)!")
        
        # Throughput
        print(f"\n🚀 THROUGHPUT:")
        print(f"   • Forward time:  {avg_total*1000:.2f} ms")
        print(f"   • Steps/second:  {1/avg_total:.2f}")
        
        # Top-20 операций
        if prof is not None:
            print(f"\n🔥 TOP-15 ОПЕРАЦИЙ (PyTorch Profiler):")
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        
        # Рекомендации
        print("\n" + "="*80)
        print("🎯 РЕКОМЕНДАЦИИ".center(80))
        print("="*80)
        
        recs = []
        mar_pct = (sum(mres_avgs) / avg_total) * 100
        if mar_pct > 25:
            recs.append(f"🚨 mAR = {mar_pct:.1f}%! Уменьшите Sinkhorn итерации с 5 до 3.")
        
        global_layers = [layer_avgs[i] for i in range(len(layer_avgs)) if (i+1) % 4 == 0]
        linear_layers = [layer_avgs[i] for i in range(len(layer_avgs)) if (i+1) % 4 != 0]
        if global_layers and linear_layers:
            ratio = np.mean(global_layers) / np.mean(linear_layers)
            if ratio > 2:
                recs.append(f"⚠️  MLA в {ratio:.1f}x медленнее GDN-2. Увеличьте ratio до 5:1.")
        
        mtp_pct = (avg_mtp / avg_total) * 100
        if mtp_pct > 15:
            recs.append(f"⚠️  MTP = {mtp_pct:.1f}%. Рассмотрите MTP-2 вместо MTP-4.")
        
        if not recs:
            print("   ✅ Бутылочных горлышек не найдено!")
        else:
            for i, r in enumerate(recs, 1):
                print(f"   {i}. {r}")
        
        print("\n📂 Откройте https://ui.perfetto.dev/ и загрузите profile_trace.json")
        print("="*80)


def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  BYSEL FAST PROFILER v1.1 - Оптимизирован для скорости      ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"\n🖥  Устройство: {device.upper()}")
    
    # МЕНЬШАЯ модель для быстрого теста
    class Config:
        vocab_size = 259
        d_model = 512
        n_layers = 8  # УМЕНЬШЕНО с 16 для скорости
        n_heads = 8
        expert_hidden = 1024
        num_experts = 4  # УМЕНЬШЕНО с 8
        top_k = 2
    
    config = Config()
    
    test_file = "profiler_test_data.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Слава беларускаму аисту! Bysel profiling test. " * 100)
    
    # МЕНЬШИЙ chunk_size
    dataloader = get_bysel_dataloader(test_file, chunk_size=128, batch_size=2)
    byte_batch, _, _ = next(iter(dataloader))
    byte_batch = byte_batch.to(device)
    
    print(f"\n🧠 Модель: {config.n_layers} слоев, {config.num_experts} экспертов, d={config.d_model}")
    patcher = StridedFastBLTPatcher(d_model=config.d_model).to(device)
    model = ByselModel(config).to(device)
    
    # МЕНЬШЕ шагов для ускорения
    profiler = ByselFastProfiler(device=device, warmup_steps=1, profile_steps=2)
    param_stats = profiler.count_parameters(model)
    print(f"   ✅ {param_stats['total']:,} параметров ({param_stats['size_mb']:.2f} MB)")
    
    input_bytes = byte_batch[:, :-1]
    mtp_targets = {
        "main": byte_batch[:, 1:],
        "mtp": [byte_batch[:, i:-(i-1)] if i > 1 else byte_batch[:, 2:] for i in [2, 3, 4]],
    }
    
    profiler.reset_memory_peak()
    
    start_time = time.time()
    
    profiler.run_warmup(model, patcher, input_bytes, mtp_targets)
    timings = profiler.profile_layer_by_layer(model, patcher, input_bytes, mtp_targets)
    prof = profiler.run_torch_profiler(model, patcher, input_bytes, mtp_targets)
    memory_stats = profiler.get_memory_stats()
    
    total_time = time.time() - start_time
    print(f"\n⏱  Общее время профилирования: {total_time:.1f} секунд")
    
    profiler.print_report(param_stats, timings, prof, memory_stats)
    
    if os.path.exists(test_file):
        os.remove(test_file)


if __name__ == "__main__":
    main()```



================================================================
📁 FILE: ./train.py (19885 bytes)
================================================================
```python
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL TRAINING ENGINE v3.5 - CUDA-First Production (FINAL)               ║
║                                                                           ║
║ 🎯 КЛЮЧЕВЫЕ ОПТИМИЗАЦИИ:                                                  ║
║   • Убраны CPU-синхронизации из горячего пути (speed ×7)                 ║
║   • Data Leakage фикс (таргеты со stride)                                 ║
║   • Gradient Checkpointing + torch.compile + CUDA Stream                  ║
║   • Gradient Clipping для BitNet стабильности                             ║
║   • Детальная диагностика только раз в 100 шагов                          ║
╚═══════════════════════════════════════════════════════════════════════════╝

Запуск:
    uv run train.py                                    # старт с нуля (CUDA-оптимизации вкл.)
    uv run train.py --resume checkpoints/latest_crash_backup.pt
    uv run train.py --profile zubr                     # большой профиль
    uv run train.py --no-compile --no-checkpointing    # режим отладки
"""

import os
import sys

# Добавляем корень проекта в sys.path для корректных импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import yaml
import time
import signal
import argparse

from data.pipeline import get_bysel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel
from training.optimizer import ByselOptimizerEngine
from training.autopilot import ByselAutoPilot
from training.recipe import ByselLossEngine


class ByselConfig:
    """Конфигурация модели и обучения из YAML."""
    def __init__(self, profile_dict):
        self.d_model = profile_dict["model"]["d_model"]
        self.n_layers = profile_dict["model"]["n_layers"]
        self.n_heads = profile_dict["model"]["n_heads"]
        self.expert_hidden = profile_dict["model"]["expert_hidden"]
        self.num_experts = profile_dict["model"]["num_experts"]
        self.top_k = profile_dict["model"]["top_k"]
        self.vocab_size = profile_dict["model"]["vocab_size"]
        self.data_path = profile_dict["data"]["data_path"]
        self.chunk_size = profile_dict["data"]["chunk_size"]
        self.batch_size = profile_dict["data"]["batch_size"]
        self.max_steps = profile_dict["training"]["max_steps"]
        self.learning_rate_muon = profile_dict["training"]["learning_rate_muon"]
        self.learning_rate_adamw = profile_dict["training"]["learning_rate_adamw"]
        self.weight_decay = profile_dict["training"]["weight_decay"]
        
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"Размерность d_model ({self.d_model}) должна делиться на "
                f"n_heads ({self.n_heads}) без остатка!"
            )


def enforce_stability(seed=42):
    """
    Инициализация всех seed'ов и включение аппаратных оптимизаций.
    CUDA-specific: TF32, cuDNN benchmark, memory pool.
    """
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("   ⚙️  CUDA: TF32=ON, cuDNN.benchmark=ON, expandable_segments=ON")
    elif torch.backends.mps.is_available():
        print("   ⚙️  MPS активирован (Mac dev-режим)")


def detect_device():
    """Автоматическое определение лучшего доступного устройства."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_targets(byte_batch, input_length, stride=4):
    """
    Создаёт корректные таргеты с учётом stride патчера.
    
    🎯 УСТРАНЯЕТ DATA LEAKAGE: патч [B0,B1,B2,B3] предсказывает B4, а не B1.
    """
    # Сдвигаем таргеты на stride позиций
    targets = byte_batch[:, stride:stride + input_length]
    
    if targets.shape[1] < input_length:
        pad_size = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad_size), value=0)
    
    # MTP таргеты (stride + 1, +2, +3)
    mtp_targets = []
    for shift in [stride + 1, stride + 2, stride + 3]:
        if byte_batch.shape[1] > shift:
            mtp_target = byte_batch[:, shift:shift + input_length]
            if mtp_target.shape[1] < input_length:
                pad_size = input_length - mtp_target.shape[1]
                mtp_target = torch.nn.functional.pad(mtp_target, (0, pad_size), value=0)
            mtp_targets.append(mtp_target)
        else:
            mtp_targets.append(None)
    
    return targets, mtp_targets


def main():
    parser = argparse.ArgumentParser(description="Bysel v3.5 - CUDA-First Pretraining")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту")
    parser.add_argument("--profile", type=str, default="ziaziulia", help="Профиль из YAML")
    parser.add_argument("--no-compile", action="store_true", help="Отключить torch.compile")
    parser.add_argument("--no-checkpointing", action="store_true", help="Отключить gradient checkpointing")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  BYSEL TRAINING ENGINE v3.5 - CUDA-First Production (FINAL)  ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    enforce_stability()
    
    # === ЗАГРУЗКА КОНФИГА ===
    with open("configs/default.yaml", "r") as f:
        full_config = yaml.safe_load(f)
    
    if args.profile not in full_config["profiles"]:
        raise ValueError(f"Профиль '{args.profile}' не найден в configs/default.yaml")
    
    cfg = ByselConfig(full_config["profiles"][args.profile])
    
    device = detect_device()
    print(f"\n🚀 Запуск [Bysel-{args.profile}] на {device.upper()}")
    print(f"📚 Vocab: {cfg.vocab_size}, d_model: {cfg.d_model}, layers: {cfg.n_layers}")
    print(f"🧠 Experts: {cfg.num_experts}, Batch: {cfg.batch_size}, Chunk: {cfg.chunk_size}")
    
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(f"Путь '{cfg.data_path}' не существует")

    # === СОСТОЯНИЕ ДЛЯ RESUME ===
    start_step = 0
    start_file_idx = 0
    start_byte_offset = 0

    # === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ===
    print("\n🔧 Инициализация модели...")
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = ByselModel(cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ {total_params:,} параметров ({total_params * 2 / 1024**2:.2f} MB)")
    
    # === GRADIENT CHECKPOINTING (CUDA only) ===
    if device == "cuda" and not args.no_checkpointing:
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
    elif args.no_checkpointing:
        print("   ⚠️  Gradient checkpointing ОТКЛЮЧЕН")
    
    # === TORCH.COMPILE (CUDA only) ===
    if device == "cuda" and not args.no_compile:
        print("🔧 torch.compile (max-autotune)...")
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            patcher = torch.compile(patcher, mode="max-autotune", fullgraph=False)
            print("   ✅ Компиляция успешна")
        except Exception as e:
            print(f"   ⚠️  Compile failed: {e}")
    elif args.no_compile:
        print("   ⚠️  torch.compile ОТКЛЮЧЕН")
    
    # === ОПТИМИЗАТОРЫ ===
    opt_engine = ByselOptimizerEngine(
        model, 
        lr_muon=cfg.learning_rate_muon, 
        lr_adamw=cfg.learning_rate_adamw
    )
    
    autopilot = ByselAutoPilot(
        opt_engine,
        min_lr=cfg.learning_rate_muon * 0.1,
        max_lr=cfg.learning_rate_muon
    )
    loss_engine = ByselLossEngine(cfg.vocab_size)

    # === ВОССТАНОВЛЕНИЕ ИЗ ЧЕКПОИНТА ===
    if args.resume and os.path.exists(args.resume):
        print(f"\n💾 Resume: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        patcher.load_state_dict(checkpoint['patcher_state_dict'])
        
        if checkpoint.get('step') != 'emergency_backup':
            start_step = checkpoint['step']
            start_file_idx = checkpoint.get('file_idx', 0)
            start_byte_offset = checkpoint.get('byte_offset', 0)

    # === ДАТАЛОАДЕР ===
    print("\n📚 Инициализация DataLoader...")
    dataloader = get_bysel_dataloader(
        cfg.data_path, 
        chunk_size=cfg.chunk_size, 
        batch_size=cfg.batch_size,
        start_file_idx=start_file_idx,
        start_byte_offset=start_byte_offset
    )

    # === ГЛОБАЛЬНОЕ СОСТОЯНИЕ ===
    global_current_file_idx = start_file_idx
    global_current_byte_offset = start_byte_offset
    global_current_step = start_step

    # === EMERGENCY CHECKPOINT HANDLER ===
    def save_emergency_checkpoint(signum, frame):
        print("\n\n💾 [SIGINT] Emergency save...")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'patcher_state_dict': patcher.state_dict(),
            'step': global_current_step,
            'file_idx': global_current_file_idx,
            'byte_offset': global_current_byte_offset,
        }, "checkpoints/latest_crash_backup.pt")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_emergency_checkpoint)
    signal.signal(signal.SIGTERM, save_emergency_checkpoint)

    # === AUTOCAST ===
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    autocast_enabled = (device == "cuda")
    
    # === CUDA STREAM PREFETCH ===
    use_cuda_stream = (device == "cuda")
    prefetch_stream = None
    dataloader_iter = iter(dataloader)
    current_batch = None
    
    if use_cuda_stream:
        print("\n⚡ CUDA Stream enabled")
        prefetch_stream = torch.cuda.Stream()
        try:
            current_batch = next(dataloader_iter)
            current_batch = (
                current_batch[0].to(device, non_blocking=True),
                current_batch[1],
                current_batch[2]
            )
        except StopIteration:
            return
    else:
        print("\n📥 Loading first batch (MPS/CPU)...")
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            return

    # === СТАРТ ТРЕНИРОВОЧНОГО ЦИКЛА ===
    print("\n🔥 Training loop started.")
    print("=" * 100)
    print(f"📐 Patch stride: {patcher.stride} (targets shifted to prevent Data Leakage)")
    print("=" * 100)
    start_time = time.time()
    
    for step_offset in range(cfg.max_steps):
        step = start_step + step_offset
        global_current_step = step
        
        # ============================================================
        # ШАГ 1: PREFETCH СЛЕДУЮЩЕГО BATCH
        # ============================================================
        next_batch = None
        if use_cuda_stream:
            with torch.cuda.stream(prefetch_stream):
                try:
                    next_batch = next(dataloader_iter)
                    next_batch = (
                        next_batch[0].to(device, non_blocking=True),
                        next_batch[1],
                        next_batch[2]
                    )
                except StopIteration:
                    next_batch = None
        else:
            try:
                next_batch = next(dataloader_iter)
            except StopIteration:
                next_batch = None
        
        # ============================================================
        # ШАГ 2: FORWARD + BACKWARD (БЕЗ СИНХРОНИЗАЦИЙ!)
        # ============================================================
        if current_batch is None:
            print("\n🎉 Dataset exhausted.")
            break
            
        byte_batch, last_file_idx, last_byte_offset = current_batch
        global_current_file_idx = last_file_idx
        global_current_byte_offset = last_byte_offset
        
        opt_engine.zero_grad(set_to_none=True)
        
        input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
        
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(input_bytes)
            T_patches = patches.shape[1]
            
            targets, mtp_targets = build_targets(
                byte_batch, T_patches, stride=patcher.stride
            )
            
            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(patches, mtp_targets)
            
            # 🎯 Loss БЕЗ .cpu() и .item() — работает полностью на GPU
            loss = loss_engine.compute_pretrain_loss(
                logits_t1, targets,
                [logits_t2, logits_t3, logits_t4],
                mtp_targets
            ) + aux_loss.float()
        
        # ============================================================
        # ШАГ 3: BACKWARD + CLIP + STEP
        # ============================================================
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        autopilot.inject_noise(model)
        # 🎯 Единственный sync на шаге: loss.item() для AutoPilot
        current_lr, noise_scale = autopilot.update_parameters(step, loss.item(), cfg.max_steps)
        opt_engine.step()
        
        # ============================================================
        # ШАГ 4: СИНХРОНИЗАЦИЯ CUDA STREAM
        # ============================================================
        if use_cuda_stream:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        
        current_batch = next_batch
        if current_batch is None:
            print("\n🎉 Dataset exhausted.")
            break
        
        # ============================================================
        # ШАГ 5: БЫСТРОЕ ЛОГИРОВАНИЕ (каждые 10 шагов, БЕЗ .cpu())
        # ============================================================
        if step % 10 == 0:
            elapsed = time.time() - start_time
            tokens_processed = (step_offset + 1) * cfg.batch_size * cfg.chunk_size
            speed = tokens_processed / elapsed
            
            # 🎯 Только 2 синхронизации вместо 6+
            loss_val = loss.item()
            aux_val = aux_loss.item()
            
            vram = ""
            if device == "cuda":
                vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                vram = f" | VRAM: {vram_mb:.0f}MB"
            
            print(
                f"Step {step:05d} | "
                f"Total: {loss_val:.2f} | "
                f"Aux: {aux_val:.2f} | "
                f"LR: {current_lr:.4f} | "
                f"{speed:.0f} tok/s{vram}"
            )
        
        # ============================================================
        # ШАГ 6: ДЕТАЛЬНАЯ ДИАГНОСТИКА (каждые 100 шагов, С .cpu())
        # ============================================================
        if step % 100 == 0 and step > 0:
            with torch.no_grad():
                # CE loss на CPU для точной диагностики
                main_ce = loss_engine.ce_loss(
                    logits_t1.float().cpu().reshape(-1, cfg.vocab_size),
                    targets.long().cpu().reshape(-1)
                ).item()
                
                mtp_losses = []
                for mtp_logits, mtp_target in zip([logits_t2, logits_t3, logits_t4], mtp_targets):
                    if mtp_logits is not None and mtp_target is not None:
                        mtp_l = loss_engine.ce_loss(
                            mtp_logits.float().cpu().reshape(-1, cfg.vocab_size),
                            mtp_target.long().cpu().reshape(-1)
                        ).item()
                        mtp_losses.append(mtp_l)
                
                mtp_info = f"[{', '.join(f'{l:.2f}' for l in mtp_losses)}]" if mtp_losses else "OFF"
                print(f"   🔬 DIAG | CE: {main_ce:.2f} | MTP: {mtp_info}")

        # ============================================================
        # ШАГ 7: ПЛАНОВЫЙ ЧЕКПОИНТ (каждые 1000 шагов)
        # ============================================================
        if step % 1000 == 0 and step > 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/bysel_{args.profile}_step_{step}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'patcher_state_dict': patcher.state_dict(),
                'step': step,
                'file_idx': last_file_idx,
                'byte_offset': last_byte_offset,
                'loss': loss_val,
                'lr_muon': current_lr,
                'profile': args.profile,
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

    # === ФИНАЛ ===
    total_time = time.time() - start_time
    total_tokens = (step_offset + 1) * cfg.batch_size * cfg.chunk_size
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 100)
    print("🎉 TRAINING COMPLETED")
    print("=" * 100)
    print(f"   Total time:   {total_time/3600:.2f} hours")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Avg speed:    {avg_speed:.1f} tok/s")
    
    os.makedirs("checkpoints", exist_ok=True)
    final_path = f"checkpoints/bysel_{args.profile}_FINAL.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'patcher_state_dict': patcher.state_dict(),
        'step': global_current_step,
        'file_idx': global_current_file_idx,
        'byte_offset': global_current_byte_offset,
        'profile': args.profile,
        'config': vars(cfg),
    }, final_path)
    print(f"💾 Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()```



================================================================
📁 FILE: ./training/autopilot.py (1433 bytes)
================================================================
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
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)
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



================================================================
📁 FILE: ./training/optimizer.py (4177 bytes)
================================================================
```python
import torch
import math

class Muon(torch.optim.Optimizer):
    """
    Встроенный оптимизатор Muon с поддержкой полуточности BF16/FP16 для импульса (Eq. 4, Moonlight) [1.1.8]
    Снижает использование ОЗУ под состояния оптимизатора ровно в 2 раза [1.1.8]!
    """
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
                if p.grad is None:
                    continue
                grad = p.grad
                
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    # Инициализируем импульс строго в полуточности (BF16 для MPS/CUDA, FP16 для CPU) [1.1.8, 1]
                    dtype = torch.bfloat16 if p.device.type in ["cuda", "mps"] else torch.float32
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=dtype)
                    
                buf = state['momentum_buffer']
                # Масштабируем градиент под точность буфера
                buf.mul_(momentum).add_(grad.to(buf.dtype))
                
                # Шаг Nesterov импульса (вычисления ведем в экономичном BF16/FP16) [1.1.8, 3]
                m_t = grad.to(buf.dtype) + momentum * buf
                
                # Newton-Schulz ортогонализация (10 итераций в два этапа) [3]
                O_t = self.hybrid_newton_schulz(m_t, steps=ns_steps)
                
                # Масштабирование Moonlight (Eq. 4) [1.1.8]
                A, B = p.shape[0], p.shape[1]
                scale = 0.2 * math.sqrt(max(A, B))
                
                # Обновляем веса (приводим O_t обратно к типу весов p)
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
            if not param.requires_grad:
                continue
            if param.ndim == 2 and "router" not in name and "proj" in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)
                
        self.opt_muon = Muon(muon_params, lr=lr_muon, momentum=0.95)
        self.opt_adamw = torch.optim.AdamW(adamw_params, lr=lr_adamw, weight_decay=0.01)

    def zero_grad(self, set_to_none: bool = True):
        """
        Обнуление градиентов.
        
        Args:
            set_to_none: Если True, градиенты устанавливаются в None вместо 
                         создания тензора нулей. Экономит память и быстрее.
        """
        self.opt_muon.zero_grad(set_to_none=set_to_none)
        self.opt_adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt_muon.step()
        self.opt_adamw.step()
```



================================================================
📁 FILE: ./training/recipe.py (3831 bytes)
================================================================
```python
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL LOSS ENGINE v3.3 - MPS-Proof                                        ║
║                                                                           ║
║ 🎯 ВРЕМЕННО: MTP loss ОТКЛЮЧЕН для диагностики основной модели            ║
║    Раскомментируйте после того как CE loss упадёт ниже 5.0               ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ByselLossEngine:
    def __init__(self, vocab_size=259):
        self.vocab_size = vocab_size
        # CrossEntropyLoss БЕЗ ignore_index для стабильности на MPS
        self.ce_loss = nn.CrossEntropyLoss()

    def compute_pretrain_loss(self, logits, targets, mtp_logits_list=None, mtp_targets_list=None):
        """
        Main pretrain loss.
        
        🎯 MTP loss ВРЕМЕННО ОТКЛЮЧЕН для диагностики.
        """
        # Main CE loss на CPU для стабильности
        logits_cpu = logits.detach().float().cpu()
        targets_cpu = targets.detach().long().cpu()
        
        loss = F.cross_entropy(
            logits_cpu.reshape(-1, self.vocab_size),
            targets_cpu.reshape(-1)
        ).to(logits.device)
        
        # 🎯 MTP LOSS ПОЛНОСТЬЮ ОТКЛЮЧЕН ДЛЯ ДИАГНОСТИКИ
        # Раскомментируйте, когда CE loss упадёт ниже 5.0
        """
        if mtp_logits_list is not None and mtp_targets_list is not None:
            mtp_loss_sum = 0.0
            mtp_count = 0
            for mtp_logits, mtp_targets in zip(mtp_logits_list, mtp_targets_list):
                if mtp_logits is not None and mtp_targets is not None:
                    mtp_logits_cpu = mtp_logits.detach().float().cpu()
                    mtp_targets_cpu = mtp_targets.detach().long().cpu()
                    mtp_loss_sum += F.cross_entropy(
                        mtp_logits_cpu.reshape(-1, self.vocab_size),
                        mtp_targets_cpu.reshape(-1)
                    )
                    mtp_count += 1
            if mtp_count > 0:
                loss = loss + 0.3 * (mtp_loss_sum / mtp_count).to(logits.device)
        """
        
        return loss

    def compute_sft_loss(self, logits, targets, thought_mask):
        masked_targets = targets.clone()
        masked_targets[thought_mask == 0] = -100
        
        logits_cpu = logits.detach().float().cpu()
        targets_cpu = masked_targets.detach().long().cpu()
        
        mask = targets_cpu != -100
        return F.cross_entropy(
            logits_cpu[mask].reshape(-1, self.vocab_size),
            targets_cpu[mask].reshape(-1)
        ).to(logits.device)

    def compute_kto_loss(self, policy_logps, reference_logps, labels, beta=0.1, kl_weight=0.1):
        log_ratios = policy_logps.float() - reference_logps.float()
        kl = torch.clamp(log_ratios, min=0.0).mean()
        
        losses = []
        for log_ratio, label in zip(log_ratios, labels):
            if label == 1:
                losses.append(-F.logsigmoid(beta * (log_ratio - kl)))
            else:
                losses.append(-F.logsigmoid(beta * (kl - log_ratio)))
        
        kto_loss = torch.stack(losses).mean() + kl_weight * kl
        return kto_loss```



