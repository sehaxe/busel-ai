"""
📚 BYSEL PIPELINE v3.6 - Stable Cross-Platform Loader
"""

import torch
import os
import json
import random
import platform
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
                            text_val = self._recursive_extract(data)
                            if text_val.strip():
                                extracted_texts.append(text_val.strip())
                        except json.JSONDecodeError:
                            continue
            full_text = "\n".join(extracted_texts)
            self.raw_bytes = full_text.encode('utf-8')
        else:
            with open(file_path, "rb") as f:
                self.raw_bytes = f.read()

    def _recursive_extract(self, obj):
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            return "\n".join([self._recursive_extract(v) for v in obj.values() if v])
        elif isinstance(obj, list):
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
        raise ValueError("Не удалось найти текстовую колонку в Parquet файле.")

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
        # Безопасное разделение файлов между воркерами
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files_to_process = [f for i, f in enumerate(self.files) if i % num_workers == worker_id]
            start_file = 0
        else:
            files_to_process = self.files
            start_file = self.start_file_idx

        shuffle_buffer = []
        buffer_size = 50
        
        for file_idx in range(start_file, len(files_to_process)):
            self.current_file_idx = file_idx
            file_path = files_to_process[file_idx]
            offset = self.start_byte_offset if file_idx == start_file else 0
            
            use_rust_streamer = (
                not file_path.endswith(('.parquet', '.jsonl')) 
                and HAS_RUST_IO
            )
            
            if use_rust_streamer:
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
    # 🎯 КЛЮЧЕВОЙ ФИКС: Создаем тензоры в формате int32 на CPU
    batch_tensors = torch.stack([torch.tensor(c, dtype=torch.int32) for c in chunks])
    return batch_tensors, file_indices[-1], byte_offsets[-1]


def get_bysel_dataloader(data_path, chunk_size, batch_size, start_file_idx=0, start_byte_offset=0, num_workers=None):
    dataset = RustByteStreamDataset(data_path, chunk_size, start_file_idx, start_byte_offset)
    use_pin = torch.cuda.is_available()
    
    # Кроссплатформенное автоопределение воркеров
    if num_workers is None:
        if platform.system() == "Linux" and torch.cuda.is_available():
            num_workers = min(4, os.cpu_count() or 1)
        else:
            num_workers = 0
            
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=use_pin,
        collate_fn=collate_bysel_batch
    )