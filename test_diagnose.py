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
