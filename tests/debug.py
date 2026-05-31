"""
🔍 СТАНДАРТНЫЙ СТЕНД NaN ДИАГНОСТИКИ BYSEL
Запуск: python debug_nan.py
"""

import os
import sys
import torch
import yaml

# Гарантируем корректность путей импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.pipeline import get_bysel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel
from training.optimizer import ByselOptimizerEngine
from training.recipe import ByselLossEngine

# 1. Автоопределение устройства
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Запуск диагностического стенда на устройстве: {device.upper()}")

# 2. Загрузка параметров ziaziulia из YAML
with open("configs/default.yaml", "r") as f:
    full_config = yaml.safe_load(f)

profile_dict = full_config["profiles"]["ziaziulia"]

class DebugConfig:
    d_model = profile_dict["model"]["d_model"]
    n_layers = profile_dict["model"]["n_layers"]
    n_heads = profile_dict["model"]["n_heads"]
    expert_hidden = profile_dict["model"]["expert_hidden"]
    num_experts = profile_dict["model"]["num_experts"]
    top_k = profile_dict["model"]["top_k"]
    vocab_size = profile_dict["model"]["vocab_size"]
    data_path = profile_dict["data"]["data_path"]
    chunk_size = profile_dict["data"]["chunk_size"]
    batch_size = profile_dict["data"]["batch_size"]
    learning_rate_muon = profile_dict["training"]["learning_rate_muon"]
    learning_rate_adamw = profile_dict["training"]["learning_rate_adamw"]
    max_steps = 15  # Достаточно для выявления NaN

cfg = DebugConfig()

# 3. Создание фиктивного файла данных (если папка data_path пуста)
dummy_file = "debug_nan_dummy.txt"
created_dummy_dir = False
if not os.path.exists(cfg.data_path) or len(os.listdir(cfg.data_path)) == 0:
    print(f"📁 Тестовая папка '{cfg.data_path}' пуста. Создаем фиктивный текстовый файл...")
    os.makedirs(cfg.data_path, exist_ok=True)
    created_dummy_dir = True
    with open(os.path.join(cfg.data_path, dummy_file), "w", encoding="utf-8") as f:
        f.write("Слава беларускаму аисту! Тэст для дыягностыкі NaN на платформе Bysel. " * 300)

# 4. Инициализация DataLoader
dataloader = get_bysel_dataloader(cfg.data_path, chunk_size=cfg.chunk_size, batch_size=cfg.batch_size)
dataloader_iter = iter(dataloader)

# 5. Инициализация моделей
patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
model = ByselModel(cfg).to(device)

# 🎯 Объявление типов autocast для стабильности (без конвертации параметров):
autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16
autocast_enabled = (device in ["cuda", "mps"])

opt_engine = ByselOptimizerEngine(model, lr_muon=cfg.learning_rate_muon, lr_adamw=cfg.learning_rate_adamw)
loss_engine = ByselLossEngine(cfg.vocab_size)

# 6. Подключение перехватчиков (Hooks) для детекции NaN в активациях
activated_hooks = []
nan_triggered = False

def check_tensor(tensor, name, step, is_input=False):
    global nan_triggered
    if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
        direction = "входе" if is_input else "выходе"
        print(f"\n❌ [FORWARD NaN]: Обнаружен NaN на {direction} модуля '{name}' на шаге {step}!")
        print(f"   Размерность тензора: {tensor.shape}")
        if not is_input:
            print(f"   Аномальные значения: {tensor.reshape(-1)[:5].tolist()}")
        nan_triggered = True
        return True
    return False

def make_hook(module_name):
    def hook_fn(module, inp, out):
        global nan_triggered
        if nan_triggered:
            return
        
        # Проверяем входы
        if inp:
            for i, x in enumerate(inp):
                if check_tensor(x, f"{module_name} (input_{i})", step, is_input=True):
                    return
        
        # Проверяем выходы
        if isinstance(out, tuple):
            for i, x in enumerate(out):
                if check_tensor(x, f"{module_name} (output_{i})", step):
                    return
        else:
            check_tensor(out, module_name, step)
    return hook_fn

# Регистрируем перехватчики на ключевых подмодулях
for name, module in model.named_modules():
    if len(list(module.children())) == 0 or any(kw in name for kw in ["attn", "moe", "norm", "mod_router", "m_residuals"]):
        hook = module.register_forward_hook(make_hook(name))
        activated_hooks.append(hook)

# 7. Казуальный выравниватель таргетов
def build_targets(byte_batch, input_length, stride=4):
    targets = byte_batch[:, stride::stride][:, :input_length]
    if targets.shape[1] < input_length:
        pad_size = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad_size), value=0)
    
    mtp_targets = []
    for shift in [1, 2, 3]:
        mtp_target = byte_batch[:, (stride + shift)::stride][:, :input_length]
        if mtp_target.shape[1] < input_length:
            pad_size = input_length - mtp_target.shape[1]
            mtp_target = torch.nn.functional.pad(mtp_target, (0, pad_size), value=0)
        mtp_targets.append(mtp_target)
    return targets, mtp_targets

# 8. Цикл пошаговой отладки
print("\n🔥 Запуск пошагового диагностического цикла...")
print("=" * 80)

try:
    for step in range(cfg.max_steps):
        if nan_triggered:
            break
            
        try:
            byte_batch, _, _ = next(dataloader_iter)
        except StopIteration:
            print("📝 Данные закончились.")
            break
            
        byte_batch = byte_batch.to(device, non_blocking=True)
        opt_engine.zero_grad(set_to_none=True)
        
        input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
        
        # --- ПРЯМОЙ ПРОХОД ---
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(input_bytes)
            if check_tensor(patches, "StridedFastBLTPatcher (output)", step):
                break
                
            T_patches = patches.shape[1]
            targets, mtp_targets = build_targets(byte_batch, T_patches, stride=patcher.stride)
            
            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(patches, mtp_targets)
            
            if nan_triggered:
                break
                
            loss = loss_engine.compute_pretrain_loss(
                logits_t1, targets,
                [logits_t2, logits_t3, logits_t4],
                mtp_targets
            ) + aux_loss.float()
            
        if torch.isnan(loss):
            print(f"\n❌ [LOSS NaN]: Лосс равен NaN на шаге {step}!")
            print(f"   Logits NaN: {torch.isnan(logits_t1).any().item()}")
            print(f"   Aux Loss: {aux_loss.item()}")
            break
            
        # --- ОБРАТНЫЙ ПРОХОД ---
        loss.backward()
        
        # --- АНАЛИЗ ГРАДИЕНТОВ ---
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"\n❌ [GRADIENT NaN]: NaN обнаружен в градиенте параметра '{name}' на шаге {step}!")
                nan_triggered = True
                break
        if nan_triggered:
            break
            
        # --- ШАГ ОБНОВЛЕНИЯ ВЕСОВ ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt_engine.step()
        
        # --- АНАЛИЗ ВЕСОВ ПОСЛЕ ОБНОВЛЕНИЯ ---
        for name, p in model.named_parameters():
            if torch.isnan(p).any():
                print(f"\n❌ [WEIGHT NaN]: NaN обнаружен в весах '{name}' ПОСЛЕ обновления оптимизатора на шаге {step}!")
                nan_triggered = True
                break
        if nan_triggered:
            break
            
        print(f"Step {step:02d} | Loss: {loss.item():.4f} | Aux: {aux_loss.item():.4f} | Status: OK")

finally:
    # Удаление хуков и чистка окружения
    for hook in activated_hooks:
        hook.remove()
    if created_dummy_dir:
        dummy_path = os.path.join(cfg.data_path, dummy_file)
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
        try:
            os.rmdir(cfg.data_path)
        except OSError:
            pass

if nan_triggered:
    print("\n🛑 ДИАГНОСТИКА: NaN успешно пойман и локализован выше.")
else:
    print("\n✅ ДИАГНОСТИКА: Все тестовые шаги пройдены успешно (NaN не обнаружен).")
print("=" * 80)