"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ BYSEL TRAINING ENGINE v3.6 - Production Cross-Platform Orchestrator      ║
║                                                                           ║
║ 🎯 КЛЮЧЕВЫЕ ОПТИМИЗАЦИИ:                                                  ║
║   • Бескомпромиссная кроссплатформенная стабильность (CUDA / MPS / CPU)    ║
║   • Исключена утечка данных (Causal Target stride slicing)                ║
║   • Логирование мгновенной интервальной скорости (без задержек шага 0)    ║
║   • Нативный float16 на macOS Mac и bfloat16 на NVIDIA CUDA               ║
║   • Выключение медленного MPS-шума в Autopilot                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import signal
import argparse
import yaml

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from data.pipeline import get_bysel_dataloader
from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel
from training.optimizer import ByselOptimizerEngine
from training.autopilot import ByselAutoPilot
from training.recipe import ByselLossEngine


class ByselConfig:
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
            raise ValueError(f"Размерность d_model ({self.d_model}) должна делиться на n_heads ({self.n_heads})!")


def enforce_stability(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("   ⚙️  CUDA: TF32=ON, cuDNN.benchmark=ON")
    elif torch.backends.mps.is_available():
        print("   ⚙️  MPS: Ускорение Metal Performance Shaders активно")


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_targets(byte_batch, input_length, stride=4):
    """
    Создает казуально корректные таргеты для MTP-4 без утечки данных.
    Использует векторизованный шаг среза по тензору.
    """
    targets = byte_batch[:, stride::stride][:, :input_length]
    
    if targets.shape[1] < input_length:
        pad_size = input_length - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, pad_size), value=0)
    
    # Сдвинутые таргеты для дополнительных голов MTP
    mtp_targets = []
    for shift in [1, 2, 3]:
        mtp_target = byte_batch[:, (stride + shift)::stride][:, :input_length]
        if mtp_target.shape[1] < input_length:
            pad_size = input_length - mtp_target.shape[1]
            mtp_target = torch.nn.functional.pad(mtp_target, (0, pad_size), value=0)
        mtp_targets.append(mtp_target)
        
    return targets, mtp_targets


def main():
    parser = argparse.ArgumentParser(description="Bysel v3.6 - Production Training")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту")
    parser.add_argument("--profile", type=str, default="ziaziulia", help="Профиль из YAML")
    parser.add_argument("--no-compile", action="store_true", help="Отключить torch.compile")
    parser.add_argument("--no-checkpointing", action="store_true", help="Отключить gradient checkpointing")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  BYSEL TRAINING ENGINE v3.6 - Stable Cross-Platform Production║")
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

    start_step = 0
    start_file_idx = 0
    start_byte_offset = 0

    # === ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ===
    print("\n🔧 Инициализация модели...")
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = ByselModel(cfg).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ {total_params:,} параметров ({total_params * 2 / 1024**2:.2f} MB)")
    
    # Gradient Checkpointing (только на CUDA)
    if device == "cuda" and not args.no_checkpointing:
        model.enable_gradient_checkpointing()
    
    # torch.compile (только на CUDA)
    if device == "cuda" and not args.no_compile:
        print("🔧 torch.compile (max-autotune)...")
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False)
            patcher = torch.compile(patcher, mode="max-autotune", fullgraph=False)
            print("   ✅ Компиляция успешна")
        except Exception as e:
            print(f"   ⚠️  Compile failed: {e}")
    
    # === ОПТИМИЗАТОРЫ И СИСТЕМА ПИТАНИЯ ===
    opt_engine = ByselOptimizerEngine(model, lr_muon=cfg.learning_rate_muon, lr_adamw=cfg.learning_rate_adamw)
    autopilot = ByselAutoPilot(opt_engine)
    loss_engine = ByselLossEngine(cfg.vocab_size)

    # Восстановление из чекпоинта
    if args.resume and os.path.exists(args.resume):
        print(f"\n💾 Восстановление чекпоинта: {args.resume}")
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

    global_current_file_idx = start_file_idx
    global_current_byte_offset = start_byte_offset
    global_current_step = start_step

    # Обработчики аварийной остановки (SIGINT)
    def save_emergency_checkpoint(signum, frame):
        print("\n\n💾 [SIGINT] Аварийное сохранение состояния...")
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

    # === ОПРЕДЕЛЕНИЕ ТОЧНОСТИ AUTOCAST ===
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16
    autocast_enabled = (device in ["cuda", "mps"])
    
    # === CUDA STREAM PREFETCH ===
    use_cuda_stream = (device == "cuda")
    prefetch_stream = None
    dataloader_iter = iter(dataloader)
    current_batch = None
    
    if use_cuda_stream:
        print("\n⚡ CUDA Stream prefetch активирован")
        prefetch_stream = torch.cuda.Stream()
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            return
    else:
        print("\n📥 Загрузка первого батча...")
        try:
            current_batch = next(dataloader_iter)
        except StopIteration:
            return

    # === СТАРТ ТРЕНИРОВОЧНОГО ЦИКЛА ===
    print("\n🔥 Обучение запущено.")
    print("=" * 100)
    
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0
    
    for step_offset in range(cfg.max_steps):
        step = start_step + step_offset
        global_current_step = step
        
        # Шаг 1. Предвыборка (Prefetch)
        next_batch = None
        if use_cuda_stream:
            with torch.cuda.stream(prefetch_stream):
                try:
                    next_batch = next(dataloader_iter)
                except StopIteration:
                    next_batch = None
        else:
            try:
                next_batch = next(dataloader_iter)
            except StopIteration:
                next_batch = None
        
        if current_batch is None:
            print("\n🎉 Датасет полностью пройден.")
            break
            
        byte_batch, last_file_idx, last_byte_offset = current_batch
        
        # 🎯 УНИВЕРСАЛЬНЫЙ ФИКС: Перенос батча на MPS/CUDA/CPU во избежание Placeholder-ошибок
        byte_batch = byte_batch.to(device, non_blocking=True)
        
        global_current_file_idx = last_file_idx
        global_current_byte_offset = last_byte_offset
        
        opt_engine.zero_grad(set_to_none=True)
        input_bytes = byte_batch[:, :-patcher.stride] if byte_batch.shape[1] > patcher.stride else byte_batch
        
        # Шаг 2. Прямой проход (Forward)
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=autocast_enabled):
            patches = patcher(input_bytes)
            T_patches = patches.shape[1]
            
            targets, mtp_targets = build_targets(
                byte_batch, T_patches, stride=patcher.stride
            )
            
            (logits_t1, logits_t2, logits_t3, logits_t4), aux_loss = model(patches, mtp_targets)
            
            loss = loss_engine.compute_pretrain_loss(
                logits_t1, targets,
                [logits_t2, logits_t3, logits_t4],
                mtp_targets
            ) + aux_loss.float()
        
        # Шаг 3. Обратный проход (Backward)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Накладываем шум только на CUDA (на MPS randn_like критически медленный)
        if device == "cuda":
            autopilot.inject_noise(model)
            
        current_lr, noise_scale = autopilot.update_parameters(step, loss.item(), cfg.max_steps)
        opt_engine.step()
        
        # Ожидание асинхронного стрима
        if use_cuda_stream:
            torch.cuda.current_stream().wait_stream(prefetch_stream)
        
        current_batch = next_batch
        
        # Шаг 4. Быстрое интервальное логирование
        if step % 10 == 0:
            current_time = time.time()
            tokens_processed = (step_offset + 1) * cfg.batch_size * cfg.chunk_size
            
            # Вычисление чистой мгновенной скорости (без задержек шага 0)
            if step_offset == 0:
                elapsed_interval = current_time - start_time
                tokens_interval = tokens_processed
            else:
                elapsed_interval = current_time - last_log_time
                tokens_interval = tokens_processed - last_log_tokens
            
            speed = tokens_interval / elapsed_interval if elapsed_interval > 0 else 0
            
            last_log_time = current_time
            last_log_tokens = tokens_processed
            
            loss_val = loss.item()
            aux_val = aux_loss.item()
            
            vram = ""
            if device == "cuda":
                vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                vram = f" | VRAM: {vram_mb:.0f}MB"
            elif device == "mps":
                vram_mb = torch.mps.current_allocated_memory() / 1024**2
                vram = f" | VRAM: {vram_mb:.0f}MB"
            
            print(
                f"Step {step:05d} | "
                f"Total: {loss_val:.2f} | "
                f"Aux: {aux_val:.2f} | "
                f"LR: {current_lr:.5f} | "
                f"{speed:.0f} tok/s{vram}"
            )

        # Шаг 5. Плановый чекпоинт
        if step % 1000 == 0 and step > 0:
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/bysel_{args.profile}_step_{step}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'patcher_state_dict': patcher.state_dict(),
                'step': step,
                'file_idx': last_file_idx,
                'byte_offset': last_byte_offset,
                'loss': loss.item(),
                'lr_muon': current_lr,
                'profile': args.profile,
            }, checkpoint_path)
            print(f"💾 Плановый чекпоинт сохранен: {checkpoint_path}")

    # === ФИНАЛ ===
    total_time = time.time() - start_time
    total_tokens = (step_offset + 1) * cfg.batch_size * cfg.chunk_size
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 100)
    print("🎉 ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
    print("=" * 100)
    print(f"   Общее время:   {total_time/3600:.2f} ч")
    print(f"   Всего токенов: {total_tokens:,}")
    print(f"   Ср. скорость:  {avg_speed:.1f} токенов/сек")
    
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
    print(f"💾 Финальный чекпоинт: {final_path}")


if __name__ == "__main__":
    main()