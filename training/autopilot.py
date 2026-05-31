"""
⚙️ BYSEL AUTOPILOT v6.0 (PREDICTIVE CYBER-ENGINE)
Содержит предиктивный подавитель взрывов, адаптивный клиппинг и динамический Weight Decay.
"""

import torch
import math


class ByselAutoPilot:
    def __init__(self, opt_engine, max_lr_muon, max_lr_adamw, target_wd=0.1, warmup_steps="auto", min_lr_ratio=0.1, noise_scale=0.01, noise_decay=0.999):
        self.opt_engine = opt_engine
        self.max_lr_muon = max_lr_muon
        self.max_lr_adamw = max_lr_adamw
        self.target_wd = target_wd
        self.warmup_steps_raw = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        
        # Внутреннее состояние
        self.loss_history = []
        self.grad_norm_history = []
        self.recovery_countdown = 0
        self.stabilization_factor = 1.0
        self.warmup_steps = 0

    def before_step(self, model, step, max_steps):
        """
        Предиктивный анализ градиентов ДО шага оптимизатора.
        Выполняет адаптивный клиппинг, подавление взрывов и адаптацию Weight Decay.
        """
        if not any(p.grad is not None for p in model.parameters()):
            return 1.0

        # 1. Рассчитываем глобальную норму градиентов на текущем шаге
        with torch.no_grad():
            grads = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
            if not grads:
                return 1.0
            current_grad_norm = torch.norm(torch.stack(grads)).item()

        # Ведем статистику норм градиентов
        self.grad_norm_history.append(current_grad_norm)
        if len(self.grad_norm_history) > 50:
            self.grad_norm_history.pop(0)

        # 2. ПРЕДИКТИВНОЕ ПОДАВЛЕНИЕ ВЗРЫВОВ (Dampening)
        # Если норма градиента внезапно подскочила выше чем rolling_mean + 3 * rolling_std
        # Это сигнализирует о неизбежном взрыве (loss spike) на следующем шаге.
        if len(self.grad_norm_history) >= 15:
            history_tensor = torch.tensor(self.grad_norm_history[:-1])
            mean_norm = history_tensor.mean().item()
            std_norm = history_tensor.std().item()
            threshold = mean_norm + 3.0 * max(1e-5, std_norm)

            if current_grad_norm > threshold:
                # Превентивно прижимаем градиенты к среднему значению
                scale_factor = mean_norm / (current_grad_norm + 1e-8)
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale_factor)
                
                print(f"\n⚡ [PREDICTIVE DAMPENING ACTIVATED]:")
                print(f"   • Detected abnormal gradient norm surge: {current_grad_norm:.4f} (Threshold: {threshold:.4f})")
                print(f"   • Preventively scaled gradients down by factor: {scale_factor:.4f} to bypass impending loss spike.")
                
                # Обновляем текущую норму в истории для корректной статистики
                current_grad_norm = mean_norm
                self.grad_norm_history[-1] = current_grad_norm

        # 3. АДАПТИВНЫЙ ДИНАМИЧЕСКИЙ КЛИППИНГ (Adaptive Gradient Clipping — AGC)
        # Рассчитываем плавающий предел клиппинга на основе скользящего среднего
        if len(self.grad_norm_history) >= 10:
            rolling_avg_grad = sum(self.grad_norm_history) / len(self.grad_norm_history)
            # Порог клиппинга плавно адаптируется под текущую фазу обучения
            clipping_threshold = min(2.0, max(0.3, rolling_avg_grad * 1.5))
        else:
            clipping_threshold = 1.0

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipping_threshold)

        # 4. ДИНАМИЧЕСКИЙ РЕГУЛЯРИЗАТОР ВЕСОВ (Dynamic Weight Decay)
        # Защищает латентные веса на старте и кристаллизует тернарную структуру на финише.
        progress = 0.0 if max_steps == 0 else min(1.0, max(0.0, float(step) / float(max_steps)))
        
        if step < self.warmup_steps:
            # На этапе прогрева WD снижен до 10%
            wd_factor = 0.1
        elif progress > 0.90:
            # На этапе финального сжатия снижаем WD до 50%, убирая избыточное давление на веса
            wd_factor = 0.5
        else:
            # Линейно растет от 10% до 100% по мере прогресса обучения
            wd_factor = 0.1 + 0.9 * progress

        current_wd = self.target_wd * wd_factor

        # Динамически инжектируем новый WD во все группы оптимизаторов
        if self.opt_engine.opt_muon is not None:
            for pg in self.opt_engine.opt_muon.param_groups:
                pg['weight_decay'] = current_wd
        for pg in self.opt_engine.opt_adamw.param_groups:
            pg['weight_decay'] = current_wd

        return clipping_threshold

    def update_parameters(self, step, current_loss, max_steps):
        # Автоматический расчет шагов прогрева на первом шаге
        if step == 0 or self.warmup_steps == 0:
            if self.warmup_steps_raw == "auto" or self.warmup_steps_raw is None:
                self.warmup_steps = max(50, int(0.05 * max_steps))
            else:
                self.warmup_steps = int(self.warmup_steps_raw)

        # Ведем скользящую историю лосса
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 30:
            self.loss_history.pop(0)

        # Реактивное спасение от аномалий (если превентивное подавление пропустило микро-всплеск)
        if len(self.loss_history) >= 15 and self.recovery_countdown == 0:
            rolling_avg = sum(self.loss_history[:-1]) / (len(self.loss_history) - 1)
            if current_loss > 1.35 * rolling_avg:
                self.recovery_countdown = 15       
                self.stabilization_factor = 0.35    
                self.noise_scale = max(0.01, self.noise_scale * 1.5)  
                print(f"\n⚠️  [AUTOPILOT SPIKE DETECTOR]: Всплеск лосса! Срезан LR до 35% на 15 шагов для выхода из ямы.\n")

        # Обработка таймера восстановления
        if self.recovery_countdown > 0:
            self.recovery_countdown -= 1
            if self.recovery_countdown == 0:
                self.stabilization_factor = 1.0  

        # Расчет базовой скорости (Warmup + Cosine)
        if step < self.warmup_steps:
            warmup_ratio = float(step + 1) / float(self.warmup_steps)
            lr_factor = warmup_ratio
        else:
            progress = float(step - self.warmup_steps) / float(max_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            
            if progress > 0.90:
                self.noise_scale = 0.0  # Выключаем шум на финише
            
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

        lr_factor *= self.stabilization_factor

        new_lr_muon = self.max_lr_muon * lr_factor
        new_lr_adamw = self.max_lr_adamw * lr_factor

        if self.opt_engine.opt_muon is not None:
            for pg in self.opt_engine.opt_muon.param_groups:
                pg['lr'] = new_lr_muon
        for pg in self.opt_engine.opt_adamw.param_groups:
            pg['lr'] = new_lr_adamw

        if self.recovery_countdown == 0:
            self.noise_scale *= self.noise_decay

        return new_lr_muon, self.noise_scale

    def inject_noise(self, model):
        if self.noise_scale < 1e-6:
            return
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm = p.grad.norm().item()
                    if grad_norm > 1e-5:
                        noise = torch.randn_like(p.grad) * (self.noise_scale * grad_norm)
                        p.grad.add_(noise)