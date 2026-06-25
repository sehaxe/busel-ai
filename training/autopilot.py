"""
⚙️ busel (PREDICTIVE CYBER-ENGINE + WSD + WSD-S + wd33)
Содержит предиктивный подавитель взрывов, адаптивный клиппинг, динамический Weight Decay,
Warmup-Stable-Decay (WSD) расписание, WSD-S с переиспользованием чекпоинтов,
и wd33 расписание для QAT (Mapping Schedule × Bit-Width, 2026).
"""
import torch
import math

class buselAutoPilot:
    def __init__(self, opt_engine, max_lr_muon, max_lr_adamw, target_wd=0.1,
                 warmup_steps="auto", min_lr_ratio=0.1, noise_scale=0.01,
                 noise_decay=0.999, lr_schedule="cosine", wsd_decay_frac=0.1,
                 grad_clip=2.0):
        self.opt_engine = opt_engine
        self.max_lr_muon = max_lr_muon
        self.max_lr_adamw = max_lr_adamw
        self.target_wd = target_wd
        self.warmup_steps_raw = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.lr_schedule = lr_schedule
        self.wsd_decay_frac = wsd_decay_frac
        self.grad_clip = grad_clip
        
        self.loss_history = []
        self.grad_norm_history = []
        self.recovery_countdown = 0
        self.stabilization_factor = 1.0
        self.warmup_steps = 0

    def before_step(self, model, step, max_steps):
        # ponytail: cache param list — avoids model.parameters() iteration every step
        if not hasattr(self, '_cached_params'):
            self._cached_params = list(model.parameters())
        if not any(p.grad is not None for p in self._cached_params):
            return 1.0
        
        if step < 50:
            torch.nn.utils.clip_grad_norm_(self._cached_params, max_norm=1.0, foreach=True)
            return 1.0

        with torch.no_grad():
            current_grad_norm = torch.nn.utils.get_total_norm(
                [p.grad for p in self._cached_params if p.grad is not None], 2.0
            ).item()
            if current_grad_norm != current_grad_norm:
                return 1.0
            
        self.grad_norm_history.append(current_grad_norm)
        if len(self.grad_norm_history) > 50:
            self.grad_norm_history.pop(0)
            
        if len(self.grad_norm_history) >= 15:
            history_tensor = torch.tensor(self.grad_norm_history[:-1])
            mean_norm = history_tensor.mean().item()
            std_norm = history_tensor.std().item()
            threshold = mean_norm + 3.0 * max(1e-5, std_norm)
            
            if current_grad_norm > threshold:
                scale_factor = mean_norm / (current_grad_norm + 1e-8)
                torch.nn.utils.clip_grad_norm_(self._cached_params, max_norm=current_grad_norm * scale_factor, foreach=True)
                print(f"\n⚡ [PREDICTIVE DAMPENING ACTIVATED]:")
                print(f"   • Detected abnormal gradient norm surge: {current_grad_norm:.4f} (Threshold: {threshold:.4f})")
                print(f"   • Preventively scaled gradients down by factor: {scale_factor:.4f} to bypass impending loss spike.")

        if len(self.grad_norm_history) >= 10:
            rolling_avg_grad = sum(self.grad_norm_history) / len(self.grad_norm_history)
            clipping_threshold = min(1.0, max(0.5, rolling_avg_grad * 1.0))
        else:
            clipping_threshold = 1.0
        torch.nn.utils.clip_grad_norm_(self._cached_params, max_norm=clipping_threshold, foreach=True)
        
        progress = 0.0 if max_steps == 0 else min(1.0, max(0.0, float(step) / float(max_steps)))
        if step < self.warmup_steps:
            wd_factor = 0.1
        elif progress > 0.90:
            wd_factor = 0.5
        else:
            wd_factor = 0.1 + 0.9 * progress
        current_wd = self.target_wd * wd_factor
        
        if self.opt_engine.opt_muon is not None:
            for pg in self.opt_engine.opt_muon.param_groups:
                pg['weight_decay'] = current_wd
            for pg in self.opt_engine.opt_adamw.param_groups:
                pg['weight_decay'] = current_wd

        return clipping_threshold

    def update_parameters(self, step, current_loss, max_steps):
        if step == 0 or self.warmup_steps == 0:
            if self.warmup_steps_raw == "auto" or self.warmup_steps_raw is None:
                self.warmup_steps = max(50, int(0.05 * max_steps))
            else:
                self.warmup_steps = int(self.warmup_steps_raw)
                
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 30:
            self.loss_history.pop(0)
            
        if len(self.loss_history) >= 15 and self.recovery_countdown == 0:
            rolling_avg = sum(self.loss_history[:-1]) / (len(self.loss_history) - 1)
            if current_loss > 1.35 * rolling_avg:
                self.recovery_countdown = 15
                self.stabilization_factor = 0.35
                self.noise_scale = max(0.01, self.noise_scale * 1.5)
                print(f"\n⚠️  [AUTOPILOT SPIKE DETECTOR]: Всплеск лосса! Срезан LR до 35% на 15 шагов.\n")
                
        if self.recovery_countdown > 0:
            self.recovery_countdown -= 1
            if self.recovery_countdown == 0:
                self.stabilization_factor = 1.0
                
        if step < self.warmup_steps:
            lr_factor = float(step + 1) / float(self.warmup_steps)
        else:
            progress = float(step - self.warmup_steps) / float(max_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            if progress > 0.90:
                self.noise_scale = 0.0
            if self.lr_schedule == "wsd":
                st_steps = max_steps - self.warmup_steps
                d_steps = int(st_steps * self.wsd_decay_frac)
                s_steps = st_steps - d_steps
                s_elapsed = int((step - self.warmup_steps))
                if s_elapsed < s_steps:
                    lr_factor = 1.0
                else:
                    d_progress = (s_elapsed - s_steps) / max(1, d_steps)
                    lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * math.sqrt(1.0 - d_progress)
            else:
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
            
        lr_factor *= self.stabilization_factor
        new_lr_muon = self.max_lr_muon * lr_factor
        new_lr_adamw = self.max_lr_adamw * lr_factor
        
        if self.opt_engine.opt_muon is not None:
            for pg in self.opt_engine.opt_muon.param_groups:
                v = new_lr_muon * pg.get('lr_mult', 1.0)
                if isinstance(pg['lr'], torch.Tensor):
                    pg['lr'].fill_(v)
                else:
                    pg['lr'] = v
            for pg in self.opt_engine.opt_adamw.param_groups:
                v = new_lr_adamw * pg.get('lr_mult', 1.0)
                if isinstance(pg['lr'], torch.Tensor):
                    pg['lr'].fill_(v)
                else:
                    pg['lr'] = v
                
        if self.recovery_countdown == 0:
            self.noise_scale *= self.noise_decay
            
        return new_lr_muon, self.noise_scale
