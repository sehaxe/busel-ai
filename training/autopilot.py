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
