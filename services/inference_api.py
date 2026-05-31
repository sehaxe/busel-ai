"""
🔥 HIGH-PERFORMANCE INFERENCE API v4.2 - Advanced Nucleus & Repetition Penalty Decoder
"""

import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.patching import StridedFastBLTPatcher
from model.backbone import ByselModel

app = FastAPI(title="Bysel Sovereign Omni-LLM API", version="4.2")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model = None
patcher = None


class Config:
    vocab_size = 259
    d_model = 384          # Согласовано с профилем Shpak
    n_layers = 8
    n_heads = 6
    expert_hidden = 768
    num_experts = 4
    top_k = 2


class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 150
    temperature: float = 0.7
    top_p: float = 0.9                   # Ядерное сэмплирование (Nucleus)
    repetition_penalty: float = 1.15     # Штраф за зацикливание


@app.on_event("startup")
def load_model():
    global model, patcher
    print(f"⚙️ Loading Bysel model onto device: {device.upper()}...")
    
    cfg = Config()
    patcher = StridedFastBLTPatcher(d_model=cfg.d_model).to(device)
    model = ByselModel(cfg).to(device)
    
    checkpoint_path = "checkpoints/bysel_shpak_FINAL.pt"
    if os.path.exists(checkpoint_path):
        print(f"💾 Loaded weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        patcher.load_state_dict(checkpoint['patcher_state_dict'])
    else:
        print("⚠️ Checkpoint not found. Running with random weights for testing.")
        
    model.eval()
    patcher.eval()


@app.post("/generate")
@torch.no_grad()
def generate(request: GenerateRequest):
    if model is None or patcher is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
        
    prompt_bytes = list(request.prompt.encode('utf-8'))
    input_ids = torch.tensor(prompt_bytes, dtype=torch.int32, device=device).unsqueeze(0)
    
    generated_bytes = []
    
    for _ in range(request.max_length):
        patches = patcher(input_ids)
        (logits, _, _, _), _ = model(patches)
        
        # Получаем логиты предсказания следующего байта
        next_token_logits = logits[0, -1, :].clone()
        
        # 1. ПРИМЕНЕНИЕ REPETITION PENALTY (Штраф за зацикливание)
        # Находим уникальные байты, которые уже встречались в диалоге
        already_generated = set(input_ids[0].tolist())
        for token in already_generated:
            if token < next_token_logits.shape[-1]:
                # Положительные логиты уменьшаем, отрицательные — увеличиваем по модулю
                if next_token_logits[token] > 0:
                    next_token_logits[token] /= request.repetition_penalty
                else:
                    next_token_logits[token] *= request.repetition_penalty

        # Применяем температурное масштабирование
        next_token_logits = next_token_logits / (request.temperature + 1e-8)
        
        # 2. ПРИМЕНЕНИЕ TOP-P (NUCLEUS) SAMPLING (Отсечение мусорного шума)
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Находим индексы, которые превышают порог累積 (cumulative_probs > top_p)
        sorted_indices_to_remove = cumulative_probs > request.top_p
        # Смещаем маску вправо, чтобы сохранить хотя бы один элемент над порогом
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Переносим маску обратно в исходный порядок индексов
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = -float('Inf')
        
        # Рассчитываем итоговое распределение вероятностей
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        if next_token < 256:
            generated_bytes.append(next_token)
            
        next_tensor = torch.tensor([[next_token]], dtype=torch.int32, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)
        
    try:
        generated_text = bytes(generated_bytes).decode('utf-8', errors='replace')
    except Exception:
        generated_text = "[UTF-8 Byte Decoding Error]"
        
    return {"prompt": request.prompt, "generated_text": generated_text}


@app.get("/health")
def health():
    return {"status": "healthy", "device": device}