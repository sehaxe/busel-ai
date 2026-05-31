"""
🎛️ BYSEL STATE MANAGER - Thread-safe training state control
Управляет состоянием обучения через JSON-файл для межпроцессного взаимодействия.
"""
import os
import json
import time
import threading
from pathlib import Path

STATE_FILE = "checkpoints/training_state.json"
# 🎯 ИСПРАВЛЕНИЕ: Используем RLock (Reentrant Lock) вместо обычного Lock, 
# чтобы избежать дедлока, когда update_state() вызывает get_state() внутри себя.
_lock = threading.RLock() 

def _ensure_dir():
    os.makedirs("checkpoints", exist_ok=True)

def get_state() -> dict:
    """Читает текущее состояние обучения."""
    _ensure_dir()
    with _lock:
        if not os.path.exists(STATE_FILE):
            return {
                "status": "idle",
                "current_step": 0,
                "max_steps": 0,
                "profile": "unknown",
                "started_at": None,
                "paused_at": None,
                "total_pause_time": 0.0,
                "last_heartbeat": None,
                "pid": None
            }
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"status": "idle"}

def update_state(**kwargs):
    """Атомарно обновляет поля состояния."""
    _ensure_dir()
    with _lock:
        state = get_state()
        state.update(kwargs)
        state["last_heartbeat"] = time.time()
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

def set_status(status: str):
    """Устанавливает статус и временные метки."""
    state = get_state()
    now = time.time()
    
    if status == "paused" and state["status"] == "running":
        state["paused_at"] = now
    elif status == "running" and state["status"] == "paused" and state.get("paused_at"):
        pause_duration = now - state["paused_at"]
        state["total_pause_time"] = state.get("total_pause_time", 0.0) + pause_duration
        state["paused_at"] = None
    
    state["status"] = status
    state["last_heartbeat"] = now
    update_state(**state)

def is_alive(timeout: float = 60.0) -> bool:
    """Проверяет, жив ли процесс обучения (heartbeat)."""
    state = get_state()
    if not state.get("last_heartbeat"):
        return False
    return (time.time() - state["last_heartbeat"]) < timeout

def get_metrics_history(max_points: int = 1000) -> list:
    """Читает историю метрик из metrics.jsonl."""
    log_path = "checkpoints/metrics.jsonl"
    if not os.path.exists(log_path):
        return []
    
    metrics = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        metrics.append(json.loads(line))
                    except Exception:
                        continue
    except Exception:
        return []
    
    return metrics[-max_points:]

def get_latest_metrics() -> dict:
    """Возвращает последние метрики."""
    history = get_metrics_history(max_points=1)
    return history[0] if history else {}

def estimate_eta() -> dict:
    """Рассчитывает ETA на основе истории метрик."""
    state = get_state()
    metrics = get_metrics_history(max_points=50)
    
    if not metrics or len(metrics) < 2:
        return {"eta_seconds": None, "eta_str": "Calculating..."}
    
    current_step = state.get("current_step", metrics[-1].get("step", 0))
    max_steps = state.get("max_steps", 0)
    
    if max_steps == 0 or current_step >= max_steps:
        return {"eta_seconds": 0, "eta_str": "Completed!"}
    
    steps_remaining = max_steps - current_step
    speeds = [m.get("speed", 0) for m in metrics if m.get("speed", 0) > 0]
    if not speeds:
        return {"eta_seconds": None, "eta_str": "No speed data"}
    
    avg_speed = sum(speeds) / len(speeds)
    tokens_per_step = 4096 * 4  # shpak default
    
    if avg_speed <= 0:
        return {"eta_seconds": None, "eta_str": "Calculating..."}
    
    steps_per_sec = avg_speed / tokens_per_step if tokens_per_step > 0 else 0
    if steps_per_sec <= 0:
        return {"eta_seconds": None, "eta_str": "Calculating..."}
    
    eta_seconds = steps_remaining / steps_per_sec
    
    if eta_seconds > 86400:
        eta_str = f"{eta_seconds / 86400:.1f} days"
    elif eta_seconds > 3600:
        eta_str = f"{eta_seconds / 3600:.1f} hours"
    elif eta_seconds > 60:
        eta_str = f"{eta_seconds / 60:.1f} minutes"
    else:
        eta_str = f"{eta_seconds:.0f} seconds"
    
    return {
        "eta_seconds": eta_seconds,
        "eta_str": eta_str,
        "avg_speed": avg_speed,
        "steps_remaining": steps_remaining
    }