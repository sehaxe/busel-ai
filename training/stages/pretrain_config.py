"""busel pretrain config — single source of truth. Only size + hyperparameters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from training.stages.base import _apply_model_profile

@dataclass
class buselPretrainConfig:
    # Model size
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    expert_hidden: int = 2048
    num_experts: int = 4
    top_k: int = 1
    vocab_size: int = 326
    n_hyper: int = 2
    num_mtp_heads: int = 3  # 4 heads (t+1..t+4). Set 1 for max speed.
    nsa_n_heads: int = 16  # NSA attention requires heads divisible by 16
    
    # Data
    data_path: str = "data_train"
    chunk_size: int = 256
    batch_size: int = 256
    mix_weights: dict | None = None  # {"fineweb": 0.6, "wiki": 0.3, ...}
    
    # Training
    max_steps: Any = "auto"
    warmup_steps: Any = "auto"
    learning_rate_muon: float = 0.001
    learning_rate_adamw: float = 0.0001
    weight_decay: float = 0.1
    grad_accum_steps: int = 1
    
    # MoE depth
    mod_capacity: float = 1.0  # MoD: 0.5 = 50% tokens through FFN
    mod_interval: int = 2      # apply MoD every N layers
    
    # Features (always ON — busel defaults)
    use_chunk_curriculum: bool = True
    selective_backward: bool = True
    backward_ratio: float = 0.5
    use_differential_attention: bool = True
    use_dispersion_loss: bool = True
    dispersion_weight: float = 0.1
    dispersion_temperature: float = 2.0
    use_tequila: bool = True
    tequila_lambda: float = 0.001
    use_rho_loss: bool = True
    rho_keep_ratio: float = 0.5
    # D5: Progressive layer freezing. OFF by default (enable for large models)
    progressive_freeze: bool = False
    freeze_threshold: float = 0.5  # progress fraction to start freezing
    # D6: Byte-level ASCII curriculum. OFF by default (enable for 1B+)
    use_ascii_curriculum: bool = False
    sct_rank: int = 0  # SCT: 0 = off, 8 = rank for Spectral Linear in FFN
    use_dropbp: bool = False
    dropbp_prob: float = 0.3
    
    # Optimizer (always SF-NorMuon)
    use_ema: bool = True
    ema_decay: float = 0.999
    lotus_rank: int = 8
    lr_multipliers: dict | None = None
    min_lr_ratio: float = 0.1
    lr_schedule: str = "cosine"
    wsd_decay_frac: float = 0.1
    grad_clip: float = 2.0
    checkpoint_interval: int = 100
    use_yarn: bool = False
    yarn_scale: float = 32.0
    yarn_start_frac: float = 0.92
    yarn_duration_frac: float = 0.08
    
    # Perf
    inductor_cache_dir: str = "~/.cache/busel/inductor"
    inductor_cache_clean: bool = False
    inductor_cache_max_gb: float = 0.0
    keep_last_n: int = 5
    dynamic_compile: bool = True
    grad_ckpt_every: int = 2  # gradient checkpointing: 2 = every other layer

    @classmethod
    def from_profile(cls, profile_dict: dict) -> buselPretrainConfig:
        cfg = cls()
        m = profile_dict.get("model", {})
        d = profile_dict.get("data", {})
        t = profile_dict.get("training", {})
        p = profile_dict.get("perf", {})
        _apply_model_profile(cfg, m)
        cfg.data_path = d.get("data_path", cfg.data_path)
        cfg.chunk_size = int(d.get("chunk_size", cfg.chunk_size))
        cfg.batch_size = int(d.get("batch_size", cfg.batch_size))
        mw = d.get("mix_weights")
        if isinstance(mw, dict):
            cfg.mix_weights = {str(k): float(v) for k, v in mw.items()}
        cfg.max_steps = t.get("max_steps", cfg.max_steps)
        cfg.warmup_steps = t.get("warmup_steps", cfg.warmup_steps)
        cfg.learning_rate_muon = float(t.get("learning_rate_muon", cfg.learning_rate_muon))
        cfg.learning_rate_adamw = float(t.get("learning_rate_adamw", cfg.learning_rate_adamw))
        cfg.weight_decay = float(t.get("weight_decay", cfg.weight_decay))
        cfg.grad_accum_steps = int(t.get("grad_accum_steps", cfg.grad_accum_steps))
        cfg.mod_capacity = float(t.get("mod_capacity", cfg.mod_capacity))
        cfg.mod_interval = int(t.get("mod_interval", cfg.mod_interval))
        cfg.use_chunk_curriculum = bool(t.get("use_chunk_curriculum", cfg.use_chunk_curriculum))
        cfg.selective_backward = bool(m.get("selective_backward", cfg.selective_backward))
        cfg.backward_ratio = float(m.get("backward_ratio", cfg.backward_ratio))
        cfg.use_differential_attention = bool(m.get("use_differential_attention", cfg.use_differential_attention))
        cfg.use_dispersion_loss = bool(t.get("use_dispersion_loss", cfg.use_dispersion_loss))
        cfg.dispersion_weight = float(t.get("dispersion_weight", cfg.dispersion_weight))
        cfg.dispersion_temperature = float(t.get("dispersion_temperature", cfg.dispersion_temperature))
        cfg.use_tequila = bool(t.get("use_tequila", cfg.use_tequila))
        cfg.use_rho_loss = bool(t.get("use_rho_loss", cfg.use_rho_loss))
        cfg.rho_keep_ratio = float(t.get("rho_keep_ratio", cfg.rho_keep_ratio))
        cfg.progressive_freeze = bool(t.get("progressive_freeze", cfg.progressive_freeze))
        cfg.freeze_threshold = float(t.get("freeze_threshold", cfg.freeze_threshold))
        cfg.use_ascii_curriculum = bool(t.get("use_ascii_curriculum", cfg.use_ascii_curriculum))
        cfg.sct_rank = int(t.get("sct_rank", cfg.sct_rank))
        cfg.use_dropbp = bool(t.get("use_dropbp", cfg.use_dropbp))
        cfg.dropbp_prob = float(t.get("dropbp_prob", cfg.dropbp_prob))
        cfg.lr_multipliers = t.get("lr_multipliers", cfg.lr_multipliers)
        if isinstance(cfg.lr_multipliers, dict):
            cfg.lr_multipliers = {str(k): float(v) for k, v in cfg.lr_multipliers.items()}
        cfg.min_lr_ratio = float(t.get("min_lr_ratio", cfg.min_lr_ratio))
        cfg.lr_schedule = str(t.get("lr_schedule", cfg.lr_schedule))
        cfg.wsd_decay_frac = float(t.get("wsd_decay_frac", cfg.wsd_decay_frac))
        cfg.grad_clip = float(t.get("grad_clip", cfg.grad_clip))
        cfg.checkpoint_interval = int(t.get("checkpoint_interval", cfg.checkpoint_interval))
        yn = profile_dict.get("yarn", {})
        cfg.use_yarn = bool(yn.get("enabled", cfg.use_yarn))
        cfg.yarn_scale = float(yn.get("target_scale", cfg.yarn_scale))
        cfg.yarn_start_frac = float(yn.get("start_step_frac", cfg.yarn_start_frac))
        cfg.yarn_duration_frac = float(yn.get("duration_frac", cfg.yarn_duration_frac))
        cfg.inductor_cache_dir = str(p.get("inductor_cache_dir", cfg.inductor_cache_dir))
        cfg.inductor_cache_clean = bool(p.get("inductor_cache_clean", cfg.inductor_cache_clean))
        cfg.inductor_cache_max_gb = float(p.get("inductor_cache_max_gb", cfg.inductor_cache_max_gb))
        cfg.dynamic_compile = bool(p.get("dynamic_compile", cfg.dynamic_compile))
        cfg.keep_last_n = int(p.get("keep_last_n", cfg.keep_last_n))
        cfg.grad_ckpt_every = int(p.get("grad_ckpt_every", cfg.grad_ckpt_every))
        if cfg.d_model % cfg.n_hyper != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_hyper ({cfg.n_hyper})!")
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})!")
        return cfg
