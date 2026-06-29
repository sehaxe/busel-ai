// YAML → BuselConfig. Читает configs/<profile>.yaml (плоский формат, без обёртки).
use burn::config::Config;

#[derive(Config, Debug)]
pub struct BuselConfig {
    #[config(default = 512)] pub d_model: usize,
    #[config(default = 128)] pub d_byte: usize,
    #[config(default = 12)] pub n_layers: usize,
    #[config(default = 8)] pub n_heads: usize,
    #[config(default = 326)] pub vocab_size: usize,
    #[config(default = 32)] pub n_patches: usize,
    #[config(default = 1024)] pub expert_hidden: usize,
    #[config(default = 6)] pub num_experts: usize,
    #[config(default = 1)] pub top_k: usize,
    #[config(default = 3)] pub num_mtp_heads: usize,
    #[config(default = 32)] pub sct_rank: usize,
    #[config(default = 0.01)] pub weight_decay: f64,
    #[config(default = 64)] pub batch_size: usize,
    #[config(default = 256)] pub chunk_size: usize,
    #[config(default = 0.01)] pub learning_rate: f64,
    #[config(default = 500)] pub max_steps: usize,
    #[config(default = 0)] pub target_tok_per_param: usize,  // 0 = не использовать
    pub data_weights: Option<std::collections::HashMap<String, f64>>,
    #[config(default = 50)] pub warmup_steps: usize,
    #[config(default = 0.1)] pub wsd_decay_frac: f64,
    #[config(default = 1)] pub grad_accum_steps: usize,
    #[config(default = true)] pub use_moe: bool,
    #[config(default = 0)] pub dtopk_k: usize,
    #[config(default = true)] pub lotus: bool,
    #[config(default = 32)] pub lotus_rank: usize,
    #[config(default = true)] pub adamw_1d: bool,
    #[config(default = true)] pub ema: bool,
    #[config(default = 0.999)] pub ema_decay: f64,
    #[config(default = false)] pub selective_backward: bool,
    #[config(default = 1.0)] pub backward_ratio: f64,
    #[config(default = false)] pub routing_free: bool,
    #[config(default = 0)] pub grad_ckpt_every: usize,
    // Decoupled LR
    #[config(default = 0.5)] pub lr_mult_embed: f64,
    #[config(default = 0.5)] pub lr_mult_router: f64,
    #[config(default = false)] pub schedule_free: bool,
    #[config(default = 0.95)] pub sf_beta: f64,
    #[config(default = r#""pretrain".to_string()"#)] pub stage: String,
    #[config(default = 0.0)] pub clip_norm: f64,
    #[config(default = 128)] pub d_c: usize,
    // DropBP + curriculum
    #[config(default = 0.3)] pub dropbp_prob: f64,
    #[config(default = false)] pub progressive_freeze: bool,
    #[config(default = false)] pub ascii_curriculum: bool,
    #[config(default = false)] pub chunk_growth: bool,
}

// Плоский YAML — только model / data / training секции, без global/profiles.
#[allow(dead_code)]
#[derive(serde::Deserialize, Default)]
struct YamlProfile {
    model: Option<YamlModel>,
    data: Option<YamlData>,
    training: Option<YamlTraining>,
    perf: Option<YamlPerf>,
}
#[allow(dead_code)]
#[derive(serde::Deserialize, Default)]
struct YamlModel {
    d_model: Option<usize>, n_layers: Option<usize>, n_heads: Option<usize>,
    expert_hidden: Option<usize>, num_experts: Option<usize>, top_k: Option<usize>,
    vocab_size: Option<usize>, num_mtp_heads: Option<usize>, n_patches: Option<usize>,
    sct_rank: Option<usize>, dtopk_k: Option<usize>,
    selective_backward: Option<bool>, backward_ratio: Option<f64>,
    routing_free: Option<bool>,
    stage: Option<String>,
    lr_mult_embed: Option<f64>, lr_mult_router: Option<f64>,
    d_c: Option<usize>,
    dropbp_prob: Option<f64>, progressive_freeze: Option<bool>,
    ascii_curriculum: Option<bool>, chunk_growth: Option<bool>,
}
#[allow(dead_code)]
#[derive(serde::Deserialize, Default)]
struct YamlData {
    batch_size: Option<usize>, chunk_size: Option<usize>,
    weighted: Option<std::collections::HashMap<String, f64>>,
}
#[allow(dead_code)]
#[derive(serde::Deserialize, Default)]
struct YamlTraining {
    learning_rate_muon: Option<f64>, weight_decay: Option<f64>,
    grad_accum_steps: Option<usize>, wsd_decay_frac: Option<f64>,
    warmup_steps: Option<serde_yaml::Value>,
    lotus_rank: Option<usize>, sct_rank: Option<usize>,
    max_steps: Option<serde_yaml::Value>,
    target_tok_per_param: Option<usize>,
    schedule_free: Option<bool>,
    sf_beta: Option<f64>,
    clip_norm: Option<f64>,
}
#[allow(dead_code)]
#[derive(serde::Deserialize, Default)]
struct YamlPerf {
    grad_ckpt_every: Option<usize>,
}

impl BuselConfig {
    pub fn from_yaml(path: &str, profile: &str, bs: usize, cs: usize, max: usize) -> Self {
        let yaml_path = if path.ends_with(".yaml") { path.to_string() }
            else { format!("../configs/{profile}.yaml") };

        let s = std::fs::read_to_string(&yaml_path)
            .unwrap_or_else(|_| panic!("config not found: {yaml_path}"));

        let p: YamlProfile = serde_yaml::from_str(&s).expect("YAML parse error");
        let m = p.model.unwrap_or_default();
        let d = p.data.unwrap_or_default();
        let t = p.training.unwrap_or_default();

        // max_steps: CLI > YAML(auto→расчёт) > 500
        let max_steps = if max > 0 {
            max
        } else if let Some(ref ms) = t.max_steps {
            match ms.as_u64() {
                Some(n) => n as usize,
                None => {
                    // "auto": tok_per_param * params / (bs * cs * ga)
                    let dm = m.d_model.unwrap_or(512);
                    let nl = m.n_layers.unwrap_or(12);
                    let ne = m.num_experts.unwrap_or(0);
                    let eh = m.expert_hidden.unwrap_or(dm * 4);
                    let rk = m.sct_rank.unwrap_or(32);
                    let tpp = t.target_tok_per_param.unwrap_or(37);
                    let vs = m.vocab_size.unwrap_or(326).max(1);

                    let attn = (4 * dm * dm) as u64;
                    let ffn = if ne > 0 { (ne * 3 * rk * (dm + eh)) as u64 } else { (3 * rk * (dm + eh)) as u64 };
                    let params = nl as u64 * (attn + ffn) + (dm * vs + vs * dm) as u64;

                    let bs_val = if bs > 0 { bs } else { d.batch_size.unwrap_or(32) };
                    let cs_val = if cs > 0 { cs } else { d.chunk_size.unwrap_or(1024) };
                    let ga_val = t.grad_accum_steps.unwrap_or(1);
                    let tok_per_step = (bs_val * cs_val * ga_val) as u64;
                    let total_tok = (tpp as u64) * (params as u64);
                    (total_tok / tok_per_step.max(1)) as usize
                }
            }
        } else { 500 };

        let warmup = t.warmup_steps.as_ref().and_then(|v| {
            v.as_str().and_then(|s| s.trim_end_matches('%').parse::<f64>().ok())
                .map(|pct| (max_steps as f64 * pct / 100.0) as usize)
                .or_else(|| v.as_u64().map(|n| n as usize))
        }).unwrap_or(max_steps.saturating_mul(5) / 100);

        let mut c = Self::new();
        c.d_model = m.d_model.unwrap_or(512);
        c.n_layers = m.n_layers.unwrap_or(12);
        c.n_heads = m.n_heads.unwrap_or(8);
        c.expert_hidden = m.expert_hidden.unwrap_or(1024);
        c.num_experts = m.num_experts.unwrap_or(6);
        c.top_k = m.top_k.unwrap_or(1);
        c.vocab_size = m.vocab_size.unwrap_or(326);
        c.num_mtp_heads = m.num_mtp_heads.unwrap_or(3);
        c.n_patches = m.n_patches.unwrap_or(32);
        c.sct_rank = t.sct_rank.or(m.sct_rank).unwrap_or(32);
        c.batch_size = if bs > 0 { bs } else { d.batch_size.unwrap_or(32) };
        c.chunk_size = if cs > 0 { cs } else { d.chunk_size.unwrap_or(1024) };
        c.max_steps = max_steps;
        c.warmup_steps = warmup;
        c.learning_rate = t.learning_rate_muon.unwrap_or(0.01);
        c.weight_decay = t.weight_decay.unwrap_or(0.01);
        c.grad_accum_steps = t.grad_accum_steps.unwrap_or(1);
        c.wsd_decay_frac = t.wsd_decay_frac.unwrap_or(0.1);
        c.use_moe = m.num_experts.unwrap_or(0) > 1;
        c.dtopk_k = m.dtopk_k.unwrap_or(0);
        c.lotus_rank = t.lotus_rank.unwrap_or(32);
        c.selective_backward = m.selective_backward.unwrap_or(false);
        c.backward_ratio = m.backward_ratio.unwrap_or(1.0);
        c.grad_ckpt_every = p.perf.as_ref().and_then(|x| x.grad_ckpt_every).unwrap_or(0);
        c.data_weights = d.weighted;
        c.routing_free = m.routing_free.unwrap_or(false);
        c.lr_mult_embed = m.lr_mult_embed.unwrap_or(0.5);
        c.lr_mult_router = m.lr_mult_router.unwrap_or(0.5);
        c.stage = m.stage.clone().unwrap_or_else(|| "pretrain".to_string());
        c.schedule_free = t.schedule_free.unwrap_or(false);
        c.sf_beta = t.sf_beta.unwrap_or(0.95);
        c.clip_norm = t.clip_norm.unwrap_or(0.0);
        c.d_c = m.d_c.unwrap_or(128);
        c.dropbp_prob = m.dropbp_prob.unwrap_or(0.3);
        c.progressive_freeze = m.progressive_freeze.unwrap_or(false);
        c.ascii_curriculum = m.ascii_curriculum.unwrap_or(false);
        c.chunk_growth = m.chunk_growth.unwrap_or(false);
        c
    }
}
