// busel-burn: entry point.
mod config; mod data; mod loss; mod mask; mod grad_clip;
mod model; mod schedule_free; mod train;
mod types; mod ckpt; mod profiler; mod tui; mod pipeline;
mod optim_wrapper; mod cube; mod logs;
mod curriculum; mod dropbp;

use std::time::Instant;
use config::BuselConfig;
use model::BuselModel;
use model::hybrid::HymOpt;
use types::{Model, Optim};
use profiler::Profiler;
use tui::TuiHandle;
use logs::JsonlLogger;

use burn::module::{ModuleVisitor, Module};
use burn::tensor::{backend::Backend, Tensor};

struct ParCnt { pub n1d: u64, pub n2d: u64, pub n3d: u64 }
impl<B: Backend> ModuleVisitor<B> for ParCnt {
    fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<B, D>>) {
        let n: u64 = param.val().dims().iter().product::<usize>() as u64;
        match D { 1 => self.n1d += n, 2 => self.n2d += n, 3 => self.n3d += n, _ => {} }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut profile = "chizh-9m";
    let mut yaml = "../configs";
    let (mut bs, mut cs, mut max) = (0usize, 0usize, 0usize);
    let (mut clip, mut ckpt) = (0.0f64, 0usize);
    let mut pos = 1;
    while pos < args.len() {
        match args[pos].as_str() {
            "--bs" if pos + 1 < args.len() => { bs = args[pos + 1].parse().unwrap(); pos += 2; }
            "--cs" if pos + 1 < args.len() => { cs = args[pos + 1].parse().unwrap(); pos += 2; }
            "--steps" if pos + 1 < args.len() => { max = args[pos + 1].parse().unwrap(); pos += 2; }
            "--yaml" if pos + 1 < args.len() => { yaml = &args[pos + 1]; pos += 2; }
            "--stage" if pos + 1 < args.len() => { pos += 2; }
            "--clip" if pos + 1 < args.len() => { clip = args[pos + 1].parse().unwrap(); pos += 2; }
            "--ckpt" if pos + 1 < args.len() => { ckpt = args[pos + 1].parse().unwrap(); pos += 2; }
            a if a.starts_with("--") => { eprintln!("unknown: {a}"); pos += 1; }
            _ => { profile = &args[pos]; pos += 1; }
        }
    }

    let mut cfg = BuselConfig::from_yaml(yaml, profile, bs, cs, max);
    if clip > 0.0 { cfg.clip_norm = clip; }
    if ckpt > 0 { cfg.grad_ckpt_every = ckpt; }
    let ga = cfg.grad_accum_steps;
    if cfg.stage != "pretrain" {
        eprintln!("[busel-burn] stage '{}' not implemented", cfg.stage);
        return;
    }

    let tui = TuiHandle::new(cfg.max_steps);
    let device = Default::default();

    let mdl: Model = BuselModel::new(&cfg, &device);
    {
        let mut c = ParCnt { n1d: 0, n2d: 0, n3d: 0 };
        mdl.visit(&mut c);
        let total = c.n1d + c.n2d + c.n3d;
        let d = cfg.d_model as f64; let e = cfg.expert_hidden as f64;
        let r = cfg.sct_rank as f64; let _ne = cfg.num_experts as f64;
        let sct_eff = (d * e * 3.0) / (3.0 * r * (d + e));
        eprintln!("[init] params: {} ({}.{}.{}) = {:.1}M actual, {:.0}M eff",
            total, c.n2d, c.n1d, c.n3d,
            total as f64 / 1_000_000.0, total as f64 * sct_eff / 1_000_000.0);
    }
    let loss_fn = loss::make_loss_fn();
    let mut optim: Optim = Optim::new(
        HymOpt::new(cfg.weight_decay, 5).with(|c| {
            c.lotus = false; c.lotus_rank = cfg.lotus_rank;
            c.adamw_1d = true;
            c.ema = cfg.ema; c.ema_decay = cfg.ema_decay;
            c.lr_mult_embed = cfg.lr_mult_embed; c.lr_mult_router = cfg.lr_mult_router;
            c.d_byte = cfg.d_byte; c.num_experts = cfg.num_experts;
            c.clip_norm = cfg.clip_norm;
        })
    );
    let streamer = if let Some(ref w) = cfg.data_weights {
        data::ByteStreamer::open_weighted(std::path::Path::new("../data_train"), w)
    } else {
        data::ByteStreamer::open(std::path::Path::new("../data_train"))
    }.expect("need ../data_train/");
    let mut streamer = streamer;

    let mdl = ckpt::load(profile, mdl);
    ckpt::load_optim(profile, &mut optim);
    let mtp_w: Vec<f64> = (0..=cfg.num_mtp_heads).map(|i| 0.5f64.powi(i as i32)).collect();
    let mut best = ckpt::BestKeeper::new();
    let mut prof = Profiler::new(cfg.max_steps);

    let curriculum = crate::curriculum::Curriculum::new(
        cfg.max_steps as u64, 0.3, cfg.chunk_growth);

    // Step 0: warmup (без SF)
    let (z0, loss0) = train::accum_step(mdl, &mut optim, &cfg, &loss_fn, 0, &mtp_w, &mut streamer, None, &curriculum);
    let mut z = z0;
    let mut w = z.clone();  // w = z after step 0
    eprintln!("[init] step 0 loss {loss0:.6}");
    prof.record(0, loss0, 0);

    // Schedule-Free params
    let sf_beta = 0.95f64;

    let t0 = Instant::now(); let mut tokens: u64 = 0; let mut ema_loss = loss0;
    let mut logger = JsonlLogger::new(profile);

    for s in 1..=cfg.max_steps {
        let beta_k = (s - 1) as f64 / (s as f64 + sf_beta);

        let t_lerp = Instant::now();
        let y = schedule_free::lerp_y(&z, &w, beta_k);
        let t_fwd = Instant::now();
        let y_copy = y.clone();
        let (z_new, loss) = train::accum_step(y, &mut optim, &cfg, &loss_fn, s, &mtp_w, &mut streamer, Some(z), &curriculum);
        let t_opt = Instant::now();
        w = schedule_free::update_w(w, &z_new, &y_copy, beta_k);
        let t_end = Instant::now();
        z = z_new;

        let actual_cs = curriculum.current_chunk_size(s as u64, cfg.chunk_size);
        tokens += (cfg.batch_size * actual_cs * ga) as u64;
        ema_loss = 0.05 * loss + 0.95 * ema_loss;
        let lr = train::lr_schedule(cfg.learning_rate, s, cfg.warmup_steps, cfg.max_steps, cfg.wsd_decay_frac);
        let elapsed = t0.elapsed().as_secs_f64().max(0.001);
        let tok_s = tokens as f64 / elapsed;

        if s <= 3 || s % 10 == 0 || s == cfg.max_steps {
            eprintln!("[tr] step {s}/{ms} loss {loss:.6} lr {lr:.6} tok/s {tok_s:.0}",
                ms = cfg.max_steps);
        }
        if s % 100 == 0 { best.update(loss, profile, &z); }
        if s % 100 == 0 || s == cfg.max_steps {
            ckpt::save_latest(profile, &z, &optim);
        }

        logger.log(s, loss, lr, tok_s, tokens,
            (t_fwd - t_lerp).as_secs_f64() * 1000.0,
            (t_opt - t_fwd).as_secs_f64() * 1000.0,
            (t_end - t_opt).as_secs_f64() * 1000.0,
            elapsed);
        prof.record(s, loss, (cfg.batch_size * actual_cs * ga) as u64);
        tui.tick(s, cfg.max_steps, ema_loss, tok_s);
    }

    prof.finish();
    tui.done();
}
