// busel-burn: entry point. YAML → модель → обучение.
mod config; mod data; mod loss; mod model; mod schedule_free; mod train;
mod types; mod ckpt; mod profiler; mod tui; mod decoupled_lr; mod pipeline;

use std::time::Instant;
use config::BuselConfig;
use model::BuselModel;
use model::hybrid::HymOpt;
use schedule_free::ScheduleFree;
use types::{Model, Optim};
use profiler::Profiler;
use tui::TuiHandle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut profile = "chizh-9m";
    let mut yaml = "../configs";
    let (mut bs, mut cs, mut max) = (0usize, 0usize, 0usize);
    let mut pos = 1;
    while pos < args.len() {
        match args[pos].as_str() {
            "--bs" if pos + 1 < args.len() => { bs = args[pos + 1].parse().unwrap(); pos += 2; }
            "--cs" if pos + 1 < args.len() => { cs = args[pos + 1].parse().unwrap(); pos += 2; }
            "--steps" if pos + 1 < args.len() => { max = args[pos + 1].parse().unwrap(); pos += 2; }
            "--yaml" if pos + 1 < args.len() => { yaml = &args[pos + 1]; pos += 2; }
            "--stage" if pos + 1 < args.len() => { pos += 2; }
            a if a.starts_with("--") => { eprintln!("unknown: {a}"); pos += 1; }
            _ => { profile = &args[pos]; pos += 1; }
        }
    }

    let cfg = BuselConfig::from_yaml(yaml, profile, bs, cs, max);
    let ga = cfg.grad_accum_steps;
    if cfg.stage != "pretrain" {
        eprintln!("[busel-burn] stage '{}' not implemented", cfg.stage);
        return;
    }

    let tui = TuiHandle::new(cfg.max_steps);
    let device = Default::default();

    let mdl: Model = BuselModel::new(&cfg, &device);
    let loss_fn = loss::make_loss_fn();
    let mut optim: Optim = burn::optim::adaptor::OptimizerAdaptor::from(
        HymOpt::new(cfg.weight_decay, 5).with(|c| {
            c.lotus = cfg.lotus; c.lotus_rank = cfg.lotus_rank;
            c.adamw_1d = cfg.adamw_1d; c.ema = cfg.ema; c.ema_decay = cfg.ema_decay;
            c.lr_mult_embed = cfg.lr_mult_embed; c.lr_mult_router = cfg.lr_mult_router;
            c.d_byte = cfg.d_byte; c.num_experts = cfg.num_experts;
        })
    );
    let streamer = if let Some(ref w) = cfg.data_weights {
        data::ByteStreamer::open_weighted(std::path::Path::new("../data_train"), w)
    } else {
        data::ByteStreamer::open(std::path::Path::new("../data_train"))
    }.expect("need ../data_train/");
    let mut streamer = streamer;

    let mdl = ckpt::load(profile, mdl);
    let mtp_w: Vec<f64> = (0..=cfg.num_mtp_heads).map(|i| 0.5f64.powi(i as i32)).collect();
    let mut best = ckpt::BestKeeper::new();
    let mut prof = Profiler::new(cfg.max_steps);

    // Schedule-Free init
    let mut sf = if cfg.schedule_free { Some(ScheduleFree::new(&mdl, cfg.sf_beta)) } else { None };

    // warmup (без SF)
    let (mdl_upd, loss0) = train::accum_step(mdl, &mut optim, &cfg, &loss_fn, 0, &mtp_w, &mut streamer, None);
    let mut mdl = mdl_upd;
    prof.record(0, loss0, 0);

    let t0 = Instant::now(); let mut tokens: u64 = 0; let mut ema_loss = loss0;

    for s in 1..=cfg.max_steps {
        let (mdl_upd, loss) = train::accum_step(mdl, &mut optim, &cfg, &loss_fn, s, &mtp_w, &mut streamer, sf.as_mut());
        mdl = mdl_upd;
        tokens += (cfg.batch_size * cfg.chunk_size * ga) as u64;
        ema_loss = 0.05 * loss + 0.95 * ema_loss;
        best.update(loss, profile, &mdl);
        prof.record(s, loss, (cfg.batch_size * cfg.chunk_size * ga) as u64);
        tui.tick(s, cfg.max_steps, ema_loss, tokens as f64 / t0.elapsed().as_secs_f64().max(0.01));

        if s % 100 == 0 || s == cfg.max_steps {
            ckpt::save_latest(profile, &mdl);
        }
    }

    prof.finish();
    tui.done();
}
