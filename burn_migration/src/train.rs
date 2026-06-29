// Шаг обучения: forward/backward, LR, градиентная аккумуляция.
use std::time::Instant;
use burn::{
    nn::loss::CrossEntropyLoss,
    optim::{GradientsParams, GradientsAccumulator, Optimizer},
    tensor::{Int, Tensor},
};
use crate::config::BuselConfig;
use crate::data::ByteStreamer;
use crate::loss::mtp_loss;
use crate::dropbp::combined_mask;
use crate::curriculum::Curriculum;
use crate::types::{Backend, Model};

/// WSD: warmup → stable → decay-to-30%.
pub fn lr_schedule(base: f64, step: usize, warmup: usize, max: usize, decay_frac: f64) -> f64 {
    if step < warmup {
        base * (step as f64 / warmup.max(1) as f64)
    } else {
        let ds = (max as f64 * (1.0 - decay_frac)) as usize;
        if step >= ds && ds < max {
            let p = (step - ds) as f64 / (max - ds) as f64;
            base * (1.0 - p * 0.7)
        } else { base }
    }
}

/// Один forward + backward на mdl.
pub fn step_inner(
    mdl: &Model, loss_fn: &CrossEntropyLoss<Backend>,
    bs: usize, cs: usize, np_yaml: usize, nmtp: usize, n_layers: usize,
    mtp_w: &[f64], streamer: &mut ByteStreamer,
    selective: bool, backward_ratio: f64, ckpt_every: usize, micro_idx: usize,
    dropbp_prob: f64, freeze_progress: f64, ascii_active: bool,
) -> (f32, GradientsParams) {
    let dev = Default::default();
    let mut flat = vec![0i64; bs * cs];
    streamer.fill_batch(&mut flat);

    // pony: ASCII curriculum in one line
    if ascii_active {
        for v in &mut flat { *v &= 0x7F; }
    }

    let ids = Tensor::<Backend, 1, Int>::from_ints(flat.as_slice(), &dev).reshape([bs, cs]);

    let stride = (cs / np_yaml).max(1);
    let np = np_yaml.min(cs / stride);
    let mut mtp_tgts: Vec<Tensor<Backend, 1, Int>> = Vec::with_capacity(1 + nmtp);
    for mi in 0..=nmtp {
        let mut tgt = vec![0i64; bs * np];
        for bi in 0..bs { for pi in 0..np {
            let idx = (bi * cs + pi * stride + mi).min(bs * cs - 1);
            tgt[bi * np + pi] = flat[idx];
        }}
        mtp_tgts.push(Tensor::from_ints(tgt.as_slice(), &dev));
    }

    let skip = if ckpt_every > 0 {
        (0..n_layers).map(|i| i % ckpt_every != 0 && i != n_layers - 1).collect()
    } else if selective || dropbp_prob > 0.0 || freeze_progress > 0.0 {
        combined_mask(n_layers, dropbp_prob, if selective { backward_ratio } else { 1.0 }, freeze_progress, micro_idx as u64)
    } else {
        vec![]
    };
    let use_mask = !skip.is_empty();
    let (logits_vec, aux_loss) = if use_mask {
        mdl.forward_mask(ids.clone(), &skip, Some(&mtp_tgts))
    } else {
        mdl.forward(ids.clone(), Some(&mtp_tgts))
    };

    let (total, val) = mtp_loss(loss_fn, &logits_vec, &mtp_tgts, mtp_w, aux_loss);
    let bwd_start = Instant::now();
    let grads = GradientsParams::from_grads(total.backward(), mdl);
    eprintln!("[perf2] step_inner bwd={:.1}ms", bwd_start.elapsed().as_secs_f64() * 1000.0);
    (val, grads)
}

/// Gradient accumulation + optim step.
pub fn accum_step(
    mdl: Model, optim: &mut impl Optimizer<Model, Backend>,
    cfg: &BuselConfig, loss_fn: &CrossEntropyLoss<Backend>, step_idx: usize,
    mtp_w: &[f64], streamer: &mut ByteStreamer,
    z_opt: Option<Model>,
    curriculum: &Curriculum,
) -> (Model, f32) {
    let bs = cfg.batch_size;
    let cs = curriculum.current_chunk_size(step_idx as u64, cfg.chunk_size);
    let ga = cfg.grad_accum_steps;
    let prog = if cfg.progressive_freeze {
        (step_idx as f64 / cfg.max_steps.max(1) as f64).min(1.0)
    } else { 0.0 };
    let ascii = cfg.ascii_curriculum && curriculum.ascii_active(step_idx as u64);

    let mut acc = GradientsAccumulator::<Model>::new();
    let mut total_loss = 0.0f32;
    for mi in 0..ga {
        let (loss, grads) = step_inner(&mdl, loss_fn, bs, cs, cfg.n_patches,
            cfg.num_mtp_heads, cfg.n_layers, mtp_w, streamer,
            cfg.selective_backward, cfg.backward_ratio, cfg.grad_ckpt_every, mi,
            cfg.dropbp_prob, prog, ascii);
        acc.accumulate(&mdl, grads);
        total_loss += loss;
    }
    let grads = acc.grads();

    let lr = lr_schedule(cfg.learning_rate, step_idx, cfg.warmup_steps, cfg.max_steps, cfg.wsd_decay_frac);

    let optim_start = Instant::now();
    let updated = if let Some(z) = z_opt {
        optim.step(lr, z, grads)
    } else {
        optim.step(lr, mdl, grads)
    };
    let mut mdl = updated;
    if cfg.num_experts > 1 { mdl.update_moe_biases(); }
    eprintln!("[perf2] optim.step={:.1}ms", optim_start.elapsed().as_secs_f64() * 1000.0);

    (mdl, total_loss / ga as f32)
}
