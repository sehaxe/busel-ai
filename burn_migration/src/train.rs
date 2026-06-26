// Шаг обучения: forward/backward, LR, градиентная аккумуляция.
use burn::{
    nn::loss::CrossEntropyLoss,
    optim::{GradientsParams, GradientsAccumulator, Optimizer},
    tensor::{Int, Tensor},
};
use crate::config::BuselConfig;
use crate::data::ByteStreamer;
use crate::loss::mtp_loss;
use crate::schedule_free::ScheduleFree;
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

fn lcsb_mask(n: usize, ratio: f64, seed: u64) -> Vec<bool> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let threshold = (n as f64 * ratio).min(n as f64) as usize;
    if threshold == 0 { return vec![false; n]; }
    if threshold >= n { return vec![true; n]; }
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (n - threshold..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        idx.swap(i, j);
    }
    let mut mask = vec![false; n];
    for &i in idx.iter().skip(n - threshold) { mask[i] = true; }
    mask
}

/// Один forward + backward.
fn step_inner(
    mdl: &Model, loss_fn: &CrossEntropyLoss<Backend>,
    bs: usize, cs: usize, stride: usize, np: usize, _nmtp: usize, n_layers: usize,
    mtp_w: &[f64], streamer: &mut ByteStreamer,
    selective: bool, backward_ratio: f64, micro_idx: usize,
) -> (f32, GradientsParams) {
    let dev = Default::default();
    let mut flat = Vec::with_capacity(bs * cs);
    for _ in 0..bs { flat.extend(streamer.next_chunk(cs).iter().map(|&b| b as i64)); }
    let ids = Tensor::<Backend, 1, Int>::from_ints(flat.as_slice(), &dev).reshape([bs, cs]);
    let skip = if selective { lcsb_mask(n_layers, backward_ratio, micro_idx as u64) } else { vec![] };
    let (logits_vec, aux_loss) = if selective { mdl.forward_mask(ids.clone(), &skip) } else { mdl.forward(ids.clone()) };

    let mut tf = Vec::with_capacity(bs * np);
    for bi in 0..bs { for pi in 0..np { tf.push(flat[bi * cs + pi * stride]); } }
    let targets = Tensor::<Backend, 1, Int>::from_ints(tf.as_slice(), &dev).reshape([bs * np]);

    let (total, val) = mtp_loss(loss_fn, &logits_vec, targets, mtp_w, aux_loss);
    let grads = GradientsParams::from_grads(total.backward(), mdl);
    (val, grads)
}

/// Gradient accumulation + optim step.
/// Если sf != None: каждый micro-batch считает градиент на z, optim обновляет y.
pub fn accum_step(
    mdl: Model, optim: &mut impl Optimizer<Model, Backend>,
    cfg: &BuselConfig, loss_fn: &CrossEntropyLoss<Backend>, step_idx: usize,
    mtp_w: &[f64], streamer: &mut ByteStreamer,
    mut sf: Option<&mut ScheduleFree>,
) -> (Model, f32) {
    let bs = cfg.batch_size; let cs = cfg.chunk_size;
    let stride = (cs / cfg.n_patches).max(1);
    let ga = cfg.grad_accum_steps;

    let mut acc = GradientsAccumulator::<Model>::new();
    let mut total_loss = 0.0f32;
    let mut mdl = mdl;
    for mi in 0..ga {
        // SF: загрузить z для этого micro-batch'а
        if let Some(sf) = sf.as_mut() { sf.load_z(&mut mdl); }
        let (loss, grads) = step_inner(&mdl, loss_fn, bs, cs, stride, cfg.n_patches,
            cfg.num_mtp_heads, cfg.n_layers, mtp_w, streamer,
            cfg.selective_backward, cfg.backward_ratio, mi);
        // SF: восстановить y (градиенты уже захвачены)
        if let Some(sf) = sf.as_mut() { sf.load_z(&mut mdl); }
        acc.accumulate(&mdl, grads);
        total_loss += loss;
    }
    let grads = acc.grads();

    let lr = lr_schedule(cfg.learning_rate, step_idx, cfg.warmup_steps, cfg.max_steps, cfg.wsd_decay_frac);
    let mut mdl = optim.step(lr, mdl, grads);
    if cfg.routing_free { mdl.update_moe_biases(); }

    // SF: z = lerp(z, y_new), загрузить z для следующего forward
    if let Some(sf) = sf.as_mut() { sf.update(&mut mdl); }

    (mdl, total_loss / ga as f32)
}
