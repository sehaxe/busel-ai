// ⚙️ Гибридный оптимизатор: Muon+ (LOTUS rank-r momentum + NS + col_norm)
// для 2D + AdamW для 1D. Все фичи через флаги. EMA встроена.
// Gradient clipping: clip_norm > 0 включает — clip градиента по норме перед NS.
use burn::{
    optim::{LearningRate, SimpleOptimizer},
    record::Record,
    tensor::{backend::Backend, Tensor, Device, Distribution},
};
use crate::grad_clip::clip_by_norm;

// ── Настройки ──
#[derive(Clone)]
pub struct HymCfg {
    pub wd: f64,
    pub ns_steps: usize,
    pub lr_muon: f64, pub lr_adam: f64,
    pub lotus: bool, pub lotus_rank: usize,
    pub adamw_1d: bool, pub b1: f64, pub b2: f64, pub eps: f64,
    pub ema: bool, pub ema_decay: f64,
    // Decoupled per-layer LR: shape-based detection
    pub lr_mult_embed: f64,   // 0.5 — embed
    pub lr_mult_router: f64,  // 0.5 — router
    pub d_byte: usize,        // для детекта embed по dims[1]
    pub num_experts: usize,   // для детекта router по dims[0]
    pub clip_norm: f64,       // 0 = отключено, >0 = clip grad по norm перед NS
}

impl Default for HymCfg {
    fn default() -> Self {
        Self { wd: 0.01, ns_steps: 5, lr_muon: 1.0, lr_adam: 1.0,
            lotus: true, lotus_rank: 32,
            adamw_1d: true, b1: 0.9, b2: 0.999, eps: 1e-8,
            ema: true, ema_decay: 0.999,
            lr_mult_embed: 0.5, lr_mult_router: 0.5,
            d_byte: 128, num_experts: 6, clip_norm: 1.0 }
    }
}

// ── State ──
#[derive(Clone, Record)]
pub struct HymState<B: Backend, const D: usize> {
    pub m: Option<Tensor<B, D>>,            // momentum / AdamW m
    pub v: Option<Tensor<B, D>>,            // AdamW v (1D only)
    pub u: Option<Tensor<B, D>>,            // LOTUS U (2D only)
    pub vt: Option<Tensor<B, D>>,           // LOTUS V (2D only)
    pub ema: Option<Tensor<B, D>>,          // EMA веса
    pub step: u64,
}

// ── Optimizer ──
#[derive(Clone)]
pub struct HymOpt {
    pub cfg: HymCfg,
}

impl HymOpt {
    pub fn new(wd: f64, ns_steps: usize) -> Self {
        Self { cfg: HymCfg { wd, ns_steps, ..Default::default() } }
    }
    pub fn with(mut self, f: impl FnOnce(&mut HymCfg)) -> Self { f(&mut self.cfg); self }
}

impl<B: Backend> SimpleOptimizer<B> for HymOpt {
    type State<const D: usize> = HymState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let c = &self.cfg;
        // Shape-based per-layer LR: detect embed/router by dims
        let lr_mult = if D < 2 { 1.0 } else {
            let s = tensor.dims();
            if s.len() >= 2 && s[1] == c.d_byte { c.lr_mult_embed }
            else if s[0] == c.num_experts { c.lr_mult_router }
            else { 1.0 }
        };
        let lr = lr * lr_mult;

        let mut s: HymState<B, D> = state.unwrap_or(HymState {
            m: None, v: None, u: None, vt: None, ema: None, step: 0 });
        s.step += 1;

        let new_param = if D == 2 {
            // ── Muon+: NS + col_norm → LOTUS rank-r ──
            let sh = grad.dims(); let (o, i) = (sh[0], sh[1]); let r = c.lotus_rank;
            // 0. Gradient clipping by norm (optional)
            let grad = if c.clip_norm > 0.0 { clip_by_norm(grad, c.clip_norm) } else { grad };
            // 1. Newton-Schulz orthogonalization of gradient (Muon)
            let g = if c.ns_steps > 0 { zpns(grad.clone(), c.ns_steps) } else { grad.clone() };
            // 2. Column norm (Muon+)
            let g = cn2(g);
            // 3. LOTUS rank-r momentum
            if c.lotus {
                let lu = c.lr_muon * 0.01 / (i as f64).sqrt();
                let lv = c.lr_muon * 0.01 / (o as f64).sqrt();
                if s.u.is_none() {
                    let proj = Tensor::random([i, r], Distribution::Normal(0.0, 1.0), &grad.device());
                    let u0 = cn2(grad.clone().matmul(proj));
                    let v0 = cn2(grad.clone().transpose().matmul(u0.clone()));
                    s.u = Some(u0); s.vt = Some(v0);
                }
                let u = s.u.as_ref().unwrap().clone();
                let v = s.vt.as_ref().unwrap().clone();
                let u2 = cn2(u + g.clone().matmul(v.clone()) * lu);
                let v2 = cn2(v + g.clone().transpose().matmul(u2.clone()) * lv);
                s.u = Some(u2.clone()); s.vt = Some(v2.clone());
                let adj = lr * c.lr_muon * (o as f64 / i as f64).max(1.0).sqrt();
                tensor * (1.0 - lr * c.wd) - u2.matmul(v2.transpose()) * adj
            } else {
                // Direct Muon+ (no rank-r compression)
                let adj = lr * c.lr_muon;
                tensor * (1.0 - lr * c.wd) - g * adj
            }
        } else if c.adamw_1d {
            // ── AdamW для 1D ──
            let b1 = c.b1; let b2 = c.b2; let ep = c.eps;
            let m = match s.m.clone() { Some(m) => m * b1 + grad.clone() * (1.0 - b1), None => grad.clone() };
            let v = match s.v.clone() { Some(v) => v * b2 + (grad.clone() * grad.clone()) * (1.0 - b2), None => grad.clone() * grad.clone() };
            let mh = m.clone() / (1.0 - b1.powi(s.step as i32));
            let vh = v.clone() / (1.0 - b2.powi(s.step as i32));
            s.m = Some(m); s.v = Some(v);
            let lre = lr * c.lr_adam;
            tensor * (1.0 - lre * c.wd) - (mh / (vh.sqrt() + ep)) * lre
        } else {
            // ── Fallback: простой momentum ──
            let dec = (lr * c.wd) as f32;
            s.m = Some(match s.m.clone() {
                Some(m) => grad.clone() * (1.0 - 0.95 * (1.0 - dec).min(1.0)) + m * 0.95,
                None => grad.clone(),
            });
            let adj = if D >= 2 { lr * c.lr_muon } else { lr * c.lr_adam };
            tensor * (1.0 - lr * c.wd) - s.m.clone().unwrap() * adj
        };

        // ── EMA (после обновления весов) ──
        if c.ema {
            s.ema = Some(match s.ema.clone() {
                Some(e) => e * c.ema_decay + new_param.clone() * (1.0 - c.ema_decay),
                None => new_param.clone(),
            });
        }

        (new_param, Some(s))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        fn mv<B: Backend, const D: usize>(t: Option<Tensor<B, D>>, d: &Device<B>) -> Option<Tensor<B, D>> {
            t.map(|x| Tensor::from_data(x.into_data(), d))
        }
        HymState {
            m: mv(state.m, device), v: mv(state.v, device),
            u: mv(state.u, device), vt: mv(state.vt, device),
            ema: mv(state.ema, device),
            step: state.step,
        }
    }
}

/// Column norm по нулевой оси (col_norm для LOTUS).
fn cn2<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let n = (x.clone() * x.clone()).sum_dim(0).sqrt().clamp_min(1e-8);
    x / n
}

/// Newton-Schulz quintic iterations для Muon+.
fn zpns<B: Backend, const D: usize>(g: Tensor<B, D>, steps: usize) -> Tensor<B, D> {
    let dims = g.dims();
    let tall = dims[D - 2] > dims[D - 1];
    let mut x = if tall { g.swap_dims(D - 2, D - 1) } else { g };
    // ponytail: tensor norm avoids GPU→CPU sync (was into_scalar = 54 drains/step).
    let ns = x.clone().powf_scalar(2.0).sum().sqrt().clamp_min(1e-7);  // [1]
    let ones: [usize; D] = core::array::from_fn(|_| 1);
    let ns = ns.reshape(ones).expand(x.dims());
    x = x / ns;
    for _ in 0..steps {
        let xt = x.clone().swap_dims(D - 2, D - 1);
        let a = x.clone().matmul(xt);
        let a2 = a.clone().matmul(a.clone());
        x = x.clone() * 3.4445f32 + (a * (-4.775f32) + a2 * 2.0315f32).matmul(x);
    }
    if tall { x.swap_dims(D - 2, D - 1) } else { x }
}
