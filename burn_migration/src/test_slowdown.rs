use std::time::Instant;
use burn::{
    module::Module,
    nn::{Linear, LinearConfig, Loss, CrossEntropyLoss},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{backend::Backend, Distribution, Int, Tensor},
};
use crate::types::{Backend, Optim};

/// Minimal test: train a 2-layer MLP for 2000 steps, measure ms/step at 100-step intervals.
/// Same pattern as vorobey: forward → backward → optim step.
/// No Schedule-Free, no custom optimizer, no checkpoints.
pub fn run_slowdown_test() {
    let device = Default::default();
    let bs = 256;
    let cs = 256;
    let n_classes = 277;

    // Two linear layers (~500K params) — smaller than vorobey's 2M
    let linear1 = LinearConfig::new(cs, 1024).init(&device);
    let linear2 = LinearConfig::new(1024, n_classes).init(&device);

    let mut optim = Optim::new(
        burn::optim::AdamWConfig::new().with_lr(3e-4)
    );

    let loss_fn = CrossEntropyLoss::new();

    let t0 = Instant::now();
    let n_steps = 2000;

    for step in 1..=n_steps {
        let x = Tensor::<Backend, 2>::random([bs, cs], Distribution::Normal(0.0, 1.0), &device);
        let y = Tensor::<Backend, 1, Int>::random([bs], Distribution::Default, &device)
            .remainder(n_classes);

        let h = linear1.forward(x).relu();
        let logits = linear2.forward(h);
        let loss = loss_fn.forward(logits, y);

        let grads = GradientsParams::from_grads(loss.backward(), &linear2);
        // Need to chain grads from both modules — simple way: update params of both
        let (linear2, _) = optim.step(3e-4, linear2, grads);

        // Get gradients for linear1 too
        let grads1 = GradientsParams::from_grads(linear2.forward(Tensor::random([1, cs], Distribution::Normal(0.0, 1.0), &device)).sum().backward(), &linear1);
        let (linear1, _) = optim.step(3e-4, linear1, grads1);

        if step % 100 == 0 {
            let dt = t0.elapsed().as_secs_f64();
            let avg_ms = dt * 1000.0 / step as f64;
            eprintln!("[test] step {step}/{n_steps} avg {avg_ms:.1}ms/step loss {loss:.4}");
        }
    }

    let total = t0.elapsed().as_secs_f64();
    eprintln!("[test] DONE: {n_steps} steps in {total:.1}s = {:.1}ms/step", total * 1000.0 / n_steps as f64);
}
