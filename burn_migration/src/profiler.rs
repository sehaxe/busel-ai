// 📊 Профайлер: ms/step, tok/s, loss per шаг.

use std::time::Instant;

pub struct Profiler {
    steps: usize,
    start: Instant,
    step_start: Instant,
    losses: Vec<f32>,
    accum_tok: u64,
    warmup_done: bool,
}

impl Profiler {
    pub fn new(steps: usize) -> Self {
        let now = Instant::now();
        Profiler {
            steps,
            start: now,
            step_start: now,
            losses: Vec::with_capacity(steps),
            accum_tok: 0,
            warmup_done: false,
        }
    }

    pub fn record(&mut self, step: usize, loss: f32, tok: u64) {
        if step == 0 { return; } // warmup skip
        if !self.warmup_done {
            self.warmup_done = true;
            self.start = Instant::now();
            self.step_start = Instant::now();
            self.accum_tok = 0;
            return;
        }

        self.accum_tok += tok;
        self.step_start = Instant::now();
        self.losses.push(loss);
    }

    pub fn finish(&self) {
        let dt = self.start.elapsed().as_secs_f64().max(0.001);
        let n = self.losses.len().max(1);
        let avg_loss: f32 = self.losses.iter().sum::<f32>() / n as f32;
        let last_loss = self.losses.last().copied().unwrap_or(0.0);

        println!("\n═══ PROFILER ═══");
        println!("  шагов:      {} (из {})", n, self.steps);
        println!("  время:      {:.1}s", dt);
        println!("  tok/s:      {:.0}", self.accum_tok as f64 / dt);
        println!("  ms/step:    {:.1}", dt * 1000.0 / n as f64);
        println!("  loss avg:   {:.4}", avg_loss);
        println!("  loss last:  {:.4}", last_loss);
        println!("═══ ═══════ ═══\n");
    }
}
