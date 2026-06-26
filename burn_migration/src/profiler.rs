// 📊 Профайлер: ms/step, GPU память, tok/s, loss per шаг. Всё в одном файле.
// Использует std::time::Instant + nvidia-smi для GPU метрик.

use std::time::Instant;
use std::process::Command;

static mut GPU_PEAK_MB: f64 = 0.0;

fn gpu_mem_mb() -> f64 {
    // nvidia-smi — самый надёжный способ без cuda-sys зависимостей
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse::<f64>().unwrap_or(0.0)
        }
        Err(_) => 0.0,
    }
}

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

        // GPU память (nvidia-smi вызов ~5ms, каждые 10 шагов)
        if step % 10 == 0 {
            let mem = gpu_mem_mb();
            unsafe {
                if mem > GPU_PEAK_MB { GPU_PEAK_MB = mem; }
            }
        }
    }

    pub fn finish(&self) {
        let dt = self.start.elapsed().as_secs_f64().max(0.001);
        let n = self.losses.len().max(1);
        let avg_loss: f32 = self.losses.iter().sum::<f32>() / n as f32;
        let last_loss = self.losses.last().copied().unwrap_or(0.0);
        let mem = unsafe { GPU_PEAK_MB };

        println!("\n═══ PROFILER ═══");
        println!("  шагов:      {} (из {})", n, self.steps);
        println!("  время:      {:.1}s", dt);
        println!("  tok/s:      {:.0}", self.accum_tok as f64 / dt);
        println!("  ms/step:    {:.1}", dt * 1000.0 / n as f64);
        println!("  loss avg:   {:.4}", avg_loss);
        println!("  loss last:  {:.4}", last_loss);
        println!("  peak GPU:   {:.0} MB", mem);
        println!("═══ ═══════ ═══\n");
    }
}
