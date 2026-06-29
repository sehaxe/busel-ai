// ASCII + Chunk curriculum.

pub struct Curriculum {
    pub max_steps: u64,
    pub ascii_frac: f64,
    pub chunk_growth_steps: usize, // кол-во фаз (0 = выкл, 5 = 1/16→1/8→1/4→1/2→1)
}

impl Curriculum {
    pub fn new(max_steps: u64, ascii_frac: f64, chunk_growth: bool) -> Self {
        Self {
            max_steps,
            ascii_frac,
            chunk_growth_steps: if chunk_growth { 5 } else { 0 },
        }
    }

    /// true пока не прошло ascii_frac тренировки
    pub fn ascii_active(&self, step: u64) -> bool {
        step < (self.max_steps as f64 * self.ascii_frac) as u64
    }

    /// Дробный chunk_size: 1/2^phase от полного
    pub fn current_chunk_size(&self, step: u64, full_chunk: usize) -> usize {
        if self.chunk_growth_steps == 0 {
            return full_chunk;
        }
        let p = step as f64 / self.max_steps.max(1) as f64;
        let phase = (p * self.chunk_growth_steps as f64).min((self.chunk_growth_steps - 1) as f64) as usize;
        let divisor = 2u64.pow((self.chunk_growth_steps - 1 - phase) as u32);
        (full_chunk / divisor as usize).max(8)
    }
}
