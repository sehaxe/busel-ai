use std::io::Write;

pub struct JsonlLogger {
    file: std::fs::File,
}

impl JsonlLogger {
    pub fn new(profile: &str) -> Self {
        let path = format!("checkpoints/{profile}/log.jsonl");
        let _ = std::fs::create_dir_all(format!("checkpoints/{profile}"));
        let file = std::fs::OpenOptions::new()
            .create(true).append(true).open(&path)
            .expect("can't open log.jsonl");
        Self { file }
    }

    pub fn log(&mut self, step: usize, loss: f32, lr: f64, tok_s: f64, tokens: u64,
               lerp_ms: f64, fwd_bck_ms: f64, w_update_ms: f64, elapsed_s: f64) {
        let _ = writeln!(self.file, r#"{{"step":{},"loss":{},"lr":{},"tok_s":{},"tokens_total":{},"lerp_ms":{:.1},"fwd_bck_ms":{:.1},"w_update_ms":{:.1},"elapsed_s":{:.1}}}"#,
            step, loss, lr, tok_s as u64, tokens, lerp_ms, fwd_bck_ms, w_update_ms, elapsed_s);
    }
}
