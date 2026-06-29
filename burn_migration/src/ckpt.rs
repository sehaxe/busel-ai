// Чекпоинты через Burn NamedMpkFileRecorder + best-loss tracking.
use std::path::Path;
use burn::{module::Module, record::{FullPrecisionSettings, NamedMpkFileRecorder}};
use crate::types::{Model, Optim};

fn rec() -> NamedMpkFileRecorder<FullPrecisionSettings> {
    NamedMpkFileRecorder::<FullPrecisionSettings>::new()
}

pub fn dir(profile: &str) -> String { format!("checkpoints/{profile}") }
fn latest_path(profile: &str) -> String { format!("{}/latest.mpk", dir(profile)) }
fn best_path(profile: &str) -> String { format!("{}/best.mpk", dir(profile)) }

/// Загрузить latest чекпоинт.
pub fn load(profile: &str, mdl: Model) -> Model {
    let path = latest_path(profile);
    if !Path::new(&path).exists() { return mdl; }
    let dev = Default::default();
    match mdl.clone().load_file(&path, &rec(), &dev) {
        Ok(m) => { eprintln!("ckpt loaded: {path}"); m }
        Err(e) => { eprintln!("ckpt load failed: {e}, fresh start"); mdl }
    }
}

/// Сохранить latest + состояние оптимизатора.
pub fn save_latest(profile: &str, mdl: &Model, optim: &Optim) {
    std::fs::create_dir_all(dir(profile)).ok();
    mdl.clone().save_file(latest_path(profile), &rec()).ok();
    optim.save_to_file(&format!("{}/optim.bin", dir(profile)));
}

/// Загрузить состояние оптимизатора.
pub fn load_optim(profile: &str, optim: &mut Optim) {
    let path = format!("{}/optim.bin", dir(profile));
    if !Path::new(&path).exists() { return; }
    if let Ok(cnt) = optim.load_from_file(&path) {
        eprintln!("optim state loaded: {cnt} params");
    }
}

/// Best-loss копия.
pub struct BestKeeper { best: f32 }
impl BestKeeper {
    pub fn new() -> Self { Self { best: f32::MAX } }
    pub fn update(&mut self, loss: f32, profile: &str, mdl: &Model) {
        if loss >= self.best { return; }
        self.best = loss;
        std::fs::create_dir_all(dir(profile)).ok();
        mdl.clone().save_file(best_path(profile), &rec()).ok();
    }
}
