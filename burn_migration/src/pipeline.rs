#![allow(dead_code)]
// Multi-stage pipeline: pretrain → SFT → DPO.
use crate::config::BuselConfig;

pub enum Stage { Pretrain, Sft, Dpo }

impl Stage {
    pub fn from_str(s: &str) -> Self {
        match s { "sft" => Self::Sft, "dpo" => Self::Dpo, _ => Self::Pretrain }
    }
}

pub fn run_stage(stage: Stage, _cfg: BuselConfig) {
    match stage {
        Stage::Pretrain => {} // handled in main
        Stage::Sft => eprintln!("[busel-burn] SFT stage — todo"),
        Stage::Dpo => eprintln!("[busel-burn] DPO stage — todo"),
    }
}
