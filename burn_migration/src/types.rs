// ⚙️ Точность вычислений — одна строка меняет точность для всех тензоров:
pub type FloatElem = burn::tensor::bf16;  // BF16 — 2× throughput, <1% loss
// pub type FloatElem = f32;               // FP32 — эталон, 100% стабильно
// pub type FloatElem = burn::tensor::f16; // FP16 — 2× throughput, риск underflow

// ⚙️ Gradient checkpointing — одна строка:
use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
// Закомментируй BalancedCheckpointing чтобы выключить:
// use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing as Checkpoint;

/// Выбранная стратегия чекпоинтинга.
pub(crate) type Checkpoint = BalancedCheckpointing;

use burn::backend::{Autodiff, cuda::Cuda};
use burn::optim::adaptor::OptimizerAdaptor;
use crate::model::BuselModel;
use crate::model::hybrid::HymOpt;

pub type Backend = Autodiff<Cuda<FloatElem, i32>, Checkpoint>;
pub type Model = BuselModel<Backend>;
pub type Optim = OptimizerAdaptor<HymOpt, Model, Backend>;
