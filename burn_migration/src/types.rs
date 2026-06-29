// ⚙️ Точность вычислений — одна строка меняет точность для всех тензоров:
pub type FloatElem = burn::tensor::bf16;  // BF16 — 2× throughput, <1% loss

// ⚙️ Gradient checkpointing — NoCheckpointing
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
pub(crate) type Checkpoint = NoCheckpointing;

use burn::backend::{Autodiff, cuda::Cuda};
use crate::model::BuselModel;
use crate::optim_wrapper::HymOptWrapper;

pub type Backend = Autodiff<Cuda<FloatElem, i32>, Checkpoint>;
pub type Model = BuselModel<Backend>;
pub type Optim = HymOptWrapper<Cuda<FloatElem, i32>>;
