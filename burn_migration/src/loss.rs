// MTP loss: кросс-энтропия с весами по головам.
use burn::{nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig}, tensor::{Int, Tensor}};
use crate::types::Backend;

pub fn make_loss_fn() -> CrossEntropyLoss<Backend> {
    CrossEntropyLossConfig::new().init(&Default::default())
}

/// MTP weighted loss: sum_i w_i * CE(logits_i, targets[i]).
/// w_i = 0.5^i (default).
pub fn mtp_loss(
    loss_fn: &CrossEntropyLoss<Backend>,
    logits_vec: &[Tensor<Backend, 3>],
    targets: &[Tensor<Backend, 1, Int>],
    mtp_w: &[f64],
    aux_loss: Tensor<Backend, 1>,
) -> (Tensor<Backend, 1>, f32) {
    let [b, _t, _v] = logits_vec[0].dims();
    let mut total = Tensor::zeros([1], &logits_vec[0].device());
    for (i, logits_i) in logits_vec.iter().enumerate() {
        let [_, ti, vi] = logits_i.dims();
        total = total + loss_fn.forward(logits_i.clone().reshape([b * ti, vi]), targets[i].clone()) * mtp_w[i];
    }
    total = total + aux_loss * 0.001;
    let val: f32 = total.clone().into_scalar().into();
    (total, val)
}
