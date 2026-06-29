// Minimal test: can we call Backward::prepare from user code?
use burn::backend::autodiff::ops::backward::Backward;
use burn::backend::autodiff::ops::base::OpsKind;
use burn::backend::Autodiff;
use burn::backend::cuda::Cuda;
use burn::tensor::backend::Backend;
use burn::backend::autodiff::checkpoint::strategy::NoCheckpointing;
use burn::backend::autodiff::tensor::AutodiffTensor;

#[derive(Debug)]
struct TestOp;

impl<B: Backend> Backward<B, 2> for TestOp {
    type State = ();
    fn backward(self, _ops: burn::backend::autodiff::ops::backward::Ops<Self::State, 2>, _grads: &mut burn::backend::autodiff::grads::Gradients, _checkpointer: &mut burn::backend::autodiff::checkpoint::base::Checkpointer) {}
}

pub fn test_prepare<B: Backend>(lhs: AutodiffTensor<B>, rhs: AutodiffTensor<B>) {
    let nodes = [lhs.node.clone(), rhs.node.clone()];
    let _result = TestOp.prepare::<NoCheckpointing>(nodes).compute_bound().stateful();
}
