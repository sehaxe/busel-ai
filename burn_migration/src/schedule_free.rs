// Schedule-Free (arXiv:2405.15682): y=(1-β)z+βw, grad на y, optim обновляет z.
// Оптимизация: collect_params через ModuleVisitor (без clone модели).
use std::{any::Any, collections::HashMap, marker::PhantomData};
use burn::{module::{Module, ModuleMapper, ModuleVisitor, Param}, tensor::{backend::Backend, Tensor}};
use crate::types::Backend as BType;
use crate::types::Model;

/// Visitor: collect tensors в HashMap без clone модели (visit берёт &self).
struct CollectorVisitor<'a, B: Backend>(&'a mut HashMap<u64, Box<dyn Any>>, PhantomData<B>);
impl<B: Backend> ModuleVisitor<B> for CollectorVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.0.insert(param.id.val(), Box::new(param.val().clone()));
    }
}

/// y = a * (1-β) + b * β
pub struct LerpMapper<'a, B: Backend> {
    b: &'a HashMap<u64, Box<dyn Any>>,
    beta: f64,
    _b: PhantomData<B>,
}
impl<B: Backend> ModuleMapper<B> for LerpMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let key = param.id.val();
        if let Some(any) = self.b.get(&key) {
            let b_t: &Tensor<B, D> = any.downcast_ref().expect("SF lerp D mismatch");
            let new_t = (param.val().clone() * (1.0 - self.beta as f32) + b_t.clone() * (self.beta as f32))
                .detach().require_grad();
            param.map(|_| new_t)
        } else { param }
    }
}

/// result = a + β * (b - c)
pub struct DeltaMapper<'a, B: Backend> {
    b: &'a HashMap<u64, Box<dyn Any>>,
    c: &'a HashMap<u64, Box<dyn Any>>,
    beta: f64,
    _b: PhantomData<B>,
}
impl<B: Backend> ModuleMapper<B> for DeltaMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let key = param.id.val();
        if let (Some(any_b), Some(any_c)) = (self.b.get(&key), self.c.get(&key)) {
            let b_t: &Tensor<B, D> = any_b.downcast_ref().expect("SF delta b D mismatch");
            let c_t: &Tensor<B, D> = any_c.downcast_ref().expect("SF delta c D mismatch");
            let delta = (param.val().clone() + (b_t.clone() - c_t.clone()) * (self.beta as f32))
                .detach().require_grad();
            param.map(|_| delta)
        } else { param }
    }
}

/// Collect all param values into a lookup map (без clone mdl — через ModuleVisitor).
pub fn collect_params(mdl: &Model) -> HashMap<u64, Box<dyn Any>> {
    let mut m = HashMap::new();
    mdl.visit(&mut CollectorVisitor::<BType>(&mut m, PhantomData));
    m
}

/// y = (1-β)·z + β·w. Returns interpolated model (clone z → map).
pub fn lerp_y(z: &Model, w: &Model, beta: f64) -> Model {
    let b = collect_params(w);
    z.clone().map(&mut LerpMapper::<BType> { b: &b, beta, _b: PhantomData })
}

/// w' = w + β·(z - y). Returns updated w.
pub fn update_w(w: Model, z: &Model, y: &Model, beta: f64) -> Model {
    let b = collect_params(z);
    let c = collect_params(y);
    w.map(&mut DeltaMapper::<BType> { b: &b, c: &c, beta, _b: PhantomData })
}
