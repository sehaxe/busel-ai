// Schedule-Free: z-weights maintain, gradient computed at z, update applied to y.
// ModuleMapper + Box<dyn Any> для swap между y/z на уровне тензоров.
use std::{any::Any, collections::HashMap, marker::PhantomData};
use burn::module::{Module, ModuleMapper};
use burn::module::Param;
use burn::tensor::{backend::Backend, Tensor};
use crate::types::Model;

struct Collector<'a, B: Backend>(&'a mut HashMap<u64, Box<dyn Any>>, PhantomData<B>);
impl<B: Backend> ModuleMapper<B> for Collector<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        self.0.insert(param.id.val(), Box::new(param.val().clone()));
        param
    }
}

struct LerpMapper<'a, B: Backend> {
    y: &'a HashMap<u64, Box<dyn Any>>,
    beta: f64,
    _b: PhantomData<B>,
}
impl<B: Backend> ModuleMapper<B> for LerpMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let key = param.id.val();
        if let Some(any) = self.y.get(&key) {
            let y_t: &Tensor<B, D> = any.downcast_ref().expect("SF lerp D mismatch");
            let new_t = (1.0 - self.beta) * param.val().clone() + self.beta * y_t.clone();
            param.map(|_| new_t)
        } else {
            param
        }
    }
}

pub struct ScheduleFree {
    pub z: Model,
    pub beta: f64,
}

impl ScheduleFree {
    pub fn new(mdl: &Model, beta: f64) -> Self {
        Self { z: mdl.clone(), beta }
    }

    /// Загрузить z в модель (для forward). model ↔ z swap.
    pub fn load_z(&mut self, model: &mut Model) {
        std::mem::swap(model, &mut self.z);
    }

    /// Обновить z = (1-β)z + β·y_new и загрузить z обратно в модель.
    pub fn update(&mut self, model: &mut Model) {
        let mut y_params: HashMap<u64, Box<dyn Any>> = HashMap::new();
        let _ = model.clone().map(&mut Collector(&mut y_params, PhantomData));
        let z_new = self.z.clone().map(&mut LerpMapper { y: &y_params, beta: self.beta, _b: PhantomData });
        self.z = z_new;
        std::mem::swap(model, &mut self.z);
    }
}
