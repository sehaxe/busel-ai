use std::collections::HashMap;
use std::any::Any;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use burn::{
    module::{AutodiffModule, ModuleMapper, Param, ParamMapper},
    optim::{GradientsParams, LearningRate, Optimizer, MultiGradientsParams, SimpleOptimizer},
    tensor::{Tensor, backend::{AutodiffBackend, Backend}},
};
use crate::model::hybrid::{HymOpt, HymState};

/// HymOptWrapper: public Optimizer impl via ModuleMapper.
#[derive(Clone)]
pub struct HymOptWrapper<IB: Backend> {
    pub inner: HymOpt,
    pub states: Arc<Mutex<HashMap<u64, Box<dyn Any + Send>>>>,
    _phantom: PhantomData<IB>,
}

impl<IB: Backend> HymOptWrapper<IB> {
    pub fn new(inner: HymOpt) -> Self {
        Self { inner, states: Arc::new(Mutex::new(HashMap::new())), _phantom: PhantomData }
    }
}

// ── Save/load optimizer state ──
use burn::tensor::DType;

impl<IB: Backend> HymOptWrapper<IB> {
    pub fn save_to_file(&self, path: &str) {
        use std::io::Write;
        use crate::model::hybrid::HymState as HS;
        let guard = self.states.lock().unwrap();
        let mut out = Vec::new();
        let mut count = 0u32;
        for (&id, state) in guard.iter() {
            if let Some(s) = state.downcast_ref::<HS<IB, 2>>() {
                let tensors = [
                    ("m", &s.m), ("v", &s.v), ("u", &s.u), ("vt", &s.vt), ("ema", &s.ema),
                ];
                let mut buf = Vec::new();
                for (name, t) in &tensors {
                    if let Some(ref x) = t {
                        let dims = x.dims();
                        let d = x.clone().into_data().convert_dtype(DType::F32);
                        let db: Vec<u8> = d.as_bytes().to_vec();
                        buf.write_all(&(name.len() as u32).to_le_bytes()).ok();
                        buf.write_all(name.as_bytes()).ok();
                        buf.write_all(&(dims.len() as u32).to_le_bytes()).ok();
                        for &s in &dims { buf.write_all(&(s as u64).to_le_bytes()).ok(); }
                        buf.write_all(&(db.len() as u32).to_le_bytes()).ok();
                        buf.write_all(&db).ok();
                    }
                }
                if !buf.is_empty() {
                    out.write_all(&id.to_le_bytes()).ok();
                    out.write_all(&s.step.to_le_bytes()).ok();
                    out.write_all(&(buf.len() as u32).to_le_bytes()).ok();
                    out.write_all(&buf).ok();
                    count += 1;
                }
            } else if let Some(s) = state.downcast_ref::<HS<IB, 1>>() {
                let tensors = [("m", &s.m), ("v", &s.v), ("ema", &s.ema)];
                let mut buf = Vec::new();
                for (name, t) in &tensors {
                    if let Some(ref x) = t {
                        let dims = x.dims();
                        let d = x.clone().into_data().convert_dtype(DType::F32);
                        let db: Vec<u8> = d.as_bytes().to_vec();
                        buf.write_all(&(name.len() as u32).to_le_bytes()).ok();
                        buf.write_all(name.as_bytes()).ok();
                        buf.write_all(&(dims.len() as u32).to_le_bytes()).ok();
                        for &s in &dims { buf.write_all(&(s as u64).to_le_bytes()).ok(); }
                        buf.write_all(&(db.len() as u32).to_le_bytes()).ok();
                        buf.write_all(&db).ok();
                    }
                }
                if !buf.is_empty() {
                    out.write_all(&id.to_le_bytes()).ok();
                    out.write_all(&s.step.to_le_bytes()).ok();
                    out.write_all(&(buf.len() as u32).to_le_bytes()).ok();
                    out.write_all(&buf).ok();
                    count += 1;
                }
            }
        }
        drop(guard);
        let mut header = Vec::new();
        header.write_all(b"OPT1").ok();
        header.write_all(&count.to_le_bytes()).ok();
        let full = [header.as_slice(), &out].concat();
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap_or(std::path::Path::new("."))).ok();
        std::fs::write(path, &full).ok();
        eprintln!("[ckpt] optim saved: {count} params");
    }

    pub fn load_from_file(&self, path: &str) -> Result<u32, String> {
        use crate::model::hybrid::HymState as HS;
        use burn::tensor::Bytes as BurnBytes;
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if bytes.len() < 8 || &bytes[0..4] != b"OPT1" {
            return Err("bad header".into());
        }
        let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let dev = Default::default();
        let mut pos = 8usize;
        let mut loaded = 0u32;
        let mut guard = self.states.lock().unwrap();
        while pos + 16 <= bytes.len() && loaded < count {
            let id = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()); pos += 8;
            let step = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()); pos += 8;
            let blk = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize; pos += 4;
            let end = pos + blk;
            let mut m2 = None::<Tensor<IB, 2>>; let mut v2 = None;
            let mut u2 = None; let mut vt2 = None; let mut ema2 = None;
            let mut m1 = None::<Tensor<IB, 1>>; let mut v1 = None; let mut ema1 = None;
            while pos + 8 <= end {
                let nl = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize; pos += 4;
                let name = String::from_utf8(bytes[pos..pos+nl].to_vec()).unwrap(); pos += nl;
                let dl = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize; pos += 4;
                let mut dims = Vec::new();
                for _ in 0..dl {
                    let s = u64::from_le_bytes(bytes[pos..pos+8].try_into().unwrap()) as usize; pos += 8;
                    dims.push(s);
                }
                let bl2 = u32::from_le_bytes(bytes[pos..pos+4].try_into().unwrap()) as usize; pos += 4;
                let raw = bytes[pos..pos+bl2].to_vec(); pos += bl2;
                let b = BurnBytes::from_bytes_vec(raw);
                if dims.len() == 2 {
                    let t: Tensor<IB, 2> = Tensor::from_data(
                        burn::tensor::TensorData::from_bytes(b, &dims[..], DType::F32), &dev);
                    match name.as_str() {
                        "m" => m2 = Some(t), "v" => v2 = Some(t),
                        "u" => u2 = Some(t), "vt" => vt2 = Some(t),
                        "ema" => ema2 = Some(t), _ => {}
                    }
                } else if dims.len() == 1 {
                    let t: Tensor<IB, 1> = Tensor::from_data(
                        burn::tensor::TensorData::from_bytes(b, &dims[..], DType::F32), &dev);
                    match name.as_str() {
                        "m" => m1 = Some(t), "v" => v1 = Some(t),
                        "ema" => ema1 = Some(t), _ => {}
                    }
                }
            }
            if m2.is_some() {
                guard.insert(id, Box::new(HS::<IB, 2> { m: m2, v: v2, u: u2, vt: vt2, ema: ema2, step }));
            } else if m1.is_some() {
                guard.insert(id, Box::new(HS::<IB, 1> { m: m1, v: v1, u: None, vt: None, ema: ema1, step }));
            }
            loaded += 1;
        }
        Ok(loaded)
    }
}

// ── HymMapper ──
struct HymMapper<'a, IB: Backend> {
    opt: &'a HymOpt,
    states: &'a Mutex<HashMap<u64, Box<dyn Any + Send>>>,
    lr: LearningRate,
    _phantom: PhantomData<IB>,
}

impl<'a, IB: Backend> HymMapper<'a, IB> {
    fn new(opt: &'a HymOpt, states: &'a Mutex<HashMap<u64, Box<dyn Any + Send>>>, lr: LearningRate) -> Self {
        Self { opt, states, lr, _phantom: PhantomData }
    }
}

// ── GradSource: enum over single/multi ──
enum GradSource<'a> {
    Single(&'a mut GradientsParams),
    Multi(&'a mut MultiGradientsParams),
}

impl GradSource<'_> {
    fn take<IB: Backend, const D: usize>(&mut self, id: burn::module::ParamId) -> Option<Tensor<IB, D>> {
        match self {
            GradSource::Single(g) => g.remove::<IB, D>(id),
            GradSource::Multi(g) => g.remove::<IB, D>(id).map(|(t, _dev)| t),
        }
    }
}

// ── ModuleMapper: применяем HymOpt к каждому параметру ──
struct ParamMapperImpl<'a, 'b, IB: Backend, B: AutodiffBackend<InnerBackend = IB>> {
    hym: &'a mut HymMapper<'b, IB>,
    grad: GradSource<'a>,
    _phantom: PhantomData<(IB, B)>,
}

impl<IB: Backend, B: AutodiffBackend<InnerBackend = IB>> ModuleMapper<B>
    for ParamMapperImpl<'_, '_, IB, B>
{
    fn map_float<const D: usize>(
        &mut self,
        param: Param<Tensor<B, D>>,
    ) -> Param<Tensor<B, D>> {
        let (id, tensor, p_mapper) = param.consume();
        let tensor = match self.grad.take::<IB, D>(id) {
            Some(grad) => step_one::<IB, B, D>(self.hym, id, tensor, grad, p_mapper),
            None => return Param::from_mapped_value(id, tensor, p_mapper),
        };
        tensor
    }
}

fn step_one<IB: Backend, B: AutodiffBackend<InnerBackend = IB>, const D: usize>(
    m: &mut HymMapper<IB>,
    id: burn::module::ParamId,
    tensor: Tensor<B, D>,
    grad: Tensor<IB, D>,
    p_mapper: ParamMapper<Tensor<B, D>>,
) -> Param<Tensor<B, D>> {
    let mut guard = m.states.lock().unwrap();
    let state = guard.remove(&id.val())
        .and_then(|b| b.downcast::<HymState<IB, D>>().ok().map(|b| *b));
    let tensor_inner: Tensor<IB, D> = tensor.inner();
    let (new_tensor, new_state) =
        <HymOpt as SimpleOptimizer<IB>>::step(m.opt, m.lr, tensor_inner, grad, state);
    if let Some(s) = new_state {
        guard.insert(id.val(), Box::new(s));
    }
    drop(guard);
    let mut t = Tensor::from_inner(new_tensor);
    t = t.require_grad();
    Param::from_mapped_value(id, t, p_mapper)
}

// ── Optimizer impl ──
impl<IB: Backend, M, B> Optimizer<M, B> for HymOptWrapper<IB>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend<InnerBackend = IB>,
{
    type Record = ();

    fn step(&mut self, lr: LearningRate, module: M, mut grads: GradientsParams) -> M {
        let mut mapper = HymMapper::new(&self.inner, &self.states, lr);
        let mut pmi = ParamMapperImpl::<IB, B> {
            hym: &mut mapper,
            grad: GradSource::Single(&mut grads),
            _phantom: PhantomData,
        };
        module.map(&mut pmi)
    }

    fn step_multi(&mut self, lr: LearningRate, module: M, mut grads: MultiGradientsParams) -> M {
        let mut mapper = HymMapper::new(&self.inner, &self.states, lr);
        let mut pmi = ParamMapperImpl::<IB, B> {
            hym: &mut mapper,
            grad: GradSource::Multi(&mut grads),
            _phantom: PhantomData,
        };
        module.map(&mut pmi)
    }

    fn to_record(&self) -> Self::Record { () }
    fn load_record(self, _rec: Self::Record) -> Self { self }
}
