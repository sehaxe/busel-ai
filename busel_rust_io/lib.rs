use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use std::fs::File;
use std::io;
use memmap2::Mmap;

// ponytail: ternary packing 5:8 — 5 ternary values {-1,0,1} in 1 byte (3^5=243<256). 20× weight compression.

#[pyfunction]
fn pack_ternary_5_8(weights: Vec<i8>) -> PyResult<Vec<u8>> {
    let mut packed = Vec::with_capacity((weights.len() + 4) / 5);
    for chunk in weights.chunks(5) {
        let mut val: u8 = 0;
        for (i, &w) in chunk.iter().enumerate() {
            let t: u8 = match w { -1 => 0, 0 => 1, 1 => 2, _ => 1 };
            val += t * 3u8.pow(i as u32);
        }
        packed.push(val);
    }
    Ok(packed)
}

#[pyfunction]
fn unpack_ternary_5_8(packed: Vec<u8>, count: usize) -> PyResult<Vec<i8>> {
    let mut weights = Vec::with_capacity(count);
    for &p in &packed {
        let mut val = p as u32;
        for _ in 0..5 {
            let t = val % 3;
            weights.push(match t { 0 => -1i8, 1 => 0, 2 => 1, _ => 0 });
            val /= 3;
        }
    }
    weights.truncate(count);
    Ok(weights)
}

#[pyclass]
struct ByteStreamer {
    mmap: Mmap,
    _file: File,
    position: usize,
    chunk_size: usize,
}

#[pymethods]
impl ByteStreamer {
    #[new]
    fn new(file_path: String, chunk_size: usize, start_offset: usize) -> PyResult<Self> {
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // ponytail: MADV_SEQUENTIAL — kernel auto-prefetches ahead, frees behind.
        // Prevents 22GB page cache from OOM-killing on 16GB RAM without thrashing.
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(mmap.as_ptr() as *mut libc::c_void, mmap.len(), libc::MADV_SEQUENTIAL);
        }
        Ok(ByteStreamer { mmap, _file: file, position: start_offset, chunk_size })
    }

    fn next_chunk(&mut self, _py: Python) -> Option<Vec<u8>> {
        if self.position >= self.mmap.len() { return None; }
        let end = std::cmp::min(self.position + self.chunk_size, self.mmap.len());
        let mut chunk = self.mmap[self.position..end].to_vec();
        // ponytail: let OS manage page cache — MADV_DONTNEED thrashing was causing RAM churn.
        // 22GB data on 15GB RAM: OS evicts LRU pages naturally under memory pressure.
        if chunk.len() < self.chunk_size { chunk.resize(self.chunk_size, 0u8); }
        self.position = end;
        Some(chunk)
    }

    fn get_position(&self) -> usize { self.position }
}

#[pyfunction]
fn ternary_matmul_cpu(input: Vec<f32>, weights: Vec<i8>, rows: usize, k: usize) -> PyResult<Vec<f32>> {
    // ponytail: y = W @ x where W is (rows, k), x is (k,). Ternary: add/sub only, no multiply.
    let mut output = vec![0f32; rows];
    for i in 0..rows {
        let mut sum = 0f32;
        for j in 0..k {
            let w = weights[i * k + j];
            if w != 0 { sum += if w > 0 { input[j] } else { -input[j] }; }
        }
        output[i] = sum;
    }
    Ok(output)
}

#[pyfunction]
fn fast_save_checkpoint(path: String, data: Vec<u8>) -> PyResult<()> {
    std::fs::write(path, data)?;
    Ok(())
}

#[pyfunction]
fn fast_load_checkpoint(path: String) -> PyResult<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

#[pyfunction]
fn append_to_binary_file(path: String, data: Vec<u8>) -> PyResult<()> {
    use std::io::Write;
    use std::fs::OpenOptions;
    let mut f = OpenOptions::new().append(true).create(true).open(path)?;
    f.write_all(&data)?;
    Ok(())
}

#[pyfunction]
fn get_cpu_count() -> PyResult<usize> {
    Ok(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
}

#[pymodule]
fn busel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteStreamer>()?;
    m.add_function(wrap_pyfunction!(pack_ternary_5_8, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_ternary_5_8, m)?)?;
    m.add_function(wrap_pyfunction!(ternary_matmul_cpu, m)?)?;
    m.add_function(wrap_pyfunction!(fast_save_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(fast_load_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(append_to_binary_file, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_count, m)?)?;
    Ok(())
}
