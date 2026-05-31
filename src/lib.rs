use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::types::PyModuleMethods;
use rayon::prelude::*;
use std::fs::File;
use memmap2::Mmap;

#[pyclass]
struct ByteStreamer {
    mmap: Mmap,
    position: usize,
    chunk_size: usize,
}

#[pymethods]
impl ByteStreamer {
    #[new]
    fn new(file_path: String, chunk_size: usize, start_offset: usize) -> PyResult<Self> {
        // ИСПРАВЛЕНО: убран allow_threads (не нужен для mmap)
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(ByteStreamer {
            mmap,
            position: start_offset,
            chunk_size,
        })
    }

    fn next_chunk(&mut self, _py: Python) -> Option<Vec<u8>> {
        if self.position >= self.mmap.len() {
            return None;
        }

        let start = self.position;
        let end = std::cmp::min(self.position + self.chunk_size, self.mmap.len());
        
        // Быстрое последовательное копирование в вектор
        let mut chunk = self.mmap[start..end].to_vec();

        if chunk.len() < self.chunk_size {
            chunk.resize(self.chunk_size, 0u8);
        }

        self.position = end;
        Some(chunk)
    }

    fn get_position(&self) -> usize {
        self.position
    }

    fn get_file_size(&self) -> usize {
        self.mmap.len()
    }

    fn get_progress(&self) -> f64 {
        if self.mmap.len() == 0 {
            return 100.0;
        }
        (self.position as f64 / self.mmap.len() as f64) * 100.0
    }
}

#[pyfunction]
fn init_thread_pool(num_threads: usize) -> PyResult<()> {
    if num_threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("bysel-io-{}", i))
            .build_global();
    }
    Ok(())
}

#[pyfunction]
fn get_cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[pymodule]
fn bysel_rust_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ByteStreamer>()?;
    m.add_function(wrap_pyfunction!(init_thread_pool, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_count, m)?)?;
    Ok(())
}