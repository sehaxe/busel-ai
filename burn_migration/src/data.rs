// pony: sequential mmap byte streamer — contiguous text, wrap-around epochs.
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};
use memmap2::Mmap;
use rand::prelude::*;

struct MmapFile {
    mmap: Mmap,
    weight: f64,
    pos: usize,
}

/// Stream chunks of raw bytes sequentially from files.
/// Picks files by weight, reads each sequentially, wraps when exhausted.
pub struct ByteStreamer {
    files: Vec<MmapFile>,
    rng: rand::rngs::ThreadRng,
}

impl ByteStreamer {
    pub fn open(dir: &Path) -> std::io::Result<Self> {
        Self::open_weighted(dir, &HashMap::new())
    }

    pub fn open_weighted(dir: &Path, weights: &HashMap<String, f64>) -> std::io::Result<Self> {
        let mut entries: Vec<(PathBuf, f64)> = Vec::new();
        Self::collect_files(dir, &mut entries)?;
        if entries.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no .bin/.txt/.jsonl files found (or all weighted to 0)",
            ));
        }
        let mut files = Vec::new();
        for (p, _) in entries {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let w = weights.get(&stem).copied().unwrap_or(1.0);
            if w <= 0.0 { continue; }
            let f = File::open(&p)?;
            let mmap = unsafe { Mmap::map(&f)? };
            files.push(MmapFile { mmap, weight: w, pos: 0 });
        }
        Ok(Self { files, rng: rand::rng() })
    }

    fn collect_files(dir: &Path, out: &mut Vec<(PathBuf, f64)>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let p = entry?.path();
            if p.is_dir() { Self::collect_files(&p, out)?; continue; }
            if p.extension().map_or(false, |e| e == "bin" || e == "txt" || e == "jsonl") {
                out.push((p, 1.0));
            }
        }
        Ok(())
    }

    fn pick_file(&mut self) -> usize {
        let total: f64 = self.files.iter().map(|f| f.weight).sum();
        if total <= 0.0 { return self.rng.random_range(0..self.files.len()); }
        let mut x = self.rng.random::<f64>() * total;
        for (i, f) in self.files.iter().enumerate() {
            x -= f.weight;
            if x <= 0.0 { return i; }
        }
        self.files.len() - 1
    }

    /// Return the next sequential chunk of `len` bytes. Wraps to next file when current is exhausted.
    pub fn next_chunk(&mut self, len: usize) -> Vec<u8> {
        loop {
            let active: Vec<usize> = self.files.iter().enumerate()
                .filter(|(_, f)| f.pos < f.mmap.len())
                .map(|(i, _)| i)
                .collect();
            if active.is_empty() {
                for f in &mut self.files { f.pos = 0; }
            }
            let i = self.pick_file();
            let f = &mut self.files[i];
            if f.pos >= f.mmap.len() { continue; }
            let end = (f.pos + len).min(f.mmap.len());
            let chunk = f.mmap[f.pos..end].to_vec();
            f.pos = end;
            return chunk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_dir() {
        let r = ByteStreamer::open(Path::new("/tmp/nonexistent_busel_data"));
        assert!(r.is_err());
    }

    #[test]
    fn test_sequential_read() {
        use std::io::Write;
        let td = std::env::temp_dir().join("busel_test_seq");
        let _ = std::fs::remove_dir_all(&td);
        std::fs::create_dir_all(&td).unwrap();
        let mut f = std::fs::File::create(td.join("a.txt")).unwrap();
        f.write_all(b"hello world foo bar baz").unwrap();
        drop(f);
        let mut s = ByteStreamer::open(&td).unwrap();
        let c1 = s.next_chunk(5);
        assert_eq!(c1, b"hello");
        let c2 = s.next_chunk(5);
        assert_eq!(c2, b" worl");
        let c3 = s.next_chunk(5);
        assert_eq!(c3, b"d foo");
        let c4 = s.next_chunk(5);
        assert_eq!(c4, b" bar ");
        let c5 = s.next_chunk(5);
        assert_eq!(c5, b"baz");
        let c6 = s.next_chunk(5);
        assert_eq!(c6, b"hello"); // wrapped
        std::fs::remove_dir_all(&td).unwrap();
    }
}
