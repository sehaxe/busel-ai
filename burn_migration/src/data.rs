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
/// Only picks from active (non-exhausted) files.
pub struct ByteStreamer {
    files: Vec<MmapFile>,
    active_mask: Vec<bool>,
    total_active_weight: f64,
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
        let n = files.len();
        let total_active_weight: f64 = files.iter().map(|f| f.weight).sum();
        Ok(Self {
            files,
            active_mask: vec![true; n],
            total_active_weight,
            rng: rand::rng(),
        })
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

    /// Pick a file from active (non-exhausted) ones, weighted.
    fn pick_active(&mut self) -> usize {
        let total = self.total_active_weight;
        if total <= 0.0 {
            return self.rng.random_range(0..self.files.len());
        }
        let mut x = self.rng.random::<f64>() * total;
        for (i, _) in self.files.iter().enumerate() {
            if !self.active_mask[i] { continue; }
            x -= self.files[i].weight;
            if x <= 0.0 { return i; }
        }
        // fallback: last active
        self.files.iter().enumerate().rposition(|(i, _)| self.active_mask[i]).unwrap_or(0)
    }

    /// Reset all positions — new epoch.
    fn reset_epoch(&mut self) {
        for f in &mut self.files {
            f.pos = 0;
        }
        self.active_mask.fill(true);
        self.total_active_weight = self.files.iter().map(|f| f.weight).sum();
    }

    /// Fill `buf` (i64) from sequential file reads.
    /// Handles multi-file interleaving: picks active file by weight,
    /// reads across file boundaries, wraps epoch when all exhausted.
    pub fn fill_batch(&mut self, buf: &mut [i64]) {
        let mut cursor = 0;
        while cursor < buf.len() {
            // If all files exhausted, reset epoch
            if self.total_active_weight <= 0.0 {
                self.reset_epoch();
            }

            let i = self.pick_active();
            let f = &mut self.files[i];
            let need = buf.len() - cursor;
            let avail = f.mmap.len() - f.pos;
            let take = need.min(avail);

            // Bulk convert u8→i64
            for (j, &b) in f.mmap[f.pos..f.pos + take].iter().enumerate() {
                buf[cursor + j] = b as i64;
            }
            cursor += take;
            f.pos += take;

            if f.pos >= f.mmap.len() {
                self.active_mask[i] = false;
                self.total_active_weight -= f.weight;
            }
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
        // "helloXworldXfooXbarXbaz" — each 5-char chunk separated by X
        let mut f = std::fs::File::create(td.join("a.txt")).unwrap();
        f.write_all(b"hello world foo bar baz").unwrap();
        drop(f);
        let mut s = ByteStreamer::open(&td).unwrap();
        let mut buf = vec![0i64; 5];

        s.fill_batch(&mut buf);
        assert_eq!(&buf[..5], &[104, 101, 108, 108, 111]); // "hello"

        s.fill_batch(&mut buf);
        assert_eq!(&buf[..5], &[32, 119, 111, 114, 108]);  // " worl"

        s.fill_batch(&mut buf);
        assert_eq!(&buf[..5], &[100, 32, 102, 111, 111]);  // "d foo"

        s.fill_batch(&mut buf);
        assert_eq!(&buf[..5], &[32, 98, 97, 114, 32]);     // " bar "

        // Only 3 bytes left ("baz"), wraps to next epoch
        s.fill_batch(&mut buf);
        assert_eq!(buf[0], 98);   // 'b' — last of "baz"
        assert!(buf[1..] != [0; 4]); // filled from next epoch

        std::fs::remove_dir_all(&td).unwrap();
    }
}
