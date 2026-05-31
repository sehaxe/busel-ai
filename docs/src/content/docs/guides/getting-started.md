---
title: Getting Started
description: Quickstart guide for Busel AI.
---

Welcome to **Busel AI** (pronounced as **[бу́сэл]**, from Belarusian *бусел* — stork). This guide will help you set up the environment, build the high-performance Rust I/O extension, download Chinchilla-optimal datasets, and launch your first from-scratch training run on consumer hardware.

---

## 🛠 Prerequisites & Installation

Busel AI leverages a modern polyglot environment (Python, Rust, and Bun).

### 1. Synchronize Dependencies

We use `uv` as our primary environment manager. Run the following command in the project root to synchronize dependencies and set up the virtual environment:

```bash
uv sync
```

*(Optional) We highly recommend installing **Docling** for native layout-aware PDF reading capabilities:*

```bash
uv add docling
```

### 2. Build the Rust NVMe-I/O Extension

Compile the multi-threaded, memory-mapped byte streamer on Rust directly into your active virtual environment:

```bash
uv run maturin develop --release
```

---

## 📥 Dataset Preparation (Presets Engine)

We implement **Generalized Chinchilla Scaling Laws** (80 bytes per parameter) to automate dataset volume calculation. You don't need to manually configure token limits.

Download the fully curated, high-density multimodal pre-training and alignment stack (SmolLM-Corpus + Smoltalk + COCO) for the **Shpak** profile with a single command:

```bash
uv run python cli.py download-all --preset shpak
```

*Note: You can copy any `.pdf` textbooks or documents directly into the `data_train/` folder. The data pipeline will automatically parse them into structured Markdown on the fly.*

---

## 🔥 Launching Training

Start the progress-driven training loop for the **Shpak (48M total / 25M active)** profile:

```bash
uv run train.py --profile shpak
```

### What happens under the hood:
* **Chinchilla Auto-Planner:** Dynamically calculates total parameters (52.8M) and plans exactly **25,000 steps** for the target 3.84B tokens.
* **Sequence Length Warmup:** Training starts on a highly efficient `1024` context window (speeding up early steps by 300%) and scales up to `4096` as training progresses.
* **ByselAutoPilot v6.0:** Monitors gradient norms to prevent loss spikes pre-emptively, applies Adaptive Gradient Clipping (AGC), and dynamically schedules weight decay.
* **Stream Interleaving:** Automatically mixes textbooks, code files, and SFT dialogue prompts on the fly within each batch to prevent catastrophic forgetting.