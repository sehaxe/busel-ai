# Бусел (Busel) — Sovereign 1.58-bit Any-to-Text LLM

**MLX/Apple Silicon × PyTorch/CUDA. 1.58-битные ternary веса. Байт-левел (без BPE).**
**Hybrid Python + Rust (PyO3). Обучается на RTX 5060 Ti 16 GB / Apple Silicon.**
**Лицензия:** CC BY-NC-SA 4.0 (некоммерческая). Автор: [sehaxe](https://github.com/sehaxe)

---

## Архитектура

| Компонент | Реализация |
|-----------|-----------|
| **Веса** | 1.58-bit ternary (Stepped ReLU STE + SCT-128) |
| **Вокабуляр** | 326 байт (256 raw + 70 special) — без BPE |
| **Активации** | INT8 (BitNet v2 spec) |
| **Внимание** | GDN-2 (3:1 ratio) + MLA (d_c=128) — 16 слоёв |
| **MoE** | Top-1 с Blackboard Memory + MoD 0.5, 6 экспертов |
| **Residuals** | Manifold-Constrained Attention Residuals (Sinkhorn-Knopp на Birkhoff polytope) |
| **Предикция** | Multi-Token Prediction (6 голов — t+24 байта) |
| **Оптимизатор** | SF-NorLotusMuon (LOTUS rank-128) + FP8 AdamW (torchao) |

### Технологии обучения (все ON по умолчанию)

| Технология | Эффект |
|-----------|--------|
| **SCT rank-128** | Сжатие FFN в 6-8× без потери качества (arXiv:2604.00733) |
| **LOTUS rank-128** | Muon-состояние ×40 меньше памяти, колоночная норм. |
| **DropBP** | 30% слоёв пропускаются в backward |
| **LCSB** | 50% слоёв без градиента (-44% времени шага) |
| **MoD 0.5** | 50% токенов пропускаются |
| **EMA** | Экспоненциальное среднее весов (decay 0.999) |
| **ASCII Curriculum** | Сначала 7-бит ASCII, потом полный 8-бит |
| **Chunk Curriculum** | Рост контекста 512→1024→2048→4096→8192 |
| **Progressive Freeze** | Заморозка 75% слоёв в конце обучения |

---

## Профили

| Профиль | d_model×layers | Параметры | VRAM | batch×accum | Для чего |
|---------|---------------|-----------|------|-------------|----------|
| `chizh-9m` | 384×2 | 9M | 1 GB | 256×2 | CI / smoke test |
| `verabey-67m` | 768×6 | 67M | 6 GB | 256×2 | ~6ч, хороший |
| `sokal-120m` | 768×10 | 120M | 10 GB | 256×2 | ~12ч, сильный |
| `kruk-210m` | 768×16 | 210M | 14 GB | 768×4 | ~24ч, мощный |
| `busel-365m` | 1024×18 | 365M | 20+ GB | 256×4 | Флагман |

Все профили: SCT rank-128, LOTUS rank-128, MoD 0.5, MTP-6, GDN-2:MLA=3:1.

---

## Быстрый старт

```bash
# Установка (авто-детект GPU)
./scripts/setup.sh

# Тест (9M params, 3 минуты)
uv run python cli.py train --profile chizh-9m --max-steps 50

# Средняя модель (67M, ~6 часов)
uv run python cli.py autopilot --profile verabey-67m

# Сильная модель (210M, ~24 часа)
uv run python cli.py autopilot --profile kruk-210m
```

---

## Ключевые цифры

| Метрика | kruk-210m | fp16 эквивалент |
|---------|-----------|----------------|
| Параметры (SCT) | 210M | — |
| Параметры (без SCT) | 1.1B | 1B compute-equivalent |
| На диске | 11 MB (1.58-бит) | 2 GB (fp16) |
| VRAM (инференс) | 1 GB | 2 GB |
| VRAM (тренировка) | 14 GB (FP8 Adam) | 28 GB (Adam fp32) |
| Muon покрытие | 99.5% параметров | — |

---

## Структура проекта

```
busel-ai/
├── model/             # BitNet v2: layers, attention, routing, backbone, patching, checkpoint
├── training/          # SF-NorLotusMuon, FP8 AdamW, AutoPilot, stages/ (pretrain→SFT→DPO→eval)
├── data/              # Stream-Interleaving: Rust mmap или Python fallback
├── multimodal/        # Any-to-token: image, video, audio, PDF, docx, text
├── ui/                # Teto animation, rich terminal
├── tools/             # CLI: orchestrator, data_manager, plotter, inference, tool_executor
├── tests/             # 175+ unit tests + profiler + scaling laws + LR-finder
├── busel_rust_io/     # PyO3 Rust: mmap ByteStreamer, ternary matmul, binary packer
├── configs/           # default.yaml — 5 профилей + pipelines
├── site/              # Astro+Starlight документация
└── checkpoints/       # *.pt + busel.log.jsonl (gitignored)
```

---

## Планы

- [ ] FSDP для multi-GPU (2×3090 + 2×5060 Ti)
- [ ] Streaming inference (бесконечный контекст на GDN-2)
- [ ] GRPO reasoning (R1-стиль)
- [ ] Distillation от больших моделей
- [ ] YaRN 100K+ контекст

---

## Ссылки

- [Документация](https://sehaxe.github.io/busel-ai/)
- [GitHub](https://github.com/anomalyco/busel-ai)
- **Контакт:** sehaxe (автор и единственный разработчик)
- **Лицензия:** CC BY-NC-SA 4.0 — некоммерческая. Для commercial use: связаться с автором.
