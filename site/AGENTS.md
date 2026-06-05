# site/ — Astro + Starlight Docs

**Scope:** Human-friendly tour of the busel codebase. The wiki at <https://sehaxe.github.io/busel-ai/>.

## STRUCTURE
```
site/
├── astro.config.mjs   # Starlight config — sidebar, integrations, site URL
├── package.json       # Bun-managed deps (astro, @astrojs/starlight)
├── src/content/docs/  # All markdown content — auto-routed
│   ├── index.mdx      # Landing page
│   ├── architecture/  # 7 docs: overview, one-bit-weights, patching, attention, mar, moe, mtp
│   ├── data/          # 3 docs: pipeline, formats, multimodal
│   ├── guides/        # 3 docs: getting-started, quick-tour, profiles
│   ├── operations/    # 3 docs: inference, troubleshooting, faq
│   ├── performance/   # 3 docs: compile-modes, hardware, profiling
│   ├── reference/     # 7 docs: model, training, config, registry, etc.
│   └── training/      # 5 docs: training-guide, optimizer, autopilot, curriculum, checkpointing
├── public/            # Static assets (logo, favicon, …)
└── dist/              # Build output (gitignored)
```

## STABLE URL SECTIONS (do not break)
The 7 sidebar groups in `astro.config.mjs` are stable URLs. Never rename a folder or move a page out of its section without an explicit deprecation path — external links break silently.

| Section | Slug | Doc count |
|---|---|---:|
| Architecture | `/architecture/<doc>/` | 7 |
| Training | `/training/<doc>/` | 5 |
| Data | `/data/<doc>/` | 3 |
| Guides | `/guides/<doc>/` | 3 |
| Operations | `/operations/<doc>/` | 3 |
| Performance | `/performance/<doc>/` | 3 |
| Reference | `/reference/<doc>/` | 7 |

## WHERE TO LOOK
| Want to... | Edit | Notes |
|---|---|---|
| Add a doc | `src/content/docs/<section>/<name>.md` | Auto-routed; add to sidebar in `astro.config.mjs` if it should appear in nav |
| Reorder sidebar | `astro.config.mjs` | Each section's `items` list, in display order |
| Change landing page | `src/content/docs/index.mdx` | The 6 `Card` components are the front door |
| Add a component | `src/components/` (create) | Astro components — `.astro` files |
| Add a static asset | `public/` | Served at `/<path>` |
| Build locally | `bun run dev` (port 4321) / `bun run build` (output → `dist/`) | |

## CONVENTIONS
- **Internal links** use the absolute `file://` URL format: `[label](file:///abs/path/to/file.md#L42-L80)`. This makes the docs usable from IDEs and AGENTS — the file link points to the local file, not the deployed URL.
- **External links** use the deployed URL: `[label](https://sehaxe.github.io/busel-ai/architecture/overview/)`.
- **Frontmatter** is required: `title`, `description`, `sidebar.order`. `order` controls the intra-section order in the sidebar.
- **Imports** at the top: `import { Aside, Tabs, TabItem, Steps } from '@astrojs/starlight/components';` for the callout/tabs/steps components.
- **Emojis in titles/labels** are OK and encouraged (matches the rest of the project: 🦩 / ⚙️ / 💡 / 📚 / 🤖 / 🎯 / 🛸).
- **Build** with `bun run build` from the `site/` directory; static output goes to `dist/`. GitHub Pages deploys from `dist/`.

## ANTI-PATTERNS
- **NEVER** rename a folder under `src/content/docs/` without an explicit deprecation — external links break. The 7 section names (`architecture`, `training`, `data`, `guides`, `operations`, `performance`, `reference`) are stable.
- **NEVER** delete a doc page that has external links — redirect to a new page instead, or leave a stub with a "this content moved" note.
- **NEVER** commit `node_modules/`, `dist/`, `.astro/`, `bun.lockb` (note: `bun.lock` IS tracked; the binary `bun.lockb` is not). The `.gitignore` is the source of truth.
- **NEVER** skip the frontmatter — Starlight won't pick up the page.
- **NEVER** mix `file://` and `https://` styles in the same paragraph — be consistent. Default to `file://` for code references, `https://` for blog/paper citations.
- **NEVER** add a `## See also` link that 404s — the cross-doc links are read by humans and crawlers. Verify each link renders.
- **NEVER** claim a feature works in docs before it lands in code. The docs are derived from code, not the other way around.
- **NEVER** use absolute paths to a user's machine in committed docs (e.g. `/Users/foo/...`). Use the repo-relative `[label](file:///home/sehaxe/busel-ai/path)` form, which is portable on Linux/macOS.

## DO
- **Follow the two-track rule:** code change → README change → site/ change. When a feature is added or a default is flipped, update both `README.md` (the elevator pitch) and the relevant `site/src/content/docs/` page (the human-friendly tour).
- **Sample 2-3 similar docs** in the same section before adding a new one. The structure is consistent — `## Overview` → `## Where to look` → `## Conventions` → `## Anti-patterns` → `## See also` is the common shape.
- **Use `Aside` from Starlight** for tips (`type="tip"`), warnings (`type="caution"`), and notes (`type="note"`). Don't reinvent the callout box.
- **Quote code with line refs** when explaining a function: `[\`buselModel.__init__\`](file:///home/sehaxe/busel-ai/model/backbone.py#L120-L150)`.
- **Use the profiling numbers** from the `tests/profiler_run.py` runs (validated end-to-end) — never invent performance numbers.
- **Link to the `AGENTS.md` files** when a doc needs more depth than the tour provides. AGENTS.md is the source of truth; the site/ is the entry point.
- **Keep doc files ≤ 350 LoC.** If a doc is longer, split it. The split should match the sidebar grouping.

## NOTES
- **Build command** is `bun run build` (run from `site/`). Dev server is `bun run dev` on `localhost:4321`.
- **Deploy** is automatic — the GitHub Actions workflow builds `dist/` and deploys to GitHub Pages on push to `main`.
- **Sidebar is configured in `astro.config.mjs`** — every doc added to `src/content/docs/` MUST also be added to the corresponding sidebar `items` list, or it won't appear in the nav (but it will be reachable via direct URL).
- **No BPE / no tokenizers in code, no BPE / no tokenizers in docs** — byte-level (vocab=326) is the model. Don't suggest BPE alternatives in any doc.
- **Numbers must match the code:** if you write "1.5× speedup", the profiler must show 1.5×. If the number drifts, update both.
