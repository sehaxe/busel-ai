# busel docs site

The Starlight (Astro) source for the busel wiki at <https://sehaxe.github.io/busel-ai/>.

## Develop

```bash
cd site
bun install
bun dev          # localhost:4321
bun run build    # static output to ./dist
```

The content lives in [`src/content/docs/`](./src/content/docs/) — every `.md`
or `.mdx` file is auto-routed. The sidebar is configured in
[`astro.config.mjs`](./astro.config.mjs).

See the project root [`README.md`](../README.md) and [`CHANGELOG.md`](../CHANGELOG.md)
for the busel project itself.
