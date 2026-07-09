# JapanFold API docs

Static, hostable documentation for the [JapanFold API](https://api.japanfold.com).
No build step, no framework — just Markdown plus two small HTML pages.

## What's here

| File | What it is |
|---|---|
| `index.html` | The docs site — a light single-page shell that renders the Markdown guides (sidebar nav, hash routing). |
| `api.html` | Interactive API reference, rendered from the live OpenAPI contract at `/v1/openapi.json` (Scalar, via CDN). |
| `overview.md` | Overview + 3-call quickstart. |
| `authentication.md` | Keyless default vs `jf_live_…` Bearer key; ownership & quotas. |
| `predictions.md` | `POST /v1/predictions` — input shapes, models, co-fold, affinity, params. |
| `designs.md` | `POST /v1/designs` — BoltzGen binder design. |
| `jobs.md` | Job lifecycle — poll, list, cancel, delete, results, logs, artifacts, archive. |
| `models-and-limits.md` | Models, parameters, protocols and every numeric limit. |
| `errors.md` | RFC 9457 problem+json shape and status codes. |
| `examples.md` | End-to-end fold / co-fold+affinity / design in curl and Python. |
| `skill.md` | Installing and using the JapanFold agent skill. |

The Markdown files render on GitHub as-is, and are the content the site loads.
The OpenAPI spec at `https://api.japanfold.com/v1/openapi.json` is the
machine-readable source of truth; the guides link to it rather than duplicating
the schema.

## Preview locally

`index.html` fetches the `.md` files, so serve the folder over HTTP — don't open
it via `file://` (the browser blocks those fetches):

```bash
cd docs/site
python3 -m http.server 8000
# open http://localhost:8000/
```

## Host it

Copy this folder (`docs/site/`) to any static host (or a CDN / object store) and
serve it at, e.g., `docs.japanfold.com`:

- **GitHub Pages** — publish from `/docs/site` on the default branch.
- **Netlify / Vercel / Cloudflare Pages** — set the publish directory to
  `docs/site`, no build command.
- **Any web server / bucket** — upload the files; `index.html` is the entry point.

Everything is static; the only runtime dependencies are the two CDN scripts
(`marked` for Markdown, `@scalar/api-reference` for the OpenAPI page) and the live
API for `api.html`.
