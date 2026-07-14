# OpenDDE → one card + checkpoint picker — deploy state

Branch: `wk/japanfold-opendde-unify-checkpoint-picker` (commit 0e133e4).
Base: `wk/japanfold-opendde-restore-abag-grouped` (35aa217, currently LIVE with
two top-level OpenDDE cards).

## Goal
Moritz (2026-07-14): one OpenDDE entry in the main web model selection, with both
checkpoints reachable via a picker. NOT flip #1 (drop opendde-abag) — the accuracy
concern that justified that is resolved (opendde-9dsg-reference-parity-confirmed-
not-port-bug). NOT flip #2 (two top-level cards) — that's the current live state
he's now asking to collapse.

## Approach (presentation-only, no engine change)
- `catalog.MODELS` stays flat (both opendde + opendde-abag) = authority for
  limits/jobs validation and `/v1/models`. Both ids remain directly submittable.
- `catalog.catalog()["models"]` (served by `/api/catalog`, web UI) collapses
  same-`family` entries into ONE card with nested `checkpoints`. Per-checkpoint
  blurbs reused verbatim (abag accuracy framing unchanged).
- `/v1/models` lists flat `catalog.MODELS` (6 entries, both ids top-level) —
  API/CLI surface unchanged.
- Frontend `FoldForm.jsx`: one OpenDDE card + checkpoint toggle; selecting a
  checkpoint sets the concrete dispatch id. `expandModels()` re-flattens for
  caps/MSA/submit logic. Bundle rebuilt into `static/`.
- OpenAPI enum, CLI `--model` choices, `clients/skill/SKILL.md`: untouched.

Avoids the catalog-collapse trap (japanfold-catalog-collapse-checkpoint-variant-
trap): no single model_id switches checkpoints per-request → no resident-model
reload. The two ids stay two resident configs; the picker just picks which id to
submit.

## Dev-verified (Flask test client on pc, no device)
- `/api/catalog` → 5 models, ONE OpenDDE card, checkpoints=[opendde=General,
  opendde-abag=Antibody-Antigen]. DONE_CHECK shape (count of family opendde or
  id opendde == 1) holds locally.
- `/v1/models` → 6 flat entries, both opendde + opendde-abag top-level.
- `/v1/openapi.json` enum → both ids.
- `POST /v1/predictions` model=opendde → 202; model=opendde-abag → 202; unknown → 400.
- `tests/test_openapi_spec_matches_routes.py`, `tests/test_limits_fasta_alphabet.py` → 5 passed.

## Pending — PRODUCTION-GATED, awaiting Moritz's Telegram go-ahead
Go-ahead requested via `~/.coworker/tg.sh` on 2026-07-14 ~16:55. This task does NOT
have blanket pre-approval — wait for an explicit yes before touching the Galaxy.

Deploy plan (reuse `japanfold-prod-deployment` mechanics):
1. Health before: `ssh japanfold-ssh` — services active, 32/32 workers, 0 running
   jobs (idle). Re-confirm idle + 0 running before restart.
2. On the Galaxy repo: fetch + checkout `wk/japanfold-opendde-unify-checkpoint-
   picker` (the branch has the built frontend, so no `npm run build` needed on the
   host — Python needs restart, the static bundle is committed). Confirm with
   Moritz whether to deploy the branch directly or wait for orchestrator merge
   into `aiand-bio-platform`.
3. `sudo systemctl restart japanfold.service` (SIGTERM, 120s). NEVER stop
   cloudflared. Wait for 32/32 workers + `/v1/health` ok.
4. DONE_CHECK on prod:
   `test "$(curl -sf -m 20 https://japanfold.com/api/catalog | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for m in d["models"] if m.get("family")=="opendde" or m.get("id")=="opendde"))')" = "1"`
5. Confirm `/v1/models` still lists 6 flat entries with both opendde ids.
6. Real end-to-end: `POST /v1/predictions` model=opendde and model=opendde-abag
   via the public HTTPS API, poll to succeeded, check result shape. NEVER stop a
   live service to test.

## Not in scope
- `esmfold2` / `esmfold2-fast` left as two entries (runtime mode selector would
  need engine resident-model reload — not presentation).
- `docs/site/models-and-limits.md` already predates OpenDDE (lists neither id);
  left as-is per "docs: keep both ids as they are now" (don't remove). Adding
  OpenDDE rows there is a separate doc pass.
