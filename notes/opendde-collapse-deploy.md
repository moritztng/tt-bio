# OpenDDE single-model collapse — deploy state

Branch: `wk/japanfold-opendde-single-model` (origin pushed, commit 98ded22).
Base: `origin/aiand-bio-platform` (df838cb).

## Done (dev-verified)
Collapsed OpenDDE from two public entries (`opendde` + `opendde-abag`) to ONE
across all three JapanFold surfaces:
- `tt_bio/platform/catalog.py` — one `opendde` entry (general checkpoint). Web UI
  is data-driven from catalog, so the second card is gone with no frontend rebuild.
- `tt_bio/platform/openapi_spec.py` — `opendde-abag` removed from PredictRequest enum.
- `tt_bio/platform/jobs.py` — `opendde-abag` now a clean 400 deprecation, never 500.
- `clients/cli/japanfold_cli.py`, `clients/skill/SKILL.md` — one OpenDDE entry.

Engine (`tt_bio/main.py`, `worker.py`, `opendde.py`, engine README/docs) KEEPS
`opendde-abag` for the ongoing `tt-bio-opendde-abag-reference-9dsg` investigation —
this is presentation/routing only, no engine/dispatch change.

Dev verification (all passed):
- `/v1/models` → 5 models, exactly one opendde.
- `/v1/openapi.json` enum matches.
- `POST /v1/predictions` model=opendde-abag → 400 deprecation; model=opendde → 202.
- platform regression tests pass (openapi/routes, limits, input validation, msa defaults).

## Pending — PRODUCTION-GATED, awaiting Moritz's Telegram go-ahead
Go-ahead request sent via `~/.coworker/tg.sh` on 2026-07-14 ~15:20. Do NOT deploy
until he replies yes.

Deploy plan (reuse `japanfold-prod-deployment` mechanics):
1. Health before: `ssh japanfold-ssh` — services active, 32/32 workers, 0 running
   jobs (idle), 213G free. Re-confirm idle + 0 running before restart.
2. `ssh japanfold-ssh 'cd <platform repo> && git fetch origin && git checkout aiand-bio-platform && git pull origin aiand-bio-platform'` (pull the branch once orchestrator merges it into aiand-bio-platform — OR if Moritz approves deploying the branch directly, pull wk/japanfold-opendde-single-model). Confirm the merge path with Moritz/orchestrator.
3. `sudo systemctl restart japanfold.service` (GracefulStop SIGTERM, 120s). NEVER
   stop cloudflared. Wait for 32/32 workers back + /v1/health ok.
4. Health after + DONE_CHECK:
   `test "$(curl -sf -m 20 https://api.japanfold.com/v1/models | python3 -c 'import json,sys; m=json.load(sys.stdin); l=m.get("data") or m.get("models") or (m if isinstance(m,list) else []); print(sum(1 for x in l if isinstance(x,dict) and "opendde" in (str(x.get("id") or x.get("name") or "")).lower()))')" = "1"`
5. Real end-to-end: `POST /v1/predictions` model=opendde, two-chain insulin, poll
   to succeeded, check result shape. Also confirm model=opendde-abag → 400 on live.

## Not in scope
esmfold2 / esmfold2-fast left as two entries — a runtime mode selector needs engine
resident-model reload work (not presentation). Both verified, not over-promoted.
