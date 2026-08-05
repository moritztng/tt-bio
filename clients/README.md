# JapanFold clients

Everything a customer needs to drive JapanFold from a script, a terminal, or an
AI agent (Claude Science / Claude Code, Codex, Gemini CLI).

```
clients/
  cli/                  # the `japanfold` CLI (dependency-free Python)
  skill/                # the agent skill (Claude Code, Claude Science, 60+ agents)
  install.sh            # one-line CLI installer
```

## 1. The `japanfold` CLI

A single-file, **stdlib-only** client for the `/v1` API. Install:

```bash
curl -fsSL https://install.japanfold.com/install.sh | sh     # prod
# or, from this repo:
pipx install ./clients/cli      # (or: pip install ./clients/cli)
```

Run it — **no key needed** for the free public demo (same limits as the web app):

```bash
japanfold models
japanfold predict --sequence MKTAYIAK... --model boltz2 --name t1 --wait --out ./out
japanfold design spec.yaml --protocol nanobody-anything --num-designs 10 --wait --out ./out
```

An optional API key raises the demo limits: `export JAPANFOLD_API_KEY=jf_live_...`
(issued by the JapanFold team — no self-serve signup yet). The CLI picks it up
automatically.

- `--wait` submits, polls to completion, and downloads results in one foreground
  command (no backgrounding needed). Or submit and poll separately:
  `japanfold predict … ` → `japanfold download <job_id> --out ./out` (resume-safe).
- `--json` gives machine-readable output on every command.
- `--base-url` (or `$JAPANFOLD_BASE_URL`) targets a specific deployment — e.g. an
  on-prem JapanFold server — instead of `https://api.japanfold.com`.

Config precedence: `--api-key` > `$JAPANFOLD_API_KEY` > `~/.config/japanfold/config.json`
(written by `japanfold auth login`).

**Built for agents:** `--json` on every command (data → stdout, human messages →
stderr), `japanfold schema` prints the OpenAPI contract for tool introspection,
and exit codes let an agent branch on the failure mode without parsing text:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | job failed / generic error |
| 2 | authentication (missing/invalid key) |
| 3 | invalid input / usage |
| 4 | not found |
| 5 | rate-limited / at capacity |
| 6 | server unreachable |
| 130 | interrupted |

## 2. The agent skill (Claude Code, Claude Science, …)

One self-contained skill teaches any agent to drive the public API — no CLI, no
key. It's published at **[github.com/japanfold/japanfold](https://github.com/japanfold/japanfold)**
(source of truth: [`skill/SKILL.md`](skill/SKILL.md)).

**Claude Code (and Cursor, Codex, +70 agents) — one command:**

```bash
npx skills add japanfold/japanfold
```

**Claude Science:** either
- *zero install* — just ask: *"Use the JapanFold API at `api.japanfold.com` to
  fold this sequence …"* (it's public + self-describing at `/v1/openapi.json`;
  approve the network host when prompted), or
- *install for repeat use* — add via **Customize → Skills** (paste `SKILL.md` or
  point at `github.com/japanfold/japanfold`) and **publish** it. Claude Science
  requires only `name` + `description` in the frontmatter (both present); a
  skill stays a *draft* — invisible to `search_skills`/`skill()` — until you
  publish it, so don't skip that step. `allowed-tools` is a Claude Code field
  and is simply ignored here.

Then just ask in natural language — *"fold this sequence with Boltz-2 and report
the confidence"* or *"design 10 nanobody binders against this target."*

## 3. The API directly

Any language can call the REST API — see the [docs site](../docs/site/overview.md)
and the OpenAPI 3.1 contract at `GET /v1/openapi.json` (usable to generate typed
SDKs or an MCP server).
