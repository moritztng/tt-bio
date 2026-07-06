# JapanFold clients

Everything a customer needs to drive JapanFold from a script, a terminal, or an
AI agent (Claude Science / Claude Code, Codex, Gemini CLI).

```
clients/
  cli/                  # the `japanfold` CLI (dependency-free Python)
  claude-plugin/        # Claude Code / Claude Science plugin (bundles the skill)
  install.sh            # one-line CLI installer
```

## 1. The `japanfold` CLI

A single-file, **stdlib-only** client for the `/v1` API. Install:

```bash
curl -fsSL https://install.japanfold.com/install.sh | sh     # prod
# or, from this repo:
pipx install ./clients/cli      # (or: pip install ./clients/cli)
```

Authenticate and run:

```bash
export JAPANFOLD_API_KEY=jf_live_...        # from https://japanfold.com/account
japanfold models
japanfold predict --sequence MKTAYIAK... --model boltz2 --name t1 --wait --out ./out
japanfold design spec.yaml --protocol nanobody-anything --num-designs 10 --wait --out ./out
```

- `--wait` submits, polls to completion, and downloads results in one foreground
  command (no backgrounding needed). Or submit and poll separately:
  `japanfold predict … ` → `japanfold download <job_id> --out ./out` (resume-safe).
- `--json` gives machine-readable output on every command.
- `--base-url` (or `$JAPANFOLD_BASE_URL`) targets a specific deployment — e.g. an
  on-prem JapanFold server — instead of `https://japanfold.com`.

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

## 2. Claude Science / Claude Code integration

JapanFold plugs into agent workbenches the same way the Boltz API and NVIDIA
BioNeMo do: a **skill** that wraps the CLI. The agent installs the CLI, reads the
skill for how/when to use it, and shells out to `japanfold` — submitting jobs,
polling, and downloading structures autonomously.

**Install the plugin** (bundles the `japanfold` skill):

```bash
# Cross-agent (Claude Code, Codex, …):
npx skills add japanfold/japanfold          # from the published skills repo
# or point Claude Code at this plugin directory / a marketplace entry.
```

Then, in Claude Science / Claude Code, just ask in natural language — e.g.
*"fold this sequence with Boltz-2 and report the confidence"* or *"design 10
nanobody binders against this target."* The agent will ensure the CLI is
installed and `JAPANFOLD_API_KEY` is set, then run the prediction/design and
read back the structures and scores.

The skill lives at [`claude-plugin/skills/japanfold/SKILL.md`](claude-plugin/skills/japanfold/SKILL.md);
plugin metadata at [`claude-plugin/.claude-plugin/plugin.json`](claude-plugin/.claude-plugin/plugin.json).

### On-prem / behind a firewall

For a self-hosted JapanFold server, agents connect over the routes that need no
public endpoint:

- **SSH host in Claude Science** — add the server as a remote compute host; the
  agent runs the `japanfold` CLI there (`--base-url http://localhost:8080`), data
  and compute stay on-prem.
- **Local Claude Code** — run the agent on a machine that can reach the server on
  the LAN; the skill drives the CLI the same way.

## 3. The API directly

Any language can call the REST API — see [`../docs/api.md`](../docs/api.md) and
the OpenAPI 3.1 contract at `GET /v1/openapi.json` (usable to generate typed
SDKs or an MCP server).
