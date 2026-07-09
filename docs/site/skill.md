# The JapanFold skill

Prefer to fold and design straight from your AI agent instead of writing HTTP
calls? Install the **JapanFold skill** — a single [`SKILL.md`](https://github.com/moritztng/japanfold/blob/master/SKILL.md)
built on the open [Agent Skills](https://agentskills.io) standard. It teaches
your agent the API so you can just ask, in plain language, to fold or design.

## Install

One line installs it into any compatible harness (Claude Code, Cursor, Codex,
Gemini CLI, Cline, Windsurf, Copilot, Amp, and more):

```bash
npx skills add moritztng/japanfold          # this project
npx skills add moritztng/japanfold -g       # global: every project / new chat
```

- Target specific agents: `-a claude-code`, `-a cursor`, `-a codex`, `-a '*'` (all).
- No installer? It's just a file — drop `SKILL.md` into your agent's skills
  directory (e.g. `~/.claude/skills/japanfold/SKILL.md`).

### Claude Code plugin marketplace

Install it as a managed plugin (auto-updates via `/plugin marketplace update`):

```bash
claude plugin marketplace add moritztng/japanfold
claude plugin install japanfold@japanfold
```

Restart Claude Code, then just ask it to fold or design.

### Claude Science

Manage skills in-app — no installer. **Customize → Skills**, add from the
[repo](https://github.com/moritztng/japanfold) (or paste `SKILL.md`), and
**publish**. If egress is sandboxed, approve the host `api.japanfold.com`.

## Use

Once installed, ask in plain language:

> *"Fold this sequence with Boltz-2 and report the confidence: MKTAYIAK…"*
>
> *"Co-fold this protein with aspirin and estimate the binding affinity."*
>
> *"Design 10 nanobody binders against this target."*

Or invoke it explicitly where supported: `/japanfold`.

The API is public and self-describing, so you can also skip the skill entirely
and just tell your agent: *"use the JapanFold API at `api.japanfold.com` to
fold …"*.
