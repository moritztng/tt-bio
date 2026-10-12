#!/usr/bin/env python3
"""What ships next: the release train's one question, answered from git and the gate runs.

The rule. A release ships the newest commit whose gate PASSED, not the newest commit on main.
A gate that passes is cut as it stands, even if main moved on while it ran: main's newer code
is the next release, gated on main's head meanwhile, and most of its legs are ledger hits
(gate_fanout.py REUSE). Waiting for a gate to pass on whatever main's head is when it ends
never ends while stagings land faster than a gate runs, which is how 0.15 sat unshipped for two
days behind 13 gates. A candidate is any gated commit that descends from the last release tag,
on main or on a release branch. Nothing ships when main's code equals the last release's
(Markdown and the version line do not count, the same key gate_fanout.py uses).

    python3 scripts/release_next.py --gates ~/gates      # gate_fanout --out dirs live under here
    python3 scripts/release_next.py --json

Answers one of:
    NOTHING   main's code is the last release's
    CUT       the gate on <sha> passed on every arch; tag it (RELEASING.md, Cut the release)
    GATE      start gate_fanout.py on <sha> (no gate has run on it)
    WAIT      a gate is running on <sha>; if it is not main's head, start the head's gate as well
    FIX       the gate on <sha> failed; read its VERDICT.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_fanout import REPO, content_hash  # noqa: E402

#: A staging merge names itself either way: "Merge wk/spd-int-next7 (staging7): ..." or
#: "Merge staging11d (wk/spd-orch-s11d ...): ...".
STAGING = re.compile(r"^Merge .*?\bstaging(\w+)\b")


def git(*a, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                          text=True).stdout.strip()


def last_release(repo: Path = REPO) -> tuple[str, str]:
    """(tag, sha) of the highest v* tag, wherever it was cut (a release branch merged later counts)."""
    tags = git("tag", "-l", "v*", "--sort=-v:refname", repo=repo).split()
    if not tags:
        return "", ""
    return tags[0], git("rev-list", "-n1", tags[0], repo=repo)


def stagings(since: str, head: str, repo: Path = REPO) -> list[str]:
    """Staging names merged on main's first-parent line after `since`, oldest first."""
    rng = f"{since}..{head}" if since else head
    subj = git("log", "--first-parent", "--reverse", "--format=%s", rng, repo=repo).splitlines()
    return [m.group(1) for s in subj for m in [STAGING.search(s)] if m]


def gate_state(gates: Path | None, content: str) -> tuple[str, Path | None]:
    """('none'|'running'|'pass'|'fail', dir) of the newest full gate_fanout run on code with this
    content key, so a docs-only commit on top of a gated one needs no gate of its own. A run
    limited to some legs (plan.json `partial`) fills the ledger but is no verdict, and a run
    stopped before its verdict is no run."""
    runs = [p.parent for p, plan in _plans(gates) if plan.get("content") == content]
    runs = [d for d in runs if (d / "verdict.json").exists() or _alive(json.loads((d / "plan.json").read_text()))]
    if not runs:
        return "none", None
    d = max(runs, key=lambda p: (p / "plan.json").stat().st_mtime)
    v = d / "verdict.json"
    if not v.exists():
        return "running", d
    return ("pass" if json.loads(v.read_text()).get("pass") else "fail"), d


def _alive(plan: dict) -> bool:
    """The run's runner is still up. A plan from another host is taken as live."""
    pid = plan.get("pid")
    if not pid or plan.get("host") != socket.gethostname():
        return bool(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _plans(gates: Path | None, partial: bool = False) -> list:
    if gates is None or not gates.is_dir():
        return []
    out = []
    for p in gates.glob("*/plan.json"):
        try:
            plan = json.loads(p.read_text())
        except ValueError:
            continue
        if bool(plan.get("partial")) == partial:
            out.append((p, plan))
    return out


def passed_since(gates: Path | None, tsha: str, repo: Path = REPO) -> tuple[str, Path] | None:
    """(sha, dir) of the newest commit after the last release whose full gate passed, or None."""
    done, released = [], content_hash(tsha, repo) if tsha else ""
    for p, plan in _plans(gates):
        v, sha = p.parent / "verdict.json", plan.get("sha", "")
        try:
            if not (v.exists() and json.loads(v.read_text()).get("pass")):
                continue
            if tsha and (sha == tsha or subprocess.run(
                    ["git", "-C", str(repo), "merge-base", "--is-ancestor", tsha, sha],
                    capture_output=True).returncode or plan.get("content") == released):
                continue
            ahead = git("rev-list", "--count", f"{tsha}..{sha}" if tsha else sha, repo=repo)
            done.append((int(git("show", "-s", "--format=%ct", sha, repo=repo)), int(ahead), sha, p.parent))
        except (ValueError, subprocess.CalledProcessError):
            continue                                   # unreadable verdict, or a commit we lack
    return max(done)[2:] if done else None


def running_gates(gates: Path | None, partial: bool = False) -> list[str]:
    return sorted({plan["sha"] for p, plan in _plans(gates, partial)
                   if not (p.parent / "verdict.json").exists() and _alive(plan)})


def decide(repo: Path = REPO, gates: Path | None = None, ref: str = "origin/main") -> dict:
    head = git("rev-parse", ref, repo=repo)
    tag, tsha = last_release(repo)
    out = {"main": head, "last_release": tag, "last_release_sha": tsha,
           "stagings_since": stagings(tsha, head, repo), "running": running_gates(gates)}
    if tsha and content_hash(head, repo) == content_hash(tsha, repo):
        return {**out, "do": "NOTHING", "sha": tsha, "why": f"main's code is {tag}'s"}
    if (cut := passed_since(gates, tsha, repo)) and cut[0] != head:
        sha, d = cut
        nxt = "" if content_hash(sha, repo) == content_hash(head, repo) else (
            f"; main's head {head[:9]} is the next release, gate it now")
        return {**out, "do": "CUT", "sha": sha, "gate": str(d),
                "why": f"gate passed on every arch: {d}/VERDICT.md{nxt}"}
    state, d = gate_state(gates, content_hash(head, repo))
    why = {"none": "no gate has run on main's head",
           "running": f"gate running in {d}",
           "fail": f"gate failed: {d}/VERDICT.md",
           "pass": f"gate passed on every arch: {d}/VERDICT.md"}[state]
    do = {"none": "GATE", "running": "WAIT", "fail": "FIX", "pass": "CUT"}[state]
    if state == "fail":
        res = json.loads((d / "verdict.json").read_text()).get("results", [])
        if owed := sorted({f for r in res if r.get("verdict") == "OWED" for f in r.get("owed", [])}):
            why += f"; record {','.join(owed)} first (gate_fanout.py --legs 'record:*' --record-lever)"
    older = [s for s in out["running"] if d is None or s != json.loads((d / "plan.json").read_text())["sha"]]
    if older:
        why += f"; finish the running gate on {', '.join(s[:9] for s in older)}, do not restart it"
    if part := running_gates(gates, partial=True):
        why += f"; partial runs on {', '.join(s[:9] for s in part)} fill the ledger the gate reuses"
    return {**out, "do": do, "sha": head, "gate": str(d or ""), "why": why}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gates", type=Path, default=None, help="directory holding gate_fanout --out dirs")
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    d = decide(gates=args.gates, ref=args.ref)
    if args.json:
        print(json.dumps(d, indent=1))
    else:
        st = ", ".join(f"staging{s}" for s in d["stagings_since"]) or "none"
        print(f"{d['do']} {d['sha'][:12]}: {d['why']}\n  main {d['main'][:12]}, last release "
              f"{d['last_release'] or '-'} ({d['last_release_sha'][:12]}), landed since: {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
