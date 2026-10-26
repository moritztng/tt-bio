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
    RECORD    the gate on <sha> failed only on size-ladder drift; re-record those models' ladders
              on <sha> (gate_fanout --legs 'record:*' --record-full, into <gates>/rec-<sha9>)
    FIX       the gate on <sha> failed; read its VERDICT.md, then fix forward (below)

Size-ladder drift (a lever started firing or fires more, or declines on a new clause) is what a staging that adds
a lever does; parity, capacity and perf judge whether it is right. So a gate whose only failures
are ladder legs reporting drift counts as passed once a record run on the same commit passed for
those models, and the release commit carries the new fragments. A lever that went dark, partial
darkness, an unmeasured rung, a crash or a slower rung is not drift: that stays FIX, unless the
leg is waived. <gate dir>/WAIVE (`<leg> <arch> <sha>  # why`, the reason required) marks a failure
of the gate on <sha> that is not a regression: a waived ladder leg is re-recorded with the drift,
any other waived leg stops counting, and a gate whose failures are all waived is cut. A waiver
judges one commit's run, so it covers that commit only; a line without a sha covers the commit of
the gate dir it is in.

Fix forward, never re-cut. A failed gate's commit stays the candidate: list the fix commits in
<gate dir>/FIXES (one sha per line, oldest first, `#` comments) and the train gates that commit with
each fix's own change applied on top. Only the legs the fixes touched, and the ones that failed,
run again; the rest are ledger hits on the frozen code. Gating main's newer head instead reuses
little, because nearly every staging edits a module every leg imports (tt_bio/tenstorrent.py).
The candidate is rebuilt from FIXES on every tick with fixed commit metadata, so the same list
always names the same commit. A fix to the gate itself (a wrong rule in scripts/release_gate.py, a
test registry gap) fixes no regression and goes on main only, with the leg waived on the failed
commit: as a FIX it reruns nearly every leg, since they all read release_gate.py (rel-a06d92c6d,
10-24: two such fixes made a 155-of-245-leg candidate on unchanged model code).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_fanout import REPO, content_hash  # noqa: E402

#: A staging merge names itself either way: "Merge wk/spd-int-next7 (staging7): ..." or
#: "Merge staging11d (wk/spd-orch-s11d ...): ...".
STAGING = re.compile(r"^Merge .*?\bstaging(\w+)\b")


def git(*a, repo: Path = REPO) -> str:
    return subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                          text=True).stdout.strip()


def _git(*a, repo: Path = REPO, env: dict | None = None, input: str | None = None, strip: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                         env={**os.environ, **(env or {})}, input=input).stdout
    return out.strip() if strip else out


def patched(sha: str, fixes: list[str], repo: Path = REPO) -> str:
    """`sha` with each fix commit's own change applied on top, oldest first. Every new commit
    takes its fix's author, committer and message, so the same list names the same commit on
    every tick. Raises CalledProcessError when a fix does not apply."""
    with tempfile.TemporaryDirectory() as tmp:
        index = {"GIT_INDEX_FILE": f"{tmp}/index"}
        for f in fixes:
            an, ae, ad, cn, ce, cd, msg = _git("show", "-s", "--format=%an%x00%ae%x00%ad%x00%cn%x00%ce%x00%cd%x00%B",
                                               "--date=raw", f, repo=repo).split("\0", 6)
            if not (diff := _git("diff", "--binary", f"{f}^", f, repo=repo, strip=False)):
                continue                                 # a merge-only or empty commit
            _git("read-tree", sha, repo=repo, env=index)
            _git("apply", "--cached", "--binary", repo=repo, env=index, input=diff)
            tree = _git("write-tree", repo=repo, env=index)
            sha = _git("commit-tree", tree, "-p", sha, "-m", f"{msg.strip()}\n\n(cherry picked from commit {f})",
                       repo=repo, env={"GIT_AUTHOR_NAME": an, "GIT_AUTHOR_EMAIL": ae, "GIT_AUTHOR_DATE": ad,
                                       "GIT_COMMITTER_NAME": cn, "GIT_COMMITTER_EMAIL": ce,
                                       "GIT_COMMITTER_DATE": cd})
    return sha


def fixes(d: Path, repo: Path = REPO) -> list[str]:
    """The fix commits <d>/FIXES lists, as full shas."""
    f = d / "FIXES"
    lines = f.read_text().splitlines() if f.exists() else []
    return [git("rev-parse", "--verify", f"{w}^{{commit}}", repo=repo)
            for ln in lines if (w := ln.split("#", 1)[0].strip())]


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


#: A ladder finding the train re-records rather than fixes (release_gate._size_ladder_compare_levers).
DRIFT_OK = re.compile(r"started firing\)|: decline clause |: resolved '")
#: A fired fraction outside the band; upward is drift (older gates labelled both ways "partial darkness").
FRAC_BAND = re.compile(r": frac ([\d.]+) -> ([\d.]+) exceeds the ")


def is_drift(line: str) -> bool:
    return bool(DRIFT_OK.search(line)) or bool((m := FRAC_BAND.search(line)) and float(m[2]) > float(m[1]))
#: Every finding a ladder leg prints, the lever lines and the timing ones alike (`m 512->768: exponent ...`, `m per-rung
#: base->run: ...`): matching only `m/<rung> LEVER:` lines let a slower rung ride along with real drift as drift.
FINDING = re.compile(r"^\s+FAIL \S")


def waivers(d: Path, gates: Path | None = None, repo: Path = REPO, ref: str = "origin/main") -> dict:
    """{"<leg> <arch>": reason} for gate <d>'s commit, from <d>/WAIVE plus the WAIVE of every gated
    commit between a fixed-forward candidate and its base on main (FIXES and WAIVE live in the base's
    gate dir). One line per leg, `<leg> <arch> <sha>  # reason`. The sha (a prefix of 7 or more) is
    the gate commit the line judged and it waives nothing on any other: rel-f62c3b67a's real rfd3
    slowdown sat under a `perf bh` line its base c7a32d7c6 had been waived for. A line without one
    covers its own dir's commit. A line without a reason waives nothing, so every waiver says why
    the failure is not a regression."""
    sha = json.loads((d / "plan.json").read_text())["sha"]
    dirs = [d]
    if gates is not None:
        gated = {plan.get("sha"): p.parent for p, plan in _plans(gates)}
        try:
            off = git("rev-list", "--first-parent", sha, f"^{ref}", repo=repo).split()
            if off:   # a candidate: every gated commit down its fix chain, and its base on main
                dirs += [gated[c] for c in off[1:] + [git("merge-base", sha, ref, repo=repo)] if c in gated]
        except subprocess.CalledProcessError:
            pass                                       # a commit this clone lacks: its own WAIVE only
    out = {}
    for x in dirs:
        f = x / "WAIVE"
        for ln in (f.read_text().splitlines() if f.exists() else []):
            what, _, why = ln.partition("#")
            words = what.split()
            if len(words) not in (2, 3) or not why.strip():
                continue
            judged = words[2] if len(words) == 3 else json.loads((x / "plan.json").read_text())["sha"]
            if len(judged) >= 7 and sha.startswith(judged):
                out[" ".join(words[:2])] = why.strip()
    return out


#: gate_fanout's ledger (its --ledger default): a <key>.json for every leg run that passed.
LEDGER = Path.home() / ".tt-bio-gate-ledger"
OK = ("PASS", "REUSED", "BLOCKED", "SEEDED")


def failures(d: Path, waived: dict | None = None) -> tuple[list[str], list[str]]:
    """(the ladder models to re-record, every other failed leg as "<leg> <arch>") of gate `d`.
    A ladder leg is re-recorded when it only drifted or when it is waived; any other waived leg
    stops counting. So does a leg whose key has passed since: a rerun of the same code, packages
    and card type (a partial gate_fanout run) after the card, not the tree, failed it. A leg is
    judged by its latest run on the gate's commit in any run dir beside `d`: the runner's own files
    are in the key, so a rerun from newer runner scripts files its pass under another key (8b2f8acaa,
    10-13: ladder:boltz2 bh passed and ladder:opendde bh drifted on rerun, and both kept blocking)."""
    res = json.loads((d / "verdict.json").read_text()).get("results", [])
    latest = rerun_results(d)
    drift, other = set(), []
    for r in res:
        r = latest.get((r["leg"], r.get("arch")), r)
        if r.get("verdict") in OK or (r.get("key") and (LEDGER / f"{r['key']}.json").is_file()):
            continue
        name = f"{r['leg']} {r.get('arch', '')}".strip()
        log = Path(r.get("log") or "")
        text = log.read_text(errors="replace") if r["leg"].startswith("ladder:") and log.is_file() else ""
        lines = [ln for ln in text.splitlines() if FINDING.match(ln)]
        if r["leg"].startswith("ladder:") and (name in (waived or {}) or (
                "size-ladder drift vs the recorded baseline" in text and lines and all(map(is_drift, lines)))):
            drift.add(r["leg"].split(":", 1)[1].split("@", 1)[0])
        elif name not in (waived or {}):
            other.append(name)
    return sorted(drift), other


def rerun_results(d: Path) -> dict:
    """{(leg, arch): row} of the latest run of each leg at gate `d`'s commit in `d` and the run dirs
    beside it, gate_fanout's results.jsonl rows, ordered by when each leg ended. `d` counts too: its
    own rerun of pytest_cpu passed at 20:20Z on 10-18 and still lost to a side run's 16:06Z FAIL."""
    sha = json.loads((d / "plan.json").read_text()).get("sha") if (d / "plan.json").is_file() else None
    rows = []
    for f in d.parent.glob("*/results.jsonl") if sha else ():
        for ln in f.read_text(errors="replace").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("sha") == sha and r.get("ended"):
                rows.append(r)
    return {(r["leg"], r.get("arch")): r for r in sorted(rows, key=lambda r: r["ended"])}


def record_run(gates: Path | None, sha: str) -> tuple[str, Path | None]:
    """('none'|'running'|'pass'|'fail', dir) of the train's record run on `sha` (<gates>/rec-<sha9>)."""
    d = gates / f"rec-{sha[:9]}" if gates else None
    if d is None or not (d / "plan.json").exists():
        return "none", None
    if (d / "verdict.json").exists():
        return ("pass" if json.loads((d / "verdict.json").read_text()).get("pass") else "fail"), d
    return ("running" if _alive(json.loads((d / "plan.json").read_text())) else "none"), d


def passed(gates: Path | None, d: Path, sha: str, repo: Path = REPO, ref: str = "origin/main") -> Path | None:
    """`d` itself when its gate passed; the record run when it failed only on drift that run
    re-recorded; else None."""
    v = json.loads((d / "verdict.json").read_text())
    if v.get("pass"):
        return d
    drift, other = failures(d, waivers(d, gates, repo, ref))
    if other or all(r.get("verdict") in OK for r in v.get("results", [])):
        return None                                    # a fail with no failed leg is no pass either
    if not drift:
        return d                                       # every failure is waived or passed since
    state, rd = record_run(gates, sha)
    return rd if state == "pass" else None


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


def passed_since(gates: Path | None, tsha: str, repo: Path = REPO, ref: str = "origin/main"
                 ) -> tuple[str, Path] | None:
    """(sha, dir) of the newest commit after the last release whose full gate passed, or None.

    After means: it holds what main held when the release branched off (the tag may sit on a
    release branch main never merged), it is not already inside the release, and its code is not
    the release's."""
    done = []
    for p, plan in _since(gates, tsha, repo, ref):
        sha = plan["sha"]
        try:
            if not ((p.parent / "verdict.json").exists() and passed(gates, p.parent, sha, repo, ref)):
                continue
            ahead = git("rev-list", "--count", f"{plan['base']}..{sha}" if plan["base"] else sha, repo=repo)
            done.append((int(git("show", "-s", "--format=%ct", sha, repo=repo)), int(ahead), sha, p.parent))
        except (ValueError, subprocess.CalledProcessError):
            continue                                   # unreadable verdict, or a commit we lack
    return max(done)[2:] if done else None


def fork_point(tsha: str, ref: str, repo: Path = REPO) -> str:
    """The commit on main the release branched from. Once main merges the release branch back
    (release_train does), merge-base(tag, main) is the tag itself, and every gate on a commit
    between the branch-off and the merge-back stopped being a candidate (rec-5e250928a passed
    10-17 03:50Z and the train gated main's head instead), so take it from the main parent of the first commit on main's line that holds the tag (the
    merge-back may come in through a work branch)."""
    after = set(git("rev-list", "--ancestry-path", f"{tsha}..{ref}", repo=repo).split())
    back = [c for c in git("rev-list", "--first-parent", f"{tsha}..{ref}", repo=repo).split() if c in after]
    return git("merge-base", tsha, f"{back[-1]}^1" if back else ref, repo=repo)


def _since(gates: Path | None, tsha: str, repo: Path = REPO, ref: str = "origin/main") -> list:
    """(plan path, plan + "base") of the full runs on commits after the last release (passed_since)."""
    def ancestor(a, b):
        return subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", a, b],
                              capture_output=True).returncode == 0
    base = fork_point(tsha, ref, repo) if tsha else ""
    released = content_hash(tsha, repo) if tsha else ""
    out = []
    for p, plan in _plans(gates):
        sha = plan.get("sha", "")
        try:
            if tsha and (not ancestor(base, sha) or ancestor(sha, tsha) or plan.get("content") == released):
                continue
        except subprocess.CalledProcessError:
            continue
        out.append((p, {**plan, "base": base}))
    return out


def lost_record(gates: Path | None, tsha: str, repo: Path = REPO, ref: str = "origin/main"
                ) -> tuple[str, Path, list] | None:
    """(sha, gate dir, models) of a candidate after the last release whose gate failed only on
    ladder drift and whose record run lost its runner before writing a verdict. Nothing else
    restarts it once main has moved on: decide() records only the commit it targets (pc's
    maintenance reboot stopped rec-5e250928a 10-16 17:25Z with 23 of 27 legs recorded, and the
    train gated main's head instead). gate_fanout resumes it in place, keeping its PASSes."""
    for p, plan in _since(gates, tsha, repo, ref):
        sha, d = plan["sha"], p.parent
        try:
            if not (d / "verdict.json").exists() or json.loads((d / "verdict.json").read_text()).get("pass"):
                continue
            state, rd = record_run(gates, sha)
            if state != "none" or rd is None:
                continue
            drift, other = failures(d, waivers(d, gates, repo, ref))
        except ValueError:
            continue
        if drift and not other:
            return sha, d, drift
    return None


#: How long a failed gate stays the candidate with no FIXES before the train gates main's head
#: instead: long enough for a fix row to land a fix, short enough that an abandoned failure does
#: not hold the train past its 48 h bar.
FIX_HOLD_S = 12 * 3600


def frozen(gates: Path | None, tsha: str, repo: Path = REPO, ref: str = "origin/main",
           now: float | None = None) -> tuple[str, Path, list] | None:
    """(sha, dir, fixes) of the newest full gate on main's line after the last release when it
    failed less than FIX_HOLD_S ago, or its FIXES changed less than that ago: the commit the train
    fixes forward. A gate on a fixed-forward candidate is not on main's line, so it never counts.
    The fixes are the gate's own FIXES after any an older gate since the release listed that the
    commit does not carry yet."""
    runs = []
    for p, plan in _since(gates, tsha, repo, ref):
        try:
            if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", plan["sha"], ref],
                              capture_output=True).returncode:
                continue
            runs.append((int(git("show", "-s", "--format=%ct", plan["sha"], repo=repo)), p.stat().st_mtime, p.parent))
        except subprocess.CalledProcessError:
            continue
    if not runs:
        return None
    d = max(runs)[2]
    v = d / "verdict.json"
    try:
        failed = v.exists() and not json.loads(v.read_text()).get("pass")
    except ValueError:
        return None
    sha = json.loads((d / "plan.json").read_text())["sha"]
    if not failed or passed(gates, d, sha, repo, ref):
        return None
    last = max(f.stat().st_mtime for f in (v, d / "FIXES") if f.exists())
    if (now or time.time()) - last >= FIX_HOLD_S:
        return None
    # An older gate's fixes that never reached main still apply: without them the newer gate fails
    # the same legs again (rel-803db0fc1, 10-17: 4 WH legs lost to the jump-host drop 35dc89179
    # fixed, listed only in rel-412d01dfe/FIXES).
    # One that no longer applies on the newer base was superseded there and is dropped.
    own, carried = fixes(d, repo), []
    for _, q in sorted((p.stat().st_mtime, p.parent) for p, _ in _since(gates, tsha, repo, ref)):
        try:
            older = fixes(q, repo) if q != d else []
        except subprocess.CalledProcessError:
            continue
        for f in older:
            if f in own or f in carried or not subprocess.run(
                    ["git", "-C", str(repo), "merge-base", "--is-ancestor", f, sha], capture_output=True).returncode:
                continue
            try:
                patched(sha, carried + [f], repo)
                carried.append(f)
            except subprocess.CalledProcessError:
                pass
    return sha, d, carried + own


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
    if (cut := passed_since(gates, tsha, repo, ref)) and cut[0] != head:
        sha, d = cut
        nxt = "" if content_hash(sha, repo) == content_hash(head, repo) else (
            f"; main's head {head[:9]} is the next release, gate it now")
        rec = passed(gates, d, sha, repo, ref)
        return {**out, "do": "CUT", "sha": sha, "gate": str(d), "records": str(rec) if rec != d else "",
                "waived": waivers(d, gates, repo, ref),
                "why": f"gate passed on every arch: {d}/VERDICT.md{nxt}"
                       + (f"; size ladders re-recorded in {rec}" if rec != d else "")}
    if lost := lost_record(gates, tsha, repo, ref):
        sha, d, models = lost
        return {**out, "do": "RECORD", "sha": sha, "gate": str(d), "models": models,
                "why": f"the record run {gates}/rec-{sha[:9]} lost its runner; resume it"}
    target, what = head, "main's head"
    if fz := frozen(gates, tsha, repo, ref):
        base, fd, fl = fz
        target, what = base, f"{base[:9]} (frozen: fix forward through {fd}/FIXES)"
        if fl:
            try:
                target = patched(base, fl, repo)
            except subprocess.CalledProcessError as e:
                return {**out, "do": "FIX", "sha": base, "gate": str(fd),
                        "why": f"{fd}/FIXES does not apply on {base[:9]}: {(e.stderr or '').strip()[-300:]}"}
            what = f"{base[:9]} + {len(fl)} fix(es) from {fd}/FIXES"
            out["fixes"] = {"base": base, "gate": str(fd), "commits": fl}
    state, d = gate_state(gates, content_hash(target, repo))
    if (out.get("fixes") and d is not None and (gsha := json.loads((d / "plan.json").read_text())["sha"]) != target
            and content_hash(gsha, repo, tests=True, baselines=True)
            != content_hash(target, repo, tests=True, baselines=True)):
        # Test-only and baseline-only fixes leave the content key alone, so the verdict found is the
        # one they fix (469ed1356, 10-15: two test fixes held the train at FIX on the base's own pytest
        # failures; 59d6264ae, 10-26: re-recorded ladder baselines took gate rel-8812d72f2's verdict,
        # graded on the old records, as their own).
        state, d = "none", None
    why = {"none": f"no gate has run on {what}",
           "running": f"gate running in {d}",
           "fail": f"gate failed: {d}/VERDICT.md",
           "pass": f"gate passed on every arch: {d}/VERDICT.md"}[state]
    do = {"none": "GATE", "running": "WAIT", "fail": "FIX", "pass": "CUT"}[state]
    if state == "fail" and passed(gates, d, gsha := json.loads((d / "plan.json").read_text())["sha"], repo, ref) == d:
        return {**out, "do": "CUT", "sha": gsha, "gate": str(d), "records": "", "waived": waivers(d, gates, repo, ref),
                "why": f"gate passed on every arch but for waived legs: {d}/VERDICT.md"}
    models, other = failures(d, waivers(d, gates, repo, ref)) if state == "fail" else ([], [])
    # The record run need not wait for a rerun of a ladder leg the card failed: when that leg's
    # model drifted on another arch it is in the record set already, so the set is final and the
    # two run side by side (gate 8b2f8acaa queued ~4 h of records behind a 2.5 h BH rerun). The
    # cut still waits for every non-drift failure to pass.
    if models and all(o.startswith("ladder:") and o.split()[0][7:].split("@")[0] in models for o in other):
        rstate, rd = record_run(gates, gsha)
        if rstate == "pass" and not other:
            return {**out, "do": "CUT", "sha": gsha, "gate": str(d), "records": str(rd),
                    "why": f"gate failed only on size-ladder drift, re-recorded in {rd}"}
        if rstate == "none" or (rstate == "running" and not other):
            do = {"none": "RECORD", "running": "WAIT"}[rstate]
            return {**out, "do": do, "sha": gsha, "gate": str(d), "models": models,
                    "why": f"gate failed only on size-ladder drift ({', '.join(models)}): "
                           + ("re-record them" if do == "RECORD" else f"record run in {rd}")
                           + (f"; {', '.join(other)} must pass before the cut" if other else "")}
        why += {"fail": f"; the record run {rd} failed too", "running": f"; record run in {rd}",
                "pass": f"; re-recorded in {rd}"}[rstate]
    if state == "fail":
        res = json.loads((d / "verdict.json").read_text()).get("results", [])
        if owed := sorted({f for r in res if r.get("verdict") == "OWED" for f in r.get("owed", [])}):
            why += f"; record {','.join(owed)} first (gate_fanout.py --legs 'record:*' --record-lever)"
    older = [s for s in out["running"] if d is None or s != json.loads((d / "plan.json").read_text())["sha"]]
    if older:
        why += f"; finish the running gate on {', '.join(s[:9] for s in older)}, do not restart it"
    if part := running_gates(gates, partial=True):
        why += f"; partial runs on {', '.join(s[:9] for s in part)} fill the ledger the gate reuses"
    if state == "fail":
        # The FIX line is the work order (flow_watch relays it): which legs, and where a fix goes. A
        # fixed-forward candidate's fixes go in its base gate's FIXES, not in the candidate's own dir.
        fd = Path(out["fixes"]["gate"]) if out.get("fixes") else d
        if other:
            why += f"; failed: {', '.join(other)}" + (
                f" (ladder drift on {', '.join(models)} is re-recorded by the train)" if models else "")
        why += f"; fix forward: append each fix commit to {fd}/FIXES" + ("" if out.get("fixes") else (
            f" (main's head is gated instead {FIX_HOLD_S // 3600} h after this verdict if none are listed)"))
        why += f"; a failure that is not a regression goes in {fd}/WAIVE as `<leg> <arch> {gsha[:12]}  # why`"
    return {**out, "do": do, "sha": target, "gate": str(d or ""), "why": why}


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
