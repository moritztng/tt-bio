#!/usr/bin/env python3
"""The release train: when release_next.py says CUT, release that commit to PyPI with no human step.

    python3 scripts/release_train.py --gates ~/gates --record ~/train.jsonl            # cron
    python3 scripts/release_train.py --gates ~/gates --dry-run                         # what it would do

On CUT <sha> it makes one release commit on top of <sha> that changes only CHANGELOG.md (the
[Unreleased] notes become the version's section), README.md's pinned install tag and pyproject's
version line, plus the abb3 records the gate's record:abb3 leg re-ran and re-stamped at the gated
commit (<gate>/abb3/, so the train's gate config carries --record-abb3). Those are exactly the files
the gate's content key leaves out, and the train refuses to tag if the key moved, so the tag is the gated code. It pushes the commit as rel/v<version> (for
main to merge back) and the tag; the tag's workflow (.github/workflows/release.yaml) builds,
publishes to PyPI and creates the GitHub release. The train then waits for PyPI to serve the
version and installs it into a fresh venv (without dependencies: pc's disk has no room for torch
per release) to prove the wheel is the one tagged.

The version is the next minor when the notes carry an Added, Changed or Removed section, else the
next patch, skipping any tag that already exists. Notes already released under an earlier tag are
dropped, so a main that has not merged the last release commit back does not ship them twice.
On GATE, with --gate-config, it starts gate_fanout.py on that commit, detached, so every push to
main is gated without anyone asking; the ledger means only the legs the push changed run. The
config is a JSON object of gate_fanout's own flags, {"hosts": "...", "workers": "...", "timed":
"...", "args": [...]}, and the run goes to <gates>/rel-<sha9>; while another full gate runs it
waits for it instead. On RECORD (the gate failed only on size-ladder drift, release_next.py) it
re-records those models' ladders on the same commit into <gates>/rec-<sha9>, and the release
commit carries the fragments. When a failed gate's dir lists fixes in FIXES (release_next.py, Fix
forward), the commit it gates is the failed one plus those fixes, pushed first as rel-cand/<sha9> so
the gate hosts can fetch it. WAIT, FIX and NOTHING are printed and left alone (a FIX is a regression
for a person or a fix row, who fixes it forward through FIXES; the train never re-cuts around it).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_fanout import REPO, content_hash  # noqa: E402
from release_next import decide, git, last_release  # noqa: E402
from splice_ladder_fragments import splice  # noqa: E402

PYPI = "https://pypi.org/pypi/tt-bio/json"
SECTION = re.compile(r"^## \[(?P<name>[^\]]+)\][^\n]*\n(?P<body>.*?)(?=^## \[|\Z)", re.M | re.S)
BULLET = re.compile(r"^- .*?(?=^- |^#|\Z)", re.M | re.S)


def notes(changelog: str, released: str) -> str:
    """The [Unreleased] body at the candidate, minus bullets an earlier release already carried."""
    secs = {m["name"]: m["body"] for m in SECTION.finditer(changelog)}
    old = {b.strip() for m in SECTION.finditer(released) if m["name"] != "Unreleased"
           for b in BULLET.findall(m["body"])}
    out, keep = [], True
    for block in re.split(r"(?=^### |^- )", secs.get("Unreleased", ""), flags=re.M):
        if block.startswith("- "):
            keep = block.strip() not in old
        if keep or block.startswith("### "):
            out.append(block)
    body = "".join(out)
    # a heading whose bullets were all dropped goes too
    return re.sub(r"^### [^\n]*\n\s*(?=^### |\Z)", "", body, flags=re.M).strip()


def next_version(tag: str, body: str, taken: set) -> str:
    major, minor, patch = (int(x) for x in tag.lstrip("v").split("."))
    if re.search(r"^### (Added|Changed|Removed)", body, re.M):
        minor, patch = minor + 1, 0
    else:
        patch += 1
    while f"v{major}.{minor}.{patch}" in taken:
        patch += 1
    return f"{major}.{minor}.{patch}"


def release_files(tree: Path, version: str, body: str, day: str) -> None:
    cl = tree / "CHANGELOG.md"
    text = cl.read_text()
    m = SECTION.search(text)
    assert m and m["name"] == "Unreleased", "CHANGELOG.md has no [Unreleased] section first"
    cl.write_text(text[:m.start()] + f"## [Unreleased]\n\n## [{version}] - {day}\n\n{body}\n\n" + text[m.end():])
    pp = tree / "pyproject.toml"
    pp.write_text(re.sub(r'^version = "[^"]*"', f'version = "{version}"', pp.read_text(), count=1, flags=re.M))
    rd = tree / "README.md"
    rd.write_text(re.sub(r"(tt-bio\.git@)v\d+\.\d+\.\d+", rf"\g<1>v{version}", rd.read_text()))


RELEASE_FILES = ("CHANGELOG.md", "README.md", "pyproject.toml")


def rerecorded_files(gate_dir: str, sha: str, records: str = "") -> dict:
    """{path: text} the release commit replaces: the abb3 records the gate's record:abb3 leg re-ran
    and re-stamped (<gate>/abb3/), and the size-ladder fragments a record run re-recorded
    (<records>/recorded/<card_type>/<model>.json, one card type's entry each)."""
    out = {}
    v, d = Path(gate_dir) / "verdict.json", Path(gate_dir) / "abb3"
    res = json.loads(v.read_text()).get("results", []) if v.exists() else []
    if any(r["leg"] == "record:abb3" and r["verdict"] == "PASS" for r in res):
        out.update({str(f.relative_to(d)): f.read_text() for f in sorted(d.rglob("*.txt"))})
    if records and (frags := sorted(Path(records).glob("recorded/*/*.json"))):
        with tempfile.TemporaryDirectory() as tmp:
            bd = Path(tmp)
            for name in {f.name for f in frags}:
                (bd / name).write_text(git("show", f"{sha}:docs/size_ladder_baseline.d/{name}") + "\n")
            for model, _ in splice([Path(records) / "recorded"], bd):
                out[f"docs/size_ladder_baseline.d/{model}.json"] = (bd / f"{model}.json").read_text()
    return out


def release_commit(sha: str, version: str, body: str, day: str, records: dict | None = None) -> str:
    """The release commit on `sha`, built in a scratch index: no checkout, no disk. `records`
    ({path: text}, rerecorded_files) replace those files."""
    records = records or {}
    with tempfile.TemporaryDirectory() as d:
        tree = Path(d)
        files = (*RELEASE_FILES, *records)
        for f in files:
            (tree / f).parent.mkdir(parents=True, exist_ok=True)
            (tree / f).write_text(git("show", f"{sha}:{f}") + "\n")
        release_files(tree, version, body, day)
        for f, text in records.items():
            (tree / f).write_text(text)
        env = {**os.environ, "GIT_INDEX_FILE": str(tree / "index")}

        def plumb(*a):
            return subprocess.run(["git", "-C", str(REPO), *a], check=True, capture_output=True, text=True,
                                  env=env).stdout.strip()
        plumb("read-tree", sha)
        for f in files:
            mode = plumb("ls-files", "-s", f).split()[0]
            plumb("update-index", "--cacheinfo", f"{mode},{plumb('hash-object', '-w', str(tree / f))},{f}")
        return plumb("commit-tree", plumb("write-tree"), "-p", sha,
                     "-m", f"Release {version}\n\nGated at {sha}.")


def pypi_has(version: str) -> bool:
    try:
        return version in json.load(urllib.request.urlopen(PYPI, timeout=30))["releases"]
    except Exception:
        return False


def install_check(version: str, tries: int = 10) -> str:
    """The installed version, or "" if it never installs. PyPI's JSON lists a release minutes before
    the simple index serves it; on 2026-10-15 v0.17.0 crashed the train here and lost its record."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["uv", "venv", "-q", d], check=True)
        for i in range(tries):
            if subprocess.run(["uv", "pip", "install", "-q", "-p", f"{d}/bin/python", "--no-deps", "--no-cache",
                               f"tt-bio=={version}"]).returncode == 0:
                break
            time.sleep(60)
        else:
            return ""
        return subprocess.run([f"{d}/bin/python", "-c", "import importlib.metadata as m; print(m.version('tt-bio'))"],
                              check=True, capture_output=True, text=True).stdout.strip()


def cut(sha: str, gate_dir: str, remote: str, dry: bool, wait_s: int, record: Path | None,
        records: str = "", waived: dict | None = None) -> int:
    tag, tsha = last_release()
    taken = set(git("tag", "-l", "v*").split())
    released = git("show", f"{tag}:CHANGELOG.md") if tag else ""
    body = notes(git("show", f"{sha}:CHANGELOG.md"), released)
    if not body:
        print(f"CUT {sha[:12]} has no release notes in CHANGELOG.md [Unreleased]; not cutting")
        return 1
    version = next_version(tag or "v0.0.0", body, taken)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"train: v{version} from {sha[:12]} (last {tag}), gate {gate_dir}, re-records {sorted(rerecorded_files(gate_dir, sha, records))}, waived {waived or {}}")
    if dry:
        print(body)
        return 0
    files = rerecorded_files(gate_dir, sha, records)
    rel = release_commit(sha, version, body, day, files)
    if content_hash(rel) != content_hash(sha):
        print(f"release commit {rel[:12]} changed gated code; not tagging")
        return 1
    git("tag", "-a", f"v{version}", rel, "-m", f"tt-bio {version}")
    git("push", "-q", remote, f"{rel}:refs/heads/rel/v{version}", f"v{version}")
    t0, rec = time.time(), {"version": version, "sha": sha, "release_commit": rel, "gate": gate_dir,
                            "waived": waived or {},
                            "tagged": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    while not pypi_has(version) and time.time() - t0 < wait_s:
        time.sleep(60)
    rec["pypi"] = pypi_has(version)
    rec["installed"] = install_check(version) if rec["pypi"] else ""
    rec["ok"] = rec["installed"] == version
    print(json.dumps(rec))
    if record:
        with open(record, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return 0 if rec["ok"] else 1


def own_runner(out: Path) -> Path:
    """A worktree at this train's commit, <out>/runner, for the gate in `out` to run from. A gate
    reads its scripts for hours; run from the train's own clone it pinned that clone, and the cron
    skipped every tick until it ended, so on 2026-10-13 a gate stalled on a returned Galaxy also held
    back the cut of an older commit that was ready. Runners of gates that have ended are removed."""
    for old in out.parent.glob("*/runner"):
        if (old.parent / "VERDICT.md").exists() and not locked(old.parent / "runner.lock"):
            subprocess.run(["git", "-C", str(REPO), "worktree", "remove", "--force", str(old)], capture_output=True)
    subprocess.run(["git", "-C", str(REPO), "worktree", "prune"], capture_output=True)
    runner = out / "runner"
    head = git("rev-parse", "HEAD", repo=REPO)
    if runner.exists():
        subprocess.run(["git", "-C", str(runner), "checkout", "-q", "--detach", head], check=True)
    else:
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "-q", "--detach", str(runner), head], check=True)
    return runner


def locked(path: Path) -> bool:
    """True while a gate_fanout runner holds `path` (gate_fanout.hold_out)."""
    try:
        with open(path) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except FileNotFoundError:
        return False
    except BlockingIOError:
        return True
    return False


def start_gate(sha: str, gates: Path, config: Path, models: list | None = None) -> None:
    """The full gate on `sha` (<gates>/rel-<sha9>, plus the config's "args"), or with `models` a
    record run of their size ladders (<gates>/rec-<sha9>)."""
    cfg = json.loads(config.read_text())
    out = gates / f"{'rec' if models else 'rel'}-{sha[:9]}"
    out.mkdir(parents=True, exist_ok=True)
    runner = own_runner(out)
    cmd = [sys.executable, "-u", str(runner / "scripts" / "gate_fanout.py"), "--sha", sha, "--out", str(out),
           *(x for k in ("hosts", "workers", "timed") if cfg.get(k) for x in (f"--{k}", cfg[k])),
           *(["--legs", "record:*", "--record-full", ",".join(models)] if models else cfg.get("args", []))]
    with open(out / "train.log", "a") as log:
        p = subprocess.Popen(cmd, cwd=runner, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONPATH": str(runner)})
    print(f"{'RECORD' if models else 'GATE'} {sha[:12]}: started gate_fanout pid {p.pid}, {out}/train.log")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gates", type=Path, required=True, help="directory holding gate_fanout --out dirs")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--record", type=Path, help="append one JSON line per release here")
    ap.add_argument("--wait", type=int, default=3600, help="seconds to wait for PyPI to serve the tag")
    ap.add_argument("--gate-config", type=Path, help="start gate_fanout on GATE with these flags (JSON)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for _ in range(3):   # a fetch racing another one on the same clone fails; a stale view is safe
        if subprocess.run(["git", "-C", str(REPO), "fetch", "-q", args.remote, "--tags"]).returncode == 0:
            break                       # (a tag that exists remotely is refused at push)
        time.sleep(20)
    d = decide(gates=args.gates, ref=f"{args.remote}/main")
    if d["do"] == "GATE" and d["running"]:
        # One full gate at a time: a second would queue the same legs on the same cards, and the
        # running one's PASSes (with their read sets) are what the next gate reuses.
        d = {**d, "do": "WAIT", "why": f"gate main's head after the running gate on "
                                       f"{', '.join(s[:9] for s in d['running'])} ends"}
    if d["do"] in ("GATE", "RECORD") and args.gate_config and not args.dry_run:
        if fx := d.get("fixes"):
            subprocess.run(["git", "-C", str(REPO), "push", "-q", "-f", args.remote,
                            f"{d['sha']}:refs/heads/rel-cand/{fx['base'][:9]}"], check=True)
        start_gate(d["sha"], args.gates, args.gate_config, d.get("models"))
        return 0
    if d["do"] != "CUT":
        print(f"{d['do']} {d['sha'][:12]}: {d['why']}")
        return 0
    return cut(d["sha"], d["gate"], args.remote, args.dry_run, args.wait, args.record, d.get("records", ""),
               d.get("waived"))


if __name__ == "__main__":
    sys.exit(main())
