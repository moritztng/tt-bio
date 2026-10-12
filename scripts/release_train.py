#!/usr/bin/env python3
"""The release train: when release_next.py says CUT, release that commit to PyPI with no human step.

    python3 scripts/release_train.py --gates ~/gates --record ~/train.jsonl            # cron
    python3 scripts/release_train.py --gates ~/gates --dry-run                         # what it would do

On CUT <sha> it makes one release commit on top of <sha> that changes only CHANGELOG.md (the
[Unreleased] notes become the version's section), README.md's pinned install tag and pyproject's
version line. Those are exactly the files the gate's content key leaves out, and the train refuses
to tag if the key moved, so the tag is the gated code. It pushes the commit as rel/v<version> (for
main to merge back) and the tag; the tag's workflow (.github/workflows/release.yaml) builds,
publishes to PyPI and creates the GitHub release. The train then waits for PyPI to serve the
version and installs it into a fresh venv (without dependencies: pc's disk has no room for torch
per release) to prove the wheel is the one tagged.

The version is the next minor when the notes carry an Added, Changed or Removed section, else the
next patch, skipping any tag that already exists. Notes already released under an earlier tag are
dropped, so a main that has not merged the last release commit back does not ship them twice.
Anything else release_next.py answers (GATE, WAIT, FIX, NOTHING) is printed and left alone.
"""
from __future__ import annotations

import argparse
import json
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


def pypi_has(version: str) -> bool:
    try:
        return version in json.load(urllib.request.urlopen(PYPI, timeout=30))["releases"]
    except Exception:
        return False


def install_check(version: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["uv", "venv", "-q", d], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "-p", f"{d}/bin/python", "--no-deps", "--no-cache",
                        f"tt-bio=={version}"], check=True)
        return subprocess.run([f"{d}/bin/python", "-c", "import importlib.metadata as m; print(m.version('tt-bio'))"],
                              check=True, capture_output=True, text=True).stdout.strip()


def cut(sha: str, gate_dir: str, remote: str, dry: bool, wait_s: int, record: Path | None) -> int:
    tag, tsha = last_release()
    taken = set(git("tag", "-l", "v*").split())
    released = git("show", f"{tag}:CHANGELOG.md") if tag else ""
    body = notes(git("show", f"{sha}:CHANGELOG.md"), released)
    if not body:
        print(f"CUT {sha[:12]} has no release notes in CHANGELOG.md [Unreleased]; not cutting")
        return 1
    version = next_version(tag or "v0.0.0", body, taken)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"train: v{version} from {sha[:12]} (last {tag}), gate {gate_dir}")
    if dry:
        print(body)
        return 0
    with tempfile.TemporaryDirectory() as d:
        tree = Path(d) / "tree"
        git("worktree", "add", "-q", "--detach", str(tree), sha)
        try:
            release_files(tree, version, body, day)
            git("add", "CHANGELOG.md", "README.md", "pyproject.toml", repo=tree)
            git("commit", "-qm", f"Release {version}\n\nGated at {sha} ({Path(gate_dir).name}).", repo=tree)
            rel = git("rev-parse", "HEAD", repo=tree)
        finally:
            git("worktree", "remove", "--force", str(tree))
    if content_hash(rel) != content_hash(sha):
        print(f"release commit {rel[:12]} changed gated code; not tagging")
        return 1
    git("tag", "-a", f"v{version}", rel, "-m", f"tt-bio {version}")
    git("push", "-q", remote, f"{rel}:refs/heads/rel/v{version}", f"v{version}")
    t0, rec = time.time(), {"version": version, "sha": sha, "release_commit": rel, "gate": gate_dir,
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gates", type=Path, required=True, help="directory holding gate_fanout --out dirs")
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--record", type=Path, help="append one JSON line per release here")
    ap.add_argument("--wait", type=int, default=3600, help="seconds to wait for PyPI to serve the tag")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for _ in range(3):   # a fetch racing another one on the same clone fails; a stale view is safe
        if subprocess.run(["git", "-C", str(REPO), "fetch", "-q", args.remote, "--tags"]).returncode == 0:
            break                       # (a tag that exists remotely is refused at push)
        time.sleep(20)
    d = decide(gates=args.gates, ref=f"{args.remote}/main")
    if d["do"] != "CUT":
        print(f"{d['do']} {d['sha'][:12]}: {d['why']}")
        return 0
    return cut(d["sha"], d["gate"], args.remote, args.dry_run, args.wait, args.record)


if __name__ == "__main__":
    sys.exit(main())
