#!/usr/bin/env python3
"""Everything a campaign folder says about what it has done, as one comparable record.

The soak has to show that a campaign interrupted mid-trajectory and restarted keeps every design
it had accepted and charges nothing twice. That is a diff, not a claim, so it needs a snapshot
taken before the interruption and another after the resume:

  state          `.campaign_state.json` verbatim -- what the campaign thinks it has charged
  trajectories   one entry per trajectory row: its number, its recipe hash, how it ended
  accepted       one entry per accepted design: its name, its hash, and the sha256 of every
                 structure file written for it, so a rewritten or truncated design shows up
  ranked         the ranked table's order, which is rewritten on every acceptance
  files          count and bytes under the folder, to catch a resume that starts a second tree

Run it twice and compare with `--against`, which prints only what changed and exits non-zero if
anything that must not change did: an accepted design that lost its structure, changed its
sha256, or disappeared, or a trajectory number that was charged a second time.

  census.py <project> --out before.json
  census.py <project> --out after.json --against before.json
"""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import sys

STRUCTURE_SUFFIXES = (".cif", ".pdb")


def rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as table:
        return list(csv.DictReader(table))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def table(project: pathlib.Path, *names: str) -> pathlib.Path | None:
    """A campaign table by any of the names BindCraft 2 has written it under."""
    for name in names:
        for candidate in (project / name, *project.glob(f"*/{name}")):
            if candidate.is_file():
                return candidate
    return None


def structures_for(project: pathlib.Path, design: str) -> dict[str, str]:
    found = {}
    for path in project.rglob("*"):
        if (path.is_file() and path.suffix.lower() in STRUCTURE_SUFFIXES
                and path.name.startswith(design)):
            found[str(path.relative_to(project))] = sha256(path)
    return found


def census(project: pathlib.Path) -> dict:
    state_path = project / ".campaign_state.json"
    state = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except ValueError:
            state = "unreadable"
    trajectory_rows = rows(table(project, "trajectories.csv", "!_Trajectories.csv") or project / "_")
    accepted_rows = rows(table(project, "accepted.csv", "!_Accepted.csv") or project / "_")
    ranked_rows = rows(table(project, "!_Ranked.csv", "ranked.csv") or project / "_")
    files = [path for path in project.rglob("*") if path.is_file()]
    return {
        "project": str(project),
        "state": state,
        "trajectories": [{"trajectory": row.get("trajectory"), "design": row.get("design"),
                          "hash": row.get("hash"), "terminated": row.get("terminated")}
                         for row in trajectory_rows],
        "accepted": [{"design": row.get("design"), "hash": row.get("hash"),
                      "structures": structures_for(project, row.get("design") or "\0")}
                     for row in accepted_rows],
        "ranked": [row.get("design") for row in ranked_rows],
        "files": {"count": len(files), "bytes": sum(path.stat().st_size for path in files)},
    }


def compare(before: dict, after: dict) -> tuple[list[str], list[str]]:
    """`(broken, moved)`: what must not have changed and did, and what legitimately grew."""
    broken, moved = [], []
    was = {design["design"]: design for design in before["accepted"]}
    now = {design["design"]: design for design in after["accepted"]}
    for name, design in was.items():
        if name not in now:
            broken.append(f"accepted design {name} is gone after the resume")
            continue
        if design["structures"] and not now[name]["structures"]:
            broken.append(f"accepted design {name} lost every structure file")
        for path, digest in design["structures"].items():
            if path not in now[name]["structures"]:
                broken.append(f"{name}: structure {path} is gone")
            elif now[name]["structures"][path] != digest:
                broken.append(f"{name}: structure {path} was rewritten "
                              f"({digest[:12]} -> {now[name]['structures'][path][:12]})")
    numbers_before = [row["trajectory"] for row in before["trajectories"]]
    numbers_after = [row["trajectory"] for row in after["trajectories"]]
    duplicated = {number for number in numbers_after if numbers_after.count(number) > 1}
    if duplicated:
        broken.append(f"trajectory numbers written twice: {sorted(duplicated)}")
    hashes_after = [row["hash"] for row in after["trajectories"] if row["hash"]]
    repeated = {value for value in hashes_after if hashes_after.count(value) > 1}
    if repeated:
        broken.append(f"recipe designed twice: {sorted(repeated)}")
    if [number for number in numbers_before if number not in numbers_after]:
        broken.append("a trajectory row present before the interruption is gone")

    moved.append(f"trajectory rows {len(numbers_before)} -> {len(numbers_after)}")
    moved.append(f"accepted designs {len(was)} -> {len(now)}")
    for key in ("trajectories", "accepted"):
        first = (before["state"] or {}).get(key) if isinstance(before["state"], dict) else None
        second = (after["state"] or {}).get(key) if isinstance(after["state"], dict) else None
        moved.append(f"state {key} {first} -> {second}")
    moved.append(f"files {before['files']['count']} -> {after['files']['count']}, "
                 f"{before['files']['bytes']} -> {after['files']['bytes']} bytes")
    return broken, moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--out")
    ap.add_argument("--against", help="an earlier census to compare this one against")
    args = ap.parse_args()

    project = pathlib.Path(args.project)
    if not project.is_dir():
        print(f"no campaign folder at {project}", file=sys.stderr)
        return 2
    now = census(project)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(now, indent=1, sort_keys=True))
    if not args.against:
        print(json.dumps({key: now[key] for key in ("state", "ranked", "files")}, indent=1))
        print(f"trajectories {len(now['trajectories'])}, accepted {len(now['accepted'])}")
        return 0
    before = json.loads(pathlib.Path(args.against).read_text())
    broken, moved = compare(before, now)
    for line in moved:
        print(f"  moved: {line}")
    for line in broken:
        print(f"BROKEN: {line}")
    print("RESUME CLEAN" if not broken else f"RESUME BROKE {len(broken)} THINGS")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
