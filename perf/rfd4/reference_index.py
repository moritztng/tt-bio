"""Index a GPU reference session's designs for the seed floor.

`rfd4-fair-gpu` generates the reference designs; this turns its output tree into the index the
floor watcher consumes (`state/rfd4/evidence/reference-designs.json`, schema in METRICS.md), so
the GPU row does not have to know the harness's shape and the floor lands on the next tick.

    python3 perf/rfd4/reference_index.py --root <session reference dir> \
        --family binder --recipe "<preset, steps, precision>" --session "vast:<id>"

Each immediate subdirectory of `--root` whose name ends in a seed (`...seed1234`, `seed=1234`,
or just `1234`) is one seed arm. Both the generation tree and any `rfd4-evaluate` tree for the
same seed are recorded, because one arm's metrics are split across the two.

THE FLOOR NEEDS TWO SEEDS. As of 2026-10-21 `perf/kitcmp/rfd4_arms.py --reference` runs the chosen
arm once with no `seed=` override, so a session that succeeds still produces one seed arm and no
floor. This script says so rather than writing a one-seed index that reads as progress.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

INDEX = pathlib.Path("/home/moritz/.coworker/state/rfd4/evidence/reference-designs.json")
SEED_RE = re.compile(r"(?:^|[^0-9])(?:seed[=_-]?)?(\d+)$")


def seed_arms(root: pathlib.Path) -> dict[str, list[str]]:
    """seed -> the roots holding that seed's designs and metrics."""
    arms: dict[str, list[str]] = {}
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        m = SEED_RE.search(child.name)
        if not m:
            continue
        has_designs = any(child.rglob("*_generated.cif")) or any(child.rglob("*_generated.cif.zst"))
        has_metrics = any(child.rglob("per_example_metrics*.csv"))
        if has_designs or has_metrics:
            arms.setdefault(m.group(1), []).append(str(child))
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the session's reference output directory")
    ap.add_argument("--family", required=True)
    ap.add_argument("--recipe", required=True, help="preset, steps, precision -- quoted into every bar")
    ap.add_argument("--session", required=True, help="box and rental id")
    ap.add_argument("--out", default=str(INDEX))
    a = ap.parse_args(argv)

    root = pathlib.Path(a.root)
    if not root.is_dir():
        print(f"{root} is not a directory")
        return 2
    arms = seed_arms(root)
    if not arms:
        print(f"no seed arms under {root}: expected subdirectories named for their seed")
        return 1

    out = pathlib.Path(a.out)
    index = json.loads(out.read_text()) if out.exists() else {}
    index.setdefault(a.family, {}).update(
        {"recipe": a.recipe, "session": a.session,
         "seeds": {s: roots for s, roots in sorted(arms.items())}})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=1) + "\n")
    print(f"{a.family}: {len(arms)} seed arm(s) indexed -> {out}")
    if len(arms) < 2:
        print("the seed floor needs TWO seeds; one arm indexes the reference but calibrates no floor")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
