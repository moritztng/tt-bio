#!/usr/bin/env python3
"""Which recorded capacity cells are stale in the row that actually governs them.

`scripts/capacity_gate.py` used to pin each cell to a hash of the WHOLE ceiling table, so a
Wormhole-only cap moving marked every Blackhole cell stale and asked for hours of card time that
would re-measure the same 1536 tokens against the same rows. The fingerprint is per (model, arch)
now. This script is the migration and the evidence behind it: for every cell in
`docs/capacity_gate_baseline.json` it reads `tt_bio/size_limits.CEILINGS` as it stood at the tree
the cell was measured at, compares the cell's own row against today's, and re-stamps only the
cells whose row did not move. A cell whose row DID move keeps its old stamp and stays stale,
which is the whole point of the pin.

    python3 perf/land_standing/capacity_cell_rows.py            # report
    python3 perf/land_standing/capacity_cell_rows.py --write    # re-stamp the unmoved cells

Run from the repo root. It reads git, not the working tree, so an uncommitted edit to
size_limits.py does not change what a historical row looked like.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import capacity_gate as cg                                                       # noqa: E402

BASELINE = ROOT / "docs" / "capacity_gate_baseline.json"


def ceilings_at(tree: str):
    """`CEILINGS` as it stood at `tree`, or None if that tree's module will not load."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{tree}:tt_bio/size_limits.py"],
                         capture_output=True, text=True)
    if src.returncode:
        return None
    ns: dict = {}
    try:
        exec(compile(src.stdout, f"size_limits.py@{tree[:9]}", "exec"), ns)
    except Exception:
        return None
    return ns.get("CEILINGS")


def row(ceilings, model: str) -> str:
    """The one row a cell of `model` on the bar's arch is measured against, as a stable string."""
    c = (ceilings or {}).get(model, {}).get(cg.BAR_ARCH)
    if c is None:
        return json.dumps([model, cg.BAR_ARCH, "no row for this arch"])
    return json.dumps([model, cg.BAR_ARCH, c.residues, c.measured, c.counts,
                       c.ladder_ligand_atoms], default=str)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="re-stamp the cells whose row is unmoved")
    a = ap.parse_args()

    doc = json.loads(BASELINE.read_text())
    now = ceilings_at("HEAD")
    cache: dict[str, object] = {}
    unmoved, moved, unreadable = [], [], []
    for card, blk in doc["cards"].items():
        for model, cell in sorted((blk.get("cells") or {}).items()):
            tree = (cell or {}).get("tree")
            if tree not in cache:
                cache[tree] = ceilings_at(tree)
            then = cache[tree]
            if then is None:
                unreadable.append((card, model, tree))
                continue
            (unmoved if row(then, model) == row(now, model) else moved).append((card, model, tree))
            if a.write and row(then, model) == row(now, model):
                cell["ceilings_fingerprint"] = cg.ceilings_fingerprint(model)

    print(f"unmoved (row identical since the cell was measured): {len(unmoved)}")
    for card, model, tree in unmoved:
        print(f"  {card}/{model:<15} {tree[:9]}  {row(now, model)}")
    print(f"\nmoved (needs card time): {len(moved)}")
    for card, model, tree in moved:
        print(f"  {card}/{model:<15} {tree[:9]}\n      was {row(cache[tree], model)}"
              f"\n      now {row(now, model)}")
    if unreadable:
        print(f"\ntree unreadable, left alone: {len(unreadable)}")
        for card, model, tree in unreadable:
            print(f"  {card}/{model} {tree}")

    if a.write:
        BASELINE.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"\nre-stamped {len(unmoved)} cells in {BASELINE.relative_to(ROOT)}")
        print(f"stale after the write: {cg.baseline_stale()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
