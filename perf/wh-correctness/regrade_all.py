#!/usr/bin/env python3
"""Re-run the checker over every artifact already on disk and report verdict changes.

`chain_geometry` was changed to classify per residue rather than per chain, so a chain
holding two polymer types is measured as two. That is a core function on every predict cell
as well as the design ones, and "it only affects mixed chains" is a claim, not a measurement.
This re-grades everything that has an artifact and prints each cell whose verdict moved, so a
change to the instrument cannot silently rewrite 150 answers.

    regrade_all.py            # report changes
    regrade_all.py --apply    # write them back into matrix.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "results" / "matrix.jsonl"
ARTIFACTS = HERE / "results" / "artifacts"
PAYLOADS = HERE / "results" / "payloads"


def design_flags() -> dict[str, list[str]]:
    sys.path.insert(0, str(HERE))
    import matrix
    out = {}
    for c in matrix.cells("design"):
        f = []
        if "design_chain" in c:
            f += ["--design-chain", c["design_chain"]]
        if "design_min" in c:
            f += ["--design-min", str(c["design_min"]), "--design-max", str(c["design_max"])]
        out[c["cell"]] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    dflags = design_flags()
    rows = [json.loads(l) for l in MATRIX.read_text().splitlines() if l.strip()]
    moved, checked = [], 0
    for r in rows:
        cell = r["cell"]
        d = ARTIFACTS / cell
        if not d.exists() or r.get("kind") == "embed":
            continue
        structs = sorted(p for p in d.rglob("*") if p.suffix in (".cif", ".pdb"))
        if not structs:
            continue
        is_design = cell in dflags
        inp = PAYLOADS / f"{cell}.input.yaml"
        fails = []
        for f in structs:
            rep = f.with_suffix(f.suffix + ".check.json")
            cmd = [sys.executable, str(HERE / "check_structure.py"), str(f),
                   "--kind", "design" if is_design else "predict",
                   "--json", str(rep), "--quiet"]
            if is_design:
                cmd += dflags[cell]
            elif inp.exists():
                cmd += ["--input", str(inp)]
            subprocess.run(cmd, check=False)
            if rep.exists():
                fails += json.loads(rep.read_text()).get("fail", [])
        checked += 1
        was, now = r.get("pass"), not fails
        if was != now:
            moved.append((cell, was, now, "; ".join(fails)[:150]))
        r["pass"], r["why"] = now, "; ".join(fails)[:400]

    print(f"{checked} cell(s) re-graded, {len(moved)} verdict change(s)")
    for cell, was, now, why in moved:
        print(f"  {cell:32s} {was} -> {now}   {why}")
    if a.apply:
        MATRIX.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"rewritten {MATRIX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
