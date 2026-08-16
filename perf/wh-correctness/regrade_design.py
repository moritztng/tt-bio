#!/usr/bin/env python3
"""Re-grade already-downloaded design artifacts after a checker fix, without re-folding.

`check_structure.py`'s designed-chain assertion originally required the chain id the spec
named. BoltzGen renames chains per protocol -- `protein-small_molecule` returns the designed
binder as chain A -- so the assertion failed ten correct designs on the instrument's own
assumption. The fold is not in question and the artifacts are on disk, so the honest repair is
to re-run the checker over them and rewrite the verdict, not to spend another 245 s of shared
production capacity reproducing a structure we already have.

    regrade_design.py            # report
    regrade_design.py --apply    # rewrite those rows in matrix.jsonl
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


def cell_specs() -> dict[str, list[str]]:
    """cell -> the extra check_structure flags matrix.py would pass it today."""
    sys.path.insert(0, str(HERE))
    import matrix
    out = {}
    for c in matrix.cells("design"):
        flags = []
        if "design_chain" in c:
            flags += ["--design-chain", c["design_chain"]]
        if "design_min" in c:
            flags += ["--design-min", str(c["design_min"]), "--design-max", str(c["design_max"])]
        out[c["cell"]] = flags
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    specs = cell_specs()
    rows = [json.loads(l) for l in MATRIX.read_text().splitlines() if l.strip()]
    changed = 0
    for r in rows:
        if r["cell"] not in specs or not (ARTIFACTS / r["cell"]).exists():
            continue
        fails, n = [], 0
        for f in sorted(p for p in (ARTIFACTS / r["cell"]).rglob("*")
                        if p.suffix in (".cif", ".pdb")):
            rep = f.with_suffix(f.suffix + ".check.json")
            subprocess.run([sys.executable, str(HERE / "check_structure.py"), str(f),
                            "--kind", "design", "--json", str(rep), "--quiet"] + specs[r["cell"]],
                           check=False)
            n += 1
            if rep.exists():
                fails += json.loads(rep.read_text()).get("fail", [])
        if not n:
            continue
        was = r.get("pass")
        r["pass"] = not fails
        r["why"] = "; ".join(fails)[:400]
        r["regraded"] = "checker design-chain fix"
        print(f"{r['cell']:20s} {n:3d} structures  {was} -> {r['pass']}  {r['why'][:120]}")
        changed += 1

    print(f"{changed} cell(s) re-graded")
    if a.apply and changed:
        MATRIX.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"rewritten {MATRIX.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
