#!/usr/bin/env python3
"""Does a batched job return N DIFFERENT designs, and are they as good as one at a time?

Counting CIFs proves a batch ran; it does not prove the batch is worth anything. Two silent
failures sit exactly here and neither shows up in seconds per design:

  * **N copies of one design.** A batch axis that broadcasts one noise draw returns N files,
    passes every count check, and multiplies throughput by nothing. Measured as the pairwise
    CA spread in the target's own frame, which needs no superposition: the designs are placed
    in that frame already, so a real batch separates them by tens of Angstrom and a broadcast
    one by zero.
  * **geometry that degrades with batch size.** Scored with
    `perf/wh-correctness/check_structure.py --kind design`, the same code
    `scripts/release_gate.py` imports for its GEOMETRY leg, so a design judged here is judged
    by what gates a release.

    python3 perf/mgxscale/batchqa.py OUT_DIR [OUT_DIR ...] --out qa.jsonl
"""
import argparse
import itertools
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = ROOT / "perf" / "wh-correctness" / "check_structure.py"


def ca_coords(p: pathlib.Path):
    import numpy as np
    cols, rows = [], []
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1].split()[0])
        elif s.startswith(("ATOM", "HETATM")):
            f = s.split()
            if len(f) == len(cols):
                rows.append(f)
    ai, xi = cols.index("label_atom_id"), cols.index("Cartn_x")
    return np.array([[float(r[xi]), float(r[xi + 1]), float(r[xi + 2])]
                     for r in rows if r[ai] == "CA"])


def spread(cifs) -> dict:
    """Pairwise CA RMS deviation, no superposition. Zero means the batch is one design."""
    import numpy as np
    X = [ca_coords(c) for c in cifs]
    pairs = [float(np.sqrt(((X[i] - X[j]) ** 2).sum(1).mean()))
             for i, j in itertools.combinations(range(len(X)), 2)
             if len(X[i]) == len(X[j]) and len(X[i])]
    if not pairs:
        return {"n": len(cifs), "pairs": 0}
    return {"n": len(cifs), "pairs": len(pairs), "min_a": round(min(pairs), 3),
            "median_a": round(sorted(pairs)[len(pairs) // 2], 3), "max_a": round(max(pairs), 3),
            "identical": bool(min(pairs) < 1e-6)}


def geometry(cif: pathlib.Path, design_chain: str | None) -> dict:
    cmd = [sys.executable, str(CHECKER), str(cif), "--kind", "design", "--quiet",
           "--json", "/dev/stdout"]
    if design_chain:
        cmd += ["--design-chain", design_chain]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    try:
        return json.loads(p.stdout[p.stdout.index("{"):])
    except Exception:
        return {"error": (p.stderr or p.stdout)[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+")
    ap.add_argument("--design-chain", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", action="store_true", help="also score every design")
    args = ap.parse_args()

    recs = []
    for d in args.out_dirs:
        p = pathlib.Path(d)
        cifs = sorted((p / "intermediate_designs").glob("*.cif")) or sorted(p.rglob("*.cif"))
        if not cifs:
            recs.append({"out_dir": p.name, "error": "no cif"})
            continue
        rec = {"out_dir": p.name, "cifs": [c.name for c in cifs], "spread": spread(cifs)}
        if args.geometry:
            rec["geometry"] = [geometry(c, args.design_chain) for c in cifs]
            bad = [g for g in rec["geometry"] if g.get("fail")]
            rec["n_fail"] = len(bad)
        recs.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "geometry"}), flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
