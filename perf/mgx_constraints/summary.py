#!/usr/bin/env python3
"""Collapse score.py's per-sample rows to one line per (model, input): n/5 that met the bar
and the min-max of the distance the bar is on.

    summary.py [out_dir ...]        # defaults to out/ and ref/
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = {"n_c": "ring_closed", "sg_ligand": "bonded", "sg_sg": None, "p_og1": None}


def main():
    dirs = sys.argv[1:] or ["out", "ref"]
    rows = []
    for d in dirs:
        out = subprocess.run([sys.executable, str(HERE / "score.py"), str(HERE / d)],
                             capture_output=True, text=True).stdout
        rows += [json.loads(x) for x in out.splitlines() if x.startswith("{")]
    seen = {}
    for r in rows:
        k = (r["model"], r["tag"], r["input"])
        seen.setdefault(k, []).append(r)
    for k in sorted(seen):
        rs = seen[k]
        bits = []
        for metric, gate in KEY.items():
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            if not vals:
                continue
            bit = f"{metric} {min(vals):.2f}-{max(vals):.2f}"
            if gate:
                bit += f" {sum(1 for r in rs if r.get(gate))}/{len(rs)}"
            bits.append(bit)
        rm = [r["ca_rmsd"] for r in rs if r.get("ca_rmsd") is not None]
        if rm:
            bits.append(f"ca_rmsd {min(rm):.2f}-{max(rm):.2f}")
        print(f"{k[0]:<15} {k[1]:<9} {k[2]:<22} " + "  ".join(bits))


if __name__ == "__main__":
    main()
