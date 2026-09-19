"""One row per rung: what ran, on which card, at what clock, and what the structure scored.

Written from `out/` rather than by hand. `out/` is gitignored (CIFs and logs), so this file
is the part of the evidence that survives in the repo, and every field in it is read back
from an artifact the run produced.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CHECK = HERE.parent / "wh-correctness" / "check_structure.py"

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting
#: the wrong node's clock is how a fold at 1350 MHz gets reported as one at 800.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}


def score(cif: Path, yaml: Path, out: Path) -> dict:
    subprocess.run([sys.executable, str(CHECK), str(cif), "--input", str(yaml),
                    "--json", str(out), "--quiet"], check=False)
    return json.loads(out.read_text()) if out.exists() else {}


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("e2_*_dev*")):
        if d.name.endswith("_census"):
            continue
        rung, dev = d.name[3:].rsplit("_dev", 1)
        res = list(d.rglob("results.json"))
        if not res:
            continue
        r = json.loads(res[0].read_text())[0]
        clk = [x.split() for x in (d / "aiclk.log").read_text().splitlines() if x.strip()]
        host = [x.split() for x in (d / "host.log").read_text().splitlines() if x.strip()]
        node = UMD_TO_NODE[int(dev)]
        row = {
            "rung": rung, "tokens": int(rung), "umd_device": int(dev), "node": node,
            "aiclk_median_mhz": statistics.median(int(x[node + 1]) for x in clk),
            "aiclk_min_mhz": min(int(x[node + 1]) for x in clk),
            "status": r["status"], "runtime_s": r.get("runtime_s"),
            "wall_s": int((d / "rung.txt").read_text().split("wall_s=")[1]),
            "msa": r.get("msa"), "plddt": r.get("plddt"), "ptm": r.get("ptm"),
            "n_residues": r.get("n_residues"), "n_chains": r.get("n_chains"),
            "host_load_median": round(statistics.median(float(x[2]) for x in host), 1),
        }
        cifs = list(d.rglob("*.cif"))
        if cifs:
            row["cif_md5"] = hashlib.md5(cifs[0].read_bytes()).hexdigest()
            rep = score(cifs[0], HERE / "inputs" / f"e2_{rung}.yaml", d / "struct.json")
            if rep:
                row["verdict"] = rep.get("verdict")
                row["chains"] = [{"chain": c["chain"], "n_res": c["n_res"],
                                  "breaks": c["breaks"], "step_median": c["step_median"],
                                  "in_band": c["in_band_frac"], "rg_ratio": c["rg_ratio"]}
                                 for c in rep["checks"]["chains"]]
                row["clashes"] = rep["checks"].get("clashes")
        rows.append(row)
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r["rung"], "dev", r["umd_device"], r["aiclk_median_mhz"], "MHz",
              r["runtime_s"], "s", r.get("cif_md5", "-")[:8], r.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
