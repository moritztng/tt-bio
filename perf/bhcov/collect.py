"""One row per rung: what ran, on which card, at what clock, and what the structure scored.

Written from `out/` rather than by hand. `out/` is gitignored (CIFs and logs), so this file is
the part of the evidence that survives in the repo, and every field in it is read back from an
artifact the run produced.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "wh-correctness"))
HERE = Path(__file__).parent

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting the
#: wrong node's clock is how a fold at 1350 MHz gets reported as one at 800.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("b2_*_dev*")):
        rung, dev = d.name[3:].rsplit("_dev", 1)
        res = list(d.rglob("results.json"))
        if not res:
            continue
        r = json.loads(res[0].read_text())[0]
        clk = [line.split() for line in (d / "aiclk.log").read_text().splitlines() if line.strip()]
        node = UMD_TO_NODE[int(dev)]
        host = [line.split() for line in (d / "host.log").read_text().splitlines() if line.strip()]
        row = {
            "rung": rung, "umd_device": int(dev), "node": node,
            "aiclk_median_mhz": statistics.median(int(x[node + 1]) for x in clk),
            "runtime_s": r["runtime_s"], "status": r["status"],
            "plddt": round(r["plddt"], 4), "ptm": round(r["ptm"], 4),
            "host_load_median": round(statistics.median(float(x[2]) for x in host), 1),
            "cif_md5": __import__("hashlib").md5(
                list(d.rglob("*.cif"))[0].read_bytes()).hexdigest(),
        }
        s = d / "struct.json"
        if s.exists():
            rep = json.loads(s.read_text())
            row["chains"] = [{"chain": c["chain"], "n_res": c["n_res"], "breaks": c["breaks"],
                              "step_median": c["step_median"], "in_band": c["in_band_frac"],
                              "rg_ratio": c["rg_ratio"]} for c in rep["checks"]["chains"]]
            row["clashes"] = rep["checks"].get("clashes")
        rows.append(row)
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r["rung"], "dev", r["umd_device"], r["aiclk_median_mhz"], "MHz",
              r["runtime_s"], "s", r["cif_md5"][:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
