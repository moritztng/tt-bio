"""One row per rung: what ran, on which card, at what clock, and what the structure scored.

Written from `out/` rather than by hand. `out/` is gitignored (CIFs and logs), so this file is
the part of the evidence that survives in the repo, and every field in it is read back from an
artifact the run produced.

Both axes of the coverage bar come out of the run's own `results.json`: `n_tokens` is
`feats["restype"].shape[0]` and `msa_depth` is `feats["msa"].shape[0]`, so each is the shape the
model was handed rather than a count of `>` records in the input a3m. Those two differ here by
a lot -- protenix-v2 does not apply `--max_msa_seqs` at its default, and the featurizer stacks
per-chain alignments block-diagonally -- which is exactly why neither is inferred.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).parent

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting the
#: wrong node's clock is how a fold at 1350 MHz gets reported as one at 800.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("ptx_*_dev*")):
        rung, dev = d.name[4:].rsplit("_dev", 1)
        node = UMD_TO_NODE[int(dev)]
        clk = [x.split() for x in (d / "aiclk.log").read_text().splitlines() if x.strip()]
        host = [x.split() for x in (d / "host.log").read_text().splitlines() if x.strip()]
        mhz = [int(x[node + 1]) for x in clk if x[node + 1] != "NA"]
        row = {"rung": rung, "umd_device": int(dev), "node": node,
               "aiclk_median_mhz": statistics.median(mhz), "aiclk_min_mhz": min(mhz),
               "host_load_median": round(statistics.median(float(x[2]) for x in host), 1)}
        for kv in ((d / "rung.txt").read_text().split() if (d / "rung.txt").exists() else []):
            if kv.startswith(("rc=", "wall_s=")):
                k, v = kv.split("=")
                row[k] = int(v)
        res = list(d.rglob("results.json"))
        if res:
            r = json.loads(res[0].read_text())[0]
            row |= {"status": r["status"], "runtime_s": r["runtime_s"],
                    "n_tokens": r["n_tokens"], "msa_depth": r["msa_depth"],
                    "n_residues": r["n_residues"], "n_atoms": r["n_atoms"],
                    "plddt": round(r["plddt"], 4), "ptm": round(r["ptm"], 4)}
            cif = sorted(d.rglob("*.cif"))
            if cif:
                row["cif_md5"] = hashlib.md5(cif[0].read_bytes()).hexdigest()
        else:
            row["status"] = "no results.json"
        if (d / "struct.json").exists():
            rep = json.loads((d / "struct.json").read_text())
            row["struct_verdict"] = rep.get("verdict")
            row["chains"] = [{"chain": c["chain"], "n_res": c["n_res"], "breaks": c["breaks"],
                              "step_median": c["step_median"], "in_band": c["in_band_frac"],
                              "rg_ratio": c["rg_ratio"]} for c in rep["checks"]["chains"]]
            row["clashes"] = rep["checks"].get("clashes")
        rows.append(row)
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r["rung"], "dev", r["umd_device"], r["aiclk_median_mhz"], "MHz",
              r.get("n_tokens"), "tok", r.get("msa_depth"), "rows",
              r.get("runtime_s"), "s", r.get("cif_md5", "")[:8], r["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
