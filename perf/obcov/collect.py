"""One row per rung: what ran, on which card, at what clock, and what the structure scored.

Written from `out/` rather than by hand. `out/` is gitignored (CIFs, logs, weights scratch), so
this file is the part of the evidence that survives in the repo, and every field in it is read
back from an artifact the run produced.

Both axes of the coverage bar come out of the run's own `results.json`: `n_tokens` is
`feats["restype"].shape[0]` and `msa_depth` is `feats["msa"].shape[0]`, so each is the shape the
model was handed rather than a count of `>` records in the input a3m. For openbind the two
differ from the input by a lot -- the OF3 family folds the resolved alignment whole, and openbind
dedups its main MSA on its own checkpoint spec -- which is why neither is inferred.

The structure is scored with the shared instrument (`perf/wh-correctness/check_structure.py`),
because a completed fold with a torn backbone is worse than a refusal: it looks like success.
`refusals_in_log` is grepped rather than assumed absent, and `progress` reports whether RSS and
the log were still moving at the end, because openbind's L1 refusals inside the diffusion
transformer are caught and retried, so a stall presents as a budget timeout and not as a throw.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
CHECK = HERE.parent / "wh-correctness" / "check_structure.py"
PY = "/home/ttuser/tt-bio/env/bin/python3"

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting the
#: wrong node's clock is how a fold at 1350 MHz gets reported as one at 800.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}

REFUSAL = re.compile(r"Out of Memory|Statically allocated circular buffers|TT_THROW", re.I)


def clock(d: Path, dev: int) -> dict:
    rows = [line.split() for line in (d / "aiclk.log").read_text().splitlines() if line.strip()]
    col = [int(r[UMD_TO_NODE[dev] + 1]) for r in rows if r[UMD_TO_NODE[dev] + 1].isdigit()]
    return {"aiclk_median_mhz": statistics.median(col) if col else None,
            "aiclk_min_mhz": min(col) if col else None,
            "aiclk_max_mhz": max(col) if col else None, "aiclk_samples": len(col)}


def progress(d: Path) -> dict:
    """Was anything still moving at the end? Distinguishes a stall from a slow fold."""
    rows = [line.split() for line in (d / "host.log").read_text().splitlines() if line.strip()]
    rss = [int(r[3]) for r in rows if len(r) > 3 and r[3].isdigit()]
    tail = rss[-30:]
    return {"host_load_median": round(statistics.median(float(r[2]) for r in rows), 1) if rows else None,
            "rss_peak_gib": round(max(rss) / 1048576, 2) if rss else None,
            "rss_flat_last_5min": bool(tail) and len(set(tail)) == 1}


def structure(d: Path, cif: Path, yaml: Path) -> dict:
    out = d / "struct.json"
    subprocess.run([PY, str(CHECK), str(cif), "--input", str(yaml), "--json", str(out),
                    "--quiet"], capture_output=True)
    if not out.exists():
        return {}
    rep = json.loads(out.read_text())
    return {"struct_verdict": rep.get("verdict"),
            "chains": [{"chain": c["chain"], "n_res": c["n_res"], "breaks": c["breaks"],
                        "step_median": c["step_median"], "in_band": c["in_band_frac"],
                        "rg_ratio": c["rg_ratio"]} for c in rep["checks"]["chains"]],
            "clashes": rep["checks"].get("clashes")}


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("ob_*_dev*")):
        rung, dev = d.name[3:].rsplit("_dev", 1)
        dev = int(dev)
        log = (d / "fold.log").read_text(errors="replace") if (d / "fold.log").exists() else ""
        row = {"rung": rung, "umd_device": dev, "node": UMD_TO_NODE[dev],
               **clock(d, dev), **progress(d),
               "refusals_in_log": sorted(set(REFUSAL.findall(log)))}
        for kv in ((d / "rung.txt").read_text().split() if (d / "rung.txt").exists() else []):
            if kv.startswith(("rc=", "wall_s=")):
                k, v = kv.split("=")
                row[k] = int(v)
        res = list(d.rglob("results.json"))
        if res:
            r = json.loads(res[0].read_text())[0]
            row |= {"status": r["status"], "runtime_s": r.get("runtime_s"),
                    "n_tokens": r.get("n_tokens"), "msa_depth": r.get("msa_depth"),
                    "n_residues": r.get("n_residues"), "n_atoms": r.get("n_atoms"),
                    "plddt": round(r["plddt"], 4) if r.get("plddt") is not None else None,
                    "ptm": round(r["ptm"], 4) if r.get("ptm") is not None else None}
            cifs = sorted(d.rglob("*.cif"))
            if cifs:
                row["cif_md5"] = hashlib.md5(cifs[0].read_bytes()).hexdigest()
                row |= structure(d, cifs[0], HERE / "inputs" / f"ob_{rung}.yaml")
        else:
            row["status"] = "no_results_json"
            row["last_log"] = "\n".join(log.splitlines()[-6:])
        rows.append(row)
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r["rung"], "dev", r["umd_device"], r["aiclk_median_mhz"], "MHz",
              r.get("n_tokens"), "tok", r.get("msa_depth"), "rows",
              r.get("runtime_s"), "s", r.get("struct_verdict"),
              (r.get("cif_md5") or "")[:8], r["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
