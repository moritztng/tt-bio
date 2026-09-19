"""One row per rung: what ran, on which card, at what clock, and what the structure scored.

Written from `out/` rather than by hand. `out/` is gitignored (CIFs and logs), so this file is
the part of the evidence that survives in the repo, and every field in it is read back from an
artifact the run produced. Scores the structure too, with the shared instrument
(`perf/wh-correctness/check_structure.py`) rather than a second opinion: a completed fold with a
torn backbone is worse than a refusal, because it looks like success.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CHECK = HERE.parent / "wh-correctness" / "check_structure.py"
PY = "/home/ttuser/tt-bio/env/bin/python3"

#: tt-smi UMD id -> /dev/tenstorrent node on qb1. They are not the same number, and quoting the
#: wrong node's clock is how a fold at 1350 MHz gets reported as one at 800.
UMD_TO_NODE = {0: 1, 1: 2, 2: 3, 3: 0}

#: The two things a freeze would have said and a completed fold does not. Grepped rather than
#: assumed absent: the 2026-09-10 row's whole claim was "no allocator refusal anywhere".
REFUSAL = re.compile(r"Out of Memory|Statically allocated circular buffers|TT_THROW", re.I)


def clock(d: Path, dev: int) -> dict:
    rows = [line.split() for line in (d / "aiclk.log").read_text().splitlines() if line.strip()]
    col = [int(r[UMD_TO_NODE[dev] + 1]) for r in rows if r[UMD_TO_NODE[dev] + 1].isdigit()]
    return {"aiclk_median_mhz": statistics.median(col) if col else None,
            "aiclk_min_mhz": min(col) if col else None,
            "aiclk_max_mhz": max(col) if col else None, "aiclk_samples": len(col)}


def structure(d: Path, cif: Path, yaml: Path) -> dict:
    out = d / "struct.json"
    subprocess.run([PY, str(CHECK), str(cif), "--input", str(yaml), "--json", str(out),
                    "--quiet"], capture_output=True)
    if not out.exists():
        return {}
    rep = json.loads(out.read_text())
    return {"verdict": rep.get("verdict"),
            "chains": [{"chain": c["chain"], "n_res": c["n_res"], "breaks": c["breaks"],
                        "step_median": c["step_median"], "in_band": c["in_band_frac"],
                        "rg_ratio": c["rg_ratio"]} for c in rep["checks"]["chains"]],
            "clashes": rep["checks"].get("clashes")}


def main() -> int:
    rows = []
    for d in sorted((HERE / "out").glob("opendde*_dev*")):
        ckpt, rest = d.name.split("_", 1)
        rung, mode, dev = rest.rsplit("_", 2)
        dev = int(dev[3:])
        log = (d / "fold.log").read_text(errors="replace") if (d / "fold.log").exists() else ""
        res = list(d.rglob("results.json"))
        row = {"ckpt": ckpt, "rung": rung, "mode": mode, "umd_device": dev,
               "node": UMD_TO_NODE[dev], **clock(d, dev),
               "refusals_in_log": sorted(set(REFUSAL.findall(log))),
               "rung_txt": (d / "rung.txt").read_text().strip() if (d / "rung.txt").exists() else ""}
        if res:
            r = json.loads(res[0].read_text())[0]
            cifs = list(d.rglob("*.cif"))
            row.update({"runtime_s": r.get("runtime_s"), "status": r.get("status"),
                        "plddt": round(r["plddt"], 4) if r.get("plddt") is not None else None,
                        "ptm": round(r["ptm"], 4) if r.get("ptm") is not None else None,
                        "n_tokens": r.get("n_tokens"),
                        "cif_md5": hashlib.md5(cifs[0].read_bytes()).hexdigest() if cifs else None})
            if cifs:
                row.update(structure(d, cifs[0], HERE / "inputs" / f"odd_{rung}.yaml"))
        else:
            row["status"] = "no_results_json"
            row["last_log"] = "\n".join(log.splitlines()[-4:])
        rows.append(row)
    (HERE / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(r["ckpt"], r["rung"], r["mode"], "dev", r["umd_device"],
              r["aiclk_median_mhz"], "MHz", r.get("runtime_s"), "s",
              r.get("status"), r.get("verdict"), (r.get("cif_md5") or "")[:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
