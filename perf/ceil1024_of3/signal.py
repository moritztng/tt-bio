#!/usr/bin/env python3
"""One line of structural signal per rung, so a PASS is never just "it returned".

    signal.py RUNGDIR [RUNGDIR ...]

RUNGDIR is a rung's out_dir (the one `rung.sh` passes to `predict --out_dir`). Reads the
mmCIF and the results JSON under it and runs perf/wh-correctness/check_structure.py, whose
thresholds are calibrated against real crystal structures rather than picked to look strict.
Prints the three things a capacity ladder has to report -- clash fraction, backbone
continuity, confidence -- plus the checker's own verdict, because a fold that completes and
is torn is a worse outcome than one that refuses.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "wh-correctness" / "check_structure.py"


def one(rung_dir: str) -> int:
    name = Path(rung_dir).name
    cif = glob.glob(f"{rung_dir}/*/structures/*.cif")
    res = glob.glob(f"{rung_dir}/*/results.json")
    if not cif:
        print(f"{name:<12} NO-STRUCTURE")
        return 1
    out = Path(f"/tmp/signal_{name}.json")
    cmd = [sys.executable, str(CHECKER), cif[0], "--json", str(out), "--quiet"]
    if res:
        cmd += ["--conf", res[0]]
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    if not out.exists():
        print(f"{name:<12} CHECKER-FAILED rc={rc}")
        return 1
    d = json.loads(out.read_text())
    c = d["checks"]
    cl, cf = c["clashes"], c["confidence"]
    verdict = "PASS" if not d["fail"] else "FAIL"
    # One line per chain, because a dropped or truncated chain is one of the failures the
    # checker exists to catch and an aggregate would hide it.
    for ch in c["chains"]:
        print(f"{name:<12} {verdict} n_res={ch['n_res']} chain={ch['chain']} "
              f"clashes={cl['n']}/{cl['heavy_atoms']} breaks={ch['breaks']} "
              f"ca_ca_med={ch['step_median']} in_band={ch['in_band_frac']} "
              f"rg_ratio={ch['rg_ratio']} "
              f"plddt={cf.get('mean')}+-{cf.get('std')}")
    if d["fail"]:
        print(f"{name:<12}   fail={d['fail']}")
    if d["warn"]:
        print(f"{name:<12}   warn={d['warn']}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(max((one(a) for a in sys.argv[1:]), default=2))
