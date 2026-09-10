#!/usr/bin/env python3
"""Score a rung whose fold finished but whose harness process did not, from the artifact.

esmfold2 at 1300 folded (`Done: 1 ok, 0 failed`, 424.9 s) as an orphan: its run_rung.py parent
was killed by hand while stopping a tangle of chains, so the CIF is on disk and no row was ever
written. The harness judges the artifact and not the exit status, and the artifact is here, so
this scores it with run_rung's own functions rather than re-spending seven minutes of card time.
The row is tagged `recovered` in `harness` so nobody reads it as a clean harness run.

    ./recover_row.py --model esmfold2 --size 1300 --out_tag p300c
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WT = HERE.parents[1]
import ladder_paths          # noqa: E402
import run_rung              # noqa: E402
PY_BIN = "/home/ttuser/tt-bio-dev/env/bin/python3"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--card", type=int, default=1)
    a = ap.parse_args()

    label = f"{a.model}_{a.size}" + (f"_{a.tag}" if a.tag else "")
    out = ladder_paths.runs_dir(a.out_tag) / label
    cifs = sorted(out.rglob("structures/*.cif")) + sorted(out.rglob("structures/*.pdb"))
    if not cifs:
        sys.exit(f"no artifact under {out}")
    nres = run_rung.cif_residues(cifs[0])
    metrics = {}
    rj = sorted(out.rglob("results.json"))
    if rj:
        metrics = json.loads(rj[0].read_text())
        if isinstance(metrics, list):
            metrics = metrics[0] if metrics else {}
    sig = subprocess.run([PY_BIN, str(WT / "perf/ceilings/struct_signal.py"), str(out)],
                         capture_output=True, text=True, cwd=str(WT),
                         env={"PYTHONPATH": str(WT), "PATH": "/usr/bin:/bin",
                              "TT_VISIBLE_DEVICES": ""})
    signal = json.loads(sig.stdout.strip().splitlines()[-1])
    log = out / "fold.log"
    text = log.read_text(errors="replace") if log.is_file() else ""
    row = {"model": a.model, "size": a.size, "tag": a.tag, "task": "predict",
           "card": a.card, **ladder_paths.where(), "code": run_rung.code_state(),
           "harness": "recovered-artifact: the fold completed, its run_rung.py parent was "
                      "killed by hand at 2026-09-10T01:20:42Z, so wall clock and host RSS are "
                      "not recorded for this rung",
           "verdict": "PASS" if nres == a.size else "FAIL",
           "wall_s": None, "engine_runtime_s": metrics.get("runtime_s"),
           "cif": str(cifs[0]), "cif_residues": nres,
           "n_tokens": next((metrics[k] for k in ("n_tokens", "num_tokens", "tokens")
                             if isinstance(metrics.get(k), int)), None),
           "plddt": signal.get("plddt"), "struct": signal,
           "exit": None, "killed": False, "single_sequence": True, "sampling_steps": 20,
           "recycling_steps": None, "peak_host_rss_gib": None, "floor_memavail_gib": None,
           "oom": run_rung._oom_of(text), "fatal_oom": False, "tail": "",
           "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    run_rung._write(row, a.out_tag)
    return 0 if row["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
