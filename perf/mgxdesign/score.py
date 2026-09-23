#!/usr/bin/env python3
"""Score every design a walk produced, with the instrument the release gate itself uses.

A design's accuracy is not an RMSD against a reference structure -- there is no reference for a
binder nobody has made. What the repo already checks, and what `tt_bio/size_limits.py`'s rfd3 row
quotes for its own ceiling ("0 backbone breaks and worst Ca-Ca 3.92-3.95 A ... clash_frac
0.016-0.051"), is that the delivered coordinates are chemically sane. So this runs
`perf/wh-correctness/check_structure.py --kind design` over each rung's CIF and puts the numbers
beside the rung, one row per design.

NOT `perf/ceilrfd3/break_score.py`, which is the obvious candidate and has been dead since
2026-09-11: commit 2b566d4aa deleted `perf/ceilings/` as an concluded campaign's directory, and
break_score.py's `from ceilings.struct_signal import ...` went with it. check_structure.py is the
live one -- `scripts/release_gate.py` imports `chain_geometry` and `clashes` from it for the
GEOMETRY leg, so a rung scored here is scored by the same code that gates a release.

    python3 perf/mgxdesign/score.py --work perf/mgxdesign/work-rfd3 --out perf/mgxdesign/rfd3.score.jsonl
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = ROOT / "perf" / "wh-correctness" / "check_structure.py"


def score_one(cif: pathlib.Path, design_chain: str | None) -> dict:
    cmd = [sys.executable, str(CHECKER), str(cif), "--kind", "design", "--quiet",
           "--json", "/dev/stdout"]
    if design_chain:
        cmd += ["--design-chain", design_chain]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    # The checker writes its report to --json and its verdict to the exit code. A non-zero exit
    # is a FINDING about the structure, not an error here, so the report is kept either way.
    try:
        start = p.stdout.index("{")
        rep = json.loads(p.stdout[start:])
    except Exception:
        return {"cif": cif.name, "error": (p.stdout + p.stderr)[-400:], "rc": p.returncode}
    ch = {c["chain"]: c for c in rep.get("checks", {}).get("chains", [])}
    cl = rep.get("checks", {}).get("clashes", {})
    # Field names are the checker's own (`n_res`, `step_median`, `in_band_frac`, `breaks`,
    # `rg_ratio`), read out of chain_geometry rather than guessed: a scorer that asks for a key
    # the report does not carry records null and reads as "clean".
    return {
        "cif": cif.name,
        "rc": p.returncode,
        "chains": {k: v.get("n_res") for k, v in ch.items()},
        "step_median": {k: v.get("step_median") for k, v in ch.items()},
        "in_band_frac": min((v.get("in_band_frac") for v in ch.values()
                             if v.get("in_band_frac") is not None), default=None),
        "rg_ratio": {k: v.get("rg_ratio") for k, v in ch.items()},
        "breaks": sum(int(v.get("breaks") or 0) for v in ch.values()),
        "clashes": cl.get("n"),
        "heavy_atoms": cl.get("heavy_atoms"),
        "clash_frac": (round(cl["n"] / cl["heavy_atoms"], 5)
                       if cl.get("n") is not None and cl.get("heavy_atoms") else None),
        "fail": rep.get("fail", []),
        "warn": rep.get("warn", [])[:3],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=pathlib.Path,
                    help="a walk's --work dir; every out_<model>_<size>/ under it is scored")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--design-chain", default=None,
                    help="the chain the spec asked to be designed, when the model names one")
    a = ap.parse_args()

    rows = []
    for out_dir in sorted(a.work.glob("out_*")):
        try:
            model, size = out_dir.name.rsplit("_", 1)
            model = model[len("out_"):]
            size = int(size)
        except ValueError:
            continue
        cifs = sorted(out_dir.rglob("*.cif"))
        if not cifs:
            rows.append({"model": model, "size": size, "scored": 0,
                         "note": "the rung wrote no CIF, so there is nothing to score"})
            continue
        for cif in cifs:
            rows.append({"model": model, "size": size, **score_one(cif, a.design_chain)})
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
