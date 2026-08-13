"""p43 — the shape sweep that has to pass before the two kernels can ship default-on.

`RFD3_SPARSE_BIAS` (L6a) and `RFD3_FUSED_SCORES` (L6b/L6d) are both `torch.equal` at the production
shape and byte-identical over a 200-step design there. What a sweep has to add is not more speed
evidence: it is that **the kernels decline cleanly on shapes they cannot serve**. Both gate
themselves -- `eligible_shape` on (batch, heads, keys, dtype) and `dense_bias is None` for the fused
path -- and a bad gate looks exactly like a good one in the output, because the fallback produces the
same CIF. So each shape is run twice in separate processes, off and on, and two things are checked:

1. the 20-step CIF sha256 is **identical** between the arms, at every shape;
2. `rfd3_bias.stats_line()` says whether the kernel served or declined, so a shape that silently
   stopped being served is visible instead of passing as "identical".

The shapes come from the contig, which is what actually varies at inference: the middle number is the
designed span, so `A1-10,<n>,A31-40` gives n+20 tokens and ~13.4 atoms each. That sweeps It, Jt and
the atom count, and the small end sweeps K itself, since k=min(128, atoms) stops being a multiple of
32 there -- which is the decline path.

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half PYTHONPATH=$PWD \\
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/rfd3_port/p43_shape_sweep.py \\
        --out perf/p43/shape_sweep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
WT = Path(__file__).resolve().parents[2]
PDB = "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"


def gate_unit() -> dict:
    """The gate on shapes no fixture produces, checked directly rather than hoped for.

    `k` is a model constant -- 128 for the atom blocks, 32 for the token DiT -- and both are already
    below the smallest contig this PDB admits (20 tokens, ~280 atoms), so no design length can make
    `n_keys` a non-multiple of 32. The reachable decline in production is `batch != 1`, which the
    sweep exercises for real below; the rest is checked here, in-process and without a device.
    """
    import ttnn
    from tt_bio import rfd3_bias
    rfd3_bias.set_enabled(True)
    rfd3_bias.REJECTS.clear()
    cases = {
        "batch2": (2, 4, 3359, 128, ttnn.bfloat16),
        "fp32": (1, 4, 3359, 128, ttnn.float32),
        "keys_not_tile_multiple": (1, 4, 3359, 40, ttnn.bfloat16),
        "keys_below_tile": (1, 4, 3359, 16, ttnn.bfloat16),
        "production": (1, 4, 3359, 128, ttnn.bfloat16),
    }
    got = {k: bool(rfd3_bias.eligible_shape(*v)) for k, v in cases.items()}
    return {"eligible": got, "rejects": {str(k): v for k, v in rfd3_bias.REJECTS.items()},
            "as_expected": got == {"batch2": False, "fp32": False,
                                   "keys_not_tile_multiple": False, "keys_below_tile": False,
                                   "production": True}}


def run_one(spec: Path, out: Path, steps: int, seed: int, fused: str, extra=()) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(WT), RFD3_SPARSE_BIAS="1", RFD3_FUSED_SCORES=fused,
               RFD3_BIAS_STATS="1")
    env.pop("RFD3_TUNE_MATMUL", None)
    cmd = [PY, "-m", "tt_bio.main", "design", str(spec), "--model", "rfd3", "--from_pdb",
           "--out_dir", str(out), "--num_timesteps", str(steps), "--seed", str(seed)]
    cmd += list(extra) if extra else ["--num_designs", "1"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, cwd=str(WT), capture_output=True, text=True)
    wall = time.perf_counter() - t0
    tail = (p.stdout or "")[-4000:] + (p.stderr or "")[-4000:]
    stats = ""
    m = re.search(r"\[rfd3_bias\][^\n]*", tail)
    if m:
        stats = m.group(0)
    atoms = ""
    m = re.search(r"\((\d+) atoms", tail)
    if m:
        atoms = m.group(1)
    shas = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()
            for q in sorted(out.rglob("*.cif"))}
    return {"rc": p.returncode, "wall_s": wall, "sha": shas, "stats": stats, "atoms": atoms,
            "err": "" if not p.returncode else tail[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", default="40,120,230,400",
                    help="designed span per shape; total tokens is span + 20")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("perf/p43/shape_sweep.json"))
    a = ap.parse_args()

    rec: dict = {"steps": a.steps, "seed": a.seed, "shapes": [], "gate_unit": gate_unit()}
    print(f"gate check: {rec['gate_unit']['eligible']}\n  as_expected="
          f"{rec['gate_unit']['as_expected']} rejects={rec['gate_unit']['rejects']}", flush=True)
    for span in [int(s) for s in a.spans.split(",")]:
        name = f"iai{span}"
        spec = Path(f"/tmp/p43_spec_{span}.json")
        spec.write_text(json.dumps({name: {"input": PDB, "contig": f"A1-10,{span},A31-40"}}))
        row = {"span": span, "tokens": span + 20}
        for arm, fused in (("off", "0"), ("on", "1")):
            row[arm] = run_one(spec, WT / f"perf/p43/{name}_{arm}", a.steps, a.seed, fused)
        row["atoms"] = row["on"]["atoms"] or row["off"]["atoms"]
        row["identical"] = (row["off"]["rc"] == 0 and row["on"]["rc"] == 0
                            and row["off"]["sha"] == row["on"]["sha"] and bool(row["on"]["sha"]))
        rec["shapes"].append(row)
        served = "FUSED" if re.search(r"fused served=[1-9]", row["on"]["stats"]) else "declined"
        print(f"span={span:>4} tokens={row['tokens']:>4} atoms={row['atoms']:>5}  "
              f"rc={row['off']['rc']}/{row['on']['rc']}  identical={row['identical']}  "
              f"on-arm={served}\n    off {row['off']['stats']}\n    on  {row['on']['stats']}",
              flush=True)
        if row["off"]["rc"] or row["on"]["rc"]:
            print("    ERR " + (row["off"]["err"] or row["on"]["err"])[-600:], flush=True)

    # The one decline a real run can hit: multiplicity batching makes batch != 1, both kernels
    # refuse, and the dense route has to produce the same design. "Declines cleanly" is not a
    # property the CIF can show on its own -- the stats line is what separates it from "served".
    spec = Path("/tmp/p43_spec_batch.json")
    spec.write_text(json.dumps({"iaib": {"input": PDB, "contig": "A1-10,230,A31-40"}}))
    row = {"span": 230, "tokens": 250, "batch": 2}
    for arm, fused in (("off", "0"), ("on", "1")):
        row[arm] = run_one(spec, WT / f"perf/p43/batch2_{arm}", a.steps, a.seed, fused,
                           extra=["--num_designs", "2", "--batch_size", "2"])
    row["atoms"] = row["on"]["atoms"] or row["off"]["atoms"]
    row["identical"] = (row["off"]["rc"] == 0 and row["on"]["rc"] == 0
                        and row["off"]["sha"] == row["on"]["sha"] and bool(row["on"]["sha"]))
    row["declined_cleanly"] = "declined=" in row["on"]["stats"] and re.search(
        r"fused served=0", row["on"]["stats"]) is not None
    rec["shapes"].append(row)
    print(f"batch=2   atoms={row['atoms']:>5}  rc={row['off']['rc']}/{row['on']['rc']}  "
          f"identical={row['identical']}  declined_cleanly={row['declined_cleanly']}\n"
          f"    off {row['off']['stats']}\n    on  {row['on']['stats']}", flush=True)
    if row["off"]["rc"] or row["on"]["rc"]:
        print("    ERR " + (row["off"]["err"] or row["on"]["err"])[-600:], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2))
    bad = [r["span"] for r in rec["shapes"] if not r["identical"]]
    print(f"\nwrote {a.out}")
    print("NOT identical / failed: " + (", ".join(map(str, bad)) if bad else "none"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
