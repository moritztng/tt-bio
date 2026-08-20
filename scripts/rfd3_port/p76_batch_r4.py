#!/usr/bin/env python3
"""p76 -- the batching confirmation at the page fixture, owed by the brief's Lever A.

`rfd3-close-the-page-gap` P3.15 closed batching NO-GO at R2 = 3844 atoms:

    | batch | s/design | vs b=1 |
    | 1 | 57.863 | -- |
    | 2 | 55.900 | 1.035x |
    | 4 | 55.666 | 1.039x |
    gate 1.05x, neither arm clears it

and its stated mechanism was "batching amortises launch overhead, it multiplies DRAM traffic".
Three things were never measured and this screen measures them:

  * R4 (6051 atoms) was only ever tried PRE-calibration (E2.1, b=2 measured 1.161x WORSE);
    `RFD3_TUNE_MATMUL` has been default-on above 2952 atoms since.
  * `_BATCH_ATOM_PAIR_BUDGET // (6051*6051)` is 2, so b=2 is reachable and b=4 is not without
    raising an OOM-gated budget. Verified against the current tree, not inherited.
  * `_BATCH_SPEED_CAP = 1` above 2952 atoms hard-forces b=1 here, so the screen overrides the
    CAP, never the budget.

Gate, written before the run: **b=2 must beat b=1 by >= 1.05x** to be worth a release-gated
default change. Below that, batching is closed for the second time with the R4 number the first
close was missing. If b=2 is WORSE, the E2.1 pothole survived calibration at R4 and that is a
defect note against `_BATCH_SPEED_CAP`, not a lever.

Protocol (`perf-page-matched-batch-protocol-recurrence`): if b=2 wins, the comparison is our b=2
s/design against the H200's b=8 amortised 12.974, which is what the 51.896 bar already encodes.
Do not re-denominate.

Batching is bit-exact by construction here (design.py:53), but siblings in one batch are different
designs, so a CIF digest across arms is not the check -- output validation is
(`design-model-output-validation-not-folding-invariants`).

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p2 -- env TT_VISIBLE_DEVICES=1 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p2 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p76_batch_r4.py \
          perf/p76/batch_r4.json 200 1,2,1,2
"""
import json
import math
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p76/batch_r4.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ARMS = [int(a) for a in (sys.argv[3] if len(sys.argv) > 3 else "1,2,1,2").split(",")]
NDESIGN = int(sys.argv[4]) if len(sys.argv) > 4 else 2
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
GATE = 1.05
CARD = int(os.environ.get("TT_VISIBLE_DEVICES", "1"))

WALLS = []
_sample = RFD3Sampler.sample


def _ttnn_version():
    """Provenance: this task is dispatched to qb2 and qb1, which carry different wheels."""
    import importlib.metadata as md
    try:
        return md.version("ttnn")
    except Exception:
        return "unknown"


def _timed(self, dm, n, *a, **k):
    t0 = time.perf_counter()
    out = _sample(self, dm, n, *a, **k)
    WALLS.append((time.perf_counter() - t0, n))
    return out


RFD3Sampler.sample = _timed


def validate(out_dir, expect_designs):
    """Every written design must parse, carry atoms and have finite coordinates."""
    cifs = sorted(pathlib.Path(out_dir).glob("*.cif"))
    report = {"n_cifs": len(cifs), "expected": expect_designs, "atoms": [], "ok": True}
    if len(cifs) != expect_designs:
        report["ok"] = False
    for c in cifs:
        n_atom, bad = 0, 0
        for line in c.read_text().splitlines():
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            parts = line.split()
            n_atom += 1
            # x/y/z are the 5th, 4th and 3rd fields from the end of a written ATOM row
            # (the two trailing fields are the entity and atom serials).
            try:
                xyz = [float(parts[i]) for i in (-5, -4, -3)]
            except (IndexError, ValueError):
                bad += 1
                continue
            if not all(math.isfinite(v) for v in xyz):
                bad += 1
        report["atoms"].append(n_atom)
        if n_atom == 0 or bad:
            report["ok"] = False
    return report


def run(specs, out_dir, batch, ndesign):
    os.system("rm -rf %s" % out_dir)
    WALLS.clear()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=ndesign,
                           batch_size=batch, verbose=False)
    total = sum(w for w, _ in WALLS)
    sizes = [n for _, n in WALLS]
    return total, sizes, validate(out_dir, ndesign)


def main():
    specs = json.loads(FIXTURE.read_text())

    # The capacity clamp, read off the current tree rather than inherited.
    L = 6051
    budget_allows = max(1, rfd3_design._BATCH_ATOM_PAIR_BUDGET // (L * L))
    ceiling = rfd3_design._BATCH_DESIGN_CEILING
    cap_before = rfd3_design._BATCH_SPEED_CAP
    print("[p76] L=%d  budget admits b<=%d  design ceiling %d  speed cap %d above %d atoms"
          % (L, budget_allows, ceiling, cap_before,
             rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS), flush=True)
    print("[p76] steps=%d card=%d arms=%s num_designs=%d  gate %.2fx"
          % (STEPS, CARD, ARMS, NDESIGN, GATE), flush=True)

    # Override the CAP, never the budget: the budget is the OOM bound.
    rfd3_design._BATCH_SPEED_CAP = max(ARMS)

    t, sizes, v = run(specs, "/tmp/rfd3_p76_warm", 1, 1)
    print("[p76] warmup %.3f s (sizes %s), discarded" % (t, sizes), flush=True)

    rows = []
    for i, b in enumerate(ARMS):
        t, sizes, v = run(specs, "/tmp/rfd3_p76_%d" % i, b, NDESIGN)
        per = t / NDESIGN
        rows.append({"arm": "b%d" % b, "batch": b, "rep": i, "total_s": round(t, 3),
                     "s_per_design": round(per, 3), "forward_sizes": sizes, "validation": v})
        print("[p76] rep%d b=%d  %8.3f s total  %8.3f s/design  forwards %s  valid %s"
              % (i, b, t, per, sizes, v["ok"]), flush=True)

    rfd3_design._BATCH_SPEED_CAP = cap_before

    def med(b):
        v = sorted(r["s_per_design"] for r in rows if r["batch"] == b)
        return statistics.median(v) if v else None

    b1, b2 = med(1), med(2)
    aa = [r["s_per_design"] for r in rows if r["batch"] == 1]
    aa_spread = abs(aa[1] - aa[0]) if len(aa) > 1 else None
    ratio = (b1 / b2) if (b1 and b2) else None
    verdict = None
    if ratio is not None:
        if ratio >= GATE:
            verdict = "GO"
        elif ratio < 1.0:
            verdict = "WORSE THAN b=1 -- the E2.1 pothole survived calibration at R4"
        else:
            verdict = "NO-GO, closed for the second time with an R4 number"

    if aa_spread is not None:
        print("\nA/A control (the two b=1 reps): %s -> spread %.3f s" % (aa, aa_spread))
    print("b=1 median %8.3f s/design" % b1)
    if b2:
        print("b=2 median %8.3f s/design   ratio %.4fx   gate %.2fx   %s"
              % (b2, ratio, GATE, verdict))
        print("vs the H200 b=8 amortised 12.974: %.3fx  (do NOT compare against their b=1)"
              % (b2 / 12.974))
    all_valid = all(r["validation"]["ok"] for r in rows)
    print("output validation: %s" % ("all designs parse with finite coords" if all_valid
                                     else "FAILED -- see the json"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "rows": rows, "num_timesteps": STEPS, "num_designs": NDESIGN, "seed": SEED,
        "arms": ARMS, "budget_allows": budget_allows, "design_ceiling": ceiling,
        "speed_cap_shipped": cap_before,
        "speed_cap_above_atoms": rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS,
        "b1_median_s_per_design": b1, "b2_median_s_per_design": b2,
        "aa_control_s": aa, "aa_spread_s": aa_spread, "ratio": ratio, "gate": GATE,
        "verdict": verdict, "all_valid": all_valid,
        "h200_b8_amortised_s": 12.974,
        "ratio_to_h200_b8": (b2 / 12.974) if b2 else None,
        "host": os.uname().nodename, "ttnn": _ttnn_version(), "card": CARD,
    }, indent=2))


if __name__ == "__main__":
    main()
