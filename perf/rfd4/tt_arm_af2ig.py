"""The oracle seam's TT arm, on a card, with real AlphaFold2 weights.

`perf/rfd4/reference_score.py` proved the host arm: one `AF2IGOracle(backend="reference")` call
returns all twelve of upstream's reward keys. This is the same adapter with `backend="tt"`, which
differs by one constructor, and it answers the three things the reference arm cannot:

  * does the device model fold through `af2ig.fold` at all, and return the same twelve keys;
  * is one `setup()` enough for a whole batch, i.e. can a beam step score its candidates without
    reopening the chip per candidate (reopening is what deadlocks UMD, `tt_bio/worker.py:861`);
  * is the same candidate scored twice the same, which a reward has to be or a beam's ordering
    is noise.

    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:rfd4-beam \
        python3 perf/rfd4/tt_arm_af2ig.py --rungs 128,256,512 \
            --params /home/ttuser/.boltz/af2/params/params_model_1_ptm.npz

The per-fold seconds here are a real device timing and carry the clock sampled during the folds,
but they are a batch-of-one-each walk, not the matched oracle protocol: `perf/rfd4/oracle_cost.py`
is what measures that.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# A lone p300 chip is a CUSTOM topology: without the mesh descriptor every ad-hoc device script
# in this tree dies in TT_FATAL. tt-bio's CLI sets it in the parent; this is not the CLI.
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402

ensure_p300_mesh_descriptor()

from perf import clocksample  # noqa: E402
from tt_bio.rfd4.oracle import OracleRequest  # noqa: E402
from tt_bio.rfd4.oracle_af2ig import AF2IGOracle  # noqa: E402


def request(rung: int) -> OracleRequest:
    pdb = ROOT / "perf" / "rfd4" / "inputs" / f"af2ig_{rung}" / "complex.pdb"
    return OracleRequest(structure=pdb.read_text(), example_id=f"af2ig_{rung}",
                         chain_of_interest="B")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="128,256,512")
    ap.add_argument("--params", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--name", default="oracle_ppi")
    ap.add_argument("--dev", type=int, default=0,
                    help="the leased card's index in the clock sampler's list; tt-smi honours "
                         "TT_VISIBLE_DEVICES and reports the granted chip as 0")
    ap.add_argument("--out", default=str(ROOT / "perf" / "rfd4" / "results" / "tt_arm_af2ig.json"))
    args = ap.parse_args()

    rungs = [int(x) for x in args.rungs.split(",") if x]
    oracle = AF2IGOracle(params=args.params, name=args.name, backend="tt",
                         recycles=args.recycles)

    with clocksample.during(period=2.0) as clk:
        t0 = time.monotonic()
        oracle.setup()
        load_s = time.monotonic() - t0

        # One setup, every candidate: this is the beam-step question.
        jobs = [request(r) for r in rungs]
        per_fold = []
        scored = []
        for job in jobs:
            t = time.monotonic()
            scored.append(oracle.fold_batch([job])[0])
            per_fold.append(round(time.monotonic() - t, 2))

        # The determinism check: the first rung again, same process, same open device.
        t = time.monotonic()
        repeat = oracle.fold_batch([request(rungs[0])])[0]
        repeat_s = round(time.monotonic() - t, 2)
        oracle.teardown()

    first = scored[0]
    drift = {k: [first.get(k), repeat.get(k)] for k in sorted(set(first) | set(repeat))
             if first.get(k) != repeat.get(k)}

    row = {
        "backend": "tt", "host": "qb2", "board_type": "p300c", "recycles": args.recycles,
        "params": args.params, "rungs": rungs,
        "load_s": round(load_s, 1),
        "fold_s_by_rung": dict(zip(rungs, per_fold)),
        "repeat_fold_s": repeat_s,
        "keys_by_rung": {r: sorted(m) for r, m in zip(rungs, scored)},
        "n_keys_by_rung": {r: len(m) for r, m in zip(rungs, scored)},
        "metrics_by_rung": dict(zip(rungs, scored)),
        "determinism": {"rung": rungs[0], "bit_identical": not drift, "drift": drift},
        "clock": clk.summary(),
        "clock_line": clk.line(args.dev),
        "protocol_note": ("one setup for the whole batch; recycles+1 trunk passes per fold. "
                          "NOT the matched oracle protocol -- see perf/rfd4/oracle_cost.py"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "metrics_by_rung"}, indent=2))

    ok = all(len(m) == 12 for m in scored) and not drift
    print("TT ARM: PROVEN" if ok else "TT ARM: NOT proven")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
