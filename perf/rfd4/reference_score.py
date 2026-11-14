"""Score one candidate through the oracle seam's reference arm, on the host, with real weights.

This is the arm `rfd4-quality` grades with while the device path is built, so it has to be a real
fold rather than a promise: AlphaFold2's own parameters, the torch `AF2Model`, the same
`af2ig.fold` the card path calls, and out the other end the reward keys a beam-search config
weights by name.

    python3 perf/rfd4/reference_score.py --rung 128 \\
        --params /home/moritz/bcx_shipped/af2_params/params_model_1_ptm.npz

No device is opened: `backend="reference"` builds the torch model, which is why this runs on a
box holding no card. It is slow by construction (CPU, 48 evoformer blocks, recycles + 1 passes)
and is a correctness instrument, never a timing one -- nothing here is a perf number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio.rfd4.oracle import OracleRequest  # noqa: E402
from tt_bio.rfd4.oracle_af2ig import AF2IGOracle  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=128)
    ap.add_argument("--params", required=True)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--name", default="oracle_ppi")
    ap.add_argument("--out", default=str(ROOT / "perf" / "rfd4" / "results" / "reference_score.json"))
    args = ap.parse_args()

    fixture = ROOT / "perf" / "rfd4" / "inputs" / f"af2ig_{args.rung}" / "complex.pdb"
    structure = fixture.read_text()

    oracle = AF2IGOracle(params=args.params, name=args.name, backend="reference",
                         recycles=args.recycles)
    t0 = time.monotonic()
    oracle.setup()
    loaded = time.monotonic() - t0

    job = OracleRequest(structure=structure, example_id=f"af2ig_{args.rung}",
                        chain_of_interest="B")
    t1 = time.monotonic()
    metrics = oracle.fold_batch([job])[0]
    folded = time.monotonic() - t1

    row = {"rung_tokens": args.rung, "backend": "reference", "recycles": args.recycles,
           "params": args.params, "load_s": round(loaded, 1), "fold_s": round(folded, 1),
           "note": "CPU correctness instrument, not a timing", "metrics": metrics}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    return 0 if metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
