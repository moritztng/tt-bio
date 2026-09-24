#!/usr/bin/env python3
"""Fold extra upstream reference seeds for an esmfold2 cell on the host CPU, fp32.

    <esm 3.4.1 venv>/bin/python perf/mgx/acc/ref_seed_cpu.py --model esmfold2 \
        --fixture 7aqx_1024 --seed 2 --out perf/mgx/acc/refseeds --threads 12

The reference has two GPU seeds per cell, so its floor is one distance. When a TT cell sits
outside that floor on residues the crystal does not resolve, more upstream seeds say whether the
floor or the port is off. This calls ref_fold.py's own esmfold2 runner (same fold() arguments,
same pinned a3m) with the model on the CPU, and writes <out>/<model>/<fixture>/s<seed>.cif
beside a record json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
import ref_fold  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("esmfold2", "esmfold2-fast"))
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    plan = json.loads((ref_fold.HERE / "plan.json").read_text())
    cfg = dict(plan["models"][args.model], device="cpu")
    dest = args.out / args.model / args.fixture
    work = dest / f"work_s{args.seed}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.perf_counter()
    cif, facts = ref_fold.RUNNERS[args.model](ref_fold.load_fixture(args.fixture), args.seed,
                                             cfg, work)
    shutil.copyfile(cif, dest / f"s{args.seed}.cif")
    rec = dict(facts, model=args.model, fixture=args.fixture, seed=args.seed, device="cpu",
               threads=args.threads, wall_s=round(time.perf_counter() - t0, 1))
    (dest / f"s{args.seed}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print("REF", json.dumps({k: rec[k] for k in ("model", "fixture", "seed", "wall_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
