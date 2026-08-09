#!/usr/bin/env python3
"""Multi-prediction throughput and parity for the batched protenix-v2 diffusion path.

Same target, same config and the same fold path as ``tt_baseline`` / ``tt_concurrency``, so
the folds/s here is directly comparable to the committed single-process number. B copies of
the 117-aa fold is also exactly what the H200 MPS leg runs, so nothing about the comparison
depends on bucketing.

Two modes, both on one card:

  --mode throughput   B folds through worker.predict_many, timed warm, against B serial
                      folds through predict_one measured in the same process.
  --mode parity       coords from fold_many(B) against coords from B separate fold() calls
                      at the same seed: max abs deviation, RMSD and PCC per member.

Usage:

    TT_VISIBLE_DEVICES=3 python3 scripts/gpu_vs_tt/tt_batch_throughput.py \
        --batch 4 --repeat 3 --out results/conc/mp_batch4_c3.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import tt_baseline  # noqa: E402


def _pcc(a, b):
    import torch
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="throughput", choices=["throughput", "parity"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3, help="timed rounds")
    ap.add_argument("--serial-repeat", type=int, default=None,
                    help="serial folds for the in-process baseline (default: --repeat)")
    ap.add_argument("--target", default=str(REPO_ROOT / "examples" / "prot.yaml"))
    ap.add_argument("--msa-a3m", default=str(HERE / "fixtures" / "prot117.a3m"))
    ap.add_argument("--msa-dir", default=None)
    ap.add_argument("--steps", type=int, default=tt_baseline.SAMPLING_STEPS,
                    help="diffusion steps; only lower it for a shape smoke test")
    ap.add_argument("--cycles", type=int, default=tt_baseline.RECYCLING_STEPS,
                    help="trunk recycling cycles; only lower it for a shape smoke test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    B = args.batch
    target = Path(args.target)
    msa_dir = Path(args.msa_dir or (Path(args.out).resolve().parent / "msa"))
    one_fold, meta, state = tt_baseline.build_fold(
        "protenix-v2", msa_dir, target, Path(args.msa_a3m), samples=1)
    job_cfg = meta["job_cfg"]
    job_cfg["sampling_steps"] = args.steps
    job_cfg["recycling_steps"] = args.cycles

    cold_s, cold_metrics = one_fold()
    assert cold_metrics.get("msa"), "fold ran without an MSA -- cache seeding failed"

    out = dict(side="tenstorrent", model="protenix-v2", mode=args.mode, batch=B,
               machine=socket.gethostname(), visible_devices=os.environ.get("TT_VISIBLE_DEVICES"),
               target=str(target), ttnn_version=tt_baseline._ttnn_version(),
               tt_bio_git=tt_baseline._git_sha(), recycling_steps=tt_baseline.RECYCLING_STEPS,
               sampling_steps=args.steps, recycling_override=args.cycles, seed=tt_baseline.SEED,
               n_msa=meta["n_msa"], load_s=meta["load_s"], cold_s=round(cold_s, 3),
               plddt_cold=cold_metrics.get("plddt"), n_tokens=cold_metrics.get("n_tokens"),
               date=time.strftime("%Y-%m-%d"), **{k: meta[k] for k in ("card_type", "aiclk_mhz")
                                                  if k in meta})

    if args.mode == "parity":
        feats, chains, specs = state._protenix_inputs(target, job_cfg)
        with torch.no_grad():
            single, single_conf = state.model.fold(
                feats, n_step=args.steps, n_sample=1, seed=tt_baseline.SEED,
                return_confidence=True, n_cycles=args.cycles)
            many, many_conf = state.model.fold_many(
                [feats] * B, n_step=args.steps, seed=tt_baseline.SEED,
                return_confidence=True, n_cycles=args.cycles)
        ref = single[0].float()
        rows = []
        for b in range(B):
            got = many[b][0].float()
            d = (got - ref)
            rows.append(dict(member=b, max_abs=float(d.abs().max()),
                             rmsd=float((d.pow(2).sum(-1).mean()).sqrt()),
                             pcc=_pcc(got, ref), bit_exact=bool(torch.equal(got, ref)),
                             plddt=round(float(many_conf[b]["plddt"]), 6)))
        out["parity"] = rows
        out["plddt_single"] = round(float(single_conf["plddt"]), 6)
    else:
        serial_n = args.serial_repeat if args.serial_repeat is not None else args.repeat
        serial = []
        for _ in range(serial_n):
            t, _m = one_fold()
            serial.append(t)
        paths = [target] * B
        state.predict_many(paths, job_cfg)          # warm the batched shapes
        batched = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            res = state.predict_many(paths, job_cfg)
            batched.append(time.perf_counter() - t0)
        out["serial_latency_s"] = [round(t, 3) for t in serial]
        out["serial_folds_per_s"] = round(1.0 / statistics.median(serial), 5)
        out["batch_wall_s"] = [round(t, 3) for t in batched]
        out["batch_folds_per_s"] = round(B / statistics.median(batched), 5)
        out["speedup_vs_serial"] = round(
            (B / statistics.median(batched)) * statistics.median(serial), 4)
        out["batch_plddt"] = [round(float(m["plddt"]), 6) for m, _b, _f in res]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("mode", "batch", "serial_folds_per_s", "batch_folds_per_s",
                               "speedup_vs_serial", "parity", "batch_plddt")}, indent=2),
          flush=True)
    state.reset()
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
