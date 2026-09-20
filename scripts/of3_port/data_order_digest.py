#!/usr/bin/env python3
"""Hash the order OpenFold3's sampler visits datapoints in.

`of3t-equivalence` needs a fixed batch AND a fixed data order. `batch_digest.py`
covers the batch; this covers the order, which comes from a different mechanism and
so needs its own check.

`OF3DistributedSampler.__iter__` seeds a private `torch.Generator` with
`self.seed + self.epoch` and draws the dataset index and then the datapoint index from
it, so the order is a pure function of (seed, epoch, dataset probabilities, per-datapoint
probabilities) and does not touch the global RNG. That is the good case, and this
demonstrates it rather than reading it off the source: the same (seed, epoch) must give
the same sequence, and a different epoch must give a different one.

Per-rank batch is 1 (their runner asserts it, LEDGER R5), so the sequence below is the
batch sequence.

Usage:
    python scripts/of3_port/data_order_digest.py --data-dir <dir> --epochs 3
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
DEFAULT_PKG = "tt_bio._vendor.openfold3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--package", default=DEFAULT_PKG)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epoch-len", type=int, default=64)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import batch_digest as bd

    sd_mod = importlib.import_module(f"{args.package}.core.data.framework.stochastic_sampler_dataset")

    bd.seed_everything(args.seed)
    ds = bd.build_dataset(args.package, args.data_dir, 4,
                          bd.STAGE_CROP[args.stage], "train", args.stage)
    sampler_ds = sd_mod.SamplerDataset(datasets=[ds], epoch_len=args.epoch_len)
    print(f"{type(ds).__name__}: {len(ds)} datapoints   epoch_len {args.epoch_len}   seed {args.seed}")

    out = {"seed": args.seed, "epoch_len": args.epoch_len, "epochs": {}}
    for epoch in range(args.epochs):
        sampler = sd_mod.OF3DistributedSampler(
            dataset=sampler_ds,
            dataset_probabilities=[1.0],
            next_dataset_indices={},
            epoch_len=args.epoch_len,
            num_replicas=1,
            rank=0,
            seed=args.seed,
        )
        sampler.set_epoch(epoch)
        pairs = list(sampler)
        d = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
        out["epochs"][epoch] = {"n": len(pairs), "sha256": d, "head": pairs[:6]}
        print(f"  epoch {epoch}  n {len(pairs):4d}  sha256 {d}  head {pairs[:4]}")

    digests = [v["sha256"] for v in out["epochs"].values()]
    print()
    print("epochs differ from each other:" , len(set(digests)) == len(digests),
          "(they must -- a sampler that returns the same order every epoch is broken)")

    # Repeat epoch 0 to show the order is reproducible, which is the actual claim.
    sampler = sd_mod.OF3DistributedSampler(
        dataset=sampler_ds, dataset_probabilities=[1.0], next_dataset_indices={},
        epoch_len=args.epoch_len, num_replicas=1, rank=0, seed=args.seed)
    sampler.set_epoch(0)
    again = hashlib.sha256(json.dumps(list(sampler)).encode()).hexdigest()
    ok = again == out["epochs"][0]["sha256"]
    print(f"epoch 0 reproduces: {ok}  {again}")
    out["epoch0_reproduces"] = ok

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
