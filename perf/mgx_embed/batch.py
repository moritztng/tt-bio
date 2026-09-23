"""Batch behaviour of one embedding model on one chip: throughput, and what batching does to a vector.

    python3 perf/mgx_embed/batch.py --model esmc-300m --pdb mtor.pdb --n 256 --out rows.jsonl

The library is what an entrant embeds between design and submission: --n distinct windows of a
real protein, lengths spread evenly over --min..--max residues (binder-sized by default). The
model loads once; each --batch_sizes value then embeds the whole library, timed with AICLK
sampled during the call. The first pass pays compile, so every batch size runs twice and the
second is the one reported as throughput.

Each batched vector is compared with the same sequence embedded alone (batch_size 1), because
the CLI documents that padding is masked but the bucketed length sets the bf16 reduction order.

Across chips: start one process per chip with the same --barrier DIR and --parties K. Each runs an
untimed compile pass, then waits until K processes have reached DIR before its timed passes, so the
passes overlap and the per-chip seq/s is what a chip gives while K-1 others share the host. A party
that is still missing after --barrier-timeout seconds is recorded in `parties_seen`, not waited on.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "mgx_embed"))
from perf.clocksample import during  # noqa: E402
from accuracy import pdb_sequence  # noqa: E402


def library(full, n, lo, hi):
    lens = np.linspace(lo, hi, n).round().astype(int)
    starts = np.linspace(0, len(full) - hi, n).round().astype(int)
    return {f"s{i:04d}": full[s:s + L] for i, (s, L) in enumerate(zip(starts, lens))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--min", type=int, default=60)
    ap.add_argument("--max", type=int, default=250)
    ap.add_argument("--batch_sizes", default="1,8,32")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--barrier", default=None, help="directory shared by the processes of one run")
    ap.add_argument("--parties", type=int, default=1)
    ap.add_argument("--barrier-timeout", type=float, default=1800)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)

    seqs = library(pdb_sequence(a.pdb), a.n, a.min, a.max)
    if a.model.startswith("saprot"):
        from tt_bio import saprot as mod
        t = time.time()
        model = mod.load_saprot(a.model, fast=a.fast)
    else:
        from tt_bio import esmc as mod
        t = time.time()
        model = mod.load_esmc(a.model, fast=a.fast)
    load_s = time.time() - t

    parties_seen = 1
    if a.barrier:
        bs0 = int(a.batch_sizes.split(",")[0])
        mod.embed_sequences(model, seqs, batch_size=bs0)  # compile, untimed
        b = Path(a.barrier); b.mkdir(parents=True, exist_ok=True)
        (b / str(os.getpid())).touch()
        deadline = time.time() + a.barrier_timeout
        while (parties_seen := len(list(b.iterdir()))) < a.parties and time.time() < deadline:
            time.sleep(0.2)

    alone = None
    with open(a.out, "a") as fh:
        for bs in [int(x) for x in a.batch_sizes.split(",")]:
            for rep in (0, 1):
                with during() as clk:
                    t = time.time()
                    out = mod.embed_sequences(model, seqs, batch_size=bs)
                    wall = time.time() - t
                t_end = time.time()
                pooled = np.stack([e.pooled for e in out]).astype(np.float64)
                if alone is None:
                    alone = pooled
                cos = (pooled * alone).sum(1) / (np.linalg.norm(pooled, axis=1)
                                                 * np.linalg.norm(alone, axis=1))
                row = {"model": a.model, "fast": a.fast, "n": a.n, "len_range": [a.min, a.max],
                       "residues": sum(map(len, seqs.values())), "batch_size": bs, "rep": rep,
                       "load_s": round(load_s, 1), "wall_s": round(wall, 2),
                       "t_start": round(t_end - wall, 2), "t_end": round(t_end, 2),
                       "parties": a.parties, "parties_seen": parties_seen,
                       "seq_per_s": round(a.n / wall, 2),
                       "pooled_cos_vs_alone_min": float(cos.min()),
                       "finite": bool(np.isfinite(pooled).all()), "aiclk": clk.summary(),
                       "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES")}
                print(json.dumps(row), flush=True)
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
