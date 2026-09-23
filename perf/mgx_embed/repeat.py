"""Does an embedding model return the same vector when it embeds the same sequence again?

    python3 perf/mgx_embed/repeat.py --model esmc-6b --pdb mtor.pdb --n 16 --passes 3 --out rows.jsonl

--n windows of the --pdb chain, lengths spread over --min..--max, embedded --passes times in one
process at batch size 1. Every pass is compared with pass 0 per sequence, and each row records
whether the sequence's token axis needed padding to the bucket, because a padded forward carries
an attention mask and a key-valid tensor that an unpadded one does not.

--perturb moves device buffers between passes without changing any input: `reverse` runs every
other pass in reverse order, `alloc` holds an extra --alloc-mb device buffer from pass 1 on. A
program-cache hit that reuses a stale buffer address is invisible when every pass allocates
the same buffers at the same addresses, which is what a roomy chip does.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "mgx_embed"))
from accuracy import pdb_sequence  # noqa: E402
from batch import library  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--min", type=int, default=60)
    ap.add_argument("--max", type=int, default=250)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--perturb", choices=("none", "reverse", "alloc"), default="none")
    ap.add_argument("--alloc-mb", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    if a.model.startswith("saprot"):
        from tt_bio import saprot as mod
        model = mod.load_saprot(a.model, fast=a.fast)
        seqs = {k: (v, "#" * len(v)) for k, v in library(pdb_sequence(a.pdb), a.n, a.min, a.max).items()}
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(a.model, fast=a.fast)
        seqs = library(pdb_sequence(a.pdb), a.n, a.min, a.max)

    passes, held = [], None
    for p in range(a.passes):
        if p == 1 and a.perturb == "alloc":
            import ttnn
            from tt_bio.tenstorrent import get_device
            n = a.alloc_mb * 2 ** 20 // 2 // 1024 // 32 * 32
            held = ttnn.from_torch(torch.ones(1, 1, n, 1024, dtype=torch.bfloat16),
                                   device=get_device(), layout=ttnn.TILE_LAYOUT)
        items = list(seqs.items())
        if a.perturb == "reverse" and p % 2:
            items.reverse()
        out = mod.embed_sequences(model, dict(items), batch_size=1)
        passes.append({e.id: e.per_residue.astype(np.float64) for e in out})
    del held
    with open(a.out, "a") as fh:
        for sid, seq in (seqs.items() if not a.model.startswith("saprot") else
                         ((k, v[0]) for k, v in seqs.items())):
            tokens = len(seq) + 2
            base = passes[0][sid]
            for p in range(1, a.passes):
                d = passes[p][sid]
                cos = (d * base).sum(1) / (np.linalg.norm(d, axis=1) * np.linalg.norm(base, axis=1))
                row = {"model": a.model, "fast": a.fast, "id": sid, "L": len(seq),
                       "tokens": tokens, "padded": tokens % mod.BUCKET != 0, "pass": p, "perturb": a.perturb,
                       "bit_equal": bool(np.array_equal(d, base)),
                       "cos_min_vs_pass0": float(cos.min()),
                       "max_abs_vs_pass0": float(np.abs(d - base).max()),
                       "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES")}
                print(json.dumps(row), flush=True)
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
