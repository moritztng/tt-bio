#!/usr/bin/env python3
"""A/B the SaProt MLM head on the embed path, which no embed caller reads.

``SaprotModel.__call__`` used to run ``self.head(emb)`` unconditionally and
``Saprot._dispatch`` used to copy the resulting ``[B, L, 446]`` back to the host, while
``embed_sequences`` looks at ``logits`` only when ``return_logits=True`` -- which the embed
CLI, the worker and the perf gate never pass. Same defect as ESMC's discarded hidden-state
readbacks, so the fix has the same shape: a flag threaded down to the one place the work is
issued (``want_logits``), default unchanged.

Arm A forces the head back on, arm B is the shipped path. Arms interleave on one resident
model in one process, with a leading A/A pair whose spread is this box's noise floor. The
embedding is produced before the head is reached, so the two arms must be bit-exact; that is
checked with ``torch.equal`` and a sha256 over the arrays rather than assumed.

Usage:
  TT_VISIBLE_DEVICES=<n> python3 perf/whde/saprot_logits_ab.py \
      --model saprot-650m --n-seqs 8 --residues 76 --out results/x.json
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")  # 76 aa, the perf gate's embed fixture, verbatim


def digest(results):
    h = hashlib.sha256()
    for e in results:
        h.update(e.id.encode())
        h.update(np.ascontiguousarray(e.per_residue).tobytes())
        h.update(np.ascontiguousarray(e.pooled).tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--residues", type=int, default=76)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from tt_bio.saprot import Saprot, embed_sequences, load_saprot

    reps = (args.residues // len(UBIQUITIN)) + 1
    seq = (UBIQUITIN * reps)[: args.residues]
    sequences = {f"seq{i}": seq for i in range(args.n_seqs)}

    t0 = time.perf_counter()
    model = load_saprot(args.model)
    load_s = time.perf_counter() - t0

    original = Saprot.forward

    def arm_forward(self, tokens, attn_mask=None, key_valid=None, embed_mask=None,
                    want_logits=True):
        # Arm A reproduces the pre-change behaviour exactly: the head always runs and its
        # output is always copied back, whatever the caller asked for.
        return original(self, tokens, attn_mask, key_valid, embed_mask,
                        want_logits=(want_logits or arm_forward.force_head))

    Saprot.forward = arm_forward

    def one(arm):
        arm_forward.force_head = (arm == "A")
        t0 = time.perf_counter()
        out = embed_sequences(model, sequences, batch_size=args.batch_size)
        return (time.perf_counter() - t0) * 1e3, out

    for _ in range(args.warmup):
        one("A")

    order = ["A", "A"] + [a for _ in range(args.repeat) for a in ("B", "A")]
    runs, keep = [], {}
    for i, arm in enumerate(order):
        ms, out = one(arm)
        runs.append({"i": i, "arm": arm, "wall_ms": round(ms, 2),
                     "digest": digest(out), "load1": round(os.getloadavg()[0], 2)})
        keep.setdefault(arm, out)

    bit_exact = all(
        torch.equal(torch.from_numpy(a.per_residue), torch.from_numpy(b.per_residue))
        and torch.equal(torch.from_numpy(a.pooled), torch.from_numpy(b.pooled))
        for a, b in zip(keep["A"], keep["B"])
    )

    med = {a: statistics.median([r["wall_ms"] for r in runs if r["arm"] == a])
           for a in ("A", "B")}
    aa = [r["wall_ms"] for r in runs[:2]]
    out = {
        "model": args.model, "n_seqs": args.n_seqs, "residues": args.residues,
        "batch_size": args.batch_size, "load_s": round(load_s, 1),
        "host": os.uname().nodename, "device": os.environ.get("TT_VISIBLE_DEVICES"),
        "median_ms": {a: round(v, 2) for a, v in med.items()},
        "speedup_B_over_A": round(med["A"] / med["B"], 4),
        "aa_floor_pct": round(abs(aa[0] - aa[1]) / max(aa) * 100, 2),
        "bit_exact_torch_equal": bit_exact,
        "digests": sorted({r["digest"] for r in runs}),
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
