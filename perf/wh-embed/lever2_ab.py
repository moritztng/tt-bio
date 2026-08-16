#!/usr/bin/env python3
"""Integrated A/B for lever 2: the 6B embed forward's discarded hidden-state readbacks.

Arm A is the pre-change behaviour -- ESMCHiddenStatesModel copies all n_layers+1 hidden states
to the host and _trunk_forward uses the last. Arm B passes last_hidden_only=True, so only the
final-norm state is copied.

Both arms call the shipped embed_sequences on one resident model in one process, arms
interleaved, with a leading A/A pair whose spread is the noise floor. The device op sequence is
identical between arms -- the only difference is which intermediates get copied to the host --
so the result must be bit-exact, and this checks that with torch.equal rather than a PCC of
convenience. If it is not bit-exact something else changed and the lever stops.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")


def digest(results):
    h = hashlib.sha256()
    for e in results:
        h.update(e.id.encode())
        h.update(np.ascontiguousarray(e.per_residue).tobytes())
        h.update(np.ascontiguousarray(e.pooled).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-6b")
    ap.add_argument("--n-seqs", type=int, default=4)
    ap.add_argument("--residues", type=int, default=76)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.esmc import ESMCLanguageModel, embed_sequences, load_esmc

    reps = (args.residues // len(UBIQUITIN)) + 1
    seq = (UBIQUITIN * reps)[:args.residues]
    sequences = {f"seq{i}": seq for i in range(args.n_seqs)}

    t0 = time.perf_counter()
    model = load_esmc(args.model, fast=args.fast)
    load_s = time.perf_counter() - t0

    # Force the arm at the one place the flag is read. Arm A reproduces the pre-change
    # behaviour exactly: last_hidden_only is ignored and every state is copied back.
    original = ESMCLanguageModel.forward

    def arm_forward(self, input_ids, attn_mask=None, last_hidden_only=False):
        return original(self, input_ids, attn_mask,
                        last_hidden_only=(last_hidden_only and arm_forward.enabled))

    ESMCLanguageModel.forward = arm_forward

    def one(arm):
        arm_forward.enabled = (arm == "B")
        t0 = time.perf_counter()
        out = embed_sequences(model, sequences, batch_size=1)
        return time.perf_counter() - t0, out

    for _ in range(args.warmup):
        one("B")
        one("A")

    order = ["A", "A"] + [a for _ in range(args.repeat) for a in ("B", "A")]
    runs, digests, ref = [], {}, {}
    for i, arm in enumerate(order):
        wall, out = one(arm)
        d = digest(out)
        digests.setdefault(arm, []).append(d)
        if arm not in ref:
            ref[arm] = out
        runs.append(dict(i=i, arm=arm, wall_ms=round(wall * 1000, 2), digest=d[:16],
                         load1=round(os.getloadavg()[0], 2)))
        print(f"  {i:2d} {arm}  {wall*1000:9.2f} ms  {d[:16]}", file=sys.stderr, flush=True)

    import torch
    bitexact = all(
        torch.equal(torch.from_numpy(a.per_residue), torch.from_numpy(b.per_residue))
        and torch.equal(torch.from_numpy(a.pooled), torch.from_numpy(b.pooled))
        for a, b in zip(ref["A"], ref["B"]))

    aa = [r["wall_ms"] for r in runs[:2]]
    A = [r["wall_ms"] for r in runs[2:] if r["arm"] == "A"]
    B = [r["wall_ms"] for r in runs[2:] if r["arm"] == "B"]
    med = statistics.median
    res = dict(
        model=args.model, fast=args.fast, n_seqs=args.n_seqs, residues=args.residues,
        arch=os.environ.get("PROBE_ARCH", "unknown"),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES", ""),
        load_s=round(load_s, 1),
        aa_control_ms=aa,
        aa_noise_pct=round(abs(aa[0] - aa[1]) / min(aa) * 100, 2) if aa[1] else None,
        arm_A_ms=[round(x, 2) for x in A], arm_B_ms=[round(x, 2) for x in B],
        median_A_ms=round(med(A), 2), median_B_ms=round(med(B), 2),
        speedup_B_over_A=round(med(A) / med(B), 4),
        ms_per_forward_A=round(med(A) / args.n_seqs, 2),
        ms_per_forward_B=round(med(B) / args.n_seqs, 2),
        saved_ms_per_forward=round((med(A) - med(B)) / args.n_seqs, 2),
        digest_A=digests["A"][0], digest_B=digests["B"][0],
        digests_all_equal=len(set(sum(digests.values(), []))) == 1,
        bit_exact_torch_equal=bool(bitexact),
        runs=runs,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
