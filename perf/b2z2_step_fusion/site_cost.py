#!/usr/bin/env python3
"""Put a us figure on every SITE in the diffusion step, by aligning two orderings of it.

`step_probe.py --mode ops` gives the ordered list of top-level ttnn calls in one
`Diffusion.__call__`; `b2z2-sampler-stall-split`'s armed capture gives the ordered list of the
device programs the same call dispatches. Neither alone says what a given line of
`tenstorrent.py` costs: the op-code histogram has lost the order, and the ttnn list has no times.

Aligning them does. The device sequence is a subsequence of the ttnn sequence -- a `deallocate`,
a view-only `reshape` and a no-op `to_memory_config` dispatch nothing -- so a greedy walk that
consumes ttnn calls until one matches the next device op code recovers the map, and the walk is
checked by requiring it to consume the whole of both.

Kernel durations come off the armed capture, whose gaps are inflated 3.81x but whose kernels are
not (armed kernel sum 40.4144 ms against a 41.4820 ms bare wall, 2.6 %). So a kernel duration is
usable here and a span is not.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

# ttnn call -> the device op code it dispatches, or None when it may dispatch nothing
EXPECT = {
    "ttnn.linear": "Matmul", "ttnn.matmul": "Matmul",
    "ttnn.add": "BinaryNg", "ttnn.add_": "BinaryNg",
    "ttnn.multiply": "BinaryNg", "ttnn.multiply_": "BinaryNg",
    "ttnn.subtract": "BinaryNg", "ttnn.subtract_": "BinaryNg",
    "ttnn.layer_norm": "LayerNorm",
    "ttnn.softmax": "Softmax",
    "ttnn.permute": ("Permute", "Transpose"),
    "ttnn.transpose": ("Transpose", "Permute"),
    "ttnn.reshape": "ReshapeView",
    "ttnn.unsqueeze": "ReshapeView", "ttnn.squeeze": "ReshapeView",
    "ttnn.to_memory_config": "Copy",
    "ttnn.Tensor.__getitem__": "Slice",
    "ttnn.pad": "Pad",
    "ttnn.transformer.scaled_dot_product_attention": "SDPA",
    "ttnn.experimental.nlp_create_qkv_heads": "NlpCreateHeads",
    "ttnn.experimental.nlp_concat_heads": "NLPConcatHeads",
    "ttnn.cos": ("UnaryNg", "Unary"),
    "ttnn.sigmoid": ("UnaryNg", "Unary"), "ttnn.silu": ("UnaryNg", "Unary"),
    "ttnn.clone": "Clone",
}
NEVER = {"ttnn.deallocate"}          # dispatches nothing, ever


def short(code):
    return code.replace("DeviceOperation", "").replace("Operation", "")


def load_device(csv_path, fence_code="Unary", fence_run=3):
    rows = list(csv.DictReader(gzip.open(csv_path, "rt") if str(csv_path).endswith(".gz")
                               else open(csv_path)))
    codes = [short(r["OP CODE"]) for r in rows]
    fences = []
    i = 0
    while i < len(codes):
        if codes[i].startswith(fence_code):
            j = i
            while j < len(codes) and codes[j].startswith(fence_code):
                j += 1
            if j - i >= fence_run:
                fences.append((i, j))
            i = j
        else:
            i += 1
    if len(fences) < 2:
        raise SystemExit(f"need two fences, found {len(fences)}")
    lo, hi = fences[-2][1], fences[-1][0]
    span = rows[lo:hi]
    return span


def split_reps(span, n_expected=None):
    """The fenced span is `reps` identical program sequences back to back."""
    codes = [short(r["OP CODE"]) for r in span]
    n = len(codes)
    for reps in (3, 2, 1, 4, 5, 10):
        if n % reps:
            continue
        k = n // reps
        if all(codes[i] == codes[i % k] for i in range(n)):
            return [span[r * k:(r + 1) * k] for r in range(reps)]
    raise SystemExit(f"span of {n} does not split into identical reps")


def align(ttnn_seq, dev_codes):
    """ttnn call index -> device op index, greedily. Returns (pairs, unmatched_ttnn)."""
    pairs, j = [], 0
    for i, name in enumerate(ttnn_seq):
        if j >= len(dev_codes):
            break
        if name in NEVER:
            continue
        want = EXPECT.get(name)
        if want is None:
            continue
        want = (want,) if isinstance(want, str) else want
        if any(dev_codes[j].startswith(w) for w in want):
            pairs.append((i, j))
            j += 1
    return pairs, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", type=Path, required=True, help="step_probe --mode ops output")
    ap.add_argument("--csv", type=Path, required=True, help="armed ops_perf CSV[.gz]")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    ops = json.loads(a.ops.read_text())["ops"]["sequence"]
    reps = split_reps(load_device(a.csv))
    codes = [short(r["OP CODE"]) for r in reps[0]]
    # median kernel duration over the reps, per program position
    dur = [sorted(float(rep[k]["DEVICE KERNEL DURATION [ns]"]) for rep in reps)[len(reps) // 2]
           for k in range(len(codes))]

    pairs, consumed = align(ops, codes)
    out = {"n_ttnn": len(ops), "n_device": len(codes), "n_reps": len(reps),
           "n_aligned": len(pairs), "device_consumed": consumed,
           "kernel_sum_ms": round(sum(dur) / 1e6, 4),
           "aligned_all": consumed == len(codes)}
    if not out["aligned_all"]:
        out["stuck_at_device"] = {"i": consumed, "code": codes[consumed] if consumed < len(codes) else None,
                                  "next_codes": codes[consumed:consumed + 8]}
    dev_of_ttnn = dict(pairs)
    # per-ttnn-op row, in order, with its device cost
    table = []
    for i, name in enumerate(ops):
        j = dev_of_ttnn.get(i)
        table.append({"i": i, "ttnn": name,
                      "code": codes[j] if j is not None else None,
                      "us": round(dur[j] / 1e3, 3) if j is not None else 0.0})
    out["table"] = table
    by = defaultdict(lambda: {"n": 0, "us": 0.0})
    for t in table:
        if t["code"]:
            b = by[t["code"]]
            b["n"] += 1
            b["us"] += t["us"]
    out["by_code"] = {k: {"n": v["n"], "ms": round(v["us"] / 1e3, 4)}
                      for k, v in sorted(by.items(), key=lambda kv: -kv[1]["us"])}
    out["unmatched_ttnn"] = dict(Counter(t["ttnn"] for t in table if t["code"] is None))
    a.out.write_text(json.dumps(out, indent=1))
    print(f"aligned {len(pairs)}/{len(codes)} device programs, kernel sum "
          f"{out['kernel_sum_ms']:.4f} ms, all={out['aligned_all']}")
    for k, v in out["by_code"].items():
        print(f"  {v['n']:5d}  {v['ms']:8.4f} ms  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
