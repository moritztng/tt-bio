#!/usr/bin/env python3
"""p79 -- gate 1a/1b for the gathered atom softmax, at the production shape.

Two questions, isolated from a fold so a wrong answer is cheap:

  1a  Are the SCORES bit-identical under gathering? They must be. The gather is a copy, so this
      gate fails only if the index or the layout is wrong -- not because fp32 is non-associative.
  1b  What does the gathered chain cost per call against the dense softmax it replaces?
      gather (r 588.6 MB, w 12.4) + softmax on 12.4 MB + scatter (w 294.3 MB) against a softmax
      that reads 588.6 fp32 and writes 294.3 bf16.

The attention delta between the arms is REPORTED, not gated: it is the quantity the accuracy
envelope exists to price (scripts/rfd3_port/p78_envelope_spec.json). A bf16-ULP-scale maxabs says
the wiring is right; the design-quality bar says whether it is shippable.

    ~/.coworker/scripts/benchlock.sh rfd3-b8-to-4x-p3 -- env TT_VISIBLE_DEVICES=2 \
      TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p3 PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p79_gathered_probe.py \
          perf/p79/gathered_probe.json 6051 128 4 5
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio import softmax_generic                                      # noqa: E402
from tt_bio.rfd3 import model as M                                      # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p79/gathered_probe.json")
L = int(sys.argv[2]) if len(sys.argv) > 2 else 6051
K = int(sys.argv[3]) if len(sys.argv) > 3 else 128
H = int(sys.argv[4]) if len(sys.argv) > 4 else 4
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
SEED = 42


def timed(fn, reps):
    """Median ms/call, synced. Warm once first: the first call compiles the program."""
    ttnn.deallocate(fn())            # warm: the first call compiles the program
    ttnn.synchronize_device(DEV)
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(DEV)
        ms.append(1000.0 * (time.perf_counter() - t0))
        ttnn.deallocate(out)
    return statistics.median(ms), min(ms), max(ms)


def main():
    global DEV
    torch.manual_seed(SEED)
    n_key = M._align_tile(L)
    # get_device(), not ttnn.open_device(): the raw open throws on this box because
    # TT_VISIBLE_DEVICES makes the cluster type CUSTOM, which needs a mesh graph descriptor.
    # get_device() also takes the card lease and enables the program cache the timings assume.
    DEV = get_device()
    print("[p79] L=%d n_key=%d K=%d H=%d reps=%d" % (L, n_key, K, H, REPS), flush=True)

    # Neighbour indices in the production form: K distinct sorted columns per row, in [0, L).
    idx = torch.stack([torch.randperm(L)[:K].sort().values for _ in range(L)]).unsqueeze(0)
    idx_h = idx.expand(1, H, L, K).contiguous()

    # The scores tensor as the shipped chain leaves it: qk*scale at neighbours, -1e4 elsewhere,
    # fp32. The exact values do not matter for either gate, the mask structure does.
    dense = torch.full((1, H, L, n_key), -1e4, dtype=torch.float32)
    dense.scatter_(3, idx_h, torch.randn(1, H, L, K) * 3.0)
    dense_dev = ttnn.from_torch(dense, layout=ttnn.TILE_LAYOUT, device=DEV,
                                dtype=ttnn.float32)
    gather_idx = M._sparse_attn_index(idx, DEV, H)
    zeros = M._zero_template(None, DEV, ttnn.bfloat16, 1, H, L)

    # ---- gate 1a: the scores under gathering -------------------------------------------------
    compact_dev = ttnn.gather(dense_dev, 3, gather_idx)
    compact = ttnn.to_torch(compact_dev)
    ref = dense.gather(3, idx_h)
    bit_exact = bool(torch.equal(compact, ref))
    print("[p79] gate 1a scores bit-identical under gathering: %s  (maxabs %.3e)"
          % (bit_exact, (compact - ref).abs().max().item()), flush=True)

    # ---- the attention delta the envelope prices --------------------------------------------
    a_dense = ttnn.to_torch(softmax_generic.softmax_bf16(dense_dev, ttnn.bfloat16)).float()
    w = softmax_generic.softmax_bf16(compact_dev, ttnn.bfloat16)
    a_gath = ttnn.to_torch(ttnn.scatter(zeros, 3, gather_idx, w)).float()
    delta = (a_dense - a_gath).abs()
    rows_dense, rows_gath = a_dense.sum(3), a_gath.sum(3)
    print("[p79] attention maxabs %.6e   nonzero-count dense %d gathered %d"
          % (delta.max().item(), int((a_dense != 0).sum()), int((a_gath != 0).sum())), flush=True)
    print("[p79] row sums: dense mean %.6f  gathered mean %.6f  max|diff| %.6e"
          % (rows_dense.mean().item(), rows_gath.mean().item(),
             (rows_dense - rows_gath).abs().max().item()), flush=True)

    # ---- gate 1b: the timings ----------------------------------------------------------------
    dense_ms = timed(lambda: softmax_generic.softmax_bf16(dense_dev, ttnn.bfloat16), REPS)

    def gathered_chain():
        c = ttnn.gather(dense_dev, 3, gather_idx)
        ww = softmax_generic.softmax_bf16(c, ttnn.bfloat16)
        ttnn.deallocate(c)
        out = ttnn.scatter(zeros, 3, gather_idx, ww)
        ttnn.deallocate(ww)
        return out

    gath_ms = timed(gathered_chain, REPS)
    gather_only = timed(lambda: ttnn.gather(dense_dev, 3, gather_idx), REPS)
    sm_only = timed(lambda: softmax_generic.softmax_bf16(compact_dev, ttnn.bfloat16), REPS)

    print("\n%-22s %9s %9s %9s" % ("", "median", "min", "max"))
    for name, v in (("dense softmax", dense_ms), ("gathered chain", gath_ms),
                    ("  gather alone", gather_only), ("  softmax alone", sm_only)):
        print("%-22s %9.4f %9.4f %9.4f ms" % (name, *v))
    ratio = dense_ms[0] / gath_ms[0]
    # 9 calls per diffusion step (decoder 6, atom encoder 3) x 200 steps.
    prize = 9 * 200 * (dense_ms[0] - gath_ms[0]) / 1000.0
    print("\nratio %.3fx   isolated prize %.3f s/design at 9 calls x 200 steps" % (ratio, prize))
    print("An isolated per-call screen is faithful on this region (the atom path's synced-to-wall "
          "ratios are 1.027 and 1.106); it is not on the DiT. The fold A/B is still the number.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "L": L, "n_key": n_key, "K": K, "H": H, "reps": REPS, "seed": SEED,
        "scores_bit_identical": bit_exact,
        "attention_maxabs": delta.max().item(),
        "row_sum_maxabs": (rows_dense - rows_gath).abs().max().item(),
        "nonzero_dense": int((a_dense != 0).sum()), "nonzero_gathered": int((a_gath != 0).sum()),
        "dense_softmax_ms": dense_ms, "gathered_chain_ms": gath_ms,
        "gather_only_ms": gather_only, "compact_softmax_ms": sm_only,
        "ratio": ratio, "isolated_prize_s_per_design": prize,
        "calls_per_step": 9, "steps": 200,
        "host": "qb2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
    }, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
