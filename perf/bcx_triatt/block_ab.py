#!/usr/bin/env python3
"""`b` for one fwd+bwd pair unit at n=256, before and after, in one process on one card.

Arms differ only in how the triangle-attention region is issued (see ARMS). Every arm runs
the census's step (`census.one_step`) on the same weights, interleaved K=1/K=2 in a rotating
order so drift and clock land on every arm alike. `b` is the census's device-side slope,
median(fwd K2 - K1) + median(bwd K2 - K1), with the paired whole-step differences beside it.

`price_default` is priced, not proposed: HiFi4 + fp32 accumulation is `precise_config`'s on
purpose, and whether the default is accurate enough is `bcx-bfp8`'s question.
"""
import argparse, json, pathlib, statistics, sys, time
import numpy as np
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.hallgrad.census import (BINS, CONTACT_BINS, C_S, C_Z, HEADS, HEAD_DIM, MIN_SEP,  # noqa: E402
                                  one_step, stamp)
from perf.hallgrad.e2e_distogram import ClockTrace, make_weights  # noqa: E402

# name: (split_verbs, bmm_config, chunk, triatt config)
ARMS = {
    "census":        (False, False, 128, None),
    "heads_only":    (True,  False, 128, None),
    "bmm_only":      (False, True,  128, None),
    "new":           (True,  True,  128, None),
    "new_c32":       (True,  True,  32,  None),
    "new_c64":       (True,  True,  64,  None),
    "new_c256":      (True,  True,  256, None),
    "census_c64":    (False, False, 64,  None),
    "census_c256":   (False, False, 256, None),
    "price_default": (True,  True,  128, "default"),
    "heads_c256":    (True,  False, 256, None),
    "bmm_c256":      (False, True,  256, None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag
    assert ag.__file__.startswith(str(ROOT)), ag.__file__
    device = tt.get_device()
    clocks = ClockTrace(period=1.0).start()
    arms = args.arms.split(",")
    default_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False,
        packer_l1_acc=True)
    blob = {"argv": " ".join(sys.argv), "stamp_start": stamp(), "n": args.n,
            "arms": {a: list(ARMS[a][:3]) + [ARMS[a][3]] for a in arms}}
    n = args.n
    idx = np.arange(n)
    mask = np.abs(idx[:, None] - idx[None, :]) >= MIN_SEP
    M = int(mask.sum())
    mask_t = torch.from_numpy(mask).to(torch.float64)
    inC = torch.zeros(BINS, dtype=torch.float64)
    inC[:CONTACT_BINS] = 1.0
    logits = (np.random.default_rng(args.seed).standard_normal((n, 21)) * 0.1).astype(np.float32)
    W = {}
    for K in (1, 2):
        Wnp = make_weights(np.random.default_rng(args.seed), C_S, C_Z, HEADS, HEAD_DIM, C_Z, BINS,
                           blocks=K, block_transition=True)
        W[K] = {k: ag.Tensor(ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(v)).to(torch.bfloat16),
                                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
                for k, v in Wnp.items()}

    def step(arm, K):
        split, bmm, chunk, tcfg = ARMS[arm]
        ag.TRIATT_BMM_CONFIG = bmm
        cfg = dict(heads=HEADS, head_dim=HEAD_DIM, hidden=C_Z, chunk=chunk, checkpoint=False,
                   blocks=K, block_transition=True, split_verbs=split,
                   triatt_config=default_cfg if tcfg == "default" else None)
        return one_step(ag, ttnn, device, W[K], cfg, logits, mask_t, M, inC)

    print("# warm-up", flush=True)
    for _ in range(args.warm):
        for a in arms:
            for K in (1, 2):
                step(a, K)
    rec = {a: {1: [], 2: []} for a in arms}
    t0 = time.time()
    for i in range(args.pairs):
        order = arms[i % len(arms):] + arms[:i % len(arms)]
        if i % 2:
            order = order[::-1]
        for a in order:
            for K in ((1, 2) if i % 2 == 0 else (2, 1)):
                rec[a][K].append(step(a, K))
    t1 = time.time()
    blob["window"] = [t0, t1]
    blob["clock_window"] = clocks.window(t0, t1)
    res = {}
    for a in arms:
        r1, r2 = rec[a][1], rec[a][2]
        f = [y["fwd_s"] - x["fwd_s"] for x, y in zip(r1, r2)]
        bw = [y["bwd_s"] - x["bwd_s"] for x, y in zip(r1, r2)]
        st = [y["step_s"] - x["step_s"] for x, y in zip(r1, r2)]
        res[a] = {
            "b": statistics.median(f) + statistics.median(bw),
            "b_fwd": statistics.median(f), "b_bwd": statistics.median(bw),
            "step_diff_median": statistics.median(st),
            "step_diff_p10": float(np.percentile(st, 10)), "step_diff_p90": float(np.percentile(st, 90)),
            "bwd_K2_p10": float(np.percentile([y["bwd_s"] for y in r2], 10)),
            "bwd_K2_p90": float(np.percentile([y["bwd_s"] for y in r2], 90)),
            "reached": [r1[0]["reached"], r2[0]["reached"]],
            "raw": {"K1": r1, "K2": r2},
        }
    base = res.get("census", {}).get("b")
    print(f"AICLK over the timed window: {blob['clock_window']}")
    for a in arms:
        r = res[a]
        sp = f"{base / r['b']:.3f}x" if base else ""
        print(f"{a:14s} b {r['b']:.4f} s (fwd {r['b_fwd']:.4f} bwd {r['b_bwd']:.4f})  "
              f"step-diff p10/50/90 {r['step_diff_p10']:.4f}/{r['step_diff_median']:.4f}/"
              f"{r['step_diff_p90']:.4f}  vs census {sp}  reached {r['reached']}", flush=True)
    blob["results"] = res
    blob["stamp_end"] = stamp()
    blob["clock_meta"] = clocks.summary()
    clocks.stop()
    json.dump(blob, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
