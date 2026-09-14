"""An L1 refusal on the above-cap route must cost speed, never correctness.

`fused_pairs` only offers pairs the CB model says fit, so on a healthy tree the device refusal
branch in `triatt_sdpa.sdpa` is never taken and "the fallback is fine" would be an assertion about
dead code. This forces it: `cb_fits_l1` is stubbed to True, so every pair is offered including the
ones that overflow, tt-metal throws, and the question is what `_tri_att_sdpa_at` returns.

Three things are checked, at a padded length above the cap where the route is live:

  1. the throw is absorbed -- `PM_L1_ERRORS` carries the device's own message with its byte figures;
  2. the refused config is memoised in `_PM_OVER_L1`, so the next call declines instead of throwing;
  3. the tensor that comes back is bit-identical to the arm with the route off.

    TT_VISIBLE_DEVICES=2 python3 perf/ttx_a3/l1_refusal_control.py --seq 1536
"""
import argparse, json, os, sys, time

import torch
import ttnn

sys.path.insert(0, os.environ.get("WT", os.getcwd()))
from tt_bio import sdpa_generic as SG          # noqa: E402
from tt_bio import tenstorrent as T            # noqa: E402
from tt_bio import triatt_sdpa as TS           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64,
                    help="batch rows; the full S rows are not needed to exercise the branch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, H, D, B = a.seq, a.heads, a.head_dim, a.batch
    dev = T.get_device()
    torch.manual_seed(0)
    scale = D ** -0.5

    def mk(shape):
        t = (torch.randn(shape, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        return ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    q, k, v = (mk([B, H, S, D]) for _ in range(3))
    bias = mk([1, H, S, S])
    ttnn.synchronize_device(dev)

    res = {"seq": S, "heads": H, "head_dim": D, "batch": B,
           "grid": list(T.COMPUTE_GRID_MAIN), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # arm 1: the route off, which is what shipped before this row
    T._SDPA_FUSED_LARGE_S = False
    T.SDPA_CHUNK_PICKS.clear()
    ref = ttnn.to_torch(T._tri_att_sdpa_at(q, k, v, bias, scale))
    res["off_pick"] = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}

    # arm 2: the route on, with the L1 model lying so every pair is offered
    TS._PM_OVER_L1.clear(); TS.PM_L1_ERRORS.clear(); TS.STATS[:] = [0, 0]
    T._SDPA_FUSED_LARGE_S = True
    T.SDPA_CHUNK_PICKS.clear()
    TS.fused_pairs.cache_clear()
    real_fits = SG.cb_fits_l1
    SG.cb_fits_l1 = lambda p, **kw: True
    try:
        n_offered = len(TS.fused_pairs(S, H, D, T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1],
                                       bias.dtype))
        got = ttnn.to_torch(T._tri_att_sdpa_at(q, k, v, bias, scale))
        # a second call, to prove the memo makes it decline instead of re-throwing
        got2 = ttnn.to_torch(T._tri_att_sdpa_at(q, k, v, bias, scale))
    finally:
        SG.cb_fits_l1 = real_fits
        TS.fused_pairs.cache_clear()

    res.update(
        pairs_offered_with_the_model_lying=n_offered,
        pm_over_l1=sorted(str(x) for x in TS._PM_OVER_L1),
        pm_l1_errors={str(kk): str(vv).splitlines()[0][:220]
                      for kk, vv in TS.PM_L1_ERRORS.items()},
        on_pick={str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()},
        bit_identical_to_off=bool(torch.equal(ref, got)),
        second_call_bit_identical=bool(torch.equal(ref, got2)),
        max_abs_diff=float((ref.float() - got.float()).abs().max()),
        triatt_served=TS.STATS[0], triatt_declined=TS.STATS[1])
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    ok = (res["pm_l1_errors"] and res["bit_identical_to_off"]
          and res["second_call_bit_identical"])
    print("CONTROL:", "PASS" if ok else "FAIL", flush=True)
    ttnn.close_device(dev)
    return 0 if ok else 1


sys.exit(main())
