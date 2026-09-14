"""An L1 refusal on the above-cap route must cost speed, never correctness.

`fused_pairs` only offers pairs the CB model says fit, so on a healthy tree the device refusal
branch in `triatt_sdpa.sdpa` is never taken and "the fallback is fine" would be an assertion about
dead code. Three arms force it, at a padded length above the cap where the route is live:

  A  route off. The stock ladder, byte for byte what shipped before this row.
  B  route on with `cb_fits_l1` stubbed to True, so every pair is offered including the ones that
     overflow. tt-metal throws, `sdpa` absorbs it, memoises the config in `_PM_OVER_L1` and
     declines; the walk continues and lands on a pair that runs. B is NOT expected to be
     bit-identical to A -- landing on a fused pair is the whole point of the route -- so it is
     judged on accuracy against an fp32 reference instead.
  C  route on with every pair `fused_pairs` offers pre-seeded into `_PM_OVER_L1`, so the fused
     route is exhausted and the call falls through to the same stock ladder A took. C MUST be
     bit-identical to A: that is the claim "an L1 refusal means no speedup here, not a wrong
     answer".

The reference is an fp32 torch evaluation of the SAME bf16 operands, so only kernel error is left.
It is taken over the first `--ref-rows` batch rows; the full [64, 4, 1536, 1536] score tensor is
2.4 GB in fp32 and the rows are independent.

    TT_VISIBLE_DEVICES=0 python3 perf/ttx_a3/l1_refusal_control.py --seq 1536 \
        --out perf/ttx_a3/l1_refusal_1536.json
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
    ap.add_argument("--ref-rows", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S, H, D, B = a.seq, a.heads, a.head_dim, a.batch
    dev = T.get_device()
    torch.manual_seed(0)
    scale = D ** -0.5

    def mk(shape):
        t = (torch.randn(shape, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        return t, ttnn.from_torch(t, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    qh, q = mk([B, H, S, D])
    kh, k = mk([B, H, S, D])
    vh, v = mk([B, H, S, D])
    bh, bias = mk([1, H, S, S])
    ttnn.synchronize_device(dev)

    # fp32 reference on the bf16 operands, row by row
    R = min(a.ref_rows, B)
    ref32 = torch.empty([R, H, S, D], dtype=torch.float32)
    b32 = bh[0].float()
    for i in range(R):
        s = (qh[i].float() @ kh[i].float().transpose(-1, -2)) * scale + b32
        ref32[i] = torch.softmax(s, dim=-1) @ vh[i].float()
    rn = ref32.norm().item()

    def rel_rms(got):
        return float((got[:R].float() - ref32).norm().item() / rn)

    res = {"seq": S, "heads": H, "head_dim": D, "batch": B, "ref_rows": R,
           "grid": list(T.COMPUTE_GRID_MAIN), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "commit": os.popen("git rev-parse HEAD").read().strip(),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def run(tag):
        T.SDPA_CHUNK_PICKS.clear()
        TS.STATS[:] = [0, 0]
        out = ttnn.to_torch(T._tri_att_sdpa_at(q, k, v, bias, scale))
        res[tag + "_pick"] = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}
        res[tag + "_served"] = TS.STATS[0]
        res[tag + "_declined"] = TS.STATS[1]
        res[tag + "_rel_rms_vs_fp32"] = round(rel_rms(out), 8)
        return out

    # A: the route off
    T._SDPA_FUSED_LARGE_S = False
    ref = run("off")

    # B: the route on, with the L1 model lying so every pair is offered and the overflows throw
    TS._PM_OVER_L1.clear()
    TS.PM_L1_ERRORS.clear()
    T._SDPA_FUSED_LARGE_S = True
    TS.fused_pairs.cache_clear()
    cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    real_fits = SG.cb_fits_l1
    SG.cb_fits_l1 = lambda p, **kw: True
    try:
        res["pairs_offered_with_the_model_lying"] = len(
            TS.fused_pairs(S, H, D, cores, bias.dtype))
        got = run("lying")
        got2 = run("lying_again")   # the memo must make it decline, not re-throw
    finally:
        SG.cb_fits_l1 = real_fits
        TS.fused_pairs.cache_clear()
    res["throws_absorbed"] = len(TS.PM_L1_ERRORS)
    res["pm_l1_errors"] = {str(kk): str(vv).splitlines()[0][:160]
                           for kk, vv in sorted(TS.PM_L1_ERRORS.items(), key=str)[:4]}
    res["lying_bit_identical_to_off"] = bool(torch.equal(ref, got))
    res["lying_max_abs_diff_vs_off"] = float((ref.float() - got.float()).abs().max())
    res["lying_second_call_matches_first"] = bool(torch.equal(got, got2))

    # C: every fused pair pre-refused, so the route is exhausted and falls through to A's ladder
    TS._PM_OVER_L1.clear()
    TS.PM_L1_ERRORS.clear()
    TS.fused_pairs.cache_clear()
    pairs = TS.fused_pairs(S, H, D, cores, bias.dtype)
    for qc, kc in pairs:
        TS._PM_OVER_L1.add((S, S, qc, kc, 2))
    res["pairs_offered_honestly"] = len(pairs)
    exhausted = run("exhausted")
    res["exhausted_bit_identical_to_off"] = bool(torch.equal(ref, exhausted))
    res["exhausted_max_abs_diff_vs_off"] = float((ref.float() - exhausted.float()).abs().max())
    TS._PM_OVER_L1.clear()
    TS.fused_pairs.cache_clear()

    ok = (res["throws_absorbed"] > 0
          and res["exhausted_bit_identical_to_off"]
          and res["lying_second_call_matches_first"]
          and res["lying_rel_rms_vs_fp32"] <= res["off_rel_rms_vs_fp32"])
    res["control"] = "PASS" if ok else "FAIL"
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    print("CONTROL:", res["control"], flush=True)
    ttnn.close_device(dev)
    return 0 if ok else 1


sys.exit(main())
