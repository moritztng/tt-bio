"""spd-bh: triangle attention chunk surface on a Blackhole grid, for the plan perf/lpx_sdpa/bench.py runs.

LPX pins the mask-reuse kernel to (q192, k384), the Wormhole optimum on an 8x8 grid. The kernel puts one
(head, q chunk) on a core and splits the batch over the rest, so the core count it lights up is
(cores // (8 * ceil(23 / q_tiles))) * 8 * ceil(23 / q_tiles): at q192 that is 64 of 64 on Wormhole,
128 of 130 on a p150a and 96 of 110 on a p300c, where q256 / q384 / q736 light 96 / 96 / 104.
This plan measures the surface instead of predicting it, in the two configurations that ship:
normal (bf16 q/k/v/mask, HiFi2) and LPX (bfp8 q/k/v, bf16 mask, bfp8 score CB, LoFi).
usage: plan_triatt.py OUT.json [TA|TT]
"""
import json, sys

out = sys.argv[1]
cls = sys.argv[2] if len(sys.argv) > 2 else "TA"
H = 8 if cls == "TA" else 2
B = S = 736; D = 32
BF = dict(q="bf16", k="bf16", v="bf16", mask="bf16")
LPX = dict(q="bfp8", k="bfp8", v="bfp8", mask="bf16")
g0 = dict(shape={"q": [B, H, S, D], "k": [B, H, S, D], "v": [B, H, S, D], "mask": [1, H, S, S]},
          scale=D ** -0.5, reps=10)
base = dict(impl="stock", qc=256, kc=256, fid="default")
arms = []
for q in (96, 128, 160, 192, 256, 384, 736):
    for k in (256, 384, 736):
        arms.append(dict(impl="fused", qc=q, kc=k, dt=BF, fid="HiFi2", name=f"reuse q{q} k{k} bf16 HiFi2"))
        arms.append(dict(impl="fused", qc=q, kc=k, dt=LPX, fid="LoFi", im="bfp8", name=f"reuse q{q} k{k} lpx"))
groups = [dict(g0, name=f"{cls}.bh{i // 12}", base=base, arms=arms[i:i + 12]) for i in range(0, len(arms), 12)]
json.dump({"groups": groups}, open(out, "w"), indent=1)
print(f"{len(groups)} groups, {len(arms)} arms -> {out}")
