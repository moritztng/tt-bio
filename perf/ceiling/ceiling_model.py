#!/usr/bin/env python3
"""Bottom-up ceiling for a 298 aa protenix-v2 fold on one Blackhole p150a (qb2 card 3).

Every input is measured and committed next to this file. Nothing is inherited.
  census_fold_p298.json    every matmul/SDPA of one real fold, real shapes, real call counts
  census_n320.json         one real PairformerLayer at N=320: per-op-kind calls, bytes, time
  roof_by_ckc_qb2c3.json   square compute roof under the production kernel config
  k_sweep_qb2c3.json       achieved TFLOP/s vs K and vs N, in DRAM and in L1; L1 copy roof
  shape_roof_qb2c3.json    achieved TFLOP/s at the 8 classes carrying 87% of the arithmetic
  floors_qb2c3.json        per-ttnn-call floor; DRAM->DRAM roof on a ladder that flattens
  weights_protenix_v2.json parameter counts per stack, off protenix-v2.pt
"""
import json
import math
import os

D = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(D, n)))

fold, n320 = J("census_fold_p298.json"), J("census_n320.json")
ckc, ks, sr, fl = J("roof_by_ckc_qb2c3.json"), J("k_sweep_qb2c3.json"), J("shape_roof_qb2c3.json"), J("floors_qb2c3.json")

SQUARE = ckc["runs"]["4096_HiFi4_fp32acc_packer(production)"]["tflops"]
L1_COPY = max(v["GBs_rw"] for v in ks["l1_copy_roof"].values())
READ_ROOF, WRITE_ROOF = 392.2, 266.3
PERCALL_US = min(fl["per_call_us"].values())
c32 = lambda x: int(math.ceil(x / 32) * 32)

# ---------------------------------------------------------------- separable rate model
# Two measured sweeps share the point (K=256, N=256). A separable model
#   rate(K,N) = rK(K) * rN(N) / rK(256)   with rK(256) == rN(256) by construction
# reproduces both sweeps exactly and needs no fitting. Log-linear interpolation between
# measured points, flat outside, capped at the square roof.
KS = sorted((int(k), v["tflops"]) for k, v in ks["k_sweep_dram"].items())
NS = sorted((int(k), v["tflops"]) for k, v in ks["n_sweep_dram"].items())
ANCHOR = (dict(KS)[256] + dict(NS)[256]) / 2


def interp(tab, x):
    if x <= tab[0][0]:
        return tab[0][1] * x / tab[0][0]  # rate falls ~linearly toward 0 below the smallest K
    if x >= tab[-1][0]:
        return tab[-1][1]
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x0 <= x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)


rate = lambda K, N: min(SQUARE, interp(KS, K) * interp(NS, N) / ANCHOR)
print(f"rate model anchored at (256,256) = {ANCHOR:.1f} TFLOP/s; rate(4096,4096) = {rate(4096,4096):.1f} "
      f"vs square roof {SQUARE}")
for lbl, v in sr["classes"].items():
    print(f"  measured {lbl:46s} best {v['best_tflops']:6.2f} TFLOP/s")
SDPA_TF = sr["sdpa_320_8_320_32"]["tflops"]
print(f"  measured SDPA [320,8,320,32] q=k=320  {SDPA_TF} TFLOP/s")
print()

# measured overrides, keyed by (K, N) of the class
OVR = {}
for lbl, v in sr["classes"].items():
    body = lbl.split()[1]
    a, b = body.split("@")
    K = int(a.strip("[]").split(",")[-1])
    N = int(b.strip("[]").split(",")[-1])
    OVR[(K, N)] = max(OVR.get((K, N), 0), v["best_tflops"])

# ---------------------------------------------------------------- executed FLOPs per class
def dims(a, b):
    A = [int(x) for x in a.split("x")]
    B = [int(x) for x in b.split("x")]
    return A, B


def pad_factor_and_shape(a, b):
    """(executed / logical) FLOP ratio from tile padding, plus the (K, N) the rate model needs."""
    A, B = dims(a, b)
    if len(B) == 2 and A[-1] == B[0]:
        K, N = A[-1], B[1]
        f = (c32(K) / K) * (c32(N) / N)
        if len(A) >= 2:
            f *= c32(A[-2]) / A[-2]
        return f, c32(K), c32(N)
    if len(B) > 2 and A[-1] == B[-2]:  # batched
        K, N, M = A[-1], B[-1], A[-2]
        return (c32(K) / K) * (c32(N) / N) * (c32(M) / M), c32(K), c32(N)
    # transposed / non-standard contraction (a few tiny classes); pad the two tiled dims of a
    f = (c32(A[-1]) / A[-1]) * (c32(A[-2]) / A[-2]) if len(A) >= 2 else 1.0
    return f, c32(A[-1]), c32(A[-1])


tot_logical = fold["total_flops"]
tot_exec = 0.0
t_ideal = t_real = 0.0
per_stage = {}
unmatched = 0.0
for e in fold["top_matmul_shapes"]:
    lg = e["flops"]
    if e["op"] == "scaled_dot_product_attention":
        A, _ = dims(e["a"], e["b"])
        f = (c32(A[-2]) / A[-2]) ** 2  # both q and k sequence dims pad
        F = lg * f
        r = SDPA_TF
    else:
        f, K, N = pad_factor_and_shape(e["a"], e["b"])
        F = lg * f
        if (K, N) in OVR:
            r = OVR[(K, N)]
        else:
            r = rate(K, N)
            unmatched += F
    tot_exec += F
    t_ideal += F / (SQUARE * 1e12)
    t_real += F / (r * 1e12)
    s = per_stage.setdefault(e["stage"], [0.0, 0.0, 0.0])
    s[0] += F
    s[1] += F / (SQUARE * 1e12)
    s[2] += F / (r * 1e12)

print(f"logical FLOPs  {tot_logical/1e12:7.1f} TFLOP")
print(f"executed       {tot_exec/1e12:7.1f} TFLOP  = +{100*(tot_exec/tot_logical-1):.1f}% tile padding (298 -> 320)")
print(f"priced by the rate model rather than a direct measurement: {100*unmatched/tot_exec:.0f}% of executed FLOPs")
print()
print(f"{'stage':20s} {'execTFLOP':>10s} {'ideal ms':>9s} {'ceil ms':>9s} {'eff TF/s':>9s}")
for k in sorted(per_stage, key=lambda x: -per_stage[x][0]):
    F, ti, tr = per_stage[k]
    print(f"{k:20s} {F/1e12:10.1f} {ti*1e3:9.0f} {tr*1e3:9.0f} {F/tr/1e12:9.1f}")
print(f"{'TOTAL arithmetic':20s} {tot_exec/1e12:10.1f} {t_ideal*1e3:9.0f} {t_real*1e3:9.0f} {tot_exec/t_real/1e12:9.1f}")
print()

# ---------------------------------------------------------------- non-arithmetic floor
bk = n320["by_kind"]
PF_BLOCKS = 480  # 48 x 10 recycles
nonmm_bytes = sum(bk[k]["dram_in"] + bk[k]["dram_out"] for k in ("norm", "move", "eltwise"))
nonmm_calls = sum(bk[k]["n"] for k in ("norm", "move", "eltwise"))
t_nonmm_blk = nonmm_bytes / (L1_COPY * 1e9)
t_nonmm = t_nonmm_blk * PF_BLOCKS
print(f"Pairformer block N=320: measured {n320['block_ms_median']:.2f} ms, {n320['total_flops']/1e9:.1f} GFLOP, "
      f"{n320['n_calls']} ttnn calls")
print(f"  arithmetic ceiling for one block {t_real*1e3*sum(per_stage[s][2] for s in per_stage if s.startswith('pf.'))/t_real/PF_BLOCKS:.2f} ms")
print(f"  non-arithmetic: {nonmm_calls} calls moving {nonmm_bytes/1e6:.0f} MB; at the measured L1 copy roof "
      f"{L1_COPY:.0f} GB/s = {t_nonmm_blk*1e3:.2f} ms/block -> {t_nonmm:.2f} s over {PF_BLOCKS} blocks")
pf_arith = sum(per_stage[s][2] for s in per_stage if s.startswith("pf."))
print(f"  block ceiling = {pf_arith/PF_BLOCKS*1e3:.2f} + {t_nonmm_blk*1e3:.2f} = "
      f"{(pf_arith/PF_BLOCKS + t_nonmm_blk)*1e3:.2f} ms vs {n320['block_ms_median']:.2f} measured = "
      f"{n320['block_ms_median']/((pf_arith/PF_BLOCKS + t_nonmm_blk)*1e3):.2f}x")
print()

# ---------------------------------------------------------------- dispatch floor
CALLS = {"diffusion": 202116, "pf": 5145 + 101784 + 10872 + 36292 + 74944 + 3872, "other": 417 + 7}
print(f"per-ttnn-call floor measured at {PERCALL_US:.2f} us (smallest of "
      f"{ {k: v for k, v in fl['per_call_us'].items()} })")
for k, n in CALLS.items():
    print(f"  {k:10s} {n:7d} calls -> {n*PERCALL_US/1e6:6.2f} s eager dispatch floor")
t_disp_diff = CALLS["diffusion"] * PERCALL_US / 1e6
print()

# ---------------------------------------------------------------- essential DRAM
W = J("weights_protenix_v2.json") if os.path.exists(os.path.join(D, "weights_protenix_v2.json")) else None
PF_W_MB, MSA_W_MB, CONF_W_MB, DIFF_W_MB = 9.498, 5.381, 9.498, 396.8
ess = {
    "pairformer weights x10 recycles": 48 * PF_W_MB * 10 / 1e3,
    "msa weights x10 recycles": 4 * MSA_W_MB * 10 / 1e3,
    "confidence weights": 4 * CONF_W_MB / 1e3,
    "diffusion transformer weights x200 steps": DIFF_W_MB * 200 / 1e3,
}
ess_tot = sum(ess.values())
PAIR_MB = 320 * 320 * 256 * 2 / 1e6
z_rt = 484 * PAIR_MB * 2 / 1e3
print("essential DRAM bytes an optimal implementation still has to move:")
for k, v in ess.items():
    print(f"  {k:44s} {v:8.2f} GB")
print(f"  {'TOTAL weights':44s} {ess_tot:8.2f} GB -> {ess_tot/READ_ROOF*1e3:.0f} ms of reads at {READ_ROOF} GB/s")
print(f"  {'+ z round-tripping DRAM once per block (upper)':44s} {z_rt:8.2f} GB -> "
      f"{(z_rt/2/READ_ROOF + z_rt/2/WRITE_ROOF)*1e3:.0f} ms")
today_mem = fold["total_dram_in"] / 1e9 / READ_ROOF + fold["total_dram_out"] / 1e9 / WRITE_ROOF
print(f"  today: {fold['total_dram_in']/1e9:.0f} GB read + {fold['total_dram_out']/1e9:.0f} GB write = "
      f"{today_mem*1e3:.0f} ms, i.e. {(fold['total_dram_in']+fold['total_dram_out'])/1e9/ess_tot:.0f}x the essential bytes")
print()
diff_ai = fold["by_stage"]["diffusion"]["flops"] / (DIFF_W_MB * 200 * 1e6)
print(f"diffusion arithmetic intensity against its own weight stream: "
      f"{diff_ai:.0f} FLOP/byte vs machine balance {SQUARE*1e12/(READ_ROOF*1e9):.0f} -> "
      f"{'MEMORY' if diff_ai < SQUARE*1e12/(READ_ROOF*1e9) else 'COMPUTE'}-bound at B=1")
print()

# ---------------------------------------------------------------- verdict
t_diff_floor = max(per_stage["diffusion"][2], DIFF_W_MB * 200 / 1e3 / READ_ROOF, t_disp_diff)
t_ceiling = pf_arith + t_nonmm + t_diff_floor + sum(
    per_stage[s][2] for s in per_stage if not s.startswith("pf.") and s != "diffusion")
print(f"IDEAL   (all executed FLOPs at the {SQUARE} TFLOP/s square roof, nothing else counted): "
      f"{t_ideal:.2f} s")
print(f"CEILING (each op at the rate its own shape achieves, non-arithmetic at the L1 copy roof, "
      f"diffusion at max(arith, weight stream, dispatch)): {t_ceiling:.2f} s")
print(f"  pairformer arithmetic {pf_arith:.2f} + pairformer non-arithmetic {t_nonmm:.2f} + "
      f"diffusion {t_diff_floor:.2f} + rest {sum(per_stage[s][2] for s in per_stage if not s.startswith('pf.') and s != 'diffusion'):.2f}")
print(f"MEMORY  floor: {ess_tot/READ_ROOF:.2f} s of essential DRAM reads")
print(f"  compute floor / memory floor = {t_ideal/(ess_tot/READ_ROOF):.0f}x -> COMPUTE-BOUND")
