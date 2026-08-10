#!/usr/bin/env python3
"""The bottom-up fold ceiling at any token count, from measured inputs only.

perf/ceiling/ceiling_model.py answers this at 298 aa with every input hard-coded. This is the same
model with the size as an argument, plus the one term that does not survive the move to 512 aa:

  the non-arithmetic floor. At 298 aa a pair tensor is 52.4 MB, an L1<->L1 copy needs 104.9 MB of
  160.8 MB, and pricing all 217 non-arithmetic block ops at the L1 copy roof is legal. At 512 aa
  the tensor is 134.2 MB and the copy needs 268.4 MB, which the chip does not have. So each op is
  priced at the L1 copy roof only if its own live byte set fits L1, and at the measured DRAM copy
  roof otherwise. Both bases are reported: `dram` reproduces the 298 aa model exactly, `fit` is
  the size-aware one and is the headline.

Self-check first, then the new size:

    python3 perf/moonshot512/ceiling_size.py --label "298 aa qb2c3 (self-check)" \
      --census-fold perf/ceiling/census_fold_p298.json \
      --census-block perf/ceiling/census_n320.json \
      --k-sweep perf/ceiling/k_sweep_qb2c3.json --roofs perf/ceiling/roof_by_ckc_qb2c3.json \
      --floors perf/ceiling/floors_qb2c3.json --legacy-shape-roof perf/ceiling/shape_roof_qb2c3.json \
      --l1-copy-roof-GBs 1152.2 --dram-copy-roof-GBs 317.2 \
      --read-roof-GBs 392.2 --write-roof-GBs 266.3 --l1-total-MB 160.8 \
      --pf-blocks 480 --diffusion-calls 202116 --fold-s 34.19 --out /tmp/ceiling_298_check.json
"""
import argparse
import json
import math
import os

c32 = lambda x: int(math.ceil(x / 32) * 32)
DT = {"BFLOAT16": 2, "FLOAT32": 4, "BFLOAT8_B": 1, "BFLOAT4_B": 1, "UINT32": 4, "INT32": 4,
      "UINT16": 2, "UINT8": 1}


def dims(s):
    return [int(x) for x in s.split("x")]


def pad_key(a, b):
    A, B = dims(a), dims(b)
    if len(B) == 2 and A[-1] == B[0]:
        f = (c32(A[-1]) / A[-1]) * (c32(B[1]) / B[1])
        if len(A) >= 2:
            f *= c32(A[-2]) / A[-2]
        return f, c32(A[-1]), c32(B[1]), tuple(A[:-2] + [c32(A[-2]), c32(A[-1])]), tuple([c32(B[0]), c32(B[1])])
    if len(B) > 2 and A[-1] == B[-2]:
        f = (c32(A[-1]) / A[-1]) * (c32(B[-1]) / B[-1]) * (c32(A[-2]) / A[-2])
        return f, c32(A[-1]), c32(B[-1]), tuple(A[:-2] + [c32(A[-2]), c32(A[-1])]), tuple(B[:-2] + [c32(B[-2]), c32(B[-1])])
    f = (c32(A[-1]) / A[-1]) * (c32(A[-2]) / A[-2]) if len(A) >= 2 else 1.0
    return f, c32(A[-1]), c32(A[-1]), tuple(A), tuple(dims(b))


def tensor_bytes(spec):
    """'1x512x512x256:BFLOAT16:DRAM' -> bytes, padding the two tiled dims."""
    parts = spec.split(":")
    d = dims(parts[0])
    if len(d) >= 2:
        d = d[:-2] + [c32(d[-2]), c32(d[-1])]
    n = 1
    for x in d:
        n *= x
    return n * DT.get(parts[1].upper() if len(parts) > 1 else "BFLOAT16", 2)


def build_rate(ks, square):
    KS = sorted((int(k), v["tflops"]) for k, v in ks["k_sweep_dram"].items())
    NS = sorted((int(k), v["tflops"]) for k, v in ks["n_sweep_dram"].items())
    anchor = (dict(KS)[256] + dict(NS)[256]) / 2

    def interp(tab, x):
        if x <= tab[0][0]:
            return tab[0][1] * x / tab[0][0]
        if x >= tab[-1][0]:
            return tab[-1][1]
        for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
            if x0 <= x <= x1:
                t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return y0 + t * (y1 - y0)
    return lambda K, N: min(square, interp(KS, K) * interp(NS, N) / anchor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--census-fold", required=True)
    ap.add_argument("--census-block", required=True)
    ap.add_argument("--k-sweep", required=True)
    ap.add_argument("--roofs", required=True, help="roof_by_ckc output, for the square compute roof")
    ap.add_argument("--floors", required=True, help="floors.py output, for the per-ttnn-call floor")
    ap.add_argument("--shape-roof", default=None, help="shape_roof_census.py output (new format)")
    ap.add_argument("--legacy-shape-roof", default=None, help="perf/ceiling/shape_roof.py output")
    ap.add_argument("--l1-copy-roof-GBs", type=float, required=True)
    ap.add_argument("--dram-copy-roof-GBs", type=float, required=True)
    ap.add_argument("--read-roof-GBs", type=float, required=True)
    ap.add_argument("--write-roof-GBs", type=float, required=True)
    ap.add_argument("--l1-total-MB", type=float, required=True)
    ap.add_argument("--pf-blocks", type=int, required=True, help="Pairformer block executions per fold")
    ap.add_argument("--diffusion-calls", type=int, required=True)
    ap.add_argument("--diff-weight-MB", type=float, default=396.8)
    ap.add_argument("--diff-steps", type=int, default=200)
    ap.add_argument("--fold-s", type=float, required=True, help="measured warm fold, same card")
    ap.add_argument("--pf-stage-s", type=float, default=None, help="measured pf_stack, for the direct questions")
    ap.add_argument("--gpu-s", type=float, default=None, help="the GPU fold the gap is quoted against")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fold, blk = json.load(open(a.census_fold)), json.load(open(a.census_block))
    ks, rk, fl = json.load(open(a.k_sweep)), json.load(open(a.roofs)), json.load(open(a.floors))
    square = rk["runs"]["4096_HiFi4_fp32acc_packer(production)"]["tflops"]
    percall_us = min(fl["per_call_us"].values())
    rate = build_rate(ks, square)

    # ---- measured per-class rates, keyed on the padded (a, b) shape, then on (K, N)
    ovr_shape, ovr_kn, sdpa_tf = {}, {}, {}
    if a.shape_roof:
        sr = json.load(open(a.shape_roof))
        for lbl, v in sr["classes"].items():
            if not v.get("best_tflops"):
                continue
            ovr_shape[(tuple(v["a"]), tuple(v["b"]))] = v["best_tflops"]
            k = (v["K"], v["N"])
            ovr_kn[k] = max(ovr_kn.get(k, 0), v["best_tflops"])
        for lbl, v in sr.get("sdpa", {}).items():
            if v.get("tflops"):
                sdpa_tf[tuple(v["shape"])] = v["tflops"]
    if a.legacy_shape_roof:
        lr = json.load(open(a.legacy_shape_roof))
        for lbl, v in lr["classes"].items():
            body = lbl.split()[1]
            aa, bb = body.split("@")
            K = int(aa.strip("[]").split(",")[-1])
            N = int(bb.strip("[]").split(",")[-1])
            ovr_kn[(K, N)] = max(ovr_kn.get((K, N), 0), v["best_tflops"])
        for k, v in lr.items():
            if k.startswith("sdpa_") and isinstance(v, dict) and "tflops" in v:
                sdpa_tf.setdefault(tuple(int(x) for x in k.split("_")[1:]), v["tflops"])

    # ---- term A: arithmetic, every class at the rate its own shape achieves
    tot_lg = fold["total_flops"]
    tot_ex = t_ideal = t_real = 0.0
    per_stage, priced_by_model = {}, 0.0
    for e in fold["top_matmul_shapes"]:
        if e["op"] == "scaled_dot_product_attention":
            A = dims(e["a"])
            F = e["flops"] * (c32(A[-2]) / A[-2]) ** 2
            key = tuple(A[:-2] + [c32(A[-2]), c32(A[-1])])
            r = sdpa_tf.get(key) or (min(sdpa_tf.values()) if sdpa_tf else rate(A[-1], A[-2]))
            if key not in sdpa_tf:
                priced_by_model += F
        else:
            f, K, N, Ap, Bp = pad_key(e["a"], e["b"])
            F = e["flops"] * f
            r = ovr_shape.get((Ap, Bp)) or ovr_kn.get((K, N))
            if r is None:
                r = rate(K, N)
                priced_by_model += F
        tot_ex += F
        t_ideal += F / (square * 1e12)
        t_real += F / (r * 1e12)
        s = per_stage.setdefault(e["stage"], [0.0, 0.0, 0.0])
        s[0] += F
        s[1] += F / (square * 1e12)
        s[2] += F / (r * 1e12)

    pf_arith = sum(v[2] for k, v in per_stage.items() if k.startswith("pf."))
    other_arith = sum(v[2] for k, v in per_stage.items() if not k.startswith("pf.") and k != "diffusion")

    # ---- term B: non-arithmetic, two bases
    L1B, DRB = a.l1_copy_roof_GBs * 1e9, a.dram_copy_roof_GBs * 1e9
    bk = blk["by_kind"]
    KINDS = ("norm", "move", "eltwise")
    dram_bytes = sum(bk[k]["dram_in"] + bk[k]["dram_out"] for k in KINDS if k in bk)
    n_nonmm = sum(bk[k]["n"] for k in KINDS if k in bk)
    t_dram_basis_blk = dram_bytes / L1B

    l1_cap = a.l1_total_MB * 1e6
    fit_b = nofit_b = 0.0
    n_fit = n_nofit = 0
    for op in blk["ops"]:
        if op.get("kind") not in KINDS:
            continue
        live = sum(tensor_bytes(s) for s in op.get("in", [])) + sum(tensor_bytes(s) for s in op.get("out", []))
        if live == 0:
            continue
        if live <= l1_cap:
            fit_b += live
            n_fit += 1
        else:
            nofit_b += live
            n_nofit += 1
    t_fit_basis_blk = fit_b / L1B + nofit_b / DRB
    t_all_basis_blk = (fit_b + nofit_b) / L1B      # all bytes, but the L1 roof granted everywhere

    t_nonmm_dram = t_dram_basis_blk * a.pf_blocks
    t_nonmm_fit = t_fit_basis_blk * a.pf_blocks
    t_nonmm_all = t_all_basis_blk * a.pf_blocks

    # ---- term C: diffusion floor
    t_disp = a.diffusion_calls * percall_us / 1e6
    t_wstream = a.diff_weight_MB * a.diff_steps / 1e3 / a.read_roof_GBs
    t_diff_arith = per_stage.get("diffusion", [0, 0, 0])[2]
    t_diff = max(t_diff_arith, t_wstream, t_disp)

    out = {"label": a.label, "square_roof_tflops": square, "percall_us": percall_us,
           "exec_tflop": round(tot_ex / 1e12, 2), "logical_tflop": round(tot_lg / 1e12, 2),
           "padding_pct": round(100 * (tot_ex / tot_lg - 1), 2),
           "priced_by_rate_model_frac": round(priced_by_model / tot_ex, 4),
           "flop_weighted_achievable_tflops": round(tot_ex / t_real / 1e12, 2),
           "arith_ideal_s": round(t_ideal, 3), "arith_at_achievable_s": round(t_real, 3),
           "pf_arith_s": round(pf_arith, 3), "other_arith_s": round(other_arith, 3),
           "nonarith": {"calls": n_nonmm,
                        "dram_basis_MB_per_block": round(dram_bytes / 1e6, 1),
                        "dram_basis_s": round(t_nonmm_dram, 3),
                        "fit_basis_calls_l1": n_fit, "fit_basis_calls_dram": n_nofit,
                        "fit_basis_MB_l1": round(fit_b / 1e6, 1),
                        "fit_basis_MB_dram": round(nofit_b / 1e6, 1),
                        "fit_basis_s": round(t_nonmm_fit, 3),
                        "allbytes_basis_MB_per_block": round((fit_b + nofit_b) / 1e6, 1),
                        "allbytes_basis_s": round(t_nonmm_all, 3),
                        "l1_fit_penalty_s": round(t_nonmm_fit - t_nonmm_all, 3)},
           "diffusion": {"arith_s": round(t_diff_arith, 3), "weight_stream_s": round(t_wstream, 3),
                         "dispatch_s": round(t_disp, 3), "floor_s": round(t_diff, 3)},
           "per_stage": {k: {"exec_tflop": round(v[0] / 1e12, 2), "ideal_ms": round(v[1] * 1e3),
                             "ceil_ms": round(v[2] * 1e3),
                             "eff_tflops": round(v[0] / v[2] / 1e12, 1)} for k, v in
                         sorted(per_stage.items(), key=lambda kv: -kv[1][0])}}

    for basis, t_non in (("dram", t_nonmm_dram), ("allbytes", t_nonmm_all), ("fit", t_nonmm_fit)):
        ceil_s = pf_arith + t_non + t_diff + other_arith
        d = {"ceiling_s": round(ceil_s, 3), "gap_at_ceiling_x": round(a.fold_s / ceil_s, 3),
             "block_ceiling_ms": round((pf_arith / a.pf_blocks + t_non / a.pf_blocks) * 1e3, 3)}
        if a.gpu_s:
            d["gpu_gap_today_x"] = round(a.fold_s / a.gpu_s, 3)
            d["gpu_gap_at_ceiling_x"] = round(ceil_s / a.gpu_s, 3)
        if a.pf_stage_s:
            rest = a.fold_s - a.pf_stage_s
            d["fold_with_pf_at_arith_floor_s"] = round(rest + pf_arith, 3)
            d["fold_with_pf_zero_s"] = round(rest, 3)
            if a.gpu_s:
                d["gpu_gap_with_perfect_pairformer_x"] = round((rest + pf_arith) / a.gpu_s, 3)
                d["gpu_gap_with_pf_zero_x"] = round(rest / a.gpu_s, 3)
                target = 4.0 * a.gpu_s
                d["fold_for_4x_s"] = round(target, 3)
                d["pf_budget_for_4x_s"] = round(target - rest, 3)
                pf_ex = sum(v[0] for k, v in per_stage.items() if k.startswith("pf."))
                if target - rest > 0:
                    d["required_tflops_for_4x_pf_only"] = round(pf_ex / (target - rest) / 1e12, 2)
                d["pf_exec_tflop"] = round(pf_ex / 1e12, 2)
        out[f"basis_{basis}"] = d

    print(f"\n=== {a.label} ===", flush=True)
    print(f"square roof {square} TFLOP/s   per-call floor {percall_us:.2f} us   "
          f"L1 copy {a.l1_copy_roof_GBs} / DRAM copy {a.dram_copy_roof_GBs} GB/s rw", flush=True)
    print(f"executed {out['exec_tflop']} TFLOP (+{out['padding_pct']}% padding), FLOP-weighted "
          f"achievable {out['flop_weighted_achievable_tflops']} TFLOP/s, "
          f"{100*out['priced_by_rate_model_frac']:.0f}% priced by the rate model", flush=True)
    print(f"{'stage':22s} {'execTF':>8s} {'ideal ms':>9s} {'ceil ms':>9s} {'eff TF/s':>9s}", flush=True)
    for k, v in out["per_stage"].items():
        print(f"{k:22s} {v['exec_tflop']:8.1f} {v['ideal_ms']:9d} {v['ceil_ms']:9d} {v['eff_tflops']:9.1f}", flush=True)
    nb = out["nonarith"]
    print(f"\nnon-arithmetic {nb['calls']} calls/block", flush=True)
    print(f"  dram basis (298 aa model): {nb['dram_basis_MB_per_block']} MB at the L1 copy roof "
          f"-> {nb['dram_basis_s']} s over {a.pf_blocks} blocks", flush=True)
    print(f"  allbytes basis: {nb['allbytes_basis_MB_per_block']} MB of all traffic at the L1 copy "
          f"roof -> {nb['allbytes_basis_s']} s", flush=True)
    print(f"  fit  basis: {nb['fit_basis_calls_l1']} calls / {nb['fit_basis_MB_l1']} MB fit L1, "
          f"{nb['fit_basis_calls_dram']} calls / {nb['fit_basis_MB_dram']} MB do not "
          f"-> {nb['fit_basis_s']} s (L1-fit penalty {nb['l1_fit_penalty_s']} s)", flush=True)
    df = out["diffusion"]
    print(f"diffusion floor {df['floor_s']} s = max(arith {df['arith_s']}, weights "
          f"{df['weight_stream_s']}, dispatch {df['dispatch_s']})", flush=True)
    for basis in ("dram", "fit"):
        d = out[f"basis_{basis}"]
        print(f"\n-- basis={basis} --", flush=True)
        print(f"  ceiling {d['ceiling_s']} s vs measured {a.fold_s} s = {d['gap_at_ceiling_x']}x headroom", flush=True)
        print(f"  block ceiling {d['block_ceiling_ms']} ms", flush=True)
        for k in ("gpu_gap_today_x", "gpu_gap_at_ceiling_x", "gpu_gap_with_perfect_pairformer_x",
                  "gpu_gap_with_pf_zero_x", "fold_for_4x_s", "pf_budget_for_4x_s",
                  "required_tflops_for_4x_pf_only", "pf_exec_tflop"):
            if k in d:
                print(f"  {k:36s} {d[k]}", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
