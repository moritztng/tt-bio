#!/usr/bin/env python3
"""r4t-ceiling step 1 (analytic): FLOPs, DRAM bytes and Blackhole roofline per op class for one RFD4
training step (crop 256, batch 1 x exploration_k 4, depth 2, attn_fp32 on, Muon+Adam over 178,448,304 params).

Hardware peak citations (knowledgebase/knowledge/):
  - GDDR6 DRAM bandwidth: 512 GB/s per BH chip (p150a, p300c, BH Galaxy = 32 x 512 = 16 TB/s):
      knowledge/infra/machines.md:150, knowledge/marketing/claims.md:60-62,233.
  - Tensix cores: p150a 140 physical (130 worker grid), p300c / BH Galaxy 120 physical (110 worker grid
      in ttnn on qb2 p300c, verified in ops.csv AVAILABLE WORKER CORE COUNT = 110; 120 on BH Galaxy):
      knowledge/infra/machines.md:150, knowledge/marketing/claims.md:60-62,233.
  - FPU rate per Tensix @ 1.35 GHz: 774 FP8 TFLOPS / (140 Tensix x 1.35 GHz x 2) = 2048 MAC/cycle LoFi
      (= 4096 FLOP/cycle); HiFi2 = 1/2, HiFi4 = 1/4, split-bf16 fp32 (3 HiFi4 passes) = 1/12:
      knowledge/infra/machines.md:150.
  - Measured ttnn single-chip achievable roofs @ 1.35 GHz (cited beside spec peaks):
      p150a: 100.55 TFLOP/s HiFi4, 140.35 HiFi2, 161.14 LoFi, 435.2 GB/s DRAM (PLAYBOOKS.md:443, state/rfd4/ARCH.md:107-111).
      qb2 p300c: 115.685 TFLOP/s dense matmul, 440 GB/s DRAM (state/of3t/BACKWARD.md:243).

Usage:
  python3 perf/r4t_ceiling/roofline.py perf/r4t_ceiling/out/count_batch00.json \
      [--iso perf/r4t_ceiling/out/iso_batch00.json] [--cores 110] [--fidelity 4]
"""
import collections
import json
import re
import sys

# Primary chip = qb2 p300c (110 worker grid, 1.35 GHz, 512 GB/s DRAM).
CORES_P300C, CORES_GALAXY, CORES_P150A = 110, 120, 140
CLK = 1.35e9
BW = 512e9
BW_MEAS_P300C = 440e9
HIFI4_MEAS_P300C = 115.685e12
BW_MEAS_P150A = 435.2e9
HIFI4_MEAS_P150A = 100.55e12

# Exact list-price inputs (G4 PROTOCOL CARD, state/r4t/rows/perf.md:18,26 & ~/r4t/sizing/proj_v1.py):
# BH Galaxy $160,000 / 32 chips vs DGX B200 $562,154 / 8 GPUs.
PRICE_RATIO = (32 / 160000.0) / (8 / 562154.0)   # 14.05385x
GPU_DEPTH2_S = 0.175

# Exact upstream opt.muon_exclude_param_patterns (tt_bio/train/rfd4_objective.py:343-344).
MUON_EXCLUDE = (
    r"_factory\.",
    r"^(nn|glue)\.(local_latents_linear|ca_linear|insertion_q_proj|insertion_k_proj)\.",
)


def peak(fid: int = 4, cores: int = CORES_P300C) -> float:
    """Matmul peak in FLOP/s for `cores` Tensix @ 1.35 GHz at MathFidelity `fid` (1=LoFi, 2=HiFi2, 4=HiFi4)."""
    return cores * 2048 * 2 * CLK / fid


def split_bf16_fp32_peak(cores: int = CORES_P300C) -> float:
    """Split-bf16 fp32 matmul peak = 3 HiFi4 passes (D34/D89 exact fp32 Muon NS & pair constant)."""
    return peak(4, cores) / 3.0


def optimizer_work(shapes, ns_iters: int = 5, fp32_passes: int = 3):
    """Exact 282-Muon / 276-Adam split over trained parameters (178,448,304 elems).

    Returns dict with:
      - n_total, n_muon, n_adam, elems_total, elems_muon, elems_adam
      - muon_ns_logical_flops, muon_ns_hifi4_flops (x3 split-bf16), muon_ns_bytes
      - eltwise_bytes: fp32 state traffic:
          Muon params (p, g, m, ema): clip-norm read g (1) + read p,g,m,ema (4) + write p,m,ema (3) = 8 x 4 B
          Adam params (p, g, m, v, ema): clip-norm read g (1) + read p,g,m,v,ema (5) + write p,m,v,ema (4) = 10 x 4 B
      - eltwise_flops: ~30 flops/param
    """
    n_muon = n_adam = elems_muon = elems_adam = 0
    ns_fl = ns_by = 0
    for shp, nm in shapes:
        k = 1
        for s in shp:
            k *= s
        if len(shp) >= 2 and not any(re.search(p, nm) for p in MUON_EXCLUDE):
            n_muon += 1
            elems_muon += k
            a, b = shp[0], k // shp[0]
            mm, nn = min(a, b), max(a, b)
            # 5 quintic NS iters: A = X X^T (2 m^2 n), B = bA + cA^2 (2 m^3), X = aX + BX (2 m^2 n)
            ns_fl += ns_iters * (4 * mm * mm * nn + 2 * mm ** 3)
            # Per iter fp32 matrix traffic: (3 mn + 5 m^2) x 4 B
            ns_by += ns_iters * (3 * mm * nn + 5 * mm * mm) * 4
        else:
            n_adam += 1
            elems_adam += k
    tot = elems_muon + elems_adam
    elt_by = elems_muon * 8 * 4 + elems_adam * 10 * 4
    return dict(
        n_total=n_muon + n_adam, n_muon=n_muon, n_adam=n_adam,
        elems_total=tot, elems_muon=elems_muon, elems_adam=elems_adam,
        muon_ns_logical_flops=ns_fl,
        muon_ns_hifi4_flops=ns_fl * fp32_passes,
        muon_ns_bytes=ns_by,
        eltwise_flops=tot * 30,
        eltwise_bytes=elt_by,
    )


def device_folds_work(elems_total: int = 178_448_304, trunk_elems: int = 175_073_664,
                      row_scale_elems: int = 16_500_000):
    """On-card DeviceFolds fold VJP + split rebuild bytes (audited on all 529 trunk params by r4t-perf).

    - fold_vjp: 178.45M fp32 grad transpose/slice/row-scale-fold + accumulate into master grad buffer
      (3 x 4 B/elem = 2.141 GB conservative upper bound).
    - split_rebuild: read 175.07M fp32 masters (700.3 MB) + write 175.07M bf16 trunk weights (350.1 MB)
      + 16.5M row-scale folds (read/write 132 MB) + K14 serving caches (_t1_w, tr_ws, bias_wts ~350 MB bf16)
      = 1.533 GB DRAM.
    """
    fold_vjp_bytes = elems_total * 3 * 4
    rebuild_bytes = trunk_elems * (4 + 2) + row_scale_elems * 8 + trunk_elems * 2
    return dict(fold_vjp_bytes=fold_vjp_bytes, rebuild_bytes=rebuild_bytes)


def unported_host_work(iso_rows, K: int = 4, depth: int = 2, n: int = 256, c_pair: int = 384):
    """Exact FLOPs and DRAM bytes of host-side components that must move onto the card:
      1. glue_heads (fwd on K candidates + 1 winner, bwd on winner, fp32)
      2. glue_cond_init (init_repr + cond_factory + transition_c_1/2: fwd K+1 x depth, bwd winner, fp32)
      3. pair_const_c1 (C1 card-built candidate pair constant + K=4 split-bf16 norm scales, const_card_probe.py)
      4. attn_fp32 extra (+2 B/elem over bf16 across 24 blocks)
    """
    by_ph = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for r in iso_rows:
        by_ph[r["phase"]][r["class"]][0] += r["flops"]
        by_ph[r["phase"]][r["class"]][1] += r["elems"]

    def scale_phase(ph, mult, bpe):
        mm_fl = mm_by = mem_fl = mem_by = 0
        for cls, (fl, el) in by_ph[ph].items():
            if cls == "matmul":
                mm_fl += fl * mult
                mm_by += el * mult * bpe
            else:
                mem_fl += fl * mult
                mem_by += el * mult * bpe
        return dict(mm_flops=mm_fl, mm_bytes=mm_by, mem_flops=mem_fl, mem_bytes=mem_by)

    hf = scale_phase("iso_glue_heads_fwd", K + 1, 4)
    hb = scale_phase("iso_glue_heads_bwd", 1, 4)
    glue_heads = {k: hf[k] + hb[k] for k in hf}

    cf = scale_phase("iso_glue_cond_init_fwd", (K + 1) * depth, 4)
    cb = scale_phase("iso_glue_cond_init_bwd", depth, 4)
    glue_cond_init = {k: cf[k] + cb[k] for k in cf}

    # C1 candidate pair constant + K=4 split-bf16 norm scales (perf/r4t_perf/const_card_probe.py):
    nn = n * n
    kp = 96
    c1_proj_fl = 3 * (2 * nn * kp * c_pair)
    c1_proj_by = 3 * (nn * kp * 2 + kp * c_pair * 2 + nn * c_pair * 4) + 2 * 3 * nn * c_pair * 4
    c1_scale_fl = 2 * (2 * K * nn * (3 * c_pair) * 32)
    el_k = K * nn * c_pair
    c1_scale_by = el_k * (24 + 46 + 13)   # x^2/xw/(xw)^2 (24 B) + 2x bits3 (46 B) + 2x rowsum matmul (13 B)
    pair_const_c1 = dict(mm_flops=c1_proj_fl + c1_scale_fl, mm_bytes=c1_proj_by,
                         mem_flops=el_k * 20, mem_bytes=c1_scale_by)

    attn_fwd_el = sum(v[1] for v in by_ph["iso_attn_fp32_1blk_fwd"].values())
    attn_bwd_el = sum(v[1] for v in by_ph["iso_attn_fp32_1blk_bwd"].values())
    attn_fp32_extra = dict(
        cand_k_bytes=K * depth * 24 * attn_fwd_el * 2,
        win_fwd_bytes=1 * depth * 24 * attn_fwd_el * 2,
        win_bwd_bytes=1 * 1 * 24 * attn_bwd_el * 2,
    )
    return dict(
        glue_heads=glue_heads,
        glue_cond_init=glue_cond_init,
        pair_const_c1=pair_const_c1,
        attn_fp32_extra=attn_fp32_extra,
    )


def main():
    cnt_path = sys.argv[1]
    iso_path = sys.argv[sys.argv.index("--iso") + 1] if "--iso" in sys.argv else None
    cores = int(sys.argv[sys.argv.index("--cores") + 1]) if "--cores" in sys.argv else CORES_P300C
    fid = int(sys.argv[sys.argv.index("--fidelity") + 1]) if "--fidelity" in sys.argv else 4

    d = json.load(open(cnt_path))
    iso = json.load(open(iso_path)) if iso_path else []
    P = peak(fid, cores)

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in d["rows"]:
        k = (r["phase"], r["class"], r["size"])
        a = agg[k]
        a[0] += r["flops"]
        a[1] += r["elems"] * 2
        a[2] += r["calls"]

    opt = optimizer_work(d["param_shapes"])
    df = device_folds_work(opt["elems_total"])
    unp = unported_host_work(iso, K=d["K"], depth=d["depth"]) if iso else None

    print("# RFD4 Step 1 Analytic FLOP/Byte Count and Blackhole Roofline")
    print(f"Batch `{d['batch']}` | crop 256 | K={d['K']} | depth={d['depth']} | attn_fp32=on | "
          f"trained params={opt['elems_total']:,} ({opt['n_muon']} Muon={opt['elems_muon']:,}, "
          f"{opt['n_adam']} Adam={opt['elems_adam']:,})")
    print()
    print("## Hardware Peaks (cited from knowledgebase/knowledge/)")
    print("| Hardware | Worker cores | LoFi TFLOP/s | HiFi2 TFLOP/s | HiFi4 TFLOP/s | Split-bf16 fp32 TFLOP/s | DRAM GB/s | HiFi4 ridge (FLOP/B) | Source |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for name, c, src in [
        ("BH p300c (qb2 measured)", CORES_P300C, "machines.md:150, claims.md:60-62, ops.csv grid=110"),
        ("BH Galaxy (32-chip target)", CORES_GALAXY, "claims.md:233 (120 Tensix/chip, 512 GB/s)"),
        ("BH p150a (physical 140c)", CORES_P150A, "machines.md:150 (140 Tensix, 774 FP8, 512 GB/s)"),
    ]:
        print(f"| {name} | {c} | {peak(1,c)/1e12:.1f} | {peak(2,c)/1e12:.1f} | {peak(4,c)/1e12:.1f} | "
              f"{split_bf16_fp32_peak(c)/1e12:.1f} | {BW/1e9:.0f} | {peak(4,c)/BW:.0f} | {src} |")
    print(f"| BH p300c measured ttnn roof | 110 | - | - | {HIFI4_MEAS_P300C/1e12:.2f} | {HIFI4_MEAS_P300C/3e12:.2f} | "
          f"{BW_MEAS_P300C/1e9:.1f} | {HIFI4_MEAS_P300C/BW_MEAS_P300C:.0f} | state/of3t/BACKWARD.md:243 |")
    print(f"| BH p150a measured ttnn roof | 130 | 161.14 | 140.35 | {HIFI4_MEAS_P150A/1e12:.2f} | {HIFI4_MEAS_P150A/3e12:.2f} | "
          f"{BW_MEAS_P150A/1e9:.1f} | {HIFI4_MEAS_P150A/BW_MEAS_P150A:.0f} | PLAYBOOKS.md:443, state/rfd4/ARCH.md:107-111 |")
    print()

    print(f"## Model Forward + Backward (p300c {cores}c HiFi{fid} = {P/1e12:.1f} TFLOP/s, {BW/1e9:.0f} GB/s)")
    print("Note: aten byte totals are UNFUSED (every PyTorch intermediate hits DRAM); L1 kernel fusion (K14 serving kernels) beats unfused DRAM time, approaching the compute + irreducible-DRAM floor.")
    print()
    print("| phase | class | size | GFLOP | unfused GB | calls | t_compute ms | t_dram_unfused ms | unfused roof ms |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    tot_comp = collections.defaultdict(float)
    tot_unfused = collections.defaultdict(float)
    tot_fl = collections.defaultdict(float)
    tot_by = collections.defaultdict(float)
    for (ph, c, s), (fl, by, calls) in sorted(agg.items()):
        tc, tb = fl / P, by / BW
        tr = max(tc, tb)
        print(f"| {ph} | {c} | {s} | {fl/1e9:.1f} | {by/1e9:.2f} | {calls} | {tc*1e3:.2f} | {tb*1e3:.2f} | {tr*1e3:.2f} |")
        tot_comp[ph] += tc
        tot_unfused[ph] += tr
        tot_fl[ph] += fl
        tot_by[ph] += by

    if unp:
        for ph, key in [("cand_k", "cand_k_bytes"), ("win_fwd", "win_fwd_bytes"), ("win_bwd", "win_bwd_bytes")]:
            eb = unp["attn_fp32_extra"][key]
            tb = eb / BW
            print(f"| {ph} | attn_fp32_extra | +2B/el | 0.0 | {eb/1e9:.2f} | 24blks | 0.00 | {tb*1e3:.2f} | {tb*1e3:.2f} |")
            tot_unfused[ph] += tb
            tot_by[ph] += eb

    print()
    print("## On-Card System Terms (Fold VJP, Split Rebuild, Optimizer)")
    print("| term | GFLOP (HiFi4-eq) | DRAM GB | t_compute ms | t_dram ms | roof 1-chip ms | roof ZeRO-1 1/32 ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    ns_tc = opt["muon_ns_hifi4_flops"] / P
    ns_tb = opt["muon_ns_bytes"] / BW
    ns_r = max(ns_tc, ns_tb)
    elt_tc = opt["eltwise_flops"] / P
    elt_tb = opt["eltwise_bytes"] / BW
    elt_r = max(elt_tc, elt_tb)
    fv_r = df["fold_vjp_bytes"] / BW
    rb_r = df["rebuild_bytes"] / BW
    print(f"| fold_vjp (DeviceFolds + fp32 grad accum) | 0.0 | {df['fold_vjp_bytes']/1e9:.2f} | 0.00 | {fv_r*1e3:.2f} | {fv_r*1e3:.2f} | {fv_r*1e3:.2f} |")
    print(f"| split_rebuild (DeviceFolds bf16 + caches) | 0.0 | {df['rebuild_bytes']/1e9:.2f} | 0.00 | {rb_r*1e3:.2f} | {rb_r*1e3:.2f} | {rb_r*1e3:.2f} |")
    print(f"| opt_muon_ns (282 mats, 5 iters, split-bf16 fp32) | {opt['muon_ns_hifi4_flops']/1e9:.1f} | {opt['muon_ns_bytes']/1e9:.2f} | {ns_tc*1e3:.2f} | {ns_tb*1e3:.2f} | {ns_r*1e3:.2f} | {ns_r/32*1e3:.2f} |")
    print(f"| opt_elementwise (282 Muon + 276 Adam + EMA + clip) | {opt['eltwise_flops']/1e9:.1f} | {opt['eltwise_bytes']/1e9:.2f} | {elt_tc*1e3:.2f} | {elt_tb*1e3:.2f} | {elt_r*1e3:.2f} | {elt_r/32*1e3:.2f} |")

    if unp:
        print()
        print("## Isolated Host-Side Submodules to Move on Card (included inside model totals above; priced separately in Step 3)")
        print("| component | matmul GFLOP | matmul GB | mem GB | t_compute ms | t_dram ms | roof ms |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label, key in [("glue_heads (fwd K+1 + bwd)", "glue_heads"),
                           ("glue_cond_init (fwd K+1 x d + bwd)", "glue_cond_init"),
                           ("pair_const_c1 (C1 proj + K scales)", "pair_const_c1")]:
            w = unp[key]
            tc = w["mm_flops"] / P
            tb_mm = w["mm_bytes"] / BW
            tb_mem = w["mem_bytes"] / BW
            r_tot = max(tc, tb_mm) + tb_mem
            print(f"| {label} | {w['mm_flops']/1e9:.2f} | {w['mm_bytes']/1e9:.2f} | {w['mem_bytes']/1e9:.2f} | "
                  f"{tc*1e3:.2f} | {(tb_mm+tb_mem)*1e3:.2f} | {r_tot*1e3:.2f} |")

    print()
    print("## Analytic Totals")
    for ph in ("cand_k", "win_fwd", "win_bwd"):
        print(f"- `{ph}`: {tot_fl[ph]/1e12:.3f} TFLOP, {tot_by[ph]/1e9:.1f} GB unfused | compute floor {tot_comp[ph]*1e3:.1f} ms | unfused DRAM roof {tot_unfused[ph]*1e3:.1f} ms")
    sys_1c = fv_r + rb_r + ns_r + elt_r
    sys_32 = fv_r + rb_r + (ns_r + elt_r) / 32.0
    comp_step_1c = sum(tot_comp.values()) + sys_1c
    comp_step_32 = sum(tot_comp.values()) + sys_32
    unfused_step_1c = sum(tot_unfused.values()) + sys_1c
    unfused_step_32 = sum(tot_unfused.values()) + sys_32
    print(f"- System terms (fold_vjp + rebuild + opt): **{sys_1c*1e3:.1f} ms** (1-chip unsharded) / **{sys_32*1e3:.1f} ms** (ZeRO-1 1/32)")
    print(f"- **Compute-floor step (fully fused model + system)**: **{comp_step_1c*1e3:.1f} ms** (1-chip, {PRICE_RATIO*GPU_DEPTH2_S/comp_step_1c:.2f}x/$) / **{comp_step_32*1e3:.1f} ms** (ZeRO-1 1/32, {PRICE_RATIO*GPU_DEPTH2_S/comp_step_32:.2f}x/$)")
    print(f"- **Unfused-aten roof step (zero fusion + system)**: **{unfused_step_1c*1e3:.1f} ms** (1-chip, {PRICE_RATIO*GPU_DEPTH2_S/unfused_step_1c:.2f}x/$) / **{unfused_step_32*1e3:.1f} ms** (ZeRO-1 1/32, {PRICE_RATIO*GPU_DEPTH2_S/unfused_step_32:.2f}x/$)")


if __name__ == "__main__":
    main()
