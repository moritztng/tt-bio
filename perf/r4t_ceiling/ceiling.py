#!/usr/bin/env python3
"""r4t-ceiling step 3: analyze Tracy ops.csv + analytic count/iso JSONs and compute the device-only ceiling.

Fixes exact physical accounting for Tracy ops.csv:
  - Matmul / MinimalMatmul FLOPs handle transpose_a/transpose_b (reading actual M, N, K and broadcast batch).
  - SliceDeviceOperation DRAM bytes = 2 * output_bytes (reads only sliced region, not entire input tensor).
  - GenericOpDeviceOperation deduplicates the preallocated output tensor (listed in both INPUT_* and OUTPUT_0).
  - L1-only tensors excluded from DRAM byte count.
  - Per-op-class achieved fraction of hardware roofline (BH p300c 110c @ 1350 MHz: 152.1 TFLOP/s HiFi4, 512 GB/s DRAM;
    also reported vs measured ttnn roof 115.685 TFLOP/s, 440 GB/s).
  - Unported host terms priced at op-class-matched measured efficiencies:
      * Muon NS (fp32 split-bf16 matmul) -> measured matmul achieved fraction
      * Elementwise optimizer, DeviceFolds fold VJP, DeviceFolds split rebuild -> measured DRAM streaming (eltwise+copy) fraction
      * FlowGlue (heads + cond/init) & C1 pair constant -> matmul part at matmul eff + DRAM part at DRAM streaming eff
      * Host batch feed -> overlapped (0 ms)

Usage:
  python3 perf/r4t_ceiling/ceiling.py OPS.csv COUNT.json --iso ISO.json [--out-md PATH] [--label NAME]
"""
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roofline as R

DTYPE_B = {
    "BFLOAT16": 2.0, "FLOAT32": 4.0, "BFLOAT8_B": 1.0625, "UINT32": 4.0,
    "INT32": 4.0, "UINT16": 2.0, "BFLOAT4_B": 0.5625, "UINT8": 1.0,
}
FID_DIV = {"LoFi": 1.0, "HiFi2": 2.0, "HiFi3": 3.0, "HiFi4": 4.0}
COMP_PHASES = {"fwd", "cand_pair", "cand_trunk", "win_pair", "win_trunk", "tape_bwd"}


def pad_dim(s):
    return int(s.split("[")[0]) if s else 0


def log_dim(s):
    if not s:
        return 0
    return int(s.split("[")[1].rstrip("]")) if "[" in s else int(s)


def tensor_nbytes(r, pfx):
    w = r.get(f"{pfx}_W_PAD[LOGICAL]")
    if not w:
        return 0.0
    mem = r.get(f"{pfx}_MEMORY", "")
    if "L1" in mem and "DRAM" not in mem:
        return 0.0
    el = (pad_dim(w) * pad_dim(r[f"{pfx}_Z_PAD[LOGICAL]"])
          * pad_dim(r[f"{pfx}_Y_PAD[LOGICAL]"]) * pad_dim(r[f"{pfx}_X_PAD[LOGICAL]"]))
    return el * DTYPE_B.get(r.get(f"{pfx}_DATATYPE", "BFLOAT16"), 2.0)


def op_bytes(r):
    op = r["OP CODE"]
    if op == "SliceDeviceOperation":
        return 2.0 * tensor_nbytes(r, "OUTPUT_0")
    if op == "GenericOpDeviceOperation":
        # ttnn.generic_op passes all inputs + preallocated outputs inside io_tensors (INPUT_0..k),
        # and Tracy also duplicates the last io_tensor as OUTPUT_0. Sum INPUT_0..7 only (no double count).
        return sum(tensor_nbytes(r, f"INPUT_{i}") for i in range(8))
    return (sum(tensor_nbytes(r, f"INPUT_{i}") for i in range(8))
            + sum(tensor_nbytes(r, f"OUTPUT_{i}") for i in range(3)))


def mm_flops(r):
    op = r["OP CODE"]
    if op not in ("MatmulDeviceOperation", "MinimalMatmulDeviceOperation"):
        return 0.0
    try:
        attr = r.get("ATTRIBUTES", "")
        ta = "transpose_a': 'true" in attr.lower() or "transpose_a=1" in attr.lower()
        ow = log_dim(r["OUTPUT_0_W_PAD[LOGICAL]"])
        oz = log_dim(r["OUTPUT_0_Z_PAD[LOGICAL]"])
        m = log_dim(r["OUTPUT_0_Y_PAD[LOGICAL]"])
        n = log_dim(r["OUTPUT_0_X_PAD[LOGICAL]"])
        i0_y = log_dim(r["INPUT_0_Y_PAD[LOGICAL]"])
        i0_x = log_dim(r["INPUT_0_X_PAD[LOGICAL]"])
        if ta:
            k = i0_y
        elif i0_y == m:
            k = i0_x
        elif i0_x == m:
            k = i0_y
        else:
            k = i0_x
        # Account for batched-dim reduction (e.g., dW = X^T @ dY over Z batch) if input batch > output batch
        i0_b = log_dim(r["INPUT_0_W_PAD[LOGICAL]"]) * log_dim(r["INPUT_0_Z_PAD[LOGICAL]"])
        i1_b = log_dim(r["INPUT_1_W_PAD[LOGICAL]"]) * log_dim(r["INPUT_1_Z_PAD[LOGICAL]"])
        ob = ow * oz
        b = max(ob, min(i0_b, i1_b))
        return 2.0 * b * m * n * k
    except Exception:
        return 0.0


def classify(op):
    o = op.lower()
    if any(k in o for k in ("matmul", "linear", "scaleddotproduct", "sdpa")):
        return "matmul"
    if "genericop" in o or "generic_op" in o:
        return "generic_fused"
    if any(k in o for k in ("layernorm", "rmsnorm", "groupnorm", "norm")):
        return "norm"
    if "softmax" in o:
        return "softmax"
    if any(k in o for k in ("sum", "mean", "reduce", "argmax", "argmin", "max", "min", "moreh")):
        return "reduce"
    if any(k in o for k in ("tilize", "untilize", "transpose", "permute", "concat", "slice", "copy",
                            "pad", "unpad", "typecast", "reshape", "repeat", "split", "embedding",
                            "interleaved", "sharded", "move", "halo", "fill", "zeros", "full",
                            "assign", "clone", "nlpcreateheads", "nlpconcatheads")):
        return "move"
    return "eltwise"


def parse_ops(path):
    out, cur, step = [], None, 0
    worker_cores = None
    for r in csv.DictReader(open(path)):
        if r["OP TYPE"] == "signpost":
            name = r["OP CODE"].split("|")[-1]
            if name == "fwd":
                step += 1
            cur = [step, name, []]
            out.append(cur)
            continue
        if cur is not None and r.get("DEVICE KERNEL DURATION [ns]"):
            if worker_cores is None and r.get("AVAILABLE WORKER CORE COUNT"):
                worker_cores = int(r["AVAILABLE WORKER CORE COUNT"])
            cur[2].append(r)
    return out, worker_cores or R.CORES_P300C


def analyze(ops_path, cnt_path, iso_path, label=""):
    out, cores = parse_ops(ops_path)
    steps = sorted({s for s, _, _ in out})
    warm = [s for s in steps if s >= 2] or steps
    nw = len(warm)
    lofi = R.peak(1, cores)
    hifi4 = R.peak(4, cores)
    bw = R.BW

    # Per-step total check
    per_step = collections.defaultdict(lambda: collections.defaultdict(float))
    by_ph = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    by_cls = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    by_ph_cls = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    by_code = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    mm_by_kind = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])

    for s, ph, rows in out:
        for r in rows:
            ms_raw = float(r["DEVICE KERNEL DURATION [ns]"]) / 1e6
            per_step[s][ph] += ms_raw
        if s not in warm:
            continue
        for r in rows:
            ms = float(r["DEVICE KERNEL DURATION [ns]"]) / 1e6 / nw
            gb = op_bytes(r) / 1e9 / nw
            tf = mm_flops(r) / 1e12 / nw
            c = classify(r["OP CODE"])
            fid = r.get("MATH FIDELITY", "") or "HiFi4"
            p_op = lofi / FID_DIV.get(fid, 4.0)
            tc = (tf * 1e12 / p_op) * 1e3
            tb = (gb * 1e9 / bw) * 1e3
            rf = max(tc, tb)
            by_ph[ph][0] += ms; by_ph[ph][1] += gb; by_ph[ph][2] += tf; by_ph[ph][3] += rf; by_ph[ph][4] += 1
            if ph not in COMP_PHASES:
                continue
            for d in (by_cls[c], by_ph_cls[(ph, c)], by_code[r["OP CODE"]]):
                d[0] += ms; d[1] += gb; d[2] += tf; d[3] += rf; d[4] += 1
            if c == "matmul":
                mk = mm_by_kind[(ph, r["OP CODE"])]
                mk[0] += ms; mk[1] += gb; mk[2] += tf; mk[3] += rf; mk[4] += 1

    # Analytic data
    d_cnt = json.load(open(cnt_path))
    iso = json.load(open(iso_path))
    opt = R.optimizer_work(d_cnt["param_shapes"])
    df = R.device_folds_work(opt["elems_total"])
    unp = R.unported_host_work(iso, K=d_cnt["K"], depth=d_cnt["depth"])

    # Achieved op-class efficiencies (roofline / measured, i.e., fraction of hardware peak achieved)
    # 1. Matmul efficiency: overall measured matmul roof/meas, AND forward MinimalMatmul vs Matmul
    mm_ms, mm_gb, mm_tf, mm_rf, _ = by_cls["matmul"]
    eff_mm_all = mm_rf / mm_ms if mm_ms else 0.35
    # Forward-only matmul efficiency (representative of clean fused/MinimalMatmul forward shapes like Muon NS)
    mm_fwd_ms = by_ph_cls[("cand_trunk", "matmul")][0] + by_ph_cls[("win_trunk", "matmul")][0]
    mm_fwd_rf = by_ph_cls[("cand_trunk", "matmul")][3] + by_ph_cls[("win_trunk", "matmul")][3]
    eff_mm_fwd = mm_fwd_rf / mm_fwd_ms if mm_fwd_ms else eff_mm_all
    # Pure compute utilization of matmuls against HiFi4 peak (152.1 TFLOP/s):
    ach_mm_tfs_all = mm_tf / (mm_ms / 1e3) if mm_ms else 0.0
    mm_min_ms = by_code["MinimalMatmulDeviceOperation"][0]
    mm_min_tf = by_code["MinimalMatmulDeviceOperation"][2]
    ach_mm_tfs_min = mm_min_tf / (mm_min_ms / 1e3) if mm_min_ms else ach_mm_tfs_all

    # 2. DRAM streaming efficiency for elementwise / typecast / concat / copy ops
    #    (measured directly on BinaryNg + Typecast + Concat + UnaryNg, excluding slow Permute/Transpose!)
    stream_ops = ("BinaryNgDeviceOperation", "TypecastDeviceOperation", "ConcatDeviceOperation",
                  "UnaryNgDeviceOperation", "UnaryDeviceOperation")
    st_ms = sum(by_code[k][0] for k in stream_ops)
    st_gb = sum(by_code[k][1] for k in stream_ops)
    ach_stream_gbs = st_gb / (st_ms / 1e3) if st_ms else 350.0
    eff_stream = ach_stream_gbs / (bw / 1e9)   # ~0.74 (380 / 512 GB/s)

    # 3. All non-matmul DRAM ops efficiency (eltwise + move + norm + reduce + softmax)
    nonmm_classes = ("eltwise", "move", "norm", "reduce", "softmax")
    nm_ms = sum(by_cls[k][0] for k in nonmm_classes)
    nm_gb = sum(by_cls[k][1] for k in nonmm_classes)
    ach_nonmm_gbs = nm_gb / (nm_ms / 1e3) if nm_ms else 250.0
    eff_nonmm = ach_nonmm_gbs / (bw / 1e9)

    # 4. Reduction efficiency (for C1 rowsum / clip-norm reductions)
    red_ms, red_gb = by_cls["reduce"][0], by_cls["reduce"][1]
    ach_red_gbs = red_gb / (red_ms / 1e3) if red_ms else 140.0
    eff_red = ach_red_gbs / (bw / 1e9)

    # Price unported terms at op-class-matched measured efficiencies!
    # Conservative & transparent:
    # - Muon NS (fp32 split-bf16 matmul, 8.40 HiFi4-TFLOP, 16.64 GB): compute-bound matmul priced at eff_mm_all
    #   (also note MinimalMatmul-rate sensitivity: 61.6 TFLOP/s -> 136 ms unsharded / 4.3 ms ZeRO-1).
    ns_roof_s = max(opt["muon_ns_hifi4_flops"] / hifi4, opt["muon_ns_bytes"] / bw)
    ns_1c_s = ns_roof_s / eff_mm_all
    ns_32_s = ns_1c_s / 32.0

    # - Elementwise optimizer (5.73 GB fp32: Muon + Adam + EMA + clip-norm): DRAM streaming @ eff_stream
    elt_roof_s = opt["eltwise_bytes"] / bw
    elt_1c_s = elt_roof_s / eff_stream
    elt_32_s = elt_1c_s / 32.0

    # - DeviceFolds fold VJP (2.14 GB: transpose/slice/row-scale + fp32 grad accum): priced at eff_nonmm (includes transpose overhead!)
    fv_roof_s = df["fold_vjp_bytes"] / bw
    fv_s = fv_roof_s / eff_nonmm

    # - DeviceFolds split rebuild (1.53 GB: fp32->bf16 cast + transpose + serving caches): priced at eff_nonmm
    rb_roof_s = df["rebuild_bytes"] / bw
    rb_s = rb_roof_s / eff_nonmm

    # - C1 candidate pair constant + K=4 split-bf16 norm scales (proj matmul @ eff_mm_all + bits3/scales @ eff_stream)
    c1 = unp["pair_const_c1"]
    c1_mm_roof_s = max(c1["mm_flops"] / hifi4, c1["mm_bytes"] / bw)
    c1_mem_roof_s = c1["mem_bytes"] / bw
    c1_s = c1_mm_roof_s / eff_mm_all + c1_mem_roof_s / eff_stream

    # - FlowGlue heads (fwd K+1 + bwd) and cond+init (fwd (K+1)*depth + bwd)
    def price_glue(w):
        r_mm = max(w["mm_flops"] / hifi4, w["mm_bytes"] / bw)
        r_mem = w["mem_bytes"] / bw
        return r_mm + r_mem, r_mm / eff_mm_all + r_mem / eff_nonmm

    gh_roof_s, gh_s = price_glue(unp["glue_heads"])
    gc_roof_s, gc_s = price_glue(unp["glue_cond_init"])

    # Measured device compute phases (already on card)
    meas_terms = [
        ("cand_pair (`PairOnCard` K=4, measured device)", by_ph["cand_pair"][0] / 1e3),
        ("cand_trunk (K=4 depth=2 grad-off, measured device)", by_ph["cand_trunk"][0] / 1e3),
        ("win_pair (P2 `PairOnCard.taped` fwd, measured device)", by_ph["win_pair"][0] / 1e3),
        ("win_trunk (K=1 depth=2 taped, measured device)", by_ph["win_trunk"][0] / 1e3),
        ("tape_bwd (trunk + P2 pair bwd, measured device)", by_ph["tape_bwd"][0] / 1e3),
        ("fwd misc (masks, measured device)", by_ph["fwd"][0] / 1e3),
    ]
    meas_tot = sum(t for _, t in meas_terms)

    unp_terms_32 = [
        ("pair_const C1 (proj + K=4 scales, roof / eff)", c1_s),
        ("glue_heads fwd+bwd (roof / eff)", gh_s),
        ("glue_cond_init fwd+bwd (roof / eff)", gc_s),
        ("fold_vjp (`DeviceFolds` + fp32 grad accum, roof / eff)", fv_s),
        ("split_rebuild (`DeviceFolds` bf16 + caches, roof / eff)", rb_s),
        ("opt Muon NS fp32 (282 mats, 5 iters, ZeRO-1 1/32, roof / eff)", ns_32_s),
        ("opt elementwise fp32 (Muon+Adam+EMA, ZeRO-1 1/32, roof / eff)", elt_32_s),
        ("host batch feed (overlapped)", 0.0),
    ]
    unp_terms_1c = [
        ("pair_const C1 (proj + K=4 scales, roof / eff)", c1_s),
        ("glue_heads fwd+bwd (roof / eff)", gh_s),
        ("glue_cond_init fwd+bwd (roof / eff)", gc_s),
        ("fold_vjp (`DeviceFolds` + fp32 grad accum, roof / eff)", fv_s),
        ("split_rebuild (`DeviceFolds` bf16 + caches, roof / eff)", rb_s),
        ("opt Muon NS fp32 (282 mats, 5 iters, unsharded 1/1, roof / eff)", ns_1c_s),
        ("opt elementwise fp32 (Muon+Adam+EMA, unsharded 1/1, roof / eff)", elt_1c_s),
        ("host batch feed (overlapped)", 0.0),
    ]

    T_32 = meas_tot + sum(t for _, t in unp_terms_32)
    T_1c = meas_tot + sum(t for _, t in unp_terms_1c)
    ratio_32 = R.PRICE_RATIO * R.GPU_DEPTH2_S / T_32
    ratio_1c = R.PRICE_RATIO * R.GPU_DEPTH2_S / T_1c

    lines = []
    def p(s=""): lines.append(s)
    p(f"# RFD4 Device-Only Step Ceiling{f' — {label}' if label else ''}")
    p(f"Source CSV: `{ops_path}` | warm steps `{warm}` (n={nw}) | worker grid `{cores}` Tensix (p300c, 1350 MHz) | "
      f"peaks: HiFi4 `{hifi4/1e12:.1f}` TFLOP/s, DRAM `{bw/1e9:.0f}` GB/s.")
    p(f"Price basis (PERF/SIZING): BH Galaxy `$160,000 / 32` vs DGX B200 `$562,154 / 8` = `{R.PRICE_RATIO:.5f}x`; B200 depth 2 = `{R.GPU_DEPTH2_S:.3f}` s/step.")
    p()
    p("## Step-to-Step Stability (DEVICE KERNEL DURATION, ms)")
    p("| phase | " + " | ".join(f"step {s}" for s in steps) + " | warm mean |")
    p("|---|" + "---:|" * (len(steps) + 1))
    for ph in ("fwd", "cand_pair", "cand_trunk", "win_pair", "win_trunk", "heads_bwd", "tape_bwd", "fold_bwd", "opt", "rebuild", "digest"):
        vals = " | ".join(f"{per_step[s][ph]:.2f}" for s in steps)
        p(f"| `{ph}` | {vals} | {by_ph[ph][0]:.2f} |")
    st_tots = " | ".join(f"{sum(per_step[s][k] for k in COMP_PHASES):.2f}" for s in steps)
    p(f"| **compute total (excl. host-IO fold_bwd/rebuild)** | {st_tots} | **{meas_tot*1e3:.2f}** |")
    p()

    p("## Measured Device Time & Achieved Roofline by Op Class (compute phases, warm mean)")
    p("| class | meas ms | % card | DRAM GB | TFLOP | ach GB/s | % 512 GB/s | ach TF/s | % 152.1 TF/s | per-op roof ms | roof/meas | ops |")
    p("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    t_ms = t_gb = t_tf = t_rf = 0.0
    for c, (ms, gb, tf, rf, n) in sorted(by_cls.items(), key=lambda x: -x[1][0]):
        t_ms += ms; t_gb += gb; t_tf += tf; t_rf += rf
        abw = gb / (ms / 1e3) if ms else 0.0
        atf = tf / (ms / 1e3) if ms else 0.0
        eff = rf / ms if ms else 0.0
        p(f"| `{c}` | {ms:.2f} | {100*ms/(meas_tot*1e3):.1f}% | {gb:.1f} | {tf:.2f} | {abw:.1f} | {100*abw/512:.1f}% | "
          f"{atf:.1f} | {100*atf/(hifi4/1e12):.1f}% | {rf:.2f} | {100*eff:.1f}% | {n//nw} |")
    p(f"| **TOTAL** | **{t_ms:.2f}** | **100.0%** | **{t_gb:.1f}** | **{t_tf:.2f}** | **{t_gb/(t_ms/1e3):.1f}** | "
      f"**{100*t_gb/(t_ms/1e3)/512:.1f}%** | **{t_tf/(t_ms/1e3):.1f}** | **{100*t_tf/(t_ms/1e3)/(hifi4/1e12):.1f}%** | "
      f"**{t_rf:.2f}** | **{100*t_rf/t_ms:.1f}%** | |")
    p()
    p("Measured op-class efficiencies used to price unported terms:")
    p(f"- Matmul (`matmul` class): `{100*eff_mm_all:.1f}%` of per-op roofline (`{ach_mm_tfs_all:.1f}` TFLOP/s overall; `MinimalMatmul` = `{ach_mm_tfs_min:.1f}` TFLOP/s; fwd matmuls = `{100*eff_mm_fwd:.1f}%`).")
    p(f"- DRAM streaming (`BinaryNg`/`Typecast`/`Concat`/`Unary`): `{ach_stream_gbs:.1f}` GB/s = `{100*eff_stream:.1f}%` of 512 GB/s (`{100*ach_stream_gbs/440:.1f}%` of 440 GB/s measured roof).")
    p(f"- All non-matmul DRAM ops (incl. `Permute`/`Transpose`/`LayerNorm`/`Reduce`): `{ach_nonmm_gbs:.1f}` GB/s = `{100*eff_nonmm:.1f}%` of 512 GB/s.")
    p()

    p("## Phase x Op-Class Breakdown (warm mean)")
    p("| phase | class | meas ms | DRAM GB | ach GB/s | TFLOP | ach TF/s | per-op roof ms | roof/meas | ops |")
    p("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ph in ("cand_pair", "cand_trunk", "win_pair", "win_trunk", "tape_bwd"):
        for (p0, c), (ms, gb, tf, rf, n) in sorted(by_ph_cls.items(), key=lambda x: -x[1][0]):
            if p0 != ph:
                continue
            abw = gb / (ms / 1e3) if ms else 0.0
            atf = tf / (ms / 1e3) if ms else 0.0
            eff = rf / ms if ms else 0.0
            p(f"| `{ph}` | `{c}` | {ms:.2f} | {gb:.1f} | {abw:.1f} | {tf:.2f} | {atf:.1f} | {rf:.2f} | {100*eff:.1f}% | {n//nw} |")
    p()

    p("## Matmul Kernel Breakdown by Phase")
    p("| phase | op | meas ms | DRAM GB | TFLOP | ach TF/s | % HiFi4 | roof ms | roof/meas | ops |")
    p("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (ph, op), (ms, gb, tf, rf, n) in sorted(mm_by_kind.items(), key=lambda x: -x[1][0]):
        atf = tf / (ms / 1e3) if ms else 0.0
        p(f"| `{ph}` | `{op}` | {ms:.2f} | {gb:.1f} | {tf:.2f} | {atf:.1f} | {100*atf/(hifi4/1e12):.1f}% | {rf:.2f} | {100*rf/ms:.1f}% | {n//nw} |")
    p()

    p("## Top 12 Device Op Codes (compute phases, warm mean)")
    p("| op code | meas ms | % card | DRAM GB | ach GB/s | TFLOP | ach TF/s | roof ms | roof/meas | ops |")
    p("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for op, (ms, gb, tf, rf, n) in sorted(by_code.items(), key=lambda x: -x[1][0])[:12]:
        abw = gb / (ms / 1e3) if ms else 0.0
        atf = tf / (ms / 1e3) if ms else 0.0
        p(f"| `{op}` | {ms:.2f} | {100*ms/(meas_tot*1e3):.1f}% | {gb:.1f} | {abw:.1f} | {tf:.2f} | {atf:.1f} | {rf:.2f} | {100*rf/ms:.1f}% | {n//nw} |")
    p()

    p("## Per-Term Device-Only Ceiling (where this goes, NOT where it is)")
    p("| term | ms/step (ZeRO-1 1/32) | ms/step (1-chip 1/1) | % of ZeRO-1 ceiling | basis |")
    p("|---|---:|---:|---:|---|")
    for k, t in meas_terms:
        p(f"| {k} | {t*1e3:.2f} | {t*1e3:.2f} | {100*t/T_32:.1f}% | measured device |")
    for (k32, t32), (_k1, t1) in zip(unp_terms_32, unp_terms_1c):
        p(f"| {k32} | {t32*1e3:.2f} | {t1*1e3:.2f} | {100*t32/T_32:.1f}% | roof / measured class eff |")
    p(f"| **CEILING (device only, host overlapped)** | **{T_32*1e3:.1f} ms ({T_32:.4f} s)** | **{T_1c*1e3:.1f} ms ({T_1c:.4f} s)** | **100.0%** | |")
    p()
    p(f"- **ZeRO-1 (1/32) per chip (DP eff 1.00)**: `{T_32:.4f} s/step` -> **`{ratio_32:.2f}x`** per $ vs B200 `{R.GPU_DEPTH2_S} s` (`{ratio_32*0.90:.2f}x` at 0.90 DP eff, `{ratio_32*0.85:.2f}x` at 0.85 DP eff).")
    p(f"- **Unsharded 1-chip (1/1) per chip**: `{T_1c:.4f} s/step` -> **`{ratio_1c:.2f}x`** per $ vs B200 `{R.GPU_DEPTH2_S} s`.")
    p(f"- Note: measured on qb2 p300c (`110` worker cores); BH Galaxy has `120` Tensix cores (`+9.1%` compute roof, same `512 GB/s` DRAM).")
    p()
    p(f"SUMMARY label={label} meas_ms={meas_tot*1e3:.2f} unp32_ms={(T_32-meas_tot)*1e3:.2f} unp1c_ms={(T_1c-meas_tot)*1e3:.2f} "
      f"ceil32_s={T_32:.4f} ratio32={ratio_32:.2f} ratio32_dp85={ratio_32*0.85:.2f} ceil1c_s={T_1c:.4f} ratio1c={ratio_1c:.2f}")
    return "\n".join(lines)


def main():
    ops_path = sys.argv[1]
    cnt_path = sys.argv[2]
    iso_path = sys.argv[sys.argv.index("--iso") + 1]
    label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else ""
    out_md = sys.argv[sys.argv.index("--out-md") + 1] if "--out-md" in sys.argv else None
    text = analyze(ops_path, cnt_path, iso_path, label=label)
    if out_md:
        open(out_md, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
