#!/usr/bin/env python3
"""Count the tile arrivals in a Boltz-2 512 aa Blackhole block and a diffusion step, from the
model's own operand shapes and the committed device-profiler dumps, and close the identity
against the measured input-tile wait.

No device is opened. Inputs are three committed artifacts from ws:b2z-kernel-cycle-census,
all taken on qb2 card 0 (one Blackhole processor of a p300c, tt-metal v0.68.0 source build,
11x10 grid):

  perf/b2z_kernel_census/ops_perf_blocksum_qb2c0.csv.gz   PairformerLayer, CB stall accumulators on
  perf/b2z_kernel_census/ops_perf_step_qb2c0.csv.gz       DiffusionStep, no stall accumulators
  perf/b2z_kernel_census/blocksum_census.json             the 272-op signature to window on

Fetch the three inputs into <artdir> with:

    for f in ops_perf_blocksum_qb2c0.csv.gz ops_perf_step_qb2c0.csv.gz \
             blocksum_census.json step_census.json; do
      git show origin/wk/b2z-kernel-cycle-census:perf/b2z_kernel_census/$f > <artdir>/$f
    done

Usage: census_tiles.py <artdir> <outdir>
"""
import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TILE_BYTES = {"BFLOAT16": 2048, "FLOAT32": 4096, "BFLOAT8_B": 1088, "UINT32": 4096,
              "UINT16": 2048, "INT32": 4096, "UINT8": 1024, "FLOAT16": 2048}
NS_PER_TILE_BH = 20.82          # MEASURED, b2z2-datum-rate-floor, BH binary shape, 1.35 GHz
NS_PER_TILE_BH_COPY = 30.44     # MEASURED, same row, BH unary copy shape
WAIT_MS_BLOCK = 18.3366         # MEASURED, b2z-kernel-cycle-census, BH, per core-averaged block


def f(r, k):
    try:
        return float(r.get(k, "") or 0)
    except (TypeError, ValueError):
        return 0.0


def _dim(v):
    m = re.match(r"\s*(\d+)", str(v) or "")
    return int(m.group(1)) if m else 0


def shape(r, slot):
    """Padded 4D extent of an operand slot, or None if the slot is empty."""
    t = tuple(_dim(r.get(f"{slot}_{a}_PAD[LOGICAL]", "")) for a in "WZYX")
    return t if (t[2] and t[3]) else None


def tiles(sh):
    w, z, y, x = sh
    return max(w, 1) * max(z, 1) * ((y + 31) // 32) * ((x + 31) // 32)


def ksrc(r):
    s = r.get("COMPUTE KERNEL SOURCE") or ""
    return "+".join(sorted(set(re.findall(r"([A-Za-z0-9_]+)\.cpp", s)))) or "-"


# How many leading slots are genuine INPUTS. Everything after them that repeats OUTPUT_0's shape
# is a pre-allocated output buffer. Read off the call sites, not guessed:
#   mm_generic.py:359      ttnn.generic_op([in0, in1, *outs], ...)          -> 2
#   trimul_tail.py:245     ttnn.generic_op([xa, wa, xb, wb, out], ...)      -> 2 distinct GEMM feeds
#   sdpa_generic.py:450    ttnn.generic_op([q, k, v, mask, out], ...)       -> 4
#   reblock_permute.py:730 ttnn.generic_op([xw, out], ...)                  -> 1
# and for stock ttnn ops, the op's own required-input count.
ARITY = {
    "compute": 2, "sdpa": 4,
    "compute_reblock_permute": 1, "compute_reblock_permute_gated": 1,
    "MatmulDeviceOperation": 2, "BinaryNgDeviceOperation": 2,
    "LayerNormDeviceOperation": 1, "SoftmaxDeviceOperation": 1,
    "TransposeDeviceOperation": 1, "PermuteDeviceOperation": 1,
}


def raw_slots(r):
    """[(slot, shape, tiles, dtype, memory)] for every populated input slot, as reported."""
    out = []
    for i in range(20):
        sh = shape(r, f"INPUT_{i}")
        if sh:
            out.append((i, sh, tiles(sh), r.get(f"INPUT_{i}_DATATYPE", ""),
                        r.get(f"INPUT_{i}_MEMORY", "")))
    return out


def split_slots(r):
    """Separate genuine input operands from pre-allocated OUTPUT buffers.

    ttnn hands every io tensor to the op as an "input", so the device profiler reports a
    pre-allocated output buffer in an INPUT_ slot as well as in OUTPUT_0. `ttnn.generic_op` is
    called as `[*inputs, *outputs]` everywhere in tt-bio (trimul_tail.py:245,
    sdpa_generic.py:450, reblock_permute.py:254/497/730, mm_generic.py:359) and binary_ng takes
    an optional output tensor in its last slot. Counting those as reads would double-count the
    op's own output as DRAM traffic and as a tile the math thread waits for, so they are moved
    to the write side. The rule: a TRAILING slot whose padded shape equals OUTPUT_0's is an
    output buffer. Returns (inputs, n_output_buffers).
    """
    slots = raw_slots(r)
    out = shape(r, "OUTPUT_0")
    if not out or not slots:
        return slots, 1 if out else 0
    floor = ARITY.get(ksrc(r)) or ARITY.get(r["OP CODE"], 1)
    n_out = 0
    while len(slots) > floor and slots[-1][1] == out:
        slots.pop()
        n_out += 1
    # The profiler appends the op's output tensor to the input slots on top of whatever the op
    # was handed, so the number of real output BUFFERS is one less than the slots that repeat
    # OUTPUT_0's shape. Checked against all six generic_op call sites: qkv_heads passes three
    # outs and shows four trailing slots; [xw, out], [q,k,v,mask,out] and [in0,in1,out] each
    # pass one and show two.
    return slots, max(n_out - 1, 1)


def operands(r):
    return split_slots(r)[0]


MM_CFG = re.compile(r"(\w*MatmulMultiCore\w*)ProgramConfig\(([^)]*)\)")


def mm_config(r):
    """Parse the matmul program config out of ATTRIBUTES. These carry per_core_M / per_core_N /
    in0_block_w verbatim, so a matmul's operand re-delivery is read off the program, not modelled."""
    m = MM_CFG.search(r.get("ATTRIBUTES") or "")
    if not m:
        return None
    cfg = {"_kind": m.group(1)}
    for kv in m.group(2).split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k, v = k.strip(), v.strip()
            if v.isdigit():
                cfg[k] = int(v)
            elif v.startswith("(x="):
                pass
    g = re.search(r"compute_with_storage_grid_size=\(x=(\d+);y=(\d+)\)", m.group(2))
    if g:
        cfg["grid_x"], cfg["grid_y"] = int(g.group(1)), int(g.group(2))
    return cfg


def matmul_arrivals(r, cfg, cores):
    """Per-core CB arrivals and tile-pair MACs for a ttnn matmul, from its own program config.

    Every ttnn multicast matmul gives each core a per_core_M x per_core_N output block and walks
    the full K: the reader delivers in0 (per_core_M x Kt) and in1 (Kt x per_core_N) tiles into the
    core's CBs, once each, and the math thread issues per_core_M x per_core_N x Kt tile-pair MACs
    off them. Batch multiplies everything when the batch is not fused into M.
    """
    ops = operands(r)
    out = shape(r, "OUTPUT_0")
    if not ops or not out or len(ops) < 2:
        return None
    in0, in1 = ops[0][1], ops[1][1]
    Kt = (in0[3] + 31) // 32
    Mt = (out[2] + 31) // 32
    Nt = (out[3] + 31) // 32
    B = max(out[0], 1) * max(out[1], 1)
    pm = cfg.get("per_core_M")
    pn = cfg.get("per_core_N")
    if not pm or not pn:
        # 1D configs name only one of them; fall back to an even split of the output block.
        pm = pm or max(1, Mt * B // max(cores, 1) // max(pn or 1, 1))
        pn = pn or max(1, Nt // max(cores // max(1, (Mt * B) // max(pm, 1)), 1))
    # Batch handling: fuse_batch=0 means the program loops the whole (Mt x Nt) grid per batch.
    fused = "fuse_batch=1" in (r.get("ATTRIBUTES") or "")
    reps = 1 if fused else B
    in0_arr = pm * Kt * reps
    in1_arr = Kt * pn * reps
    macs = pm * pn * Kt * reps
    packs = pm * pn * reps
    # A bias operand, when present, is one row of N tiles per block.
    bias = pn * reps if len(ops) > 2 else 0
    return {"in0": in0_arr, "in1": in1_arr, "bias": bias, "macs": macs, "packs": packs,
            "Mt": Mt, "Nt": Nt, "Kt": Kt, "B": B, "per_core_M": pm, "per_core_N": pn}


# Programs that dispatch no compute kernel at all: they are reader/writer only, so the math
# thread neither waits nor runs. They still move DRAM bytes.
NO_COMPUTE = {"ReshapeViewDeviceOperation", "SliceDeviceOperation", "ConcatDeviceOperation",
              "NlpCreateHeadsDeviceOperation"}

# tt-bio's own fused kernels. Each entry says how many times the kernel re-delivers an operand
# into a CB relative to a single pass, derived by reading the kernel source on this branch.
#   trimul_tail/compute.cpp      : TRIMUL_TAIL_PASSES=2 matmul passes over the same in0/in1 block,
#                                  then a gate epilogue that re-reads p and g from CB.
#   reblock_permute[_gated]      : one streaming pass, one output tile per input tile.
#   triatt_sdpa/compute/sdpa.cpp : flash-style, K and V re-walked per query block.
GENERIC_PASSES = {"compute": 2, "compute_reblock_permute_gated": 1,
                  "compute_reblock_permute": 1, "sdpa": 1}


def arrivals(r, cores):
    """Per-core CB tile arrivals (what cb_wait_front blocks on) and math-pipeline tile passes."""
    code = r["OP CODE"]
    out = shape(r, "OUTPUT_0")
    ops = operands(r)
    out_t = tiles(out) if out else 0
    if code in NO_COMPUTE or not out:
        return 0, 0, 0
    if code == "MatmulDeviceOperation":
        cfg = mm_config(r)
        if cfg:
            a = matmul_arrivals(r, cfg, cores)
            if a:
                return a["in0"] + a["in1"] + a["bias"], a["macs"], a["packs"]
    # Everything else is split by output tile across the grid: each core takes out_t/cores output
    # tiles and needs the matching input tiles. An operand smaller than the output is a broadcast
    # and is delivered whole to every core that needs it, capped at the operand's own size.
    per_core_out = out_t / max(cores, 1)
    arr = 0.0
    for _, sh, t, _dt, _mem in ops:
        arr += min(t, per_core_out) if t < out_t else per_core_out
    mult = GENERIC_PASSES.get(ksrc(r), 1)
    return arr * mult, arr * mult, per_core_out


def traffic(r):
    """Bytes this program must move, split by where they live.

    A DRAM-interleaved operand costs DRAM bandwidth; an L1-interleaved one costs NOC round trips
    into other cores' L1 and no DRAM at all. 213 of the block's 637 operand slots are
    L1-interleaved, so the two cannot be pooled.

    A reader/writer-only program (slice, concat, reshape, nlp_create_heads) reads exactly the
    region it writes, never the whole source tensor, so its read side is sized from the output.
    The profiler reports the source tensor's full shape and counting that is a 32x over-read on
    the block's 32 slices.
    """
    t = {"dram_rd": 0, "dram_wr": 0, "l1_rd": 0, "l1_wr": 0}
    out = shape(r, "OUTPUT_0")
    out_t = tiles(out) if out else 0
    out_b = out_t * TILE_BYTES.get((r.get("OUTPUT_0_DATATYPE") or "").strip(), 2048)
    slots, n_out = split_slots(r)
    if r["OP CODE"] in NO_COMPUTE:
        for _, _, _, _, mem in slots[:1] or [(0, 0, 0, "", "DRAM")]:
            t["dram_rd" if "DRAM" in (mem or "") else "l1_rd"] += out_b
    else:
        for _, _sh, tl, dt, mem in slots:
            b = tl * TILE_BYTES.get(dt.strip(), 2048)
            t["dram_rd" if "DRAM" in (mem or "") else "l1_rd"] += b
    if out:
        t["dram_wr" if "DRAM" in (r.get("OUTPUT_0_MEMORY") or "") else "l1_wr"] += n_out * out_b
    return t


def window(csv_gz, census_json):
    cen = json.load(open(census_json))
    sig = [o["op"] for o in cen["ops"]]
    rows = list(csv.DictReader(gzip.open(csv_gz, "rt")))
    codes = [r["OP CODE"] for r in rows]
    n = len(sig)
    hits = [i for i in range(len(codes) - n) if codes[i:i + n] == sig]
    if not hits:
        raise SystemExit(f"signature of {census_json} not in {csv_gz}")
    win = next((h for h, nx in zip(hits, hits[1:]) if nx - h == n), hits[-1])
    return cen, rows[win:win + n]


def analyse(tag, cen, rows):
    span = cen["median_ms"]["span_ms"]
    tot = defaultdict(float)
    per_class = defaultdict(lambda: defaultdict(float))
    sites = defaultdict(lambda: defaultdict(float))
    for r in rows:
        cores = max(int(f(r, "CORE COUNT")), 1)
        arr, mac, pack = arrivals(r, cores)
        tr = traffic(r)
        wait = f(r, "DEVICE COMPUTE CB WAIT FRONT [ns]") / cores
        resv = f(r, "DEVICE COMPUTE CB RESERVE BACK [ns]") / cores
        t1 = f(r, "DEVICE TRISC1 KERNEL DURATION [ns]")
        ker = f(r, "DEVICE KERNEL DURATION [ns]")
        whole_in = sum(t for _, _, t, _, _ in operands(r))
        out = shape(r, "OUTPUT_0")
        whole_out = tiles(out) if out else 0
        key = (r["OP CODE"], ksrc(r))
        skey = key + (tuple("x".join(map(str, s)) for _, s, _, _, _ in operands(r)),
                      "x".join(map(str, out)) if out else "", cores)
        for d, k in ((tot, None), (per_class[key], None), (sites[skey], None)):
            d["n"] += 1
            d["arrivals_pc"] += arr
            d["macs_pc"] += mac
            d["packs_pc"] += pack
            d["whole_in_tiles"] += whole_in
            d["whole_out_tiles"] += whole_out
            for tk, tv in tr.items():
                d[tk] += tv
            d["wait_ns"] += wait
            d["resv_ns"] += resv
            d["trisc1_ns"] += t1
            d["kernel_ns"] += ker
    return {"tag": tag, "span_ms": span, "programs": len(rows),
            "total": dict(tot),
            "by_class": {f"{k[0]}::{k[1]}": dict(v) for k, v in per_class.items()},
            "sites": [dict(v, op=k[0], kernel=k[1], inputs=list(k[2]), output=k[3], cores=k[4])
                      for k, v in sites.items()]}


CORES = 110          # the cell's 11x10 compute grid
ROOF_GBS = 444.9     # MEASURED chip DRAM bandwidth on this BH processor (b2z-arch-deficit)


def lstsq(A, y):
    """Least squares by normal equations. The systems here are 2-4 wide."""
    n = len(A[0])
    M = [[sum(r[i] * r[j] for r in A) for j in range(n)] for i in range(n)]
    b = [sum(r[i] * v for r, v in zip(A, y)) for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(M[k][i]))
        M[i], M[p], b[i], b[p] = M[p], M[i], b[p], b[i]
        for k in range(i + 1, n):
            f = M[k][i] / M[i][i]
            for j in range(i, n):
                M[k][j] -= f * M[i][j]
            b[k] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def r2(A, y, x):
    m = sum(y) / len(y)
    ssr = sum((sum(a * c for a, c in zip(r, x)) - v) ** 2 for r, v in zip(A, y))
    return 1 - ssr / sum((v - m) ** 2 for v in y)


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for p, i in enumerate(o):
            r[i] = p
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def report(d):
    t = d["total"]
    L = [f"\n=== {d['tag']}: {d['programs']} programs, span {d['span_ms']:.4f} ms (BH, qb2 card 0)"]
    L.append(f"  CB tile arrivals per core-averaged block   {t['arrivals_pc']:14,.0f}")
    L.append(f"  tile-pair MACs per core                    {t['macs_pc']:14,.0f}")
    L.append(f"  pack events per core                       {t['packs_pc']:14,.0f}")
    dram_pc = t["dram_rd"] / 2048 / CORES
    l1_pc = t["l1_rd"] / 2048 / CORES
    L.append(f"    of the arrivals, crossing the DRAM interface  {dram_pc:10,.0f}"
             f"  ({100*dram_pc/t['arrivals_pc']:.1f} %)")
    L.append(f"    read out of L1-interleaved                   {l1_pc:10,.0f}"
             f"  ({100*l1_pc/t['arrivals_pc']:.1f} %)")
    L.append(f"    multicast or CB replications of those        "
             f"{t['arrivals_pc']-dram_pc-l1_pc:10,.0f}"
             f"  ({100*(t['arrivals_pc']-dram_pc-l1_pc)/t['arrivals_pc']:.1f} %)")
    L.append(f"  DRAM  read {t['dram_rd']/1e6:9,.1f} MB   write {t['dram_wr']/1e6:9,.1f} MB")
    L.append(f"  L1    read {t['l1_rd']/1e6:9,.1f} MB   write {t['l1_wr']/1e6:9,.1f} MB")
    w = t["wait_ns"]
    L.append(f"  MEASURED CB wait-front/core {w/1e6:.4f} ms   reserve-back {t['resv_ns']/1e6:.4f} ms"
             f"   TRISC1 {t['trisc1_ns']/1e6:.4f} ms")
    datum = t["arrivals_pc"] * NS_PER_TILE_BH
    dram_t = t["dram_rd"] / (ROOF_GBS * 1e9) * 1e9
    L.append(f"  arrivals x {NS_PER_TILE_BH} ns/tile        = {datum/1e6:8.4f} ms"
             + (f"  = {100*datum/w:6.1f} % of the wait" if w else ""))
    L.append(f"  DRAM read at the {ROOF_GBS} GB/s roof  = {dram_t/1e6:8.4f} ms"
             + (f"  = {100*dram_t/w:6.1f} % of the wait" if w else ""))
    if not w:
        return L, None
    S = [s for s in d["sites"] if s["trisc1_ns"] > 0]
    y = [s["wait_ns"] for s in S]
    fits = {
        "tile arrivals only": lambda s: [s["arrivals_pc"]],
        "tile-pair MACs only": lambda s: [s["macs_pc"]],
        "DRAM bytes only": lambda s: [s["dram_rd"]],
        "DRAM + L1 bytes": lambda s: [s["dram_rd"], s["l1_rd"]],
        "DRAM + L1 + per-program": lambda s: [s["dram_rd"], s["l1_rd"], s["n"]],
        "DRAM + L1 + tile arrivals": lambda s: [s["dram_rd"], s["l1_rd"], s["arrivals_pc"]],
    }
    L.append(f"\n  What predicts the per-site wait? {len(S)} sites that run a compute kernel:")
    best = None
    for name, fn in fits.items():
        A = [fn(s) for s in S]
        x = lstsq(A, y)
        L.append(f"    R2 = {r2(A, y, x):7.4f}   {name:28s} " +
                 "  ".join(f"{c:.5g}" for c in x))
        if name == "DRAM + L1 bytes":
            best = x
    L.append(f"\n  Two-bandwidth fit: DRAM {1/best[0]:,.1f} GB/s"
             f" ({100/best[0]/1e9/ROOF_GBS*1e9:.0f} % of the {ROOF_GBS} GB/s roof),"
             f" L1-interleaved {1/best[1]:,.1f} GB/s")
    dterm = sum(s["dram_rd"] for s in S) * best[0]
    lterm = sum(s["l1_rd"] for s in S) * best[1]
    L.append(f"    DRAM term {dterm/1e6:.4f} ms ({100*dterm/w:.1f} %)"
             f" + L1 term {lterm/1e6:.4f} ms ({100*lterm/w:.1f} %)"
             f" = {100*(dterm+lterm)/w:.1f} % of the wait")
    L.append("\n  Spearman rank correlation against the measured per-site wait:")
    for name, key in (("bytes delivered (DRAM + L1 read)", lambda s: s["dram_rd"] + s["l1_rd"]),
                      ("DRAM bytes read", lambda s: s["dram_rd"]),
                      ("CB tile arrivals per core", lambda s: s["arrivals_pc"]),
                      ("tile-pair MACs per core", lambda s: s["macs_pc"]),
                      ("programs in the site", lambda s: s["n"])):
        L.append(f"    {spearman(y, [key(s) for s in S]):+.3f}   {name}")
    return L, best


def main():
    art, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    cen, rows = window(art / "ops_perf_blocksum_qb2c0.csv.gz", art / "blocksum_census.json")
    res["PairformerLayer"] = analyse("PairformerLayer", cen, rows)
    cen, rows = window(art / "ops_perf_step_qb2c0.csv.gz", art / "step_census.json")
    res["DiffusionStep"] = analyse("DiffusionStep", cen, rows)
    (outdir / "tile_census.json").write_text(json.dumps(res, indent=1))

    lines = []
    for tag, d in res.items():
        L, _ = report(d)
        lines += L
    txt = "\n".join(lines)
    (outdir / "REPORT.txt").write_text(txt + "\n")
    print(txt)
    return
    for tag, d in res.items():
        t = d["total"]
        print(f"\n=== {tag}: {d['programs']} programs, span {d['span_ms']:.4f} ms (BH, qb2 card 0)")
        print(f"  whole-tensor input tiles  {t['whole_in_tiles']:15,.0f}")
        print(f"  whole-tensor output tiles {t['whole_out_tiles']:15,.0f}")
        print(f"  CB arrivals per core      {t['arrivals_pc']:15,.0f}")
        print(f"  tile-pair MACs per core   {t['macs_pc']:15,.0f}")
        print(f"  pack events per core      {t['packs_pc']:15,.0f}")
        print(f"  DRAM read {t['dram_rd']/1e6:9,.1f} MB  write {t['dram_wr']/1e6:9,.1f} MB"
              f"   |  L1 read {t['l1_rd']/1e6:9,.1f} MB  write {t['l1_wr']/1e6:9,.1f} MB")
        print(f"  measured CB wait-front/core {t['wait_ns']/1e6:8.4f} ms"
              f"   reserve-back {t['resv_ns']/1e6:.4f} ms   TRISC1 {t['trisc1_ns']/1e6:.4f} ms")
        if t["wait_ns"]:
            print(f"  arrivals x 20.82 ns = {t['arrivals_pc']*NS_PER_TILE_BH/1e6:.4f} ms"
                  f"  =  {100*t['arrivals_pc']*NS_PER_TILE_BH/t['wait_ns']:.1f} % of the wait")
            print(f"  MACs     x 20.82 ns = {t['macs_pc']*NS_PER_TILE_BH/1e6:.4f} ms"
                  f"  =  {100*t['macs_pc']*NS_PER_TILE_BH/t['wait_ns']:.1f} % of the wait")
        print(f"  DRAM read time at the measured 444.9 GB/s roof:"
              f" {t['dram_rd']/444.9e9*1e3:.4f} ms = {100*t['dram_rd']/444.9e9*1e3/(t['wait_ns']/1e6 or 1):.1f} % of the wait")
        gbps = (t["dram_rd"] + t["dram_wr"]) / (d["span_ms"] / 1e3) / 1e9
        print(f"  implied DRAM bandwidth over the span: {gbps:,.1f} GB/s"
              f"   ({100*gbps/444.9:.1f} % of the measured 444.9 GB/s roof)")


if __name__ == "__main__":
    main()
