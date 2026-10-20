"""Per-kernel roofline of every generic_op the BindCraft 2 Evoformer issues, from a census.

The census (`perf/spd/bc2_evo_devprof.py --census`) credits generic_op 0 FLOPs and counts bytes
from the signature, which double-counts a preallocated output passed as an input. This labels each
signature with the tt-bio kernel that issues it, counts the bytes that kernel actually moves and
the matmul FLOPs it performs, and prices it against a FLOP roof and a DRAM roof:

    floor = max(bytes / BW, flops / PEAK)      time to floor = (ms - CORR - floor) x calls

Usage: python3 perf/bc4_trikern/roofline.py devprof.json [--peak 45e12] [--bw 438e9] [--json out]
"""
import argparse
import json
import re

TILE = 32
CORR_MS = 0.030          # per-call sync overhead the census includes (LEDGER convention)


def dims(sig):
    """[(shape, dtype)] of a census signature, in order."""
    return [(tuple(int(x) for x in s.split(",")), d)
            for s, d in re.findall(r"\[([\d, ]+)\]:(\w+)", sig)]


def nbytes(shape, dt):
    n = 1
    for s in shape:
        n *= s
    return n * (4 if dt == "FLOAT32" else 2)


def label(phase, sig, out):
    """(kernel, bytes moved, matmul FLOPs) for one census row, or None if unrecognised."""
    ins = dims(sig)
    o = dims(out)[0]
    shapes = [s for s, _ in ins]
    P = nbytes(o[0], o[1]) if len(o[0]) >= 3 else None
    if len(ins) == 9 and o[0][0] != 1 and len(o[0]) == 4 and o[1] == "FLOAT32":
        # q, k, v, dO, bias in; dqkv (packed [S, S, 3HD], the three slots are one tensor) and the
        # float32 dbias partial out. Recompute S = QK^T, dP = dO V^T, dV = P^T dO, dQ = dS K, dK = dS^T Q.
        S, H, N, D = ins[0][0]
        q = nbytes(*ins[0])
        by = 4 * q + nbytes(*ins[4]) + 3 * q + nbytes(*o)
        return "triatt_bw", by, 5 * 2 * S * H * N * N * D
    if len(ins) == 5 and len(shapes[0]) == 4 and shapes[3][0] == 1 and shapes[0][1] in (4, 8):
        S, H, N, D = shapes[0]
        return "triatt_sdpa", 4 * nbytes(*ins[0]) + nbytes(*ins[3]), 2 * 2 * S * H * N * N * D
    if len(ins) == 5 and len(shapes[1]) == 2 and len(shapes[0]) == 3:
        S1, S2, C = shapes[0]
        N3 = shapes[1][1]
        return "triatt_qkv", nbytes(*ins[0]) + 3 * nbytes(*ins[2]) + nbytes(*ins[1]), 2 * S1 * S2 * C * N3
    if len(ins) == 8 and shapes[1] == (128, 512):
        M = shapes[0][1] * shapes[0][2]
        return "pair_relu", 2 * nbytes(*ins[0]), 2 * 2 * M * 128 * 512
    if len(ins) == 8 and shapes[1][1] == 1024:
        M = shapes[0][1] * shapes[0][2]
        return "pair_relu_bw", 3 * nbytes(*ins[0]), 3 * 2 * M * 128 * 512
    if len(ins) == 2 and shapes[0][-1] == 512 and shapes[1][1] == 128:
        # p and g slices (slice_c channels each) of the projection in, the moved product out
        return "gated_move", 3 * P, 0
    if len(ins) == 3 and shapes[0][1] == 128 and shapes[1][-1] == 512:
        # cotangent in, p and g in, dp and dg out (into the packed slab)
        return "gated_move_bw", 5 * nbytes(*ins[0]), 0
    if len(ins) == 2 and shapes[0][1] == 128 and len(shapes[0]) == 4 and shapes[1][-1] == 128:
        return "permute_back", 2 * P, 0
    if len(ins) == 2 and shapes[0][-1] == 128 and shapes[1][1] == 128 and len(shapes[1]) == 4:
        return "permute_back_vjp", 2 * P, 0
    if len(ins) == 2 and len(shapes[0]) == 3 and shapes[0] == shapes[1] and shapes[0][-1] == 128:
        return "pair_transpose", 2 * P, 0
    if len(ins) == 2 and ins[0][1] == "FLOAT32" and shapes[0][0] > 1 and len(shapes[0]) == 4:
        return "lead_sum", nbytes(*ins[0]) + nbytes(*o), 0
    if any(s == (128,) or s == (256,) for s in shapes) and len(ins) == 4:
        return "lnbw", 3 * P, 0
    if len(ins) == 4 and any(s[-2:] == (1, 224, 128)[-2:] and s[1] == 1 for s in shapes if len(s) == 4):
        return "rne_add_masked", 3 * P, 0
    if len(ins) == 5 and len(set(shapes)) == 1:
        return "gate_bw", 5 * P, 0
    if len(ins) == 3 and len(set(shapes)) == 1:
        return ("rne_add" if shapes[0][-1] == 128 else "msa_add"), sum(nbytes(*i) for i in ins[:2]) + nbytes(*o), 0
    return "other", sum(nbytes(*i) for i in ins[:-1]) + nbytes(*o), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census")
    ap.add_argument("--peak", type=float, default=45e12)
    ap.add_argument("--bw", type=float, default=438e9)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = json.load(open(a.census))
    rows = []
    for op in d["census"]["ops"]:
        if op["name"] != "generic_op" or not op["phase"].startswith("evo"):
            continue
        k, by, fl = label(op["phase"], op["sig"], op["out"])
        ms = op["synced_s"] / op["calls"] * 1e3 - CORR_MS
        floor = max(by / a.bw, fl / a.peak) * 1e3
        rows.append(dict(kernel=k, phase=op["phase"].split(".")[1], calls=op["calls"], ms=ms,
                         s_it=ms * op["calls"] / 1e3, mb=by / 1e6, gflop=fl / 1e9,
                         gbs=by / ms / 1e6, tflops=fl / ms / 1e9, floor_ms=floor,
                         ttf_s=(ms - floor) * op["calls"] / 1e3, sig=op["sig"][:120]))
    rows.sort(key=lambda r: -r["ttf_s"])
    print(f"{'kernel':16} {'phase':8} {'calls':>5} {'ms':>6} {'s/it':>6} {'MB':>6} {'GFLOP':>6} "
          f"{'GB/s':>5} {'TF/s':>5} {'floor':>6} {'to-floor s/it':>13}")
    for r in rows:
        print(f"{r['kernel']:16} {r['phase']:8} {r['calls']:5d} {r['ms']:6.3f} {r['s_it']:6.4f} "
              f"{r['mb']:6.1f} {r['gflop']:6.2f} {r['gbs']:5.0f} {r['tflops']:5.1f} "
              f"{r['floor_ms']:6.3f} {r['ttf_s']:13.4f}")
    by_k = {}
    for r in rows:
        t = by_k.setdefault(r["kernel"], [0.0, 0.0, 0])
        t[0] += r["s_it"]; t[1] += r["ttf_s"]; t[2] += r["calls"]
    print("\nby kernel (s/it synced net of sync, time to floor s/it):")
    for k, (s, f, c) in sorted(by_k.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:16} calls {c:5d}  {s:6.4f}  to-floor {f:6.4f}")
    tot = sum(r["s_it"] for r in rows); ttf = sum(r["ttf_s"] for r in rows)
    print(f"  {'TOTAL':16} {sum(r['calls'] for r in rows):11d}  {tot:6.4f}  to-floor {ttf:6.4f}")
    if a.json:
        json.dump(dict(census=a.census, git_sha=d.get("git_sha"), host=d.get("host"),
                       peak=a.peak, bw=a.bw, corr_ms=CORR_MS, rows=rows, by_kernel=by_k),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
