"""Amdahl projection of the TT_BIO_LPX prototype onto a Protenix-v2 fold, from lpx-census's ops.json.

usage: python project.py OPS_JSON [FOLD_S]

Every census signature is matched to the first lever below whose class and op match; its device
seconds are divided by that lever's measured per-op speedup. Signatures no lever matches, the
signatures census did not list (coverage < 1), and the fold's non-kernel time (FOLD_S minus the
census device kernel time: host, dispatch, idle) are the unchanged remainder.

The factors are what the PROTOTYPE runs, not the best arm a sweep found: e.g. trimul einsums get
acc-off only (1.18x), not the 768-pad block-width arm (3.82x), which the prototype does not carry.
"""
import json
import sys

# (class substring, op substring, speedup, source). First match wins; "" matches anything.
LEVERS = [
    ("triangle attention", "scaled_dot_product_attention", 41.7 / 20.92,
     "lpx-sdpa reuse1: mask-reuse kernel q192 k384 bfp8 LoFi, 41.7 -> 20.92 ms"),
    ("transitions", "linear", 1.75,
     "lpx-matmul r1 #0/#4/#6/#8: bfp8 weights + hidden, LoFi, acc off, 1.39-1.83x"),
    ("diffusion attention", "matmul", 10.6 / 0.93,
     "lpx-sdpa smoke: DiT fp32 explicit chain 10.6 ms -> stock bf16 SDPA 0.93 ms"),
    ("diffusion attention", "addalpha", 10.6 / 0.93, "same chain"),
    ("diffusion attention", "softmax", 10.6 / 0.93, "same chain"),
    ("diffusion attention", "linear", 2.1, "lpx-matmul: diffusion projections fp32 -> bf16 LoFi no acc, 28.3 -> 13.5 s"),
    ("linear/projection (other)", "linear", 2.1, "same (diffusion conditioning projections)"),
    ("outer product mean", "matmul", 22.6 / 9.9, "lpx-matmul #3: z_rows b8b8 + standalone cast"),
    ("triangle multiplication", "matmul", 1.18, "lpx-matmul r3 #10: einsum fp32 acc off"),
    ("pair-weighted averaging", "linear", 30.0 / 28.9, "lpx-matmul: PWA ckc only"),
    ("pair-weighted averaging", "matmul", 30.0 / 28.9, "same"),
    ("atom transformer", "linear", 12.8 / 9.0, "lpx-matmul: atom matmuls bf16"),
    ("atom transformer", "matmul", 18.9 / 14.8, "lpx-sdpa smoke: atom attention core bf16"),
    ("pairformer attention", "linear", 1.0, "projections unchanged"),
    ("pairformer attention", "matmul", 3.20 / 1.19, "lpx-sdpa smoke: APB explicit -> fused SDPA"),
    ("pairformer attention", "softmax", 3.20 / 1.19, "same"),
]


def lever(cls, op):
    for c, o, f, src in LEVERS:
        if c in cls and o in op:
            return f, src
    return 1.0, None


def main(path, fold_s=623.9):
    d = json.load(open(path))
    kern = d["device_kernel_s"]
    listed = sum(r["device_s_full_fold"] for r in d["ops"])
    rows = {}
    for r in d["ops"]:
        f, src = lever(r["cls"], r["op"])
        if src is None:
            continue
        key = (r["cls"], r["op"].replace("ttnn.", ""))
        a = rows.setdefault(key, [0.0, 0.0, f])
        a[0] += r["device_s_full_fold"]
        a[1] += r["device_s_full_fold"] / f
    moved = sum(a[0] for a in rows.values())
    after = sum(a[1] for a in rows.values())
    print(f"census {path}: device kernel {kern:.1f} s, listed {listed:.1f} s "
          f"(coverage {d['coverage']:.3f}), AICLK {d['aiclk']}")
    print("| class | op | today s | x | prototype s | saved s |\n|---|---|---|---|---|---|")
    for (c, o), (s0, s1, f) in sorted(rows.items(), key=lambda kv: kv[1][1] - kv[1][0]):
        print(f"| {c} | {o} | {s0:.1f} | {f:.2f} | {s1:.1f} | {s0 - s1:.1f} |")
    print(f"| **moved** | | **{moved:.1f}** | {moved / after:.2f} | **{after:.1f}** | **{moved - after:.1f}** |")
    rest_k = kern - moved
    print(f"unchanged: {listed - moved:.1f} s listed kernels no lever moves, {kern - listed:.1f} s "
          f"kernels outside the listed signatures, {fold_s - kern:.1f} s non-kernel (fold {fold_s} - kernel)")
    proj = fold_s - (moved - after)
    print(f"projected fold {proj:.1f} s = {fold_s / proj:.3f}x; kernel-only {kern:.1f} -> "
          f"{rest_k + after:.1f} s = {kern / (rest_k + after):.3f}x; "
          f"ceiling (moved ops free) {fold_s / (fold_s - moved):.3f}x")


if __name__ == "__main__":
    main(sys.argv[1], *(float(x) for x in sys.argv[2:]))
