#!/usr/bin/env python3
"""Cross-rung census table: each item's bytes at every rung, its exponent measured from the
bytes (log-log slope between the smallest and largest rung that list it), and its bytes at
512/768/864 from that exponent anchored at the largest rung.

    table.py out/bh_hIL7RA_100/attrib.json out/bh_hIL2R_90/attrib.json ...
"""
import math, sys
import analyze as A

GROUP = {  # census item -> fold() items
    "Evoformer checkpoint pins (48 block inputs)": ["pin:evoformer"],
    "extra-MSA + template pins": ["pin:extra_msa", "pin:template_stack"],
    "one block's recompute: triangle multiplication x2": ["block:trimul"],
    "one block's recompute: triangle attention x2": ["block:triatt"],
    "one block's recompute: pair projections (trimul out, triatt bias/out)":
        ["block:pair_proj(trimul out, triatt bias/out)"],
    "one block's recompute: pair transition": ["block:pair_transition"],
    "one block's recompute: outer product mean": ["block:opm"],
    "one block's recompute: residual adds": ["block:residual"],
    "one block's recompute: MSA row attention pair bias + fp32 scores": ["block:msa_row_bias"],
    "one block's recompute: MSA track (2 rows)": None,   # every other block:* item
    "gradient accumulators": ["grads"],
    "leaf inputs (embedded pair, template, masks fed in)": ["leaf_inputs"],
    "weights + constants (no N)": ["weights+consts"],
    "raw device handles with N (masks, kernel scratch)": ["raw_N1", "raw_N2"],
    "unlisted (C++-held + page rounding)": ["unlisted(C++ held + page rounding)",
                                            "below top-200 groups"],
}


# Items whose exponent is known from the code rather than fitted: weights carry no N (a weight
# with 512 rows reads as N^1 at N=512), and the unlisted remainder, grads and raw handles are
# sampled at whatever node the walk caught, so their slope between two rungs is noise.
FIXED = {"weights + constants (no N)": 0, "unlisted (C++-held + page rounding)": 0,
         "gradient accumulators": 2, "raw device handles with N (masks, kernel scratch)": 2}


def grouped(r):
    taken, out = set(), {}
    for name, keys in GROUP.items():
        if keys is None:
            continue
        out[name] = sum(r["items"].get(k, 0) for k in keys)
        taken |= set(keys)
    out["one block's recompute: MSA track (2 rows)"] = sum(
        v for k, v in r["items"].items() if k not in taken and k.startswith("block:"))
    rest = sum(v for k, v in r["items"].items() if k not in taken and not k.startswith("block:"))
    if rest:
        out["other"] = rest
    return out


def main():
    rs = sorted((A.fold(p) for p in sys.argv[1:]), key=lambda r: r["axis"])
    gs = [(r["axis"], grouped(r), r) for r in rs]
    axes = [a for a, _, _ in gs]
    print("| item | " + " | ".join(f"{a}" for a in axes) + " | exp | @512 | @768 | @864 |")
    print("|---|" + "---|" * (len(axes) + 4))
    tot = {512: 0, 768: 0, 864: 0}
    for name in list(GROUP) + ["other"]:
        vals = [(a, g.get(name, 0)) for a, g, _ in gs]
        have = [(a, v) for a, v in vals if v > 1e6]
        if not have:
            continue
        if len(have) >= 2 and have[-1][0] != have[0][0]:
            (a0, v0), (a1, v1) = have[0], have[-1]
            e = math.log(v1 / v0) / math.log(a1 / a0)
        else:
            e = float("nan")
        a1, v1 = have[-1]
        if name in FIXED:
            e = FIXED[name]
        ex = round(e) if not math.isnan(e) and abs(e - round(e)) < 0.35 else e
        proj = {n: v1 * (n / a1) ** (ex if not math.isnan(ex) else 0) for n in tot}
        for n in tot:
            tot[n] += proj[n]
        print(f"| {name} | " + " | ".join(f"{v / 1e9:.3f}" for _, v in vals)
              + f" | {e:.2f} | {proj[512] / 1e9:.2f} | {proj[768] / 1e9:.2f} | {proj[864] / 1e9:.2f} |")
    print("| **walked peak (allocator `used`)** | " + " | ".join(
        f"**{r['peak'] / 1e9:.3f}**" for _, _, r in gs) + " | | "
        + " | ".join(f"**{tot[n] / 1e9:.2f}**" for n in tot) + " |")
    print("| largest free block at the peak | " + " | ".join(
        f"{r['largest_free'] / 1e6:.0f} MB" for _, _, r in gs) + " | | | | |")


if __name__ == "__main__":
    main()
