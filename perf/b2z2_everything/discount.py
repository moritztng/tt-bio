#!/usr/bin/env python3
"""The additivity discount, from the committed folds and against a bootstrapped floor.

Two things this deliberately does NOT do, both because `b2z2-trunk-fold-ab-bh` measured them wrong
on 2026-09-13:

  * it never multiplies a BLOCK or STEP ratio. Every factor below is a paired FOLD ratio measured
    in a session with its own interleaved base. Block ratios for three read-deleting levers were
    additive to 0.1 pp and their fold ratios were not, so that arithmetic is not available.
  * it never compares a ratio to a split-half or folded-median floor. The floor comes from
    `floor.py`, bootstrapped at the n the ratio is quoted at.

Two constructions, because no single session timed every arm:

  A  the product of the four in-session stage singles against the measured union.
  B  the host trio's MARGINAL inside one session -- ALL / BX -- against the trio standing alone.
     Both terms of the numerator clear their own session's floor, which is not true of A.

And the classification the amendment asks for: which levers delete a READER of a tensor somebody
else still reads. A lever pays when it deletes the LAST reader; one that deletes a proportional
share of readers does not reach the fold at its block ratio.

Host only. Reads the committed JSONs, invents nothing.
"""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
fold = json.loads((H / "fold_everything_wh_c8.json").read_text())
sing = json.loads((H / "singles_everything_wh_c8.json").read_text())
floors = {s["file"]: s for s in json.loads((H / "floor.json").read_text())["sessions"]}

F, S = fold["paired_ratio"], sing["paired_ratio"]
FF = floors["fold_everything_wh_c8.json"]["floor_all_pairs"]["hi"]
SF = floors["singles_everything_wh_c8.json"]["floor_all_pairs"]["hi"]

# Does the lever delete the LAST reader of what it removes, or a share of the readers?
READERS = {
    "SHG": ("shared", "replaces the one-hot gather matmul; the gathered window still has "
                      "the attention behind it"),
    "KVP": ("shared", "projects each atom once instead of four times -- deletes 3 of 4 readers "
                      "of the same rows, not the last"),
    "L1": ("shared", "residency: the DRAM readers go away, the tensor's consumers do not"),
    "SDPAQ": ("none", "occupancy, not movement -- it deletes no reader at all"),
    "QKVG": ("shared", "the normed pair tensor is read once instead of twice, and a THIRD reader "
                       "remains (the 32-wide pair-bias projection)"),
    "PWA": ("shared", "the normed MSA rows stop being re-read 2*n_heads times; the pair tensor "
                      "keeps its other consumers"),
    "COND": ("last", "deletes a 201 MB host download outright -- nothing else reads it"),
    "ZINIT": ("last", "z_init never crosses PCIe; the resident trunk consumes it where it is built"),
    "CONF": ("last", "the confidence head reads the pair tensor the trunk already left on device"),
}
ARMS = {"SAMP": ("SHG", "KVP", "L1", "SDPAQ"), "TRUNK": ("QKVG",), "MSA": ("PWA",),
        "HOSTONLY": ("COND", "ZINIT", "CONF")}

out = {
    "fold_session": {"floor_hi": FF, "ratios": F},
    "singles_session": {"floor_hi": SF, "ratios": S},
    "readers": {k: {"class": v[0], "why": v[1]} for k, v in READERS.items()},
}

prod = 1.0
for a in ARMS:
    prod *= S[a]
out["A_in_session_singles"] = {
    "product": round(prod, 5), "measured_union": F["ALL"],
    "discount_pct": round(100 * (1 - F["ALL"] / prod), 3),
    "caveat": "three of the four factors are inside their session's bootstrapped floor"}

bx_prod = S["SAMP"] * S["TRUNK"] * S["MSA"]
out["A_bitexact_subset"] = {
    "product": round(bx_prod, 5), "measured": F["BX"],
    "discount_pct": round(100 * (1 - F["BX"] / bx_prod), 3),
    "note": "BX itself clears its session's floor by %.1fx; none of its three factors clears "
            "theirs" % ((F["BX"] - 1) / (FF - 1))}

marg = F["ALL"] / F["BX"]
out["B_host_marginal"] = {
    "marginal_on_top_of_BX": round(marg, 5), "alone_same_box": S["HOSTONLY"],
    "alone_published": 1.06495,
    "discount_vs_alone_pct": round(100 * (1 - marg / S["HOSTONLY"]), 3),
    "discount_vs_published_pct": round(100 * (1 - marg / 1.06495), 3)}

out["readable_vs_bootstrapped_floor"] = {
    a: {"ratio": S[a], "floor_hi": SF, "readable": S[a] > SF,
        "reader_class": sorted({READERS[k][0] for k in ARMS[a]})}
    for a in ARMS}
out["readable_vs_bootstrapped_floor"]["BX"] = {
    "ratio": F["BX"], "floor_hi": FF, "readable": F["BX"] > FF,
    "reader_class": ["shared", "none"]}
out["readable_vs_bootstrapped_floor"]["ALL"] = {
    "ratio": F["ALL"], "floor_hi": FF, "readable": F["ALL"] > FF,
    "reader_class": ["last", "none", "shared"]}

out["falsifier"] = {
    "bar": "union more than 5 % BELOW the product of the surviving singles",
    "worst_discount_pct": max(out["A_in_session_singles"]["discount_pct"],
                              out["B_host_marginal"]["discount_vs_alone_pct"]),
    "fires": max(out["A_in_session_singles"]["discount_pct"],
                 out["B_host_marginal"]["discount_vs_alone_pct"]) > 5.0}

(H / "discount.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in (
    "A_in_session_singles", "A_bitexact_subset", "B_host_marginal",
    "readable_vs_bootstrapped_floor", "falsifier")}, indent=1))
