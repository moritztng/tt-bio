#!/usr/bin/env python3
"""Re-score the 298/768 aa screens with the roof that actually applies to each key.

Those two runs predate the L1 roof, so their L1-resident keys were gated against the measured DRAM
roof and every arm above it was dropped from the candidate set -- which is how the fc2 768 aa row
came to report 1.1147x when its own recorded arms contain 1.2439x. Nothing here is a new
measurement: the per-arm ms/TFLOPs/GB/s in the JSON are the numbers that were taken at the time, and
this only re-applies the gate. The L1 roof used is the one measured on the SAME card at the SAME
forced clock in the two later sessions of this pass (564.1 and 581.7 GB/s, the lower is used); it is
quoted as a cross-session gate rather than an in-session one, and it refuses nothing below, so the
choice does not decide any row.
"""
import json
import sys

L1_ROOF_CROSS_SESSION = 564.1     # lower of the two measured in-pass L1 roofs, same card, 1350 MHz


def rescore(path):
    d = json.load(open(path))
    R = d["roofs"]
    l1 = R.get("l1_rw_GBs", L1_ROOF_CROSS_SESSION)
    out = []
    for r in d["shapes"]:
        all_dram = r["amc"] == "DRAM" and r["omc"] == "DRAM"
        roof = R["dram_rw_GBs"] if all_dram else l1
        prod = r["arms"].get("prod", {}).get("ms")
        cand = {}
        for k, v in r["arms"].items():
            if k in ("prod", "prod_aa") or prod is None:
                continue
            if v["TFLOPs"] > 1.05 * R["compute_TFLOPs"] or v["GBs"] > 1.05 * roof:
                continue
            cand[k] = v
        if not cand:
            continue
        best = min(cand, key=lambda k: cand[k]["ms"])
        aa = r.get("aa_ratio")
        x = round(prod / cand[best]["ms"], 4)
        was = r["best"]["arm"] if r.get("best") else None
        out.append({"key": r["key"], "tokens": r.get("tokens", 512), "roof_GBs": roof,
                    "best": best, "x": x, "aa": aa, "was": was,
                    "was_x": r["best"]["x_vs_prod"] if r.get("best") else None,
                    "TFLOPs": cand[best]["TFLOPs"], "GBs": cand[best]["GBs"],
                    "result": (aa is None or abs(x - 1) > abs(aa - 1))})
    return d, out


for path in sys.argv[1:]:
    d, rows = rescore(path)
    print("==", path, "| roofs compute %.2f TF/s dram %.1f l1 %s GB/s"
          % (d["roofs"]["compute_TFLOPs"], d["roofs"]["dram_rw_GBs"],
             d["roofs"].get("l1_rw_GBs", "%.1f (cross-session)" % L1_ROOF_CROSS_SESSION)))
    for r in rows:
        moved = "" if r["was"] == r["best"] else "   <- WAS %s %.4fx" % (r["was"], r["was_x"])
        print("  %-16s tok=%-4d %-22s %.4fx  %6.2f TF/s %7.1f GB/s  aa=%-7s %s%s"
              % (r["key"], r["tokens"], r["best"], r["x"], r["TFLOPs"], r["GBs"],
                 r["aa"], "RESULT" if r["result"] else "inside floor", moved))
