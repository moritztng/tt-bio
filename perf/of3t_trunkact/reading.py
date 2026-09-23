#!/usr/bin/env python3
"""The reading of of3t-trunkact's forward walk: absolute error per block, its first difference,
and the pad/real split. Shares are published only beside the absolute number they came from."""
from __future__ import annotations
import argparse, json, statistics as st

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = json.load(open(a.walk))
    rows = d["per_block"]
    arms = list(d["arms"]) + ["bf16"]
    R = {"what": "of3t-trunkact: the forward divergence per block at padded N=384",
         "walk": a.walk, "host": d["host"], "tree": d["tree"],
         "tokens": d["tokens"], "real_tokens": d["real_tokens"],
         "controls": {
             "ref64_final_s_norm": d["final_ref64"]["s_norm"],
             "ref64_final_z_norm": d["final_ref64"]["z_norm"],
             "bf16_final_s_norm": d["final_bf16"]["s_norm"],
             "bf16_final_z_norm": d["final_bf16"]["z_norm"],
             "frame_REF_F64_N384_s_norm": 2547074.6682744888,
             "frame_REF_F64_N384_z_norm": 15590710.480237307,
             "frame_REF_BF16AUTO_N384_s_norm": 2545792.434019603,
             "frame_REF_BF16AUTO_N384_z_norm": 15590158.54011093}}
    R["controls"]["ref64_reproduces_frame"] = (
        R["controls"]["ref64_final_s_norm"] == R["controls"]["frame_REF_F64_N384_s_norm"]
        and R["controls"]["ref64_final_z_norm"] == R["controls"]["frame_REF_F64_N384_z_norm"])
    R["controls"]["bf16_reproduces_frame"] = (
        R["controls"]["bf16_final_s_norm"] == R["controls"]["frame_REF_BF16AUTO_N384_s_norm"]
        and R["controls"]["bf16_final_z_norm"] == R["controls"]["frame_REF_BF16AUTO_N384_z_norm"])

    T = {}
    for arm in arms:
        for leaf in ("s", "z"):
            for scope in ("padded", "pad", "real"):
                key = f"{arm}_{leaf}_{scope}"
                v = [r[scope][f"{arm}_{leaf}"]["abs_err"] for r in rows]
                T[key] = {"abs_err": v,
                          "first_difference": [None] + [v[i] - v[i-1] for i in range(1, len(v))]}
    R["series"] = T

    # the step statistic, on the window the brief names and on the whole ladder
    steps = {}
    for arm in arms:
        for leaf in ("s", "z"):
            for scope in ("padded", "real", "pad"):
                fd = T[f"{arm}_{leaf}_{scope}"]["first_difference"][1:]
                w = fd[40:48]                       # increments INTO blocks 40..47
                med = st.median([abs(x) for x in w])
                worst = max(range(len(fd)), key=lambda i: fd[i])
                steps[f"{arm}_{leaf}_{scope}"] = {
                    "window_40_47_increments": w,
                    "window_median_abs_increment": med,
                    "increment_into_45": fd[44],
                    "increment_into_45_over_window_median": (fd[44] / med) if med else None,
                    "largest_increment_block": worst + 1,
                    "largest_increment": fd[worst],
                    "H2_falsified": bool(med and fd[44] / med >= 2.0)}
    R["step"] = steps

    # ours over upstream's own bf16, per block, as a RATIO OF ABSOLUTE ERRORS
    R["over_upstream_bf16"] = {}
    for arm in d["arms"]:
        for leaf in ("s", "z"):
            for scope in ("padded", "real", "pad"):
                o = T[f"{arm}_{leaf}_{scope}"]["abs_err"]
                b = T[f"bf16_{leaf}_{scope}"]["abs_err"]
                R["over_upstream_bf16"][f"{arm}_{leaf}_{scope}"] = [
                    (o[i] / b[i]) if b[i] > 0 else None for i in range(len(o))]

    R["pad_share_of_squared_abs_err"] = {
        f"{arm}_{leaf}": [
            (T[f"{arm}_{leaf}_pad"]["abs_err"][i] ** 2)
            / (T[f"{arm}_{leaf}_padded"]["abs_err"][i] ** 2)
            for i in range(len(rows))]
        for arm in arms for leaf in ("s", "z")}

    R["final_block47"] = {
        f"{arm}_{leaf}_{scope}": rows[-1][scope][f"{arm}_{leaf}"]
        for arm in arms for leaf in ("s", "z") for scope in ("padded", "real", "pad")}

    json.dump(R, open(a.out, "w"), indent=2)

    def f(x): return "      -   " if x is None else f"{x:10.4g}"
    print("block | ship_s_pad_abs  d(ship_s)  | bf16_s_pad_abs | ship_s_real_abs  bf16_s_real")
    for i in range(len(rows)):
        print(f"{i:5d} | {f(T['ship_s_pad']['abs_err'][i])} {f(T['ship_s_pad']['first_difference'][i])} "
              f"| {f(T['bf16_s_pad']['abs_err'][i])} | {f(T['ship_s_real']['abs_err'][i])} "
              f"{f(T['bf16_s_real']['abs_err'][i])}")
    print()
    for k in ("ship_s_padded", "ship_s_pad", "ship_s_real", "ship_z_padded", "ship_z_real",
              "main_s_padded", "main_s_real", "bf16_s_padded", "bf16_s_real"):
        s = steps[k]
        print(f"{k:16s} inc_into_45={s['increment_into_45']:14.6g} "
              f"median|inc|(40..47)={s['window_median_abs_increment']:12.6g} "
              f"ratio={s['increment_into_45_over_window_median']!s:>22.22} "
              f"H2_falsified={s['H2_falsified']}")
    print()
    print("block47:", json.dumps({k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                      for kk, vv in v.items()}
                                  for k, v in R["final_block47"].items()
                                  if k.startswith(("ship_s", "bf16_s", "main_s"))}, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
