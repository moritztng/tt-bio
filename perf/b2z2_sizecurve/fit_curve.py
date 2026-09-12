#!/usr/bin/env python3
"""Fit both arms' size curves from whatever `size_curve.py` has written, and say where they part.

The deliverable of this row is a PAIR OF EXPONENTS with their fit quality, not a single ratio, so
the fit is the artifact and the per-size ratios are the supporting table. Three things this does
that a bare `polyfit` would not:

  * **Every exponent comes with a standard error and the two arms are compared against it.** The
    pre-registered falsifier is "if the two arms' exponents agree within their fit error, the
    768 aa screen was load noise", and a slope without an error bar cannot fire it.
  * **Each size's ratio is quoted beside the A/A floor of ITS OWN session**, and both are the same
    estimator: a median over that size's own interleaved reps. Quoting a per-position floor beside
    a median-of-n ratio is the mistake `CONTEXT.md` names; the floor here is the worst
    base-position-vs-base-position median ratio inside the same process.
  * **A LOCAL exponent per adjacent pair**, which is what a knee actually is. A global fit averages
    a cliff away; `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa` found its cliff by comparing
    N^2.03 on one interval against N^3.62 on the next, and that only shows per interval.

Runs offline on the JSONs. Takes no device and asserts nothing about the box it runs on.
"""
import argparse
import json
import math
import statistics as st
from pathlib import Path


def loglog_fit(xs, ys):
    """Least-squares slope of log y on log x, with its standard error and R^2.

    Two points give a slope and no error bar, which is reported as such rather than as a zero.
    """
    n = len(xs)
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((v - mx) ** 2 for v in lx)
    sxy = sum((lx[i] - mx) * (ly[i] - my) for i in range(n))
    k = sxy / sxx
    c = my - k * mx
    resid = [ly[i] - (c + k * lx[i]) for i in range(n)]
    sst = sum((v - my) ** 2 for v in ly)
    sse = sum(r * r for r in resid)
    r2 = 1 - sse / sst if sst else None
    se = math.sqrt(sse / (n - 2) / sxx) if n > 2 else None
    return {"exponent": round(k, 4), "stderr": round(se, 4) if se is not None else None,
            "r2": round(r2, 5) if r2 is not None else None, "n_points": n,
            "prefactor": round(math.exp(c), 8)}


def per_size(path):
    d = json.loads(Path(path).read_text())
    warm = [r for r in d["runs"] if not r.get("cold")]
    if not warm:
        return None
    size = d.get("size") or warm[0]["size"]
    by = {}
    for r in warm:
        by.setdefault(r["arm"], []).append(r)
    if "base" not in by or "L1" not in by:
        return None

    # A/A floor: base at position 0 against base at every later position, same estimator as the
    # ratio (median over reps), taken the expensive way round so it can never read below 1.000x.
    pos = sorted({r["pos"] for r in by["base"]})
    p0 = [r["fold_s"] for r in by["base"] if r["pos"] == pos[0]]
    floors = {}
    for p in pos[1:]:
        v = [r["fold_s"] for r in by["base"] if r["pos"] == p]
        if v and p0:
            floors[p] = round(max(st.median(p0) / st.median(v), st.median(v) / st.median(p0)), 5)
    aa = max(floors.values()) if floors else None

    row = {"size": size, "arch": d["env"].get("arch"), "host": d["env"].get("host"),
           "grid": d["env"].get("grid"), "card": d["env"].get("tt_visible_devices"),
           "aa_floor": aa, "aa_floor_per_position": floors,
           "reps_complete": len(by["L1"]),
           "loadavg_1m_min_max": [min(r["loadavg_before"][0] for r in warm),
                                  max(r["loadavg_after"][0] for r in warm)],
           "arms": {}}
    for arm, rs in sorted(by.items()):
        fs = [r["fold_s"] for r in rs]
        sp = [r["step_s"] for r in rs if r["step_n"]]
        row["arms"][arm] = {
            "n": len(fs), "median_fold_s": round(st.median(fs), 3),
            "min_fold_s": round(min(fs), 3), "max_fold_s": round(max(fs), 3),
            "spread_pct": round(100 * (max(fs) - min(fs)) / st.median(fs), 2),
            "median_step_s": round(st.median(sp), 3) if sp else None,
            "atom_l1_l1": sorted({r["atom_l1"]["l1"] for r in rs}),
            "atom_l1_dram": sorted({r["atom_l1"]["dram"] for r in rs}),
            "step_n": sorted({r["step_n"] for r in rs}),
            "sha256": sorted({r["sha256"] for r in rs}),
        }
    b, l = row["arms"]["base"], row["arms"]["L1"]
    ratio = b["median_fold_s"] / l["median_fold_s"]
    row["ratio_vs_base"] = round(ratio, 5)
    row["above_aa_floor"] = bool(aa is None or ratio > aa)
    row["ratio_resolved"] = round(ratio, 5) if row["above_aa_floor"] else 1.0
    if b["median_step_s"] and l["median_step_s"]:
        row["step_ratio_vs_base"] = round(b["median_step_s"] / l["median_step_s"], 5)
    # Bit-exactness against BASE ON THIS BOX AND THIS SIZE. Never against tt_bio.reference.
    row["bit_exact_vs_base"] = (len(b["sha256"]) == 1 and b["sha256"] == l["sha256"])
    row["sha256"] = b["sha256"][0] if len(b["sha256"]) == 1 else b["sha256"]
    row["gate_declined"] = any(v for v in l["atom_l1_dram"])
    row["gate_took_l1_every_call"] = (l["atom_l1_dram"] == [0] and l["atom_l1_l1"] != [0])
    row["protocol_ok"] = (b["step_n"] == [200] and l["step_n"] == [200])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", help="keep only this arch, e.g. Wormhole_B0 or Blackhole")
    args = ap.parse_args()

    allrows = [r for r in (per_size(p) for p in args.jsons) if r]
    allrows.sort(key=lambda r: r["size"])
    groups = {}
    for r in allrows:
        groups.setdefault(str(r["arch"]), []).append(r)
    if args.arch:
        groups = {k: v for k, v in groups.items() if args.arch.lower() in k.lower()}
    assert groups, "no rows after grouping"
    if len(groups) > 1:
        # Two parts in one invocation: fit each separately and hand back a dict of results, so a
        # Wormhole point and a Blackhole point can never land on the same regression line.
        outs = {}
        for arch, rows in groups.items():
            sub = args.out.with_name(args.out.stem + "_" + arch.split(".")[-1] + args.out.suffix)
            outs[arch] = str(sub)
            _one(rows, sub)
        print(json.dumps({"per_arch_outputs": outs}, indent=1))
        return 0
    rows = next(iter(groups.values()))
    return _one(rows, args.out)


def _one(rows, outpath):
    out = {"doc": __doc__, "arch": rows[0]["arch"], "host": rows[0]["host"],
           "grid": rows[0]["grid"], "sizes": rows}

    fits = {}
    for arm in ("base", "L1"):
        xs = [r["size"] for r in rows]
        ys = [r["arms"][arm]["median_fold_s"] for r in rows]
        if len(xs) >= 2:
            fits[arm] = loglog_fit(xs, ys)
    out["fits"] = fits

    if len(fits) == 2 and fits["base"]["stderr"] and fits["L1"]["stderr"]:
        d = fits["base"]["exponent"] - fits["L1"]["exponent"]
        se = math.hypot(fits["base"]["stderr"], fits["L1"]["stderr"])
        out["exponent_separation"] = {
            "base_minus_l1": round(d, 4), "combined_stderr": round(se, 4),
            "sigma": round(d / se, 2) if se else None,
            # The brief's falsifier, evaluated rather than described.
            "falsifier_fired": bool(abs(d) <= se),
            "reading": ("exponents agree within fit error: the 768 aa screen was load noise"
                        if abs(d) <= se else
                        "exponents separate beyond fit error: the lever's value grows with N")}

    # A knee is a LOCAL thing. Global fits average it away.
    local = {}
    for arm in ("base", "L1"):
        seq = []
        for a, b in zip(rows, rows[1:]):
            ta = a["arms"][arm]["median_fold_s"]
            tb = b["arms"][arm]["median_fold_s"]
            seq.append({"interval": f"{a['size']}->{b['size']}",
                        "exponent": round(math.log(tb / ta) / math.log(b["size"] / a["size"]), 4),
                        "seconds": [ta, tb]})
        local[arm] = seq
    out["local_exponents"] = local

    out["gate"] = {r["size"]: {"declined": r["gate_declined"],
                               "l1": r["arms"]["L1"]["atom_l1_l1"],
                               "dram": r["arms"]["L1"]["atom_l1_dram"]} for r in rows}
    out["bit_exact"] = {r["size"]: r["bit_exact_vs_base"] for r in rows}
    out["ratio_curve"] = {r["size"]: {"ratio": r["ratio_vs_base"], "aa_floor": r["aa_floor"],
                                      "resolved": r["ratio_resolved"], "n": r["reps_complete"]}
                          for r in rows}
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("arch", "host", "grid", "fits", "exponent_separation", "local_exponents",
                       "gate", "bit_exact", "ratio_curve") if k in out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
