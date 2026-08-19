#!/usr/bin/env python3
"""Rung table, per-region exponents and the 2026-08-13/08-14 comparison for one ladder JSON.

Exponent = ln(t_i/t_{i-1}) / ln(N_i/N_{i-1}) per consecutive rung pair, per region. The wall
hides a capacity cliff (boltz2 measured N^2.03 at the wall while TriAtt went N^2.03 -> N^3.62
in the same interval), so the diagnostic printed here is the PAIR: TriAtt against TriMul in the
SAME interval. TriMul carries no capacity gate in 128-1024 and sat at N^1.95-2.11 across all
four 08-13 intervals, so it is the control; a TriAtt exponent exceeding its interval TriMul
exponent by more than ~0.5 is a dark gate.

References are literals copied from the two source docs, with their host/card/ttnn recorded,
because a cross-host absolute is not a result:
  boltz2   state/boltz2-sizes-perf.md 2.1           qb2 card 1, ttnn 0.68.0, 11x10
  esmfold2 state/esmfold2-sizes-perf.md + 8b6493b8  qb2 card 1, ttnn 0.68.0, 11x10
The esmfold2 512/768/1024 rows use the 08-14 re-measurement from 8b6493b8, not the 08-13
pre-lever walls; quoting 36.103/84.770/149.08 against today would manufacture a phantom win.
"""
import json, math, statistics as st, sys
from collections import OrderedDict

REF = {
 "boltz2":   {"host": "qb2 card 1, ttnn 0.68.0, 11x10 (boltz2-sizes-perf 2.1, 2026-08-13)",
              "wall": {128: 5.639, 256: 9.357, 512: 24.822, 768: 53.265, 1024: 95.315},
              "block": {128: 1018.5, 256: 3039.5, 512: 11798.0, 768: 31978.0, 1024: 60705.0},
              "triatt": {128: 231.8, 256: 703.2, 512: 2872.0, 768: 12456.1, 1024: 25323.3},
              "trimul": {128: 438.6, 256: 1456.4, 512: 5785.3, 768: 12741.3, 1024: 23382.6}},
 "esmfold2": {"host": "qb2 card 1, ttnn 0.68.0, 11x10 (esmfold2-sizes-perf 08-13 for 128/256; "
                      "8b6493b8 08-14 for 512/768/1024)",
              "wall": {128: 3.949, 256: 11.914, 512: 32.228, 768: 81.984, 1024: 147.241}},
}
REGIONS = ["wall", "block", "triatt", "trimul", "swiglu"]
WALLKEY = {"block": ("block:PairformerLayer", "block:PairUpdateBlock"),
           "triatt": ("body:TriangleAttention",),
           "trimul": ("body:TriangleMultiplication",),
           "swiglu": ("body:SwiGLUFFN", "body:SwiGLU")}


def expo(t0, t1, n0, n1):
    if not (t0 and t1):
        return None
    return math.log(t1 / t0) / math.log(n1 / n0)


def load(path):
    d = json.load(open(path))
    per = OrderedDict()
    for r in d["runs"]:
        if r.get("arm") == "cold" or "error" in r:
            per.setdefault(r["size"], {}).setdefault("errors", []).append(
                (r.get("arm"), r.get("error")))
            continue
        s = per.setdefault(r["size"], {})
        s.setdefault("folds", []).append(r["fold_s"])
        c = r.get("cif_sha256")
        s.setdefault("cifs", []).append(
            "/".join("{}={}".format(k, v) for k, v in sorted(c.items())) if isinstance(c, dict)
            else (c or "")[:16])
        s.setdefault("plddt", []).append(r.get("plddt"))
        for reg, keys in WALLKEY.items():
            for k in keys:
                w = r.get("wall_ms", {}).get(k)
                if w:
                    s.setdefault("reg", {}).setdefault(reg, []).append(w["ms"])
                    s.setdefault("calls", {})[reg] = w["calls"]
        s["last"] = r
    return d, per


def main(path, model):
    d, per = load(path)
    ref = REF.get(model, {})
    print("# {}  host={} card={} ttnn={} grid={} rec={} steps={}".format(
        model, d.get("host"), d.get("card"), d.get("ttnn"), d.get("grid"),
        d.get("recycling_steps"), d.get("sampling_steps")))
    print("# reference: {}".format(ref.get("host", "none")))
    sizes = sorted(per)
    med, a512 = {}, None
    for n in sizes:
        f = per[n].get("folds")
        if f:
            med[n] = st.median(f)
            if n == 512:
                a512 = med[n]
    r512 = ref.get("wall", {}).get(512)
    print()
    print("| size | warm folds s | median s | A/A spread s (%) | ratio vs own 512 | CIF sha256 | "
          "08-13/14 s | delta % | ref ratio | plDDT |")
    print("|---:|---|---:|---:|---:|---|---:|---:|---:|---|")
    for n in sizes:
        if n not in med:
            print("| {} | **FAILED** {} | | | | | | | | |".format(n, per[n].get("errors")))
            continue
        f, m = per[n]["folds"], med[n]
        sp = max(f) - min(f)
        rat = (m / a512) if a512 else float("nan")
        rw = ref.get("wall", {}).get(n)
        dl = ((m - rw) / rw * 100) if rw else None
        rr = (rw / r512) if (rw and r512) else None
        cif = per[n]["cifs"]
        cifs = cif[0] if len(set(cif)) == 1 else "MISMATCH " + "/".join(sorted(set(cif)))
        pl = list(map(str, per[n]["plddt"]))
        pls = pl[0] if len(set(pl)) == 1 else "/".join(pl)
        print("| {} | {} | **{:.3f}** | {:.3f} ({:.1f} %) | {:.4f} | `{}` | {} | {} | {} | {} |".format(
            n, " / ".join("{:.3f}".format(x) for x in f), m, sp, sp / m * 100, rat, cifs,
            "{:.3f}".format(rw) if rw else "-",
            "{:+.1f}".format(dl) if dl is not None else "-",
            "{:.4f}".format(rr) if rr else "-", pls))
    print()
    print("| interval | " + " | ".join(r + " exp" for r in REGIONS) + " | TriAtt-TriMul | verdict |")
    print("|---|" + "---:|" * (len(REGIONS) + 1) + "---|")
    for i in range(1, len(sizes)):
        n0, n1 = sizes[i - 1], sizes[i]
        if n0 not in med or n1 not in med:
            continue
        row, vals = [], {}
        for r in REGIONS:
            if r == "wall":
                e = expo(med[n0], med[n1], n0, n1)
            else:
                v0 = per[n0].get("reg", {}).get(r)
                v1 = per[n1].get("reg", {}).get(r)
                e = expo(st.median(v0), st.median(v1), n0, n1) if (v0 and v1) else None
            vals[r] = e
            row.append("{:.3f}".format(e) if e is not None else "-")
        gap = (vals["triatt"] - vals["trimul"]) if (vals["triatt"] and vals["trimul"]) else None
        vd = "-" if gap is None else ("**DARK GATE**" if gap > 0.5 else "clean")
        print("| {}->{} | {} | {} | {} |".format(
            n0, n1, " | ".join(row), "{:+.3f}".format(gap) if gap is not None else "-", vd))
    print()
    print("## reference exponents, same intervals, from the 08-13 doc")
    for r in ("wall", "block", "triatt", "trimul"):
        tab = ref.get(r)
        if not tab:
            continue
        ks = sorted(tab)
        es = ["{}->{}: {:.3f}".format(ks[i-1], ks[i], expo(tab[ks[i-1]], tab[ks[i]], ks[i-1], ks[i]))
              for i in range(1, len(ks))]
        print("  {:8s} ".format(r) + "  ".join(es))
    print()
    print("## lever census, by effect, per rung")
    for n in sizes:
        r = per[n].get("last")
        if not r:
            print("  {:5d}  NO WARM ARM: {}".format(n, per[n].get("errors")))
            continue
        hm, pm = r["head_major_qkv"], r["persistent_mask"]
        print("  {:5d} aa  K1 {}/{} {}  K1tail {}/{}".format(
            n, hm["served"], hm["declined"], hm["rejects"] or "", hm["tail_served"],
            hm["tail_declined"]))
        print("         K2 {}/{} q_split={} {} pm_over_l1={}".format(
            pm["served"], pm["declined"], pm["q_split"], pm["rejects"] or "",
            pm["pm_over_l1"] or []))
        print("         E6 gated {}  reblock_fwd {}  transpose_headroom={}".format(
            r["gated_kernel"], r["reblock_fwd"], r["transpose_l1_headroom"]))
        print("         sdpa_q_chunk_over_l1={}  pair_proj_l1_out={}  pair_proj_mm={}".format(
            r["sdpa_q_chunk_over_l1"] or [], r["pair_proj_l1_out"], r["pair_proj_mm"]))
        print("         fp32_softmax_chain={}".format(r["fp32_softmax_chain"]))
        for k, v in sorted(r.get("decisions", {}).items()):
            print("         DEC {:44s} {}".format(k, v))
        print("         loadavg={} maxrss={} MB  calls={}".format(
            r["loadavg"], r["maxrss_mb"], r.get("calls", {})))
        top = sorted(r.get("wall_ms", {}).items(), key=lambda kv: -kv[1]["ms"])[:8]
        for k, v in top:
            print("         WALL {:40s} {:10.2f} ms over {} calls".format(k, v["ms"], v["calls"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
