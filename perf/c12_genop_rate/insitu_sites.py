#!/usr/bin/env python3
"""In-situ per-site generic_op table, read out of c12-profiled-fold ops reports.

The 14 generic_op programs inside one PairformerLayer / MSALayer are identified by
COMPUTE KERNEL SOURCE, not by a census key label. Durations are DEVICE KERNEL
DURATION medians over the leg reps at a during-sampled 1350 MHz. FLOP per call comes
from c12-generic-op-attribution, which matched three of the four rates to the shipped
kernel arms (ship_in 36.507222016, ship_sdpa 68.719476736, ship_out 8.589934592 GFLOP)
to the byte.
"""
import csv, gzip, json, statistics as st, sys
from pathlib import Path

# site order inside one layer, confirmed by COMPUTE KERNEL SOURCE
SITES = [
    ("trimul_in", "minimal_matmul"),
    ("reblock_gated", "reblock_permute_gated"),
    ("reblock_gated", "reblock_permute_gated"),
    ("reblock_back", "reblock_permute"),
    ("trimul_in", "minimal_matmul"),
    ("reblock_gated", "reblock_permute_gated"),
    ("reblock_gated", "reblock_permute_gated"),
    ("reblock_back", "reblock_permute"),
    ("triatt_in", "minimal_matmul"),
    ("triatt_sdpa", "triatt_sdpa"),
    ("triatt_out", "minimal_matmul"),
    ("triatt_in", "minimal_matmul"),
    ("triatt_sdpa", "triatt_sdpa"),
    ("triatt_out", "minimal_matmul"),
]
# GFLOP per call, per site class
GFLOP = {
    "trimul_in": 42.949672960,      # 24.052 TFLOP / 560 calls
    "triatt_in": 36.507222016,      # == ship_in
    "triatt_sdpa": 68.719476736,    # == ship_sdpa
    "triatt_out": 8.589934592,      # == ship_out
    "reblock_gated": 0.0,
    "reblock_back": 0.0,
}
# MB moved per call, from genop_audit TB/fold / calls
MB = {
    "trimul_in": 0.2256e6 / 560,
    "triatt_in": 0.1974e6 / 560,
    "triatt_sdpa": 0.1515e6 / 560,
    "triatt_out": 0.0376e6 / 560,
    "reblock_gated": 0.2255e6 / 1120,
    "reblock_back": 0.0752e6 / 560,
}
LEGS = [("pfl_prof2", 264), ("msal_prof", 16)]


def leg(path, ncalls):
    rows = list(csv.DictReader(gzip.open(path, "rt")))
    g = [r for r in rows if r["OP CODE"] == "GenericOpDeviceOperation"]
    assert len(g) % 14 == 0, len(g)
    reps = len(g) // 14
    out = []
    for j in range(14):
        d = [float(g[k * 14 + j]["DEVICE KERNEL DURATION [ns]"]) / 1e6 for k in range(reps)]
        src = g[j]["COMPUTE KERNEL SOURCE"]
        want = SITES[j][1]
        assert want in src, (j, want, src)
        out.append(dict(idx=j, site=SITES[j][0], ms=st.median(d), lo=min(d), hi=max(d),
                        reps=reps, cores=int(g[j]["CORE COUNT"]),
                        avail=int(g[j]["AVAILABLE WORKER CORE COUNT"]),
                        fid=g[j]["MATH FIDELITY"], calls_fold=ncalls))
    return out


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gr")
    per_site = {}
    legs = {}
    for name, ncalls in LEGS:
        rec = leg(base / name / "ops.csv.gz", ncalls)
        legs[name] = rec
        for r in rec:
            s = per_site.setdefault(r["site"], dict(calls=0, sec=0.0, ms_each=[],
                                                    cores=set(), fid=set()))
            s["calls"] += ncalls
            s["sec"] += r["ms"] * ncalls / 1000.0
            s["ms_each"].append(r["ms"])
            s["cores"].add(r["cores"])
            s["fid"].add(r["fid"])
    hdr = "site             calls   ms/call   fold s       Mc  TFLOP/s     GB/s cores fidelity"
    print(hdr)
    tot_s = tot_c = 0.0
    table = []
    for site, s in sorted(per_site.items(), key=lambda kv: -kv[1]["sec"]):
        ms = st.median(s["ms_each"])
        tf = GFLOP[site] / (ms * 1e-3) / 1000.0 if GFLOP[site] else 0.0
        gbs = MB[site] / (ms * 1e-3) / 1000.0
        mc = s["sec"] * 1350.0
        tot_s += s["sec"]
        tot_c += mc
        table.append(dict(site=site, calls=s["calls"], ms=ms, sec=s["sec"], mc=mc,
                          tflops=tf, gbs=gbs, cores=sorted(s["cores"]),
                          fid=sorted(s["fid"]), gflop_call=GFLOP[site], mb_call=MB[site]))
        print("%-16s %5d %9.4f %8.4f %8.1f %8.2f %8.1f %s %s"
              % (site, s["calls"], ms, s["sec"], mc, tf, gbs,
                 sorted(s["cores"]), sorted(s["fid"])))
    print("%-16s %5d %9s %8.4f %8.1f"
          % ("TOTAL", sum(t["calls"] for t in table), "", tot_s, tot_c))
    fl = sum(t["gflop_call"] * t["calls"] for t in table) / 1000.0
    print("aggregate: %.3f TFLOP over %.4f s = %.2f TFLOP/s" % (fl, tot_s, fl / tot_s))
    json.dump(dict(table=table, legs=legs, total_s=tot_s, total_mc=tot_c,
                   tflop_fold=fl, agg_tflops=fl / tot_s),
              open(Path(__file__).parent / "insitu_sites.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
