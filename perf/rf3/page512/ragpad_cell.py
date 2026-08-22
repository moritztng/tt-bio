#!/usr/bin/env python3
"""The missing cell, built from the run JSONs rather than transcribed.

Fills in `a1 + TT_BIO_SDPA_RAGGED_PAD` on RF3's 512 aa page row and reports both readings against
the page's existing GPU cells. The GPU side is NOT re-measured and NOT re-denominated
(`perf-page-matched-batch-protocol-recurrence`): H200 whole fold 22.794 s and device-only 7.746 s
are read straight off `perf/rf3/page512/page512_h200.json`.

The A/A control is printed FIRST and it is the pair of same-arm processes, not the within-process
spread: a fix that fires on nothing must land inside it.
"""
import argparse, json, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
H_WHOLE, H_DEV = 22.794, 7.746
# The page's own subtraction of TT host featurisation. Two figures, both measured, because the
# published one was taken at loadavg 25-26 and the quiet one applies to these folds.
HOST = {"page (loadavg 25-26)": 8.328, "quiet (loadavg 1.07)": 3.172}


def load(tag):
    p = HERE / f"rp_{tag}_qb2c1.json"
    d = json.loads(p.read_text())
    dig = sorted({v for f in d["folds"] for v in f["cif_sha256"].values()})
    cen = HERE / "census" / tag
    site, padded = {}, None
    for c in sorted(cen.glob("*.json")):
        j = json.loads(c.read_text())
        for k, v in j["sites"].items():
            s = site.setdefault(k, [0, 0])
            s[0] += v[0]; s[1] += v[1]
        padded = (padded or 0) + j["padded"]
    return dict(tag=tag, arm=d["arm"]["arm"], median=d["median_s"], warm=d["warm_walls_s"],
                spread_pct=d["spread_pct"], cif=dig, plddt=d["plddts"], sites=site, padded=padded,
                pad_flag=str(d["env_flags"].get("TT_BIO_SDPA_RAGGED_PAD", "0")).lower()
                in ("1", "true", "yes", "on"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    runs = {t: load(t) for t in a.tags}

    print("== per process ==")
    for t, r in runs.items():
        print(f"  {t:10s} arm={r['arm']} pad={r['pad_flag']} median={r['median']:8.3f} "
              f"warm={[round(x, 3) for x in r['warm']]} within={r['spread_pct']:.2f}% "
              f"cif={','.join(x[:8] for x in r['cif'])} sites={r['sites']} padded={r['padded']}")

    # A/A: same arm, same flag, different process.
    print("\n== A/A control, reported first ==")
    aa = {}
    keys = {}
    for t, r in runs.items():
        keys.setdefault((r["arm"], r["pad_flag"]), []).append(r)
    for k, rs in sorted(keys.items(), key=lambda kv: str(kv[0])):
        if len(rs) < 2:
            continue
        lo, hi = min(x["median"] for x in rs), max(x["median"] for x in rs)
        aa[k] = (hi - lo) / lo * 100
        print(f"  arm={k[0]} pad={k[1]}  {[round(x['median'], 3) for x in rs]}  "
              f"process-to-process spread {aa[k]:.2f}%")
    aa_max = max(aa.values()) if aa else float("nan")
    print(f"  -> A/A spread to beat: {aa_max:.2f}%")

    print("\n== the cell ==")
    hdr = f"{'arm':22s} {'s/fold':>8s} {'whole/H200':>10s}"
    for h in HOST:
        hdr += f" {'dev/H200 ' + h.split()[0]:>18s}"
    print(hdr)
    rows = {}
    for k, rs in sorted(keys.items(), key=lambda kv: str(kv[0])):
        med = statistics.median([w for r in rs for w in r["warm"]])
        name = f"{k[0]} + ragged pad" if k[1] else f"{k[0]} (pad off)"
        line = f"{name:22s} {med:8.3f} {med / H_WHOLE:9.3f}x"
        for h, host in HOST.items():
            line += f" {(med - host) / H_DEV:17.3f}x"
        print(line)
        rows[name] = dict(s_per_fold=round(med, 3), whole_x=round(med / H_WHOLE, 4),
                          dev_x={h: round((med - host) / H_DEV, 4) for h, host in HOST.items()},
                          n_warm=sum(len(r["warm"]) for r in rs))

    print("\n== the fix's cost at this cell ==")
    for arm in sorted({k[0] for k in keys}):
        off, on = keys.get((arm, False)), keys.get((arm, True))
        if not (off and on):
            continue
        mo = statistics.median([w for r in off for w in r["warm"]])
        mn = statistics.median([w for r in on for w in r["warm"]])
        pad = sum(r["padded"] or 0 for r in on)
        cif_same = sorted({c for r in off for c in r["cif"]}) == sorted({c for r in on for c in r["cif"]})
        verdict = ("INSIDE the A/A spread" if abs(mn / mo - 1) * 100 <= aa_max
                   else "OUTSIDE the A/A spread")
        print(f"  {arm}: off {mo:.3f} -> on {mn:.3f} = {mn / mo:.4f}x  ({verdict}); "
              f"ragged calls padded = {pad}; CIF bit-identical = {cif_same}")
        rows.setdefault("_cost", {})[arm] = dict(off=round(mo, 3), on=round(mn, 3),
                                                 ratio=round(mn / mo, 4), padded=pad,
                                                 cif_identical=cif_same,
                                                 aa_spread_pct=round(aa_max, 3),
                                                 inside_aa=abs(mn / mo - 1) * 100 <= aa_max)
    if a.out:
        a.out.write_text(json.dumps(dict(h200_whole=H_WHOLE, h200_device=H_DEV, tt_host=HOST,
                                         runs={t: r for t, r in runs.items()}, cell=rows),
                                    indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
