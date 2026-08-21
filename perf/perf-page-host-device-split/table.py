#!/usr/bin/env python3
"""The six-row table: published fold ratio against the device-only reading.

Reads the published cells straight out of site/data/perf-512aa.json (never edits it) and
the measured TT host shares out of this directory's tt_<model>_*.json. RF3 is cited from
state/rf3-perf-page-cell.md rather than re-measured.

Device-only ratio is (P - Htt) / G. For five of the six rows G is ALREADY device-only --
the published GPU cell times the network forward with featurization outside it -- so the
only thing the reading needs from us is Htt. RF3 is the exception: both of its sides are
whole folds, so its device-only ratio uses the measured GPU device time.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BAR = 4.000

# RF3, from state/rf3-perf-page-cell.md: p150a 81.051 s whole fold with host featurization
# <= 8.33 s on the same Ryzen 7 9700X; H200 22.794 s whole fold = 12.459 s host + 7.746 s device.
RF3 = dict(p=81.051, g=22.794, htt=8.33, g_device=7.746, hgpu=12.459)


def main():
    page = json.loads((ROOT / "site" / "data" / "perf-512aa.json").read_text())
    cells = {m["id"]: m["cells"] for m in page["models"]}

    rows = []
    for f in sorted(HERE.glob("tt_*_qb2c*.json")):
        d = json.loads(f.read_text())
        m, s = d["model"], d.get("summary")
        if not s or m not in cells:
            continue
        p = cells[m]["p150a"]["s_per_fold"]
        g = cells[m]["h200"]["s_per_fold"]
        rows.append(dict(
            model=m, p=p, g=g, published=p / g,
            htt=s["host_s"], device=s["device_s"], transfer=s["transfer_s"],
            residual=s["residual_s"], dev_only=(p - s["host_s"]) / g,
            htt_to_flip=p - BAR * g,
            aa_floor_pct=s["aa_floor_pct"], plain=s["plain_median_s"],
            drift_pct=100 * (s["plain_median_s"] - p) / p,
        ))
    rows.append(dict(
        model="rf3", p=RF3["p"], g=RF3["g"], published=RF3["p"] / RF3["g"],
        htt=RF3["htt"], device=RF3["p"] - RF3["htt"], transfer=0.0, residual=0.0,
        dev_only=(RF3["p"] - RF3["htt"]) / RF3["g_device"],
        htt_to_flip=RF3["p"] - BAR * RF3["g"],
        aa_floor_pct=0.48, plain=RF3["p"], drift_pct=0.0,
    ))
    rows.sort(key=lambda r: r["published"])

    w = "| model | P (p150a) | G (H200) | published | Htt | device-only | Htt to flip in | verdict |"
    print(w)
    print("|" + "---|" * 8)
    for r in rows:
        v = ("in, unchanged" if r["published"] < BAR and r["dev_only"] < BAR else
             "out, unchanged" if r["published"] >= BAR and r["dev_only"] >= BAR else
             "**flips IN**" if r["dev_only"] < BAR else "**flips OUT**")
        need = ("already in" if r["htt_to_flip"] < 0 else f"{r['htt_to_flip']:.3f} s")
        htt = "<= %.3f" % r["htt"]
        print(f"| {r['model']} | {r['p']:.3f} | {r['g']:.3f} | {r['published']:.3f}x | {htt} | "
              f"**{r['dev_only']:.3f}x** | {need} | {v} |")

    print()
    print("| model | plain median here | published | drift | A/A floor | residual |")
    print("|" + "---|" * 6)
    for r in rows:
        if r["model"] == "rf3":
            continue
        print(f"| {r['model']} | {r['plain']:.3f} s | {r['p']:.3f} s | {r['drift_pct']:+.1f} % | "
              f"{r['aa_floor_pct']:.2f} % | {r['residual']:+.3f} s |")


if __name__ == "__main__":
    main()
