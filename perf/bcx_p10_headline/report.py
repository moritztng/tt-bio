#!/usr/bin/env python3
"""The bcx-p10-headline sitting: the shipped tree, no lever env, N=3 on one qb2 chip, four times.

    report.py out/h1 out/h2 out/h3 out/h4 [--ref DIR ...] [--json out/report.json]

Per arm `perf/bcx_p10_tritraj/report.py::one` (pro-rata round, round 1 dropped, AICLK and
loadavg1 DURING the window). GO: median round <= 6.958 s. Levers: `build.fast` stamped and every
`lever_stats()` counter > 0. Equality: each (trajectory, round) digest identical across the arms,
and against every --ref arm (devtop's harness-armed N=3 lever arms) whose seed matches.
"""
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from perf.bcx_p10_tritraj.report import digests, one  # noqa: E402

BAR = 6.958


def args():
    a = sys.argv[1:]
    js = refs = None
    paths, refs = [], []
    it = iter(a)
    for x in it:
        if x == "--json":
            js = next(it)
        elif x == "--ref":
            refs.append(next(it))
        else:
            paths.append(x)
    return paths, refs, js


def served(ls):
    """Each lever's served counter, by name. The other slots are declines and fall-throughs."""
    ls = ls or {}
    tk = ls.get("taped_kernels", {})
    return {"mm_layout": ls.get("mm_layout", {}).get("served", 0),
            "taped_channel_move.fwd": (ls.get("taped_channel_move") or [0] * 4)[0],
            "taped_channel_move.back": (ls.get("taped_channel_move") or [0] * 4)[2],
            "widen_add": sum(v for k, v in ls.get("widen_add", {}).items() if k.startswith("served")),
            "rne_add": (ls.get("rne_add") or [0])[0],
            "qkv_grad_join": ls.get("qkv_grad_join", {}).get("served", 0),
            "triatt_bw": ls.get("triatt_bw", {}).get("served", 0),
            "triatt_fused_hifi": ls.get("triatt_fused_hifi", {}).get("served", 0),
            "taped_kernels.rne_add": (tk.get("rne_add") or [0])[0],
            "taped_kernels.tri_att_sdpa_hifi": (tk.get("tri_att_sdpa_hifi") or [0])[0]}


def main():
    paths, refs, js = args()
    rows, raw = [], {}
    for p in paths:
        r, d = one(p)
        rows.append(r)
        raw[r["arm"]] = d
        c = r.get("clock", {})
        print(f"{r['arm']:4s}  round {r.get('round_s', float('nan')):.3f} s  "
              f"held {r.get('held_per_round', float('nan')):.3f}  "
              f"idle {r.get('idle_per_round', float('nan')):.3f}  "
              f"busy {100 * r.get('gate_busy', float('nan')):.1f} %  "
              f"AICLK med {c.get('aiclk_med')} min {c.get('aiclk_min')}  "
              f"load1 {c.get('load1')}  hwm {r.get('host_hwm_gb')} GB  {r.get('error', '')}")
    ok = [r for r in rows if "round_s" in r]
    rs = [r["round_s"] for r in ok]
    s = {"arms": [r["arm"] for r in ok], "median_s": statistics.median(rs),
         "mean_s": statistics.mean(rs), "spread": [min(rs), max(rs)],
         "aiclk_med": statistics.median(r["clock"]["aiclk_med"] for r in ok),
         "aiclk_min": min(r["clock"]["aiclk_min"] for r in ok),
         "commit": sorted({r["commit"] for r in ok}), "card": sorted({str(r["card"]) for r in ok})}
    s["go"] = s["median_s"] <= BAR
    print(f"median {s['median_s']:.3f} s  mean {s['mean_s']:.3f}  spread "
          f"{s['spread'][0]:.3f}-{s['spread'][1]:.3f}  AICLK med {s['aiclk_med']} min "
          f"{s['aiclk_min']}  -> {'GO' if s['go'] else 'NO-GO'} (bar {BAR})")

    s["levers"] = {}
    for r in ok:
        st = raw[r["arm"]]["stamp"]
        sv = served(st.get("lever_stats"))
        s["levers"][r["arm"]] = {"fast": st.get("fast"), "served": sv,
                                 "lever_stats": st.get("lever_stats"),
                                 "unserved": sorted(k for k, v in sv.items() if not v > 0)}
        print(f"{r['arm']}: fast armed {sum(bool(v) for v in (st.get('fast') or {}).values())}"
              f"/{len(st.get('fast') or {})}  served {sv}  unserved: "
              f"{s['levers'][r['arm']]['unserved'] or 'none'}")
    s["all_served"] = all(not v["unserved"] and v["fast"] for v in s["levers"].values())

    ref = {}
    for r in ok:
        for k, h in digests(raw[r["arm"]]).items():
            ref.setdefault(k, set()).add(h)
    s["aa_equal"] = {"keys": len(ref), "outputs": sum(len(digests(raw[r["arm"]])) for r in ok),
                     "all_identical": all(len(v) == 1 for v in ref.values())}
    print(f"A/A: {s['aa_equal']['outputs']} outputs over {len(ref)} (trajectory, round) keys, "
          f"identical across arms: {s['aa_equal']['all_identical']}")

    seed = {raw[r["arm"]]["stamp"].get("seed") for r in ok}
    same = diff = 0
    used = []
    for p in refs:
        d = json.loads((pathlib.Path(p) / "round_events.json").read_text())
        if {d["stamp"].get("seed")} != seed:
            print(f"ref {p}: seed {d['stamp'].get('seed')} != {seed}, skipped")
            continue
        used.append(pathlib.Path(p).name)
        for k, h in digests(d).items():
            if k in ref:
                same, diff = (same + 1, diff) if ref[k] == {h} else (same, diff + 1)
    s["ref_equal"] = {"refs": used, "equal": same, "differ": diff}
    print(f"vs refs {used}: {same} equal, {diff} differ")
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": s}, indent=1,
                                               default=str))


if __name__ == "__main__":
    main()
