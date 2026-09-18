#!/usr/bin/env python3
"""CPU reducer: per-class curves, the A/A floor, and the fold-level conversion.

An op-level ratio is not a fold-level second. Any win here is converted through the 512 aa call
census (`perf/roof_launch/op_census_512.json`, via `percall_residual.json`) into fold seconds and
Mcycles at 1350 MHz before it is called anything.
"""
import argparse, json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
MHZ = 1350.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(HERE / "out" / "sweep.json"))
    ap.add_argument("--out", default=str(HERE / "out" / "curves.json"))
    a = ap.parse_args()
    d = json.loads(Path(a.sweep).read_text())
    rows = [r for r in d["rows"] if not r["warm"]]
    meta = d["meta"]

    dropped = [r for r in rows if "per_call_us" not in r or not r.get("clock", {}).get("pass")]
    good = [r for r in rows if r not in dropped]

    curves = {}
    for arm, info in meta["arms"].items():
        per = {}
        for r in good:
            if r["arm"] != arm:
                continue
            per.setdefault(r["grid"], []).append(r)
        entry = {"family": info["family"], "calls_per_fold": info.get("calls"),
                 "GFLOP_per_call": info["flops"] / 1e9,
                 "shape": info.get("shape"), "grid_mode": info["grid_mode"], "points": {}}
        for grid, rs in per.items():
            v = [r["per_call_us"] for r in rs]
            entry["points"][grid] = {
                "cores": rs[0]["cores"], "R": rs[0]["R"], "n_reps": len(v),
                "min_us": min(v), "median_us": st.median(v),
                "spread_pct": 100 * (max(v) - min(v)) / min(v),
                "issue_us": min(r["issue_us"] for r in rs),
                "TFLOPs_at_min": info["flops"] / (min(v) * 1e-6) / 1e12,
                "clock_min_MHz": min(r["clock"]["min_MHz"] for r in rs),
                "clock_max_MHz": max(r["clock"]["max_MHz"] for r in rs),
                "W_max": max((r["clock"]["W_max"] or 0) for r in rs)}
        full = entry["points"].get("11x10")
        aa = entry["points"].get("11x10#AA")
        if full and aa:
            entry["AA_floor_pct"] = 100 * abs(full["min_us"] - aa["min_us"]) / min(
                full["min_us"], aa["min_us"])
        scored = {k: v for k, v in entry["points"].items() if not k.endswith("#AA")}
        if scored:
            best = min(scored.items(), key=lambda kv: kv[1]["min_us"])
            entry["optimum_grid"] = best[0]
            entry["optimum_cores"] = best[1]["cores"]
            entry["optimum_us"] = best[1]["min_us"]
            entry["full_grid_us"] = scored["11x10"]["min_us"]
            entry["win_x"] = scored["11x10"]["min_us"] / best[1]["min_us"]
            entry["win_us_per_call"] = scored["11x10"]["min_us"] - best[1]["min_us"]
            c = sorted(scored.values(), key=lambda p: p["cores"])
            entry["scaling_16_to_110_x"] = c[0]["min_us"] / c[-1]["min_us"]
            entry["ideal_16_to_110_x"] = c[-1]["cores"] / c[0]["cores"]
            entry["occupancy_efficiency"] = entry["scaling_16_to_110_x"] / entry["ideal_16_to_110_x"]
            # what a 72-core grid costs or saves against the shipped 110
            if "9x8" in scored:
                entry["ratio_110_over_72"] = scored["11x10"]["min_us"] / scored["9x8"]["min_us"]
            if info.get("calls"):
                s = entry["win_us_per_call"] * 1e-6 * info["calls"]
                entry["fold_win_s"] = s
                entry["fold_win_Mcycles"] = s * MHZ
        # Amdahl fit t(c) = A/c + B over the scored ladder. B is the core-INDEPENDENT part of
        # the per-call cost: the part no number of cores removes. That is the whole mechanism
        # question, so it gets a fit rather than an eyeball on two endpoints
        # (`c10-two-points-cannot-measure-a-scaling-exponent`).
        pts = [(v["cores"], v["min_us"]) for k, v in entry["points"].items()
               if not k.endswith("#AA")]
        if len(pts) >= 3:
            xs = [1.0 / c for c, _t in pts]
            ys = [t for _c, t in pts]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            A = sxy / sxx
            B = my - A * mx
            pred = [A * x + B for x in xs]
            ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
            ss_tot = sum((y - my) ** 2 for y in ys)
            t110 = entry["points"]["11x10"]["min_us"]
            entry["amdahl"] = {
                "A_us_cores": A, "B_us_fixed": B, "r2": 1 - ss_res / ss_tot,
                "fixed_share_of_110_pct": 100 * B / t110,
                "parallel_part_at_110_us": A / 110.0,
                "headroom_to_infinite_cores_us": t110 - B,
                "headroom_to_infinite_cores_x": t110 / B if B > 0 else None}
        curves[arm] = entry

    # fold-level total over the matmul arms whose call counts are real census counts
    mm = {k: v for k, v in curves.items() if k.startswith("mm_")}
    total_s = sum(v.get("fold_win_s", 0.0) for v in mm.values())
    tri = {k: curves[k] for k in ("trimul", "triatt") if k in curves}
    tri_s = sum(v.get("fold_win_s", 0.0) for v in tri.values())

    # What the withdrawn Wormhole transfer would have cost if it had been applied here. The
    # ledger's 810 Mcycle entry said trimul wants 32 cores. On Blackhole both tri classes want
    # the full grid, so the transfer has the wrong sign and this is the size of the error.
    transfer = {}
    for target in ("8x4", "9x8"):
        loss = 0.0
        detail = {}
        for arm in ("trimul", "triatt"):
            e = curves.get(arm)
            if not e or target not in e["points"]:
                continue
            d_us = e["points"][target]["min_us"] - e["points"]["11x10"]["min_us"]
            s_ = d_us * 1e-6 * e["calls_per_fold"]
            detail[arm] = {"delta_us_per_call": d_us, "fold_s": s_, "Mcycles": s_ * MHZ}
            loss += s_
        transfer[target] = {"per_arm": detail, "fold_s": loss, "Mcycles": loss * MHZ,
                            "cores": int(target.split("x")[0]) * int(target.split("x")[1])}

    aa_floors = {k: v["AA_floor_pct"] for k, v in curves.items() if "AA_floor_pct" in v}
    out = {
        "meta": {k: meta[k] for k in ("host", "arch", "device_grid", "node", "card_env",
                                      "reps", "warm", "target_MHz", "started_utc",
                                      "finished_utc", "loadavg_start", "loadavg_end",
                                      "boot_id", "srcversion", "clock_samples",
                                      "force_response", "release_response")},
        "scored_rows": len(good), "dropped_rows": len(dropped),
        "dropped_detail": [{"arm": r["arm"], "grid": r["grid"],
                            "why": r.get("refused") or r.get("clock")} for r in dropped],
        "AA_floor_pct": aa_floors,
        "AA_floor_worst_pct": max(aa_floors.values()) if aa_floors else None,
        "curves": curves,
        "wormhole_transfer_cost": transfer,
        "fold_level": {
            "matmul_arms_win_s": total_s, "matmul_arms_win_Mcycles": total_s * MHZ,
            "tri_units_win_s": tri_s, "tri_units_win_Mcycles": tri_s * MHZ,
            "note": "Per-call win at each arm's own measured optimum times that arm's 512 aa call "
                    "count. An upper bound: it assumes every call of a shape can take that grid "
                    "and that nothing else in the fold changes. Cycles at 1350 MHz."},
    }
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"scored {len(good)} rows, dropped {len(dropped)}, worst A/A "
          f"{out['AA_floor_worst_pct']:.3f} %\n")
    hdr = f"{'arm':<14}{'family':<38}" + "".join(f"{g:>10}" for g in meta["ladder"])
    print(hdr)
    for arm, e in curves.items():
        line = f"{arm:<14}{e['family'][:37]:<38}"
        for g in meta["ladder"]:
            p = e["points"].get(g)
            line += f"{p['min_us']:>10.2f}" if p else f"{'-':>10}"
        print(line)
    print()
    print(f"{'arm':<14}{'opt grid':>10}{'opt us':>10}{'110 us':>10}{'win x':>8}"
          f"{'110/72':>9}{'16->110':>9}{'ideal':>7}{'occ eff':>9}{'fold s':>9}{'Mcyc':>8}")
    for arm, e in curves.items():
        if "optimum_grid" not in e:
            continue
        print(f"{arm:<14}{e['optimum_grid']:>10}{e['optimum_us']:>10.2f}"
              f"{e['full_grid_us']:>10.2f}{e['win_x']:>8.4f}"
              f"{e.get('ratio_110_over_72', float('nan')):>9.4f}"
              f"{e['scaling_16_to_110_x']:>9.3f}{e['ideal_16_to_110_x']:>7.2f}"
              f"{e['occupancy_efficiency']:>9.3f}"
              f"{e.get('fold_win_s', 0):>9.4f}{e.get('fold_win_Mcycles', 0):>8.1f}")
    print()
    print(f"{'arm':<14}{'issue us @110':>15}{'wall us @110':>14}{'host-bound?':>13}")
    for arm, e in curves.items():
        p = e["points"].get("11x10")
        if not p:
            continue
        print(f"{arm:<14}{p['issue_us']:>15.2f}{p['min_us']:>14.2f}"
              f"{('YES' if p['issue_us'] >= 0.9 * p['min_us'] else 'no'):>13}")
    print()
    print(f"{'arm':<14}{'A (us*cores)':>14}{'B fixed us':>12}{'r2':>8}{'B/t110 %':>10}"
          f"{'t110 us':>10}{'inf-core us':>12}")
    for arm, e in curves.items():
        m = e.get("amdahl")
        if not m:
            continue
        print(f"{arm:<14}{m['A_us_cores']:>14.1f}{m['B_us_fixed']:>12.2f}{m['r2']:>8.4f}"
              f"{m['fixed_share_of_110_pct']:>10.1f}"
              f"{e['points']['11x10']['min_us']:>10.2f}{m['B_us_fixed']:>12.2f}")
    print(f"\nmatmul arms, summed at each arm's own optimum: {total_s:.4f} s / "
          f"{total_s*MHZ:.1f} Mcycles of the 512 aa fold")
    print(f"tri units:  {tri_s:.4f} s / {tri_s*MHZ:.1f} Mcycles")
    for k, v in transfer.items():
        print(f"applying the withdrawn Wormhole optimum {k} ({v['cores']} cores) to both tri "
              f"classes would COST {v['fold_s']:.4f} s / {v['Mcycles']:.1f} Mcycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
