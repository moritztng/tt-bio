#!/usr/bin/env python3
"""Read a `perf/bwx_perf/sit.py` sitting and pair its arms by CONFIGURATION, not by order.

    report.py OUT [--json OUT/report.json]

`perf/bcx_default/report.py` groups by trajectory count, which is the question that sitting
asked. This one groups by `(extra_msa, template, trajectories)`, because the Wormhole question
is where the extra-MSA stack and the template embedder should run: on card, as every Blackhole
headline had them, or in JAX on the host, which is the shipped `campaign_predictor()` default.
On a Galaxy box the host is slow and shared, so the two arms can rank the other way round.

Per arm it prints the warm median wall and the host and device columns of the same round, from
the round meter's own events (`perf/bcx_round/analyze.py`'s attribution), with the AICLK
sampled DURING those rounds and the loadavg beside it. A round without its clock is not a
measurement on this workload, and on a shared Galaxy host neither is one without its load.
"""
import json
import pathlib
import statistics
import sys

H200 = 0.6958

#: Clock the arms are expected to hold. 1000 MHz is a Wormhole chip's ceiling, so a sample
#: under it is a throttled or contended chip; Blackhole's equivalent floor is 1200.
CLOCK_FLOOR = 1000


def warm_rounds(d):
    """(wall, device, host, aiclk, load) for every round after the compile round.

    device = the seam's own seconds (taped + backward + primal) inside `sequence_gradients`;
    host = everything else in the round. One slot only: an interleaved arm's rounds overlap
    and this attribution would double-count them, so N>1 arms are reported on wall alone.
    """
    ev = d["events"]
    clk = d.get("aiclk", [])
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        if starts[i]["round"] < 2:
            continue
        inside = [e for e in ev if e.get("t0") is not None and e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        dev = sum(e["dt"] for e in inside if e["kind"] == "device")
        s = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        out.append({"wall": t1 - t0, "device": dev, "host": (t1 - t0) - dev,
                    "aiclk": sorted(c for c, _ in s),
                    "load1": statistics.mean([l for _, l in s]) if s else None})
    return out


def one(path):
    p = pathlib.Path(path)
    d = json.loads((p / "round_events.json").read_text())
    st = d["stamp"]
    rounds = warm_rounds(d)
    clk = [c for r in rounds for c in r["aiclk"]]
    row = {"arm": p.name, "extra_msa": st.get("extra_msa_on_device"),
           # `perf/bcx_round/run_round.py` stamps no trajectory count: it runs exactly one.
           "template": st.get("template_on_device"), "n": st.get("trajectories", 1) or 1,
           "interleave": bool(st.get("interleave")), "commit": (st.get("commit") or "")[:9],
           "card": st.get("card"), "warm": len(rounds)}
    if not rounds:
        row["error"] = "no warm round"
        return row
    row.update(
        wall=statistics.median(r["wall"] for r in rounds),
        device=statistics.median(r["device"] for r in rounds),
        host=statistics.median(r["host"] for r in rounds),
        aiclk_med=statistics.median(clk) if clk else None,
        aiclk_min=min(clk) if clk else None,
        under_floor=sum(1 for c in clk if c < CLOCK_FLOOR), clk_n=len(clk),
        per_round=[{"wall": round(r["wall"], 3),
                    "load1": round(r["load1"], 2) if r["load1"] is not None else None}
                   for r in rounds],
        load1=round(statistics.median([r["load1"] for r in rounds if r["load1"] is not None]), 2)
        if any(r["load1"] is not None for r in rounds) else None)
    return row


def main():
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    root = pathlib.Path(sys.argv[1])
    rows = [one(p) for p in sorted(root.iterdir())
            if (p / "round_events.json").exists()]
    for r in rows:
        tag = f"{r['arm']:6s} extra_msa={int(bool(r['extra_msa']))} template={int(bool(r['template']))} N={r['n']}"
        if "error" in r:
            print(f"{tag}  {r['error']}")
            continue
        print(f"{tag}  wall {r['wall']:7.3f}  device {r['device']:7.3f}  host {r['host']:7.3f}  "
              f"warm {r['warm']:2d}  AICLK {r['aiclk_med']}/{r['aiclk_min']} "
              f"(<{CLOCK_FLOOR}: {r['under_floor']} of {r['clk_n']})  load1 {r['load1']}")
        # The round against the load that round ran at. On a shared Galaxy box an arm's own
        # XLA:CPU threads move loadavg1 by more than a co-tenant does, so a flat wall across a
        # rising load is what says the arm was not measuring the box's load.
        print("       per round: " + "  ".join(f"{x['wall']:.2f}s@{x['load1']}"
                                               for x in r["per_round"]))
    groups = {}
    for r in rows:
        if "error" not in r:
            groups.setdefault((bool(r["extra_msa"]), bool(r["template"]), r["n"]), []).append(r)
    summary = {}
    for key, rs in sorted(groups.items()):
        name = f"extra_msa={int(key[0])} template={int(key[1])} N={key[2]}"
        walls = [r["wall"] for r in rs]
        summary[name] = {
            "arms": [r["arm"] for r in rs], "wall": statistics.median(walls),
            "spread": [min(walls), max(walls)],
            "device": statistics.median([r["device"] for r in rs]),
            "host": statistics.median([r["host"] for r in rs]),
            "ratio_vs_h200": statistics.median(walls) / H200,
            "aiclk_med": statistics.median([r["aiclk_med"] for r in rs]),
            "aiclk_min": min(r["aiclk_min"] for r in rs),
            "load1": statistics.median([r["load1"] for r in rs if r["load1"] is not None] or [0])}
        s = summary[name]
        print(f"\n{name}: {s['wall']:.3f} s a round ({s['ratio_vs_h200']:.1f}x H200), spread "
              f"{s['spread'][0]:.3f}-{s['spread'][1]:.3f} over {len(rs)} arms, "
              f"host {s['host']:.3f} + device {s['device']:.3f}, "
              f"AICLK {s['aiclk_med']}/{s['aiclk_min']}, load1 {s['load1']}")
    if len(summary) == 2:
        (an, a), (bn, b) = sorted(summary.items(), key=lambda kv: -kv[1]["wall"])
        print(f"\n{bn} is {a['wall'] / b['wall']:.4f}x {an} on the round "
              f"({a['host'] - b['host']:+.3f} s host, {a['device'] - b['device']:+.3f} s device)")
        summary["ratio"] = {"faster": bn, "slower": an, "x": a["wall"] / b["wall"]}
    if js:
        pathlib.Path(js).write_text(json.dumps({"arms": rows, "summary": summary}, indent=1,
                                               default=str))


if __name__ == "__main__":
    main()
