#!/usr/bin/env python3
"""Turn `tapeprof.py`'s event logs into the one number this row owes: a per-node rate.

The brief's complaint about the reading it inherited is that +4.86 GiB between two phase tags
is a BOUNDARY, and a boundary names nothing. So everything here is divided by the walk's own
progress -- retired nodes, and checkpoint recomputes -- and the artifact carries `avail_at_start`
and the load beside every peak, because pc's available memory swings ~9.8 to ~26 GiB and a peak
without it is not comparable to anything.

`--pair off,on` prints the A/B the shipped lever is graded on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load(tag: str) -> dict:
    recs = [json.loads(l) for l in (OUT / f"tape_{tag}.jsonl").read_text().splitlines() if l.strip()]
    hdr = next(r for r in recs if r.get("header"))
    ftr = next((r for r in recs if r.get("footer")), None)
    breach = next((r for r in recs if r.get("BREACH")), None)
    enters = [r for r in recs if r.get("ev") == "bw_enter" and r["d"] == 0]
    exits = [r for r in recs if r.get("ev") == "bw_exit" and r["d"] == 0]
    rcx = [r for r in recs if r.get("ev") == "bw_exit" and r["d"] > 0]
    cens = [r["census"] for r in rcx if r.get("census")]
    row = {"tag": tag, "arm": hdr["arm"], "tokens": hdr["tokens"], "cycles": hdr["cycles"],
           "samples": hdr["samples"], "exact": hdr["exact"], "trim_flag": hdr.get("trim_flag_arg"),
           "host": hdr["host"], "commit": None,
           "avail_at_start_gib": hdr["avail_at_start_gib"],
           "loadavg_at_start": hdr["loadavg_at_start"],
           "mem_total_gib": hdr["mem_total_gib"], "glibc": hdr.get("glibc"),
           "breached": bool(breach), "breach": (breach or {}).get("BREACH")}
    if enters and exits:
        e, x = enters[0], exits[0]
        dn = x["i"] - e["i"]
        row["backward"] = {
            "entry_rss_gib": e["rss_gib"], "exit_rss_gib": x["rss_gib"],
            "delta_gib": round(x["rss_gib"] - e["rss_gib"], 4),
            "nodes": dn, "blocks": max(r["rc"] for r in rcx) if rcx else 0,
            "wall_s": round(x["t"] - e["t"], 3),
            "mib_per_node": round((x["rss_gib"] - e["rss_gib"]) * 1024 / dn, 4) if dn else None}
        b = row["backward"]["blocks"]
        row["backward"]["mib_per_block"] = (
            round((x["rss_gib"] - e["rss_gib"]) * 1024 / b, 2) if b else None)
    elif enters:
        row["backward"] = {"entry_rss_gib": enters[0]["rss_gib"],
                           "note": "the walk did not return -- see `breach`"}
    if cens:
        row["census_first"], row["census_last"] = cens[0], cens[-1]
        row["census_host_gib_drift"] = round(cens[-1]["host_gib"] - cens[0]["host_gib"], 4)
        row["census_f64_gib_drift"] = round(cens[-1]["host_f64_gib"] - cens[0]["host_f64_gib"], 4)
        row["census_tape_node_drift"] = cens[-1]["tape_nodes"] - cens[0]["tape_nodes"]
    if ftr:
        c = ftr["counts"]
        row["vmhwm_gib"] = ftr["vmhwm_gib"]
        row["wall_s"] = ftr["wall_s"]
        row["trim_effective"] = ftr.get("trim_host_heap_effective")
        row["trim_calls"] = c.get("trim_calls", 0)
        row["trim_s"] = round(c.get("trim_s", 0.0), 4)
        row["trim_ms_per_call"] = (round(1000 * c["trim_s"] / c["trim_calls"], 2)
                                   if c.get("trim_calls") else None)
        row["gc_objects_reclaimed"] = c.get("gc_freed")
        row["gc_gib_reclaimed"] = round(c.get("gc_gib", 0.0), 4)
        row["malloc_trim_gib_reclaimed"] = round(c.get("trim_gib", 0.0), 4)
        row["mallinfo_at_end"] = ftr["mallinfo_at_end"]
        row["softmax_calls"] = {"forward": c["sm_fwd"], "backward": c["sm_bwd"]}
    step = OUT / f"step_{tag}.json"
    if step.is_file():
        s = json.loads(step.read_text())
        row["commit"] = (s.get("env") or {}).get("commit")
        row["aiclk_during"] = (s.get("env") or {}).get("aiclk_during") or s.get("clock")
        reps = s.get("reps") or []
        if reps:
            r = reps[-1]
            row["parts_s"] = {k: r[k] for k in
                              ("trunk_s", "diffusion_s", "losses_s", "seed_upload_s",
                               "backward_s", "optimizer_s", "step_s") if k in r}
            row["tape_nodes"] = r.get("tape_nodes")
            row["params_with_grad"] = r.get("params_with_grad")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--pair", default="", help="two tags, control first, for the A/B block")
    ap.add_argument("--out", default=str(OUT / "ANSWER.json"))
    a = ap.parse_args()
    rows = {t: load(t) for t in a.tags}
    ans = {"rows": rows}
    if a.pair:
        c, f = a.pair.split(",")
        rc, rf = rows[c], rows[f]
        ans["ab"] = {
            "control": c, "fixed": f,
            "tokens": rc["tokens"], "blocks": rc.get("backward", {}).get("blocks"),
            "peak_gib": [rc.get("vmhwm_gib"), rf.get("vmhwm_gib")],
            "peak_delta_gib": (round(rf["vmhwm_gib"] - rc["vmhwm_gib"], 4)
                               if rc.get("vmhwm_gib") and rf.get("vmhwm_gib") else None),
            "backward_delta_gib": [rc.get("backward", {}).get("delta_gib"),
                                   rf.get("backward", {}).get("delta_gib")],
            "mib_per_node": [rc.get("backward", {}).get("mib_per_node"),
                             rf.get("backward", {}).get("mib_per_node")],
            "mib_per_block": [rc.get("backward", {}).get("mib_per_block"),
                              rf.get("backward", {}).get("mib_per_block")],
            "backward_s": [rc.get("parts_s", {}).get("backward_s"),
                           rf.get("parts_s", {}).get("backward_s")],
            "trim_cost_s": rf.get("trim_s"), "trim_calls": rf.get("trim_calls"),
            "avail_at_start_gib": [rc["avail_at_start_gib"], rf["avail_at_start_gib"]],
            "NOTE": ("avail_at_start is published per arm because pc's available memory swings "
                     "and a peak read against a different amount of free memory is a different "
                     "measurement, not a comparison."),
        }
    Path(a.out).write_text(json.dumps(ans, indent=1, sort_keys=True))
    for t, r in rows.items():
        bw = r.get("backward", {})
        print(f"{t:>16} tok={r['tokens']} trim={r.get('trim_effective')} "
              f"peak={r.get('vmhwm_gib')} GiB bw_delta={bw.get('delta_gib')} GiB "
              f"{bw.get('mib_per_node')} MiB/node {bw.get('mib_per_block')} MiB/block "
              f"bw_s={r.get('parts_s', {}).get('backward_s')} "
              f"avail0={r['avail_at_start_gib']} breach={r['breached']}")
    if "ab" in ans:
        print("\nA/B:", json.dumps(ans["ab"], indent=1))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
