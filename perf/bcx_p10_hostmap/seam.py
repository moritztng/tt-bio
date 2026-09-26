#!/usr/bin/env python3
"""The round table and the JAX/device seam census, off `perf/bcx_round/meter.py`'s log.

Two things `attrib.py` cannot say, because both live outside the XLA program:

* the round table. Wall, `sequence_gradients`, the device callbacks, the host remainder,
  AICLK sampled DURING the round, loadavg1, and this process's CPU seconds and run-queue
  wait. Rounds inside the profiler's window are flagged, so the instrument's own cost is
  read off the same run rather than assumed.
* the seam census. How many times a round crosses into the callback, the bytes handed each
  way at the JAX face, and the split of the callback's own wall into marshalling,
  dispatch, `synchronize_device` and whatever is left (the autograd backward's dispatch,
  which is not wrapped because wrapping it would time tt-bio's own tape, not the seam).

A round is bounded by two consecutive entries into `sequence_gradients`, so the last round
collected has no closing boundary and is dropped; round 1 carries the jit compile and is
reported apart.
"""
import collections
import json
import statistics as st
import sys

MB = 1 << 20


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(path, out=None):
    d = json.load(open(path))
    ev, stamp, clk = d["events"], d["stamp"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    tr0 = next((e["t0"] for e in ev if e["kind"] == "trace_start"), float("inf"))
    tr1 = next((e["t0"] for e in ev if e["kind"] == "trace_stop"), float("inf"))
    sched = {e["round"]: e for e in ev if e["kind"] == "sched"}

    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        rnd = starts[i]["round"]
        inside = [e for e in ev if e.get("t0") is not None and e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        sg_t = sum(e["dt"] for e in sg)
        win = (sg[0]["t0"], sg[0]["t1"]) if sg else (t0, t0)
        dev = [e for e in inside if e["kind"] == "device"
               and win[0] <= e["t0"] and e["t1"] <= win[1]]
        cross = [e for e in inside if e["kind"] == "crossing"]
        seam = [e for e in inside if e["kind"] == "seam"]
        marshal = sum(e["dt"] for e in seam if e["phase"].startswith("marshal"))
        sync = sum(e["dt"] for e in seam if e["phase"] == "dispatch:sync")
        disp = sum(e["dt"] for e in seam if e["phase"] == "dispatch:evoformer")
        samples = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in samples)
        a, b = sched.get(rnd), sched.get(rnd + 1)
        rows.append({
            # Profiled means this round's `sequence_gradients` ran inside the trace window,
            # which is what `attrib.py` attributes. The round's own wall is wider than that
            # on both ends: the trace starts just after round A's boundary and stopping it
            # writes the xplane inside round B's wall.
            "round": rnd, "profiled": bool(tr0 <= win[0] and win[1] <= tr1),
            "wall": round(t1 - t0, 3), "sg": round(sg_t, 3),
            "device_s": round(sum(e["dt"] for e in dev), 3),
            "taped_n": sum(1 for e in dev if e["phase"] == "taped"),
            "bwd_n": sum(1 for e in dev if e["phase"] == "backward"),
            "host_in_sg": round(sg_t - sum(e["dt"] for e in dev), 3),
            "opt": round(sum(e["dt"] for e in inside if e["kind"] == "optimizer"), 3),
            "lower_compile": round(sum(e["dt"] for e in inside
                                       if e["phase"] == "lower_compile"), 3),
            "crossings": len(cross),
            "mb_to_host": round(sum(e["bytes_to_host"] for e in cross) / MB, 2),
            "mb_to_jax": round(sum(e["bytes_to_jax"] for e in cross) / MB, 2),
            "marshal_s": round(marshal, 3), "sync_s": round(sync, 3),
            "evo_dispatch_s": round(disp, 3),
            "cpu_s": round(b["cpu_s"] - a["cpu_s"], 1) if a and b else None,
            "wait_s": round(b["wait_s"] - a["wait_s"], 1) if a and b else None,
            "aiclk_med": aiclk[len(aiclk) // 2] if aiclk else None,
            "aiclk_min": aiclk[0] if aiclk else None, "n_clk": len(aiclk),
            "load1": round(sum(l for _, l in samples) / len(samples), 1) if samples else None})

    body = [r for r in rows if r["round"] > 1]
    prof = [r for r in body if r["profiled"]]
    ctrl = [r for r in body if not r["profiled"]]

    def block(rs):
        if not rs:
            return {}
        return {"n": len(rs), "wall_med": med([r["wall"] for r in rs]),
                "wall_min": min(r["wall"] for r in rs),
                "wall_max": max(r["wall"] for r in rs),
                "sg_med": med([r["sg"] for r in rs]),
                "device_med": med([r["device_s"] for r in rs]),
                "host_in_sg_med": med([r["host_in_sg"] for r in rs]),
                "aiclk_med": med([r["aiclk_med"] for r in rs]),
                "load1_med": med([r["load1"] for r in rs]),
                "cpu_s_med": med([r["cpu_s"] for r in rs if r["cpu_s"] is not None]),
                "wait_s_med": med([r["wait_s"] for r in rs if r["wait_s"] is not None])}

    per_call = collections.defaultdict(lambda: collections.Counter())
    for e in ev:
        if e["kind"] == "seam":
            per_call[e["call"]][e["phase"]] += e["dt"]
    n = max(len(body), 1)
    res = {"stamp": stamp,
           "round1_wall": rows[0]["wall"] if rows else None,
           "profiled": block(prof), "control_unprofiled": block(ctrl),
           "seam": {"crossings_per_round": med([r["crossings"] for r in body]),
                    "mb_to_host_per_round": med([r["mb_to_host"] for r in body]),
                    "mb_to_jax_per_round": med([r["mb_to_jax"] for r in body]),
                    "marshal_s_per_round": med([r["marshal_s"] for r in body]),
                    "sync_s_per_round": med([r["sync_s"] for r in body]),
                    "evo_dispatch_s_per_round": med([r["evo_dispatch_s"] for r in body]),
                    "by_call_s_per_round": {c: {k: round(v / n, 4)
                                                for k, v in sorted(t.items())}
                                            for c, t in per_call.items()}},
           "rounds": rows}
    print(json.dumps({k: res[k] for k in
                      ("round1_wall", "profiled", "control_unprofiled", "seam")}, indent=1))
    cols = ("round", "profiled", "wall", "sg", "device_s", "host_in_sg", "opt",
            "lower_compile", "crossings", "mb_to_host", "mb_to_jax", "marshal_s",
            "sync_s", "evo_dispatch_s", "cpu_s", "wait_s", "aiclk_med", "load1")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
