#!/usr/bin/env python3
"""Read one bc2_overlap.py run: steady-state round period, card-gate occupancy and the py-spy states.

    python3 perf/spd/bc2_overlap_read.py RUN [--skip 5]

The steady window starts once every slot has `--skip` gradient rounds behind it and ends at the
first slot's last gradient round; py-spy windows are cut out of it. Per slot it reports the round
period, and per round: time inside the card gate (held), time standing at its door (waited), and
the rest (outside). `gate busy` is the union of held intervals over the window: at 1.0 the card
gate is the serialiser and throughput is 1 / held-per-round whatever N is.
"""
import argparse, collections, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--skip", type=int, default=5)
a = ap.parse_args()
R = a.run
load = lambda f: [json.loads(l) for l in open(R / f)]  # noqa: E731
rounds, card, clk = load("rounds.jsonl"), load("card.jsonl"), load("aiclk.jsonl")
spy = json.loads((R / "spy.json").read_text())["windows"] if (R / "spy.json").exists() else []
cuts = [(w["t0"], w["t1"] + 2.0) for w in spy]
inspy = lambda t0, t1: any(t0 < c1 and t1 > c0 for c0, c1 in cuts)  # noqa: E731

by = collections.defaultdict(list)
for r in rounds:
    by[r["slot"]].append(r)
slots = sorted(by)
w0 = max(by[s][min(a.skip, len(by[s]) - 1)]["t0"] for s in slots)
w1 = min(by[s][-1]["t1"] for s in slots)


segs = [(w0, w1)]
if cuts:
    pts = sorted([w0, w1] + [x for c in cuts for x in c if w0 < x < w1])
    segs = [(x, y) for x, y in zip(pts, pts[1:]) if not any(x >= c0 and y <= c1 for c0, c1 in cuts)]
span = sum(y - x for x, y in segs)


def inside(t0, t1):
    return any(t0 >= x and t1 <= y for x, y in segs)


out = dict(run=str(R), window=[round(w0, 1), round(w1, 1)], span_s=round(span, 1), slots={})
total_rounds = 0
for s in slots:
    rr = [r for r in by[s] if inside(r["t0"], r["t1"])]
    total_rounds += len(rr)
    cc = [c for c in card if c["slot"] == s and inside(c["t_req"], c["t_rel"])]
    held = sum(c["t_rel"] - c["t_acq"] for c in cc)
    waited = sum(c["t_acq"] - c["t_req"] for c in cc)
    per = sum(r["t1"] - r["t0"] for r in rr) / max(len(rr), 1)
    n = max(len(rr), 1)
    tags = collections.Counter()
    for c in cc:
        tags[c["tag"]] += c["t_rel"] - c["t_acq"]
    out["slots"][s or "-"] = dict(rounds=len(rr), round_s=round(per, 3), held_per_round=round(held / n, 3),
                                  waited_per_round=round(waited / n, 3),
                                  outside_per_round=round(per - (held + waited) / n, 3),
                                  held_by_tag={k: round(v / n, 3) for k, v in tags.most_common()})
# gate occupancy: union of held intervals in the window
iv = sorted((c["t_acq"], c["t_rel"]) for c in card if inside(c["t_req"], c["t_rel"]))
busy, cur = 0.0, None
for x, y in iv:
    if cur and x <= cur[1]:
        cur[1] = max(cur[1], y)
    else:
        if cur:
            busy += cur[1] - cur[0]
        cur = [x, y]
if cur:
    busy += cur[1] - cur[0]
out["gate_busy"] = round(busy / span, 3) if span else None
out["chip_s_per_round"] = round(span / total_rounds, 3) if total_rounds else None
win = [c for c in clk if inside(c["t"], c["t"])]
chip_clk = sorted(c["clk"].get(str(3), c["clk"].get(3, 0)) for c in win) if win else []
out["aiclk"] = dict(median=chip_clk[len(chip_clk) // 2] if chip_clk else None, min=chip_clk[0] if chip_clk else None,
                    n=len(chip_clk))
out["load1"] = [round(min(c["load"] for c in win), 2), round(max(c["load"] for c in win), 2)] if win else None


# ---- py-spy. Seams run on jax host-callback threads (a pool, "Dummy-N"), so states are pooled over
# them; at N=2 each slot owns half by symmetry. Fractions are of wall time (sampler ticks), so in the
# --gil window the total is the fraction of wall that some trajectory thread holds the GIL.
def classify(stack):
    fr = stack.split(";")
    leaf, tail = fr[-1], ";".join(fr[-5:])
    # the card door is DeviceGate's RLock (PyThread_acquire_lock); CPython's take_gil is stripped
    # from the binary and shows as a bare pthread_cond_timedwait under whatever wanted the GIL back
    if "PyThread_acquire_lock" in tail:
        return "card_door"
    if re.search(r"take_gil|PyEval_RestoreThread|PyGILState_Ensure|pthread_cond_timedwait", tail):
        return "gil_wait"
    if re.search(r"synchroniz|Finish|wait_for_completion|read_buffer|EnqueueReadBuffer|read_from_device|"
                 r"to_torch|sync \(tt_bio", stack):
        return "device_wait"
    if re.search(r"xla::|XlaCpu|Eigen|tsl::|jaxlib|PjRt", tail):
        return "xla_cpu"
    if ".py:" not in leaf and re.search(r"ttnn|tt_metal|tt::|_ttnncpp|libtt_metal", stack):
        return "ttnn_native"
    if re.search(r"pthread_cond|futex|epoll|nanosleep|sem_wait", leaf):
        return "idle"
    return "python"


def spy_read(f):
    th = collections.defaultdict(collections.Counter)
    for line in open(f):
        stack, _, n = line.rstrip("\n").rpartition(" ")
        if stack:
            th[stack.split(";")[0]][classify(stack)] += int(n)
    return th


idle = spy_read(R / "spy_idle.txt") if (R / "spy_idle.txt").exists() else {}
ticks = max((sum(v.values()) for v in idle.values()), default=0)
for w in ("idle", "gil"):
    f = R / f"spy_{w}.txt"
    if f.exists() and ticks:
        pool = collections.Counter()
        for k, v in spy_read(f).items():
            if "Dummy" in k:
                pool.update(v)
        out[f"spy_{w}"] = dict(ticks=ticks, pooled_frac={k: round(v / ticks, 3) for k, v in pool.most_common()},
                               total=round(sum(pool.values()) / ticks, 3))
print(json.dumps(out, indent=1))
