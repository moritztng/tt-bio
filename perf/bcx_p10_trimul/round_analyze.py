#!/usr/bin/env python3
"""The round meter's event log split by arm: the device column per arm, and the wall beside it.

`perf/bcx_round/analyze.py` builds the rows; this only groups them by the arm the round ran
and reports both columns, because the wall and the device column do not drift together on a
loud box and only one of them can carry a 3 % lever.
"""
import json
import statistics as st
import sys


def rows_of(d):
    ev, clk = d["events"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t0") is not None and e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        sg_t = sum(e["dt"] for e in sg)
        dev = {p: [e for e in inside if e["kind"] == "device" and e["phase"] == p]
               for p in ("taped", "backward", "primal")}
        samples = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in samples)
        out.append({
            "round": starts[i]["round"], "arm": starts[i].get("arm", "-"),
            "wall": t1 - t0, "sg": sg_t,
            "taped_s": sum(e["dt"] for e in dev["taped"]),
            "bwd_s": sum(e["dt"] for e in dev["backward"]),
            "primal_s": sum(e["dt"] for e in dev["primal"]),
            "taped_n": len(dev["taped"]), "bwd_n": len(dev["backward"]),
            "aiclk_med": aiclk[len(aiclk) // 2] if aiclk else None,
            "aiclk_min": aiclk[0] if aiclk else None,
            "load1": round(sum(l for _, l in samples) / len(samples), 1) if samples else None,
        })
    for r in out:
        r["device"] = r["taped_s"] + r["bwd_s"] + r["primal_s"]
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def main(path):
    d = json.load(open(path))
    rows = rows_of(d)
    print(json.dumps({k: d["stamp"].get(k) for k in
                      ("commit", "card", "host", "arms", "arm_order", "dgrad_2d_stats",
                       "DEVICE_ZEROS", "DGRAD_2D_MINIMAL_end", "seed",
                       "length_bucket_size")}, default=str))
    print(f'\n{"rnd":>4}{"arm":>5}{"wall":>9}{"device":>9}{"taped":>9}{"bwd":>9}'
          f'{"host_sg":>9}{"clk":>6}{"load":>7}')
    for r in rows:
        print(f'{r["round"]:>4}{r["arm"]:>5}{r["wall"]:>9.3f}{r["device"]:>9.3f}'
              f'{r["taped_s"]:>9.3f}{r["bwd_s"]:>9.3f}{r["sg"] - r["device"]:>9.3f}'
              f'{str(r["aiclk_med"]):>6}{str(r["load1"]):>7}')
    body = [r for r in rows if r["round"] > 1]
    print(f'\nround 1 dropped for BindCraft 2 jit; {len(body)} rounds in the medians')
    arms = sorted({r["arm"] for r in body})
    hdr = "".join(f'{a:>12}' for a in arms)
    print(f'{"median":<12}{hdr}{"x (off/on)":>14}')
    for key, lab in (("wall", "round wall"), ("device", "device column"),
                     ("taped_s", "taped fwd"), ("bwd_s", "backward")):
        v = {a: med([r[key] for r in body if r["arm"] == a]) for a in arms}
        x = (v.get("off", 0) / v["on"]) if v.get("on") else 0
        print(f'{lab:<12}' + "".join(f'{v[a]:>12.3f}' for a in arms) + f'{x:>14.4f}')
    for a in arms:
        xs = sorted(r["device"] for r in body if r["arm"] == a)
        print(f'  device, arm {a}: n={len(xs)} min {xs[0]:.3f} max {xs[-1]:.3f} '
              f'p10 {xs[max(0, len(xs) // 10)]:.3f} p90 {xs[min(len(xs) - 1, 9 * len(xs) // 10)]:.3f}')
    o = [r["device"] for r in body if r["arm"] == "off"]
    n = [r["device"] for r in body if r["arm"] == "on"]
    if o and n:
        print(f'\n  every OFF round above every ON round: {min(o) > max(n)}')
        print(f'  delta of medians: {med(o) - med(n):+.3f} s of device per round')


if __name__ == "__main__":
    main(sys.argv[1])
