#!/usr/bin/env python3
"""Split one arm`s host seconds into the four intervals of a round, and check they sum.

    PYTHONPATH=. python3 perf/bcx_p10_hostfloor/split.py perf/bcx_p10_hostfloor/out/composed

Round 1 carries BindCraft 2`s jit compile and is dropped, the rule every row of this
campaign uses. `host` is `sequence_gradients` wall minus the device callbacks, which is
the column `perf/bcx_p10_stack/compare.py` prints, so the two are the same number.
"""
import json
import pathlib
import statistics as st
import sys


def med(xs):
    return round(st.median(xs), 4) if xs else 0.0


def rng(xs):
    return f"{min(xs):.3f}-{max(xs):.3f}" if xs else "-"


def rounds_of(path):
    d = json.load(open(path))
    ev, stamp, clk = d["events"], d["stamp"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        disp = [e for e in inside if e["phase"] == "xla_dispatch"]
        blk = [e for e in inside if e["phase"] == "xla_block"]
        if not (sg and disp and blk):
            continue
        w0, w1 = sg[0]["t0"], sg[0]["t1"]
        dev = [e for e in inside if e["kind"] == "device"]
        devcpu = [e for e in inside if e["kind"] == "devcpu"]
        r = {"wall": t1 - t0, "sg": sg[0]["dt"],
             "dev": sum(e["dt"] for e in dev),
             "dev_cpu": sum(e["cpu"] for e in devcpu),
             "pre": disp[0]["t0"] - w0,
             "dispatch": disp[0]["dt"], "block": blk[0]["dt"],
             "exec": disp[0]["dt"] + blk[0]["dt"],
             "post": w1 - blk[0]["t1"],
             "outside_sg": (t1 - t0) - sg[0]["dt"],
             "exec_cpu": disp[0]["cpu"] + blk[0]["cpu"]}
        r["host"] = r["sg"] - r["dev"]
        r["xla_host"] = r["exec"] - r["dev"]
        r["xla_host_cpu"] = r["exec_cpu"] - r["dev_cpu"]
        # named Python, bucketed by which interval it fell in
        for e in inside:
            if e["kind"] not in ("pyfn", "loop", "host"):
                continue
            where = ("pre" if e["t1"] <= disp[0]["t0"] + 1e-6 else
                     "post" if e["t0"] >= blk[0]["t1"] - 1e-6 else
                     "outside" if (e["t0"] < w0 or e["t1"] > w1) else "exec")
            if e["phase"] in ("xla_dispatch", "xla_block"):
                continue
            r.setdefault("named", {}).setdefault(f"{where}:{e['phase']}", 0.0)
            r["named"][f"{where}:{e['phase']}"] += e["dt"]
        # where the host seconds sit relative to the device calls: the round is a chain of
        # `gap, callback, gap, callback, ...` and a gap is only overlappable if nothing in
        # it depends on the callback before it.
        seq, cur = [], w0
        for e in sorted(dev, key=lambda e: e["t0"]):
            seq.append(("gap", e["t0"] - cur))
            seq.append((f"dev:{e['phase']}:{e.get('module', '?')}", e["dt"]))
            cur = e["t1"]
        seq.append(("gap", w1 - cur))
        r["seq"] = seq
        s = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        aiclk = sorted(c for c, _ in s)
        r["aiclk"] = aiclk[len(aiclk) // 2] if aiclk else None
        r["aiclk_min"] = aiclk[0] if aiclk else None
        r["load1"] = round(sum(l for _, l in s) / len(s), 1) if s else None
        # process CPU over the round, from the sched stamps at the boundaries
        out.append(r)
    out = out[1:] or out                       # round 1 carries the jit compile
    # A round that blocked on BindCraft 2's compile flock measured the lock, not the
    # round. `tt_bio/main.py`'s stderr-filter fork leaves a child holding the open file
    # description the lock lives on, so `one_worker_compiles` does not release it and a
    # later round with the same compile shape waits on it forever.
    blocked = [r for r in out if r["pre"] > 1.0]
    return [r for r in out if r["pre"] <= 1.0], stamp, ev, blocked


def main(root):
    p = pathlib.Path(root) / "round_events.json"
    rows, stamp, ev, blocked = rounds_of(p)
    n = len(rows)
    ncpu = stamp.get("nproc")
    print(f"{root}: {n} timed rounds, round 1 dropped as the compile")
    if blocked:
        print(f"  DROPPED {len(blocked)} round(s) that blocked on the compile flock: "
              + ", ".join(f"{r['wall']:.1f} s wall, {r['pre']:.1f} s in the lock"
                          for r in blocked))
    print(f"  levers msa={stamp.get('extra_msa_on_device')} "
          f"tmpl={stamp.get('template_on_device')} "
          f"tri={stamp.get('triatt_taped_sdpa')}  card={stamp.get('card')} "
          f"host={stamp.get('host')} commit={stamp.get('commit', '')[:9]}")
    print(f"  AICLK med {med([r['aiclk'] for r in rows])} min "
          f"{min([r['aiclk_min'] for r in rows])}  load1 med {med([r['load1'] for r in rows])}")
    print()
    print(f"  {'interval':<34}{'med s':>9}{'range':>16}{'share of host':>15}")
    hostm = med([r["host"] for r in rows])
    def row(label, key, of_host=True):
        xs = [r[key] for r in rows]
        sh = f"{100 * med(xs) / hostm:>13.1f} %" if of_host and hostm else ""
        print(f"  {label:<34}{med(xs):>9.3f}{rng(xs):>16}{sh}")
    row("round wall", "wall", False)
    row("  sequence_gradients", "sg", False)
    row("  outside sequence_gradients", "outside_sg")
    row("device callbacks (the card)", "dev", False)
    row("HOST = sg - device", "host", False)
    print("  " + "-" * 60)
    row("  pre: argument Python", "pre")
    row("  exec: dispatch", "dispatch")
    row("  exec: block until ready", "block")
    row("    of which device callbacks", "dev", False)
    row("    XLA host compute = exec - dev", "xla_host")
    row("  post: answer Python", "post")
    row("  outside sg: BC2 loop body", "outside_sg")
    s = med([r["pre"] for r in rows]) + med([r["xla_host"] for r in rows]) + \
        med([r["post"] for r in rows]) + med([r["outside_sg"] for r in rows])
    print(f"  {'SUM of host intervals':<34}{s:>9.3f}{'':>16}"
          f"{100 * s / (hostm + med([r['outside_sg'] for r in rows])):>13.1f} %")
    print()
    print("  Where this process burns CPU, all threads, `os.times`:")
    dw, dc = med([r["dev"] for r in rows]), med([r["dev_cpu"] for r in rows])
    hw, hc = med([r["xla_host"] for r in rows]), med([r["xla_host_cpu"] for r in rows])
    print(f"    inside the device callbacks : wall {dw:6.3f} s  CPU {dc:7.3f} s  "
          f"= {dc / dw:5.2f} cores of {ncpu}")
    print(f"    XLA host compute           : wall {hw:6.3f} s  CPU {hc:7.3f} s  "
          f"= {hc / hw:5.2f} cores of {ncpu}")
    print()
    print("  named Python, median s per round:")
    keys = sorted({k for r in rows for k in r.get("named", {})},
                  key=lambda k: -med([r.get("named", {}).get(k, 0.0) for r in rows]))
    for k in keys:
        xs = [r.get("named", {}).get(k, 0.0) for r in rows]
        if med(xs) < 0.0005:
            continue
        print(f"    {k:<48}{med(xs):>8.4f}")
    tot = sum(med([r.get("named", {}).get(k, 0.0) for r in rows]) for k in keys)
    print(f"    {'named total':<48}{tot:>8.4f}")


    print()
    print("  the round as a chain, median s over the timed rounds (gap = host):")
    width = max(len(r["seq"]) for r in rows)
    same = [r for r in rows if len(r["seq"]) == width]
    for i in range(width):
        label = same[0]["seq"][i][0]
        xs = [r["seq"][i][1] for r in same]
        print(f"    {i:>2}  {label:<40}{med(xs):>8.3f}{rng(xs):>16}")
    ghost = sum(med([r["seq"][i][1] for r in same])
                for i in range(width) if same[0]["seq"][i][0] == "gap")
    print(f"        {'host gaps total':<40}{ghost:>8.3f}   over {len(same)} rounds")


if __name__ == "__main__":
    main(sys.argv[1])
