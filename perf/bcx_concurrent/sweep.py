#!/usr/bin/env python3
"""N BindCraft 2 arms at once on one QuietBox, one card each, and what each round costs.

Every arm is `perf/bcx_tracewire/round_ab.py --arm eager --extra-msa`: BindCraft 2's own campaign
drives the shipped device predictor with the extra-MSA stack on card, same seed, same settings,
same number of rounds, so the work per arm is identical at every N. Nothing here changes what a
round does; only how many run beside it.

The driver adds what one process cannot see about its neighbours, sampled every second for the
whole run: loadavg, whole-box CPU busy, each arm's process CPU (utime+stime over all threads) and
the AICLK of every card in use, read from the class node.

Rounds are compared only inside the window where every arm is past its compile rounds and none
has finished, so an N=3 figure is never a round that happened to run alone.

  sweep.py --cards 3,0,2 --n 1,2,3 --rounds 16
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARM = ROOT / "perf" / "bcx_tracewire" / "round_ab.py"
PY = "/home/ttuser/bcx_e2e_venv/bin/python"
TICK = os.sysconf("SC_CLK_TCK")


def aiclk(card):
    try:
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def proc_cpu(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return (int(f[11]) + int(f[12])) / TICK
    except (OSError, IndexError):
        return None


def box_busy():
    f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle = f[3] + f[4]
    return sum(f) - idle, sum(f)


def holders():
    r = subprocess.run("fuser -v /dev/tenstorrent/* 2>&1", shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def top_cpu():
    r = subprocess.run(["ps", "-eo", "pid,etime,pcpu,args", "--sort=-pcpu"],
                       capture_output=True, text=True)
    return [line[:200] for line in r.stdout.splitlines()[1:9]]


class Sampler(threading.Thread):
    def __init__(self, cards, pids):
        super().__init__(daemon=True)
        self.cards, self.pids, self.rows, self.halt = cards, pids, [], False

    def run(self):
        while not self.halt:
            busy, total = box_busy()
            self.rows.append({"t": time.time(), "load1": os.getloadavg()[0], "busy": busy,
                              "total": total, "aiclk": {c: aiclk(c) for c in self.cards},
                              "cpu": {c: proc_cpu(p) for c, p in self.pids.items()}})
            time.sleep(1.0)


def window_stats(samples, t0, t1, cards):
    xs = [s for s in samples if t0 <= s["t"] <= t1]
    if len(xs) < 2:
        return {}
    a, b = xs[0], xs[-1]
    dt = b["t"] - a["t"]
    ncpu = os.cpu_count()
    out = {"seconds": round(dt, 1), "n_samples": len(xs),
           "load1": {"min": min(s["load1"] for s in xs), "median": st.median(s["load1"] for s in xs),
                     "max": max(s["load1"] for s in xs)},
           "box_busy_threads": round(ncpu * (b["busy"] - a["busy"]) / (b["total"] - a["total"]), 2),
           "arm_cores": {}, "aiclk": {}}
    for c in cards:
        ca, cb = a["cpu"].get(c), b["cpu"].get(c)
        if ca is not None and cb is not None:
            out["arm_cores"][c] = round((cb - ca) / dt, 2)
        clk = sorted(s["aiclk"][c] for s in xs if s["aiclk"].get(c))
        if clk:
            out["aiclk"][c] = {"min": clk[0], "median": clk[len(clk) // 2], "max": clk[-1],
                               "n": len(clk)}
    return out


def run_level(n, cards, args, outdir):
    use = cards[:n]
    tag = f"n{n}"
    procs, pids, logs = {}, {}, {}
    pre = {"holders": holders(), "top": top_cpu(), "load": os.getloadavg(),
           "utc": time.strftime("%FT%TZ", time.gmtime())}
    for c in use:
        env = dict(os.environ, TT_VISIBLE_DEVICES=str(c), TT_BIO_LEASE_CARDS=f"{args.grant},{c}",
                   TT_BIO_LEASE_HOLDER="worker:bcx-concurrent", OMP_NUM_THREADS=str(args.omp),
                   MKL_NUM_THREADS=str(args.omp), PYTHONUNBUFFERED="1")
        out = outdir / f"{tag}_card{c}.json"
        proj = outdir / "projects" / f"{tag}_card{c}"
        logs[c] = open(outdir / f"{tag}_card{c}.log", "w")
        procs[c] = subprocess.Popen(
            [PY, "-X", "faulthandler", str(ARM), "--arm", "eager", "--extra-msa",
             "--seed", str(args.seed), "--rounds", str(args.rounds), "--card", str(c),
             "--project", str(proj), "--out", str(out)],
            cwd=ROOT, env=env, stdout=logs[c], stderr=subprocess.STDOUT)
        pids[c] = procs[c].pid
    sampler = Sampler(use, pids)
    sampler.start()
    rc = {c: p.wait() for c, p in procs.items()}
    sampler.halt = True
    sampler.join()
    for f in logs.values():
        f.close()
    if any(rc.values()):
        raise RuntimeError(f"{tag}: arm exit codes {rc}, logs in {outdir}")

    arms = {c: json.loads((outdir / f"{tag}_card{c}.json").read_text()) for c in use}
    body = {c: [r for r in a["rounds"] if r["i"] >= 2] for c, a in arms.items()}
    lo = max(rs[0]["span"][0] for rs in body.values())
    hi = min(rs[-1]["span"][1] for rs in body.values())
    per_arm = {}
    for c, rs in body.items():
        inside = [r["s"] for r in rs if r["span"][0] >= lo and r["span"][1] <= hi]
        per_arm[c] = {"rounds_in_window": len(inside),
                      "round_s": {"median": round(st.median(inside), 3), "min": round(min(inside), 3),
                                  "max": round(max(inside), 3)} if inside else None,
                      "all_body_median": round(st.median(r["s"] for r in rs), 3),
                      "trunk_cpu_s_median": round(st.median(
                          rs[k]["trunk"]["cpu"] - rs[k - 1]["trunk"]["cpu"]
                          for k in range(1, len(rs))), 3) if len(rs) > 1 else None,
                      "wall_s": arms[c]["wall_s"]}
    meds = [a["round_s"]["median"] for a in per_arm.values() if a["round_s"]]
    level = {"n": n, "cards": use, "window": [lo, hi], "pre": pre,
             "post": {"holders": holders(), "top": top_cpu(), "load": os.getloadavg()},
             "per_arm": per_arm, "box": window_stats(sampler.rows, lo, hi, use),
             "round_s_median_across_arms": round(st.median(meds), 3) if meds else None,
             "rounds_per_hour_box": round(sum(3600 / m for m in meds), 1) if meds else None}
    (outdir / f"{tag}_samples.json").write_text(json.dumps(sampler.rows))
    return level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="3,0,2", help="in the order N takes them")
    ap.add_argument("--n", default="1,2,3")
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--omp", type=int, default=6)
    ap.add_argument("--grant", default="3", help="this worker's leased card")
    ap.add_argument("--out", default=str(HERE / "runs" / time.strftime("%Y%m%dT%H%MZ", time.gmtime())))
    args = ap.parse_args()
    cards = [int(c) for c in args.cards.split(",")]
    if 1 in cards:
        sys.exit("card 1 is of3t-infab's hard pin")
    outdir = pathlib.Path(args.out).resolve()
    (outdir / "projects").mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    res = {"stamp": {"host": os.uname().nodename, "commit": head, "argv": sys.argv,
                     "nproc": os.cpu_count(), "omp": args.omp, "extra_msa_on_device": True,
                     "arm": "round_ab.py --arm eager --extra-msa"}, "levels": []}
    for n in [int(x) for x in args.n.split(",")]:
        lvl = run_level(n, cards, args, outdir)
        res["levels"].append(lvl)
        print(json.dumps({k: lvl[k] for k in ("n", "cards", "round_s_median_across_arms",
                                              "rounds_per_hour_box", "per_arm", "box")},
                         default=str), flush=True)
        (outdir / "sweep.json").write_text(json.dumps(res, indent=1, default=str))
    base = res["levels"][0]["round_s_median_across_arms"]
    for lvl in res["levels"]:
        lvl["per_arm_slowdown_vs_n1"] = round(lvl["round_s_median_across_arms"] / base, 3)
        lvl["box_speedup_vs_n1"] = round(lvl["rounds_per_hour_box"] / res["levels"][0]["rounds_per_hour_box"], 3)
    (outdir / "sweep.json").write_text(json.dumps(res, indent=1, default=str))
    print("wrote", outdir / "sweep.json", flush=True)


if __name__ == "__main__":
    main()
