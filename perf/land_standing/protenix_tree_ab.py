#!/usr/bin/env python3
"""Does the D10+D24 ranking change cost protenix-v1 fold time?

The size-ladder arm reads protenix-v1 at 1.18-1.44x its recorded baseline, and
wk/land-standing is the one branch that touches tt_bio/protenix.py, so that row is the
only ladder failure the candidate could plausibly own. The ladder compares a run against
a RECORDED baseline taken on another tree at another time; this compares the two trees
against each other, now, on one card, with the arm order alternating per rep so a
position effect cannot masquerade as an effect (the trap that refuted narrow-q).

Arms are two checkouts, not an env flag: main = origin/main, cand = wk/land-standing.
Scores runtime_s out of the fold's own results.json, which excludes model load and
process startup -- the same quantity the ladder's timing check reads.
"""
import argparse, json, os, statistics, subprocess, threading, time
from pathlib import Path

PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
TREES = {"main": "/home/ttuser/mainctl-land-standing",
         "cand": "/home/ttuser/.coworker/wt/land-standing"}


def aiclk_sampler(stop, out, card, errs):
    """AICLK lives at device_info[card]["telemetry"]["aiclk"], as a RIGHT-ALIGNED STRING.

    The first version probed board_info.aiclk then chip_telemetry.aiclk. Neither key
    exists, so every sample raised KeyError into a bare ``except: pass`` and the run
    recorded n=0 on all six legs while still printing a ratio. A clock-blind run that
    still prints a number is the failure worth guarding, so the first exception is kept
    and the summary refuses to call a cell measured when no leg carries samples.
    """
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=20)
            dev = json.loads(r.stdout)["device_info"][card]
            out.append(float(str(dev["telemetry"]["aiclk"]).strip()))
        except Exception as e:
            if not errs:
                errs.append(repr(e))
        stop.wait(1.0)


def leg(tree, rung, card, workdir, tag):
    d = workdir / ("out_" + tag)
    subprocess.run(["rm", "-rf", str(d)], check=False)
    env = dict(os.environ)
    env.update(PYTHONPATH=TREES[tree], TT_VISIBLE_DEVICES=str(card),
               TT_BIO_LEASE_CARDS=str(card), TT_BIO_LEASE_HOLDER="worker:land-standing")
    cmd = [PY, "-m", "tt_bio.main", "predict",
           TREES[tree] + "/perf/size512/fixtures/cdk2x2_" + str(rung) + ".yaml",
           "--model", "protenix-v1", "--single_sequence",
           "--sampling_steps", "6", "--diffusion_samples", "1", "--seed", "0",
           "--out_dir", str(d)]
    clk, errs, stop = [], [], threading.Event()
    t = threading.Thread(target=aiclk_sampler, args=(stop, clk, card, errs),
                         daemon=True)
    t.start()
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=TREES[tree], env=env, capture_output=True, text=True)
    wall = time.monotonic() - t0
    stop.set()
    t.join(timeout=5)
    rs = None
    for j in d.rglob("results.json"):
        try:
            rs = json.loads(j.read_text()).get("runtime_s", rs)
        except Exception:
            pass
    return {"tree": tree, "rung": rung, "rc": p.returncode, "wall": round(wall, 3),
            "runtime_s": rs,
            "aiclk_median": (round(statistics.median(clk), 1) if clk else None),
            "aiclk_n": len(clk), "aiclk_min": (min(clk) if clk else None),
            "aiclk_err": (errs[0] if errs else None),
            "tail": (p.stdout + p.stderr)[-1500:] if p.returncode else p.stdout[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, default=640)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--out",
                    default="perf/land_standing/out/ladder_attrib/protenix_tree_ab.json")
    a = ap.parse_args()
    wd = Path("/home/ttuser/.coworker/wt/land-standing/perf/land_standing/out/"
              "ladder_attrib/work")
    wd.mkdir(parents=True, exist_ok=True)
    legs = []
    for rep in range(a.reps):
        # alternate which arm goes first: a fixed order is what made narrow-q reproduce
        order = ["main", "cand"] if rep % 2 == 0 else ["cand", "main"]
        for slot, tree in enumerate(order):
            r = leg(tree, a.rung, a.card, wd, tree + "_r" + str(rep))
            r.update(rep=rep, slot=slot)
            legs.append(r)
            print("rep%d slot%d %-5s rc=%s runtime_s=%s wall=%s aiclk=%s (n=%s)"
                  % (rep, slot, tree, r["rc"], r["runtime_s"], r["wall"],
                     r["aiclk_median"], r["aiclk_n"]), flush=True)
    ok = [l for l in legs if l["rc"] == 0 and l["runtime_s"]]
    res = {"legs": legs, "host": os.uname().nodename, "rung": a.rung}
    for t in ("main", "cand"):
        v = [l["runtime_s"] for l in ok if l["tree"] == t]
        if v:
            res[t] = {"n": len(v), "median": round(statistics.median(v), 3),
                      "min": min(v), "max": max(v),
                      "spread_s": round(max(v) - min(v), 3)}
    if "main" in res and "cand" in res:
        res["ratio_cand_over_main"] = round(res["cand"]["median"] / res["main"]["median"], 4)
        res["floor_s"] = round(max(res["main"]["spread_s"], res["cand"]["spread_s"]), 3)
        res["effect_s"] = round(res["cand"]["median"] - res["main"]["median"], 3)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "legs"}, indent=2))


if __name__ == "__main__":
    main()
