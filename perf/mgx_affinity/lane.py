#!/usr/bin/env python3
"""Run affinity jobs through the shipped CLI, one pinned chip per lane, one JSON per job.

    python3 perf/mgx_affinity/lane.py --pool "2 3 4" --cards 1 plan.jsonl
    python3 perf/mgx_affinity/lane.py --pool "2 3 4 5" --cards 4 plan.jsonl   # multi-chip job

A plan line is {"tag", "surface": "nesso1"|"boltz2", "input", "args": [...]}. A job whose
results/<tag>.json exists is skipped, so a relaunch continues. Per job it records the exit code,
wall, the scalars the CLI wrote, host load and AICLK sampled DURING, and the DRAM census when the
plan sets "census": true (the census drains the pipeline at every tag, so a census job is never a
timing).

Chips: a card is taken only when its lease JSON says released more than 120 s ago, or its holder
pid is dead, and its flock is free. The lane re-writes its own lease between jobs (the engine marks the card released
when each job exits), so a sibling row cannot take the chip mid-plan. The five cardblocked chips
are never candidates. Every job runs with 2 host threads: nesso1 through OMP/MKL, since
`tt-bio affinity` has no --host_threads, and boltz2 through predict's own flag.
"""
import argparse
import csv
import fcntl
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from perf.clocksample import during  # noqa: E402

HOST = os.uname().nodename
LEASES = pathlib.Path(os.environ.get("TT_BIO_LEASE_DIR", str(pathlib.Path.home() / "leases")))
ME = "worker:mgx-affinity-scale"
BLOCKED = {1, 24, 25, 26, 27}
PY = os.environ.get("LANE_PY") or sys.executable
RES, OUT = HERE / "results", HERE / "out"


def lease(c):
    return LEASES / f"{HOST}-card{c}.json"


def free(c):
    try:
        d = json.loads(lease(c).read_text())
    except (OSError, ValueError):
        d = {"released": 1}
    mine = d.get("holder") == ME and d.get("pid") == os.getpid()
    # A sibling chain releases between its folds and re-claims within seconds; a release is
    # only an idle chip once it is two minutes old.
    idle = d.get("released") and time.time() - float(d["released"]) > 120
    if not (mine or idle or not os.path.exists(f"/proc/{d.get('pid')}")):
        return False
    with open(lease(c), "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
    return True


def claim(c):
    lease(c).write_text(json.dumps({"host": HOST, "card": str(c), "holder": ME, "pid": os.getpid(),
                                    "acquired": time.time(), "released": None,
                                    "note": "held between jobs by a mgx-affinity-scale lane"}) + "\n")


def release(c):
    d = json.loads(lease(c).read_text())
    if d.get("holder") == ME:
        d["released"] = time.time()
        lease(c).write_text(json.dumps(d) + "\n")


def take(pool, n, held, avoid):
    while len(held) < n:
        for c in pool:
            if c in BLOCKED or c in held or avoid.get(c, 0) > time.time():
                continue
            if free(c):
                claim(c)
                held.append(c)
                print(f"{time.strftime('%H:%M:%S')} took card {c}", flush=True)
                if len(held) == n:
                    break
        else:
            time.sleep(30)
    return held


class loadwatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.xs, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.xs.append(float(open("/proc/loadavg").read().split()[0]))
            self.stop.wait(10)

    def summary(self):
        s = sorted(self.xs) or [float("nan")]
        return {"min": s[0], "median": s[len(s) // 2], "max": s[-1], "nproc": os.cpu_count()}


def census(path):
    """Largest 'GiB used' line and the smallest largest-free-block the census saw."""
    if not path.exists():
        return None
    used, tag, maxfree = 0.0, None, None
    for ln in path.read_text().splitlines():
        m = re.search(r"\[DRAM\] (.*?): ([\d.]+) GiB used .*?maxfree=(\d+)MiB", ln)
        if m:
            if float(m.group(2)) > used:
                used, tag = float(m.group(2)), m.group(1)
            mf = int(m.group(3))
            maxfree = mf if maxfree is None else min(maxfree, mf)
    return {"peak_gib": used, "peak_tag": tag, "min_maxfree_mib_per_bank": maxfree}


def scalars(surface, out):
    rows = []
    if surface == "nesso1":
        f = out / "affinity.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f)))
    else:
        for f in out.rglob("results.json"):
            d = json.loads(f.read_text())
            rows += d if isinstance(d, list) else d.get("results", d.get("jobs", [d]))
    return rows


def command(job, out):
    if job["surface"] == "nesso1":
        return [PY, "-m", "tt_bio.main", "affinity", job["input"], "--out_dir", str(out), *job.get("args", [])]
    return [PY, "-m", "tt_bio.main", "predict", job["input"], "--model", "boltz2", "--out_dir", str(out),
            "--single_sequence", "--host_threads", "2", "--accelerator", "tenstorrent", "--seed", "0",
            *job.get("args", [])]


def run(job, cards):
    tag = job["tag"]
    out, log, dram = OUT / tag, OUT / f"{tag}.log", OUT / f"{tag}.dram"
    OUT.mkdir(parents=True, exist_ok=True)
    dram.unlink(missing_ok=True)
    grant = ",".join(map(str, cards))
    env = dict(os.environ, PYTHONPATH=str(ROOT), TT_VISIBLE_DEVICES=grant, TT_BIO_LEASE_CARDS=grant,
               TT_BIO_LEASE_HOLDER=ME, TT_BIO_LEASE_DIR=str(LEASES), OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", TT_METAL_LOGGER_LEVEL="FATAL",
               TT_METAL_CACHE=str(pathlib.Path.home() / ".cache/tt-metal-cache-mgxa"))
    if job.get("census"):
        env["TT_BIO_DRAM_PEAK"] = str(dram)
    env.update(job.get("env", {}))
    cmd = command(job, out)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.environ["TT_VISIBLE_DEVICES"] = grant  # so tt-smi reports the granted chips
    lw = loadwatch()
    lw.start()
    t0 = time.time()
    with during(period=5.0) as clk, open(log, "w") as fh:
        fh.write(f"START {time.strftime('%FT%TZ', time.gmtime())} cards={grant} commit={commit}\n"
                 f"CMD {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    lw.stop.set()
    text = log.read_text(errors="replace")
    row = {"tag": tag, "surface": job["surface"], "input": job["input"], "args": job.get("args", []),
           "host": HOST, "cards": cards, "commit": commit, "rc": rc, "wall_s": round(wall, 1),
           "aiclk": clk.summary(), "load": lw.summary(), "census": census(dram),
           "rows": scalars(job["surface"], out),
           "contention": bool(re.search(r"DeviceInUseError|is in use by worker:", text)),
           "errors": [ln[:400] for ln in text.splitlines()
                      if re.search(r"(?i)out of memory|Out of Memory|allocat|Traceback|Error:|error:", ln)][-12:],
           "tail": text.splitlines()[-25:] if rc else []}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=pathlib.Path)
    ap.add_argument("--pool", required=True, help="candidate cards, space separated")
    ap.add_argument("--cards", type=int, default=1, help="chips per job")
    a = ap.parse_args()
    pool = [int(x) for x in a.pool.split()]
    RES.mkdir(parents=True, exist_ok=True)
    held, avoid = [], {}
    for line in a.plan.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        job = json.loads(line)
        if (RES / f"{job['tag']}.json").exists():
            continue
        while True:
            held = take(pool, a.cards, held, avoid)
            row = run(job, held)
            for c in held:
                claim(c)
            if not row["contention"]:
                break
            for c in held:
                avoid[c] = time.time() + 600
            held = []
        (RES / f"{job['tag']}.json").write_text(json.dumps(row, indent=1) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {job['tag']} rc={row['rc']} wall={row['wall_s']}s "
              f"aiclk={row['aiclk']} rows={len(row['rows'])}", flush=True)
    for c in held:
        release(c)
    print("LANE_DONE", flush=True)


if __name__ == "__main__":
    main()
