#!/usr/bin/env python3
"""Run affinity jobs through the shipped CLI, one pinned chip per lane, one JSON per job.

    python3 perf/mgx_affinity/lane.py --pool "2 3 4 5" a.jsonl b.jsonl c.jsonl:2

One process, one lane (thread) per plan, `:N` for N chips per job. Claims go through one lock, so
two lanes never take the same chip.

A plan line is {"tag", "surface": "nesso1"|"boltz2", "input", "args": [...]}. A job whose
results/<tag>.json exists is skipped, so a relaunch continues. Per job it records the exit code,
wall, the scalars the CLI wrote, host load and AICLK sampled DURING, and the DRAM census when the
plan sets "census": true (the census drains the pipeline at every tag, so a census job is never a
timing).

Chips: a card is taken only when its lease JSON says released more than 120 s ago, or its holder
pid is dead, and its flock is free. The lane re-writes its own lease between jobs (the engine
marks the card released when each job exits), so a sibling row cannot take the chip mid-plan. The five cardblocked chips
are never candidates. Every job runs at --host_threads 2.
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
from perf.clocksample import TT_SMI  # noqa: E402

HOST = os.uname().nodename
LEASES = pathlib.Path(os.environ.get("TT_BIO_LEASE_DIR", str(pathlib.Path.home() / "leases")))
ME = "worker:mgx-affinity-scale"
BLOCKED = {1, 24, 25, 26, 27}
PY = os.environ.get("LANE_PY") or sys.executable
RES, OUT = HERE / "results", HERE / "out"
# card -> lane, for every chip a lane of this process holds. The lease JSON is shared with every
# engine child and sibling row, so it cannot tell two lanes of one process apart; this can.
HELD = {}
STOP = OUT / "STOP"  # touch to stop every lane after its current job


def lease(c):
    return LEASES / f"{HOST}-card{c}.json"


def free(c, name):
    if HELD.get(c, name) != name:
        return False
    try:
        d = json.loads(lease(c).read_text())
    except OSError:
        d = {"released": 1}
    except ValueError:
        return False  # caught mid-rewrite by its holder
    # Lanes are threads of one process, so pid alone would call a sibling lane's chip ours.
    mine = d.get("holder") == ME and d.get("pid") == os.getpid() and d.get("lane") == name
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


def claim(c, name):
    HELD[c] = name
    lease(c).write_text(json.dumps({"host": HOST, "card": str(c), "holder": ME, "pid": os.getpid(),
                                    "lane": name,
                                    "acquired": time.time(), "released": None,
                                    "note": "held between jobs by a mgx-affinity-scale lane"}) + "\n")


def release(c):
    HELD.pop(c, None)
    d = json.loads(lease(c).read_text())
    if d.get("holder") == ME:
        d["released"] = time.time()
        lease(c).write_text(json.dumps(d) + "\n")


def take(pool, n, held, avoid, name):
    """One pass over the pool; the caller sleeps outside the lock and comes back."""
    for c in pool:
        if len(held) == n:
            break
        if c in BLOCKED or c in held or avoid.get(c, 0) > time.time():
            continue
        if free(c, name):
            claim(c, name)
            held.append(c)
            print(f"{time.strftime('%H:%M:%S')} took card {c}", flush=True)
    return held


class watch(threading.Thread):
    """Host load every 10 s and each granted chip's AICLK every 20 s, for the job's duration.
    tt-smi is pinned per call (lanes are threads of one process, so the env is not theirs)."""

    def __init__(self, grant):
        super().__init__(daemon=True)
        self.grant, self.load, self.clk, self.stop = grant, [], {}, threading.Event()

    def run(self):
        n = 0
        while not self.stop.is_set():
            self.load.append(float(open("/proc/loadavg").read().split()[0]))
            if n % 2 == 0:
                try:
                    r = subprocess.run([TT_SMI, "-s"], capture_output=True, text=True, timeout=60,
                                       env=dict(os.environ, TT_VISIBLE_DEVICES=self.grant))
                    for i, dev in enumerate(json.loads(r.stdout).get("device_info", [])):
                        self.clk.setdefault(i, []).append(int(dev["telemetry"]["aiclk"]))
                except Exception:
                    pass
            n += 1
            self.stop.wait(10)

    def summary(self):
        s = sorted(self.load) or [float("nan")]
        clk = {i: {"min": min(v), "median": sorted(v)[len(v) // 2], "max": max(v), "n": len(v)}
               for i, v in self.clk.items() if v}
        return clk, {"min": s[0], "median": s[len(s) // 2], "max": s[-1], "nproc": os.cpu_count()}


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
        return [PY, "-m", "tt_bio.main", "affinity", job["input"], "--out_dir", str(out),
                "--host_threads", "2", *job.get("args", [])]
    return [PY, "-m", "tt_bio.main", "predict", job["input"], "--model", "boltz2", "--out_dir", str(out),
            "--single_sequence", "--host_threads", "2", "--accelerator", "tenstorrent", "--seed", "0",
            *job.get("args", [])]


def run(job, cards, adopt=None):
    """Run the job on ``cards``; or, with ``adopt=(pid, t0)``, watch an already running one."""
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
    w = watch(grant)
    w.start()
    if adopt:
        pid, t0 = adopt
        while os.path.exists(f"/proc/{pid}"):
            time.sleep(10)
        rc = None  # not our child; the rows below are the outcome
    else:
        t0 = time.time()
        with open(log, "w") as fh:
            fh.write(f"START {time.strftime('%FT%TZ', time.gmtime())} cards={grant} commit={commit}\n"
                     f"CMD {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            # A relaunched lane adopts this job from here instead of starting it a second time.
            running = OUT / f"{tag}.running"
            running.write_text(json.dumps({"cards": cards, "pid": proc.pid, "t0": t0}) + "\n")
            rc = proc.wait()
            running.unlink(missing_ok=True)
    wall = time.time() - t0
    w.stop.set()
    aiclk, load = w.summary()
    text = log.read_text(errors="replace")
    row = {"tag": tag, "surface": job["surface"], "input": job["input"], "args": job.get("args", []),
           "host": HOST, "cards": cards, "commit": commit, "rc": rc, "wall_s": round(wall, 1),
           "aiclk": aiclk, "load": load, "census": census(dram), "adopted": bool(adopt),
           "rows": scalars(job["surface"], out),
           "contention": bool(re.search(r"DeviceInUseError|is in use by worker:", text)),
           "errors": [ln[:400] for ln in text.splitlines()
                      if re.search(r"(?i)out of memory|Out of Memory|allocat|Traceback|Error:|error:", ln)][-12:],
           "tail": text.splitlines()[-25:] if rc != 0 else []}
    return row


LOCK = threading.Lock()


ADOPT = {}


def lane(plan, pool, n, adopt):
    held, avoid = [], {}
    for line in plan.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        job = json.loads(line)
        if (RES / f"{job['tag']}.json").exists():
            continue
        if STOP.exists():
            break
        while True:
            if job["tag"] in adopt:
                held = adopt.pop(job["tag"])[0]
                with LOCK:
                    HELD.update({c: plan.stem for c in held})
                row = run(job, held, adopt=ADOPT[job["tag"]][1:])
                break
            while len(held) < n:
                with LOCK:
                    held = take(pool, n, held, avoid, plan.stem)
                if len(held) < n:
                    time.sleep(30)
            row = run(job, held)
            if not row["contention"]:
                break
            # Refused: some other holder owns at least one of these now. Never write a lease we
            # may not own; drop them all and come back to them later.
            with LOCK:
                for c in held:
                    avoid[c] = time.time() + 600
                    HELD.pop(c, None)
            held = []
        for c in held:
            claim(c, plan.stem)
        (RES / f"{job['tag']}.json").write_text(json.dumps(row, indent=1) + "\n")
        print(f"{time.strftime('%H:%M:%S')} {plan.stem} {job['tag']} rc={row['rc']} "
              f"wall={row['wall_s']}s aiclk={row['aiclk']} rows={len(row['rows'])}", flush=True)
    for c in held:
        release(c)
    print(f"LANE_DONE {plan.stem}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plans", nargs="+", help="plan.jsonl, or plan.jsonl:N for N chips per job")
    ap.add_argument("--pool", required=True, help="candidate cards, space separated")
    ap.add_argument("--adopt", action="append", default=[],
                    help="TAG:CARDS:PID:EPOCH -- watch an already running job instead of starting "
                         "it (CARDS comma separated); repeatable. Jobs a lane started are adopted "
                         "from out/<tag>.running without this")
    a = ap.parse_args()
    pool = [int(x) for x in a.pool.split()]
    RES.mkdir(parents=True, exist_ok=True)
    for spec in a.adopt:
        tag, cards, pid, t0 = spec.split(":")
        ADOPT[tag] = ([int(c) for c in cards.split(",")], int(pid), float(t0))
        HELD.update({int(c): f"adopt:{tag}" for c in cards.split(",")})  # until its lane gets there
    for f in OUT.glob("*.running"):
        d = json.loads(f.read_text())
        if f.stem not in ADOPT and os.path.exists(f"/proc/{d['pid']}"):
            ADOPT[f.stem] = (d["cards"], d["pid"], d["t0"])
            HELD.update({c: f"adopt:{f.stem}" for c in d["cards"]})
    adopt = dict(ADOPT)  # shared: a lane pops the tag it adopts
    threads = []
    for k, spec in enumerate(a.plans):
        path, _, n = spec.partition(":")
        th = threading.Thread(target=lane, args=(pathlib.Path(path), pool, int(n or 1), adopt))
        th.start()
        threads.append(th)
        time.sleep(1)
    for th in threads:
        th.join()


if __name__ == "__main__":
    main()
