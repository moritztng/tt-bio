#!/usr/bin/env python3
"""Where does the 32-worker embed fanout lose its ceiling: per-card speed, or idle cards?

Dispatches an esmc embed through the LIVE controller and, while it runs, samples at 4 Hz how
many of the 32 resident pool workers are actually burning CPU. Sum(per-shard compute) vs
wall-clock only tells you there is a gap; the busy-worker trace tells you WHEN the cards are
idle -- at the front (dispatch/lease), the back (result transfer/reassembly), or throughout
(per-card slowness). Those need different fixes, and the existing doc inferred the answer
from the gap instead of measuring it.
"""
import json, os, subprocess, sys, threading, time, random
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")
ENV = "/home/cust-team/mthuening/tt-bio/env/bin"
CTRL = "http://UF-EV-A13-GWH02:8770"


def pool_worker_pids():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        if "multiprocessing.spawn" in line and "tt-bio/env" in line:
            pids.append(int(line.split()[0]))
    return pids


def ticks(pid):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return (float(f[11]) + float(f[12])) / CLK
    except (OSError, IndexError):
        return None


class Trace(threading.Thread):
    def __init__(self, pids, period=0.25):
        super().__init__(daemon=True)
        self.pids, self.period, self.stop = pids, period, threading.Event()
        self.rows = []

    def run(self):
        prev = {p: ticks(p) for p in self.pids}
        t0 = time.monotonic()
        while not self.stop.wait(self.period):
            now = time.monotonic()
            cur = {p: ticks(p) for p in self.pids}
            dt = now - t0
            busy = sum(1 for p in self.pids
                       if prev.get(p) is not None and cur.get(p) is not None
                       and (cur[p] - prev[p]) / dt > 0.2)
            tot = sum((cur[p] - prev[p]) for p in self.pids
                      if prev.get(p) is not None and cur.get(p) is not None) / dt
            self.rows.append((round(now, 3), busy, round(tot, 2)))
            prev, t0 = cur, now


def make_seqs(n, seed=7):
    rnd = random.Random(seed)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return {f"s{i:05d}": "".join(rnd.choice(aa) for _ in range(rnd.randint(150, 450)))
            for i in range(n)}


def run(n, model="esmc-600m", batch_size=8):
    import yaml
    d = Path(f"/home/cust-team/mthuening/embedprobe/n{n}")
    d.mkdir(parents=True, exist_ok=True)
    src = d / "seqs.yaml"
    src.write_text(yaml.safe_dump(make_seqs(n)))
    pids = pool_worker_pids()
    tr = Trace(pids)
    tr.start()
    t0 = time.monotonic()
    p = subprocess.run([f"{ENV}/tt-bio", "embed", str(src), "--model", model,
                        "--out_dir", str(d / "out"), "--controller", CTRL,
                        "--batch_size", str(batch_size)],
                       capture_output=True, text=True)
    wall = time.monotonic() - t0
    tr.stop.set(); tr.join(timeout=3)
    rows = tr.rows
    peak = max((r[1] for r in rows), default=0)
    # a worker counts as engaged from the first sample it is busy in; the fraction of the
    # run during which >=90% of the pool is busy is the number that separates "cards slow"
    # from "cards idle"
    hi = sum(1 for r in rows if r[1] >= 0.9 * len(pids))
    npz = len(list((d / "out").glob("*.npz")))
    rec = {"n": n, "model": model, "batch_size": batch_size, "rc": p.returncode,
           "wall_s": round(wall, 2), "seq_per_s": round(n / wall, 2),
           "workers": len(pids), "peak_busy": peak,
           "frac_samples_full_pool": round(hi / max(1, len(rows)), 3),
           "mean_busy": round(sum(r[1] for r in rows) / max(1, len(rows)), 2),
           "mean_pool_cores": round(sum(r[2] for r in rows) / max(1, len(rows)), 2),
           "npz_written": npz,
           "tail": p.stdout.strip().splitlines()[-2:] + p.stderr.strip().splitlines()[-2:]}
    (d / "trace.json").write_text(json.dumps(rows))
    return rec


if __name__ == "__main__":
    out = Path("/home/cust-team/mthuening/embedprobe/cells.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    for n in [int(x) for x in sys.argv[1:]] or [1024]:
        r = run(n)
        with out.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
