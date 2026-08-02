#!/usr/bin/env python3
"""Is the controller the serialization point the idle pool is waiting on?

Runs one embed and samples, at 4 Hz, the CPU of (a) the controller process, (b) the client, and
(c) how many pool workers are busy. If the pool goes idle while the controller pegs a core, the
bottleneck is result upload through one process, not the cards -- and that is a different fix from
"the cards are slow".
"""
import json, os, subprocess, sys, threading, time, random
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")
ENV = "/home/cust-team/mthuening/tt-bio/env/bin"
CTRL = "http://UF-EV-A13-GWH02:8770"
CTRL_PID = int(sys.argv[2])


def ticks(pid):
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return (float(f[11]) + float(f[12])) / CLK
    except (OSError, IndexError):
        return None


def workers():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines()
            if "multiprocessing.spawn" in l and "tt-bio/env" in l]


class T(threading.Thread):
    def __init__(self, wpids):
        super().__init__(daemon=True)
        self.w, self.stop, self.rows = wpids, threading.Event(), []

    def run(self):
        prev = {p: ticks(p) for p in self.w + [CTRL_PID]}
        t0 = time.monotonic()
        while not self.stop.wait(0.25):
            now = time.monotonic(); cur = {p: ticks(p) for p in self.w + [CTRL_PID]}
            dt = now - t0
            def d(p):
                a, b = prev.get(p), cur.get(p)
                return 0.0 if a is None or b is None else (b - a) / dt
            self.rows.append((round(now - 0, 3), sum(1 for p in self.w if d(p) > 0.2),
                              round(d(CTRL_PID), 2)))
            prev, t0 = cur, now


def main():
    n, fmt = int(sys.argv[1]), (sys.argv[3] if len(sys.argv) > 3 else "npz")
    rnd = random.Random(7)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    seqs = {f"s{i:05d}": "".join(rnd.choice(aa) for _ in range(rnd.randint(150, 450)))
            for i in range(n)}
    import yaml
    d = Path(f"/home/cust-team/mthuening/embedctrl/n{n}_{fmt}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "seqs.yaml").write_text(yaml.safe_dump(seqs))
    tr = T(workers()); tr.start()
    t0 = time.monotonic()
    p = subprocess.run([f"{ENV}/tt-bio", "embed", str(d / "seqs.yaml"), "--model", "esmc-600m",
                        "--out_dir", str(d / "out"), "--controller", CTRL,
                        "--batch_size", "8", "--format", fmt], capture_output=True, text=True)
    wall = time.monotonic() - t0
    tr.stop.set(); tr.join(timeout=3)
    rows = tr.rows
    dur = rows[-1][0] - rows[0][0] if len(rows) > 1 else wall
    # split the run into deciles so "when" is visible, not just "how much"
    dec_w, dec_c = [[] for _ in range(10)], [[] for _ in range(10)]
    t_start = rows[0][0]
    for t, b, c in rows:
        i = min(9, int(10 * (t - t_start) / dur)) if dur > 0 else 0
        dec_w[i].append(b); dec_c[i].append(c)
    f = lambda L: [round(sum(x)/len(x), 1) if x else 0 for x in L]
    print(json.dumps({"n": n, "format": fmt, "rc": p.returncode, "wall_s": round(wall, 2),
                      "seq_per_s": round(n / wall, 2), "workers": len(tr.w),
                      "pool_busy_deciles": f(dec_w), "controller_cores_deciles": f(dec_c),
                      "controller_cores_mean": round(sum(c for _, _, c in rows)/len(rows), 2),
                      "controller_core_seconds": round(sum(c for _, _, c in rows)*0.25, 1)}))


main()
