"""Host memory bandwidth as the exchange sees it: P processes each copying a 714 MB fp32 vector.

STREAM-style copy (a[:] = b), counted as read + write bytes, aggregate over all processes,
best of the timed reps after one warm rep. Not the STREAM binary: numpy's memcpy, which is
what the exchange's copies and adds run on.

    python3 perf/r4t_scale/stream_bw.py --procs 16 --out bw16.json
"""
import argparse, json, multiprocessing as mp, os, socket, time
import numpy as np


def work(n, reps, barrier, q):
    a = np.ones(n, np.float32); b = np.empty_like(a); b[...] = 0
    ts = []
    for _ in range(reps + 1):
        barrier.wait(); t0 = time.perf_counter(); b[...] = a; ts.append(time.perf_counter() - t0)
    q.put(ts[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n", type=int, default=178_448_304)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx = mp.get_context("spawn"); bar = ctx.Barrier(a.procs); q = ctx.Queue()
    ps = [ctx.Process(target=work, args=(a.n, a.reps, bar, q)) for _ in range(a.procs)]
    load0 = os.getloadavg()
    for p in ps: p.start()
    res = [q.get(timeout=900) for _ in ps]
    for p in ps: p.join()
    per_rep = [max(r[i] for r in res) for i in range(a.reps)]  # slowest proc bounds the rep
    gb = a.procs * a.n * 4 * 2 / 1e9
    out = {"host": socket.gethostname(), "procs": a.procs, "bytes_per_proc": a.n * 4,
           "slowest_proc_s": per_rep, "copy_GBps_best": gb / min(per_rep),
           "copy_GBps_median": gb / float(np.median(per_rep)), "loadavg_before": load0}
    open(a.out, "w").write(json.dumps(out, indent=1)); print(json.dumps(out))


if __name__ == "__main__":
    main()
