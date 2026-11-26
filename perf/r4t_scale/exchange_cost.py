"""Cost of one data-parallel gradient sync at RFD4-Proteina's size, on the launcher's own path.

Runs `tt_bio.train.launcher.ProcessAxis._exchange`, the collective `finetune` uses between
ranks, with `world` processes on this host and a gradient of N fp32 elements each (default
178,448,304 = every trainable RFD4-Proteina parameter, 714 MB). Host only: the gradient is
already read back for the host optimizer (optim.py), so the device->host copy is not part of
the sync and is priced separately.

Per step: publish / wait / add (the launcher's own timers) and the call's wall. Every rank's sum
must be bit-identical and equal a rank-order reference sum (rank 0 rebuilds the inputs on the
first and last step). --broken makes the last rank publish a perturbed gradient on step 2 and
must FAIL the reference check (shows the gate can fail).

    python3 perf/r4t_scale/exchange_cost.py --world 4 --steps 6 --out ~/r4t/scale/x4.json
"""
import argparse, hashlib, json, multiprocessing as mp, os, shutil, socket, sys, time
from pathlib import Path

import numpy as np

N_RFD4 = 178_448_304


def grad(rank, step, n):
    """Deterministic per-(rank, step) fill, no RNG cost on 178 M elements."""
    v = np.arange(n, dtype=np.float32)
    v *= np.float32(1e-9 * (rank + 1))
    v += np.float32(step * 0.25 + rank)
    return v


def rank_main(rank, world, n, steps, run, broken, q, zero_copy=False):
    from tt_bio.train.launcher import ProcessAxis
    ax = ProcessAxis(name="dp", device_ids=tuple(range(world)), dp_rank=rank, run=str(run))
    rows, out = [], None
    for s in range(1, steps + 1):
        v = grad(rank, s, n)
        if broken and rank == world - 1 and s == 2:
            v[n // 2] += np.float32(1.0)
        t0 = time.perf_counter()
        if zero_copy:
            # The optimizer's path: fill the send slot instead of a vector of its own (the fill
            # is the gradient readback it does anyway, so it stays outside the timed wall),
            # and gather into a buffer kept across steps.
            buf = ax.send_buffer(n)
            buf[...] = v
            t0 = time.perf_counter()
            if out is None:
                out = np.empty(n, np.float32)
            tot = ax.all_gather(ax.reduce_scatter(buf), out=out)
            for k in ("_pub", "_wait"):
                ax.state.pop(k, None)
        else:
            tot = ax._exchange(v)
        wall = time.perf_counter() - t0
        row = {"step": s, "wall_s": wall,
               "digest": hashlib.blake2b(tot.tobytes(), digest_size=8).hexdigest()}
        if rank == 0 and s in (1, 2, steps):
            ref = np.zeros(n, np.float32)
            for r in range(world):
                ref += grad(r, s, n)
            row["ref_ok"] = bool(np.array_equal(ref, tot))
            del ref
        rows.append(row)
        del tot, v
    st = ax.state
    q.put({"rank": rank, "rows": rows, "publish_s": st["publish_s"], "wait_s": st["wait_s"],
           "add_s": st["add_s"], "bytes": st["bytes"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--n", type=int, default=N_RFD4)
    ap.add_argument("--shm", default="/dev/shm")
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--zero-copy", action="store_true",
                    help="fill the send slot (send_buffer) and gather into a kept buffer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.shm) / f"r4t-scale-x-{os.getpid()}"
    run.mkdir(parents=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    load0 = os.getloadavg()
    t0 = time.perf_counter()
    ps = [ctx.Process(target=rank_main, args=(r, a.world, a.n, a.steps, run, a.broken, q, a.zero_copy))
          for r in range(a.world)]
    try:
        for p in ps:
            p.start()
        res = sorted((q.get(timeout=1800) for _ in ps), key=lambda r: r["rank"])
        for p in ps:
            p.join()
    finally:
        shutil.rmtree(run, ignore_errors=True)
    wall = time.perf_counter() - t0
    agree = all(len({r["rows"][i]["digest"] for r in res}) == 1 for i in range(a.steps))
    ref_ok = all(row.get("ref_ok", True) for row in res[0]["rows"])

    def med(key):  # steady state: step 1 pays page faults on fresh /dev/shm and spawn skew
        xs = [x for r in res for x in r[key][1:]]
        return float(np.median(xs)) if xs else None
    walls = [row["wall_s"] for r in res for row in r["rows"][1:]]
    gb = a.n * 4 / 1e9
    mw = float(np.median(walls)) if walls else None
    out = {"host": socket.gethostname(), "world": a.world, "n": a.n, "bytes_per_rank": a.n * 4,
           "steps": a.steps, "broken": a.broken, "ranks_agree": agree, "ref_sum_exact": ref_ok,
           "PASS": agree and ref_ok,
           "median_publish_s": med("publish_s"), "median_wait_s": med("wait_s"),
           "median_add_s": med("add_s"), "median_exchange_wall_s": mw,
           "max_exchange_wall_s": float(max(walls)) if walls else None,
           "algbw_GBps": gb / mw if mw else None,
           "run_wall_s": wall, "loadavg_before": load0, "per_rank": res}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rank"}))
    sys.exit(0 if out["PASS"] != a.broken else 1)


if __name__ == "__main__":
    main()
