"""Host optimizer cost per DP rank at RFD4-Proteina size, with `world` ranks contending on one host.

Replays the per-parameter arithmetic of tt_bio.train.optim.AdamW.step (moments, bias-corrected
update, decoupled decay, theta -= upd, bf16 round of before/after for the kept control, update
norm) on 178,448,304 fp32 elements split into tensors of RFD4's typical size, with the launcher's
thread split (host_threads). No device: the device write-back is priced separately. --shard
gives each rank only 1/world of the elements, the ZeRO-1 arm.

    python3 perf/r4t_scale/hostopt_cost.py --world 4 --steps 4 --out ~/r4t/scale/opt4.json
"""
import argparse, json, multiprocessing as mp, os, socket, time
from pathlib import Path
import numpy as np

N_RFD4 = 178_448_304
TENSOR = 768 * 1536   # a typical RFD4 linear, 1.18 M elements


def rank_main(rank, world, n, steps, shard, q):
    from tt_bio.train.launcher import host_threads
    import torch
    torch.set_num_threads(host_threads(world))
    mine = n // world if shard else n
    sizes = [TENSOR] * (mine // TENSOR) + ([mine % TENSOR] if mine % TENSOR else [])
    rng = np.random.default_rng(rank)
    theta = [rng.standard_normal(s, dtype=np.float32) * 0.02 for s in sizes]
    m = [np.zeros(s, np.float32) for s in sizes]
    v = [np.zeros(s, np.float32) for s in sizes]
    b1, b2, lr, eps, wd = 0.9, 0.999, 1e-4, 1e-8, 0.0
    times = []
    for st in range(1, steps + 1):
        g = [rng.standard_normal(s, dtype=np.float32) * 1e-3 for s in sizes]
        t0 = time.perf_counter()
        bc1, bc2 = 1 - b1 ** st, 1 - b2 ** st
        gn = float(np.sqrt(sum(float(x @ x) for x in g)))
        for i in range(len(sizes)):
            mi, vi, th, gi = m[i], v[i], theta[i], g[i]
            mi *= b1; mi += (1 - b1) * gi
            vi *= b2; vi += (1 - b2) * (gi * gi)
            upd = lr * ((mi / bc1) / (np.sqrt(vi / bc2) + eps) + wd * th)
            before = torch.from_numpy(th).to(torch.bfloat16)
            th -= upd
            after = torch.from_numpy(th).to(torch.bfloat16)
            _ = float(np.linalg.norm(upd)); _ = bool(torch.equal(after, before))
        times.append(time.perf_counter() - t0)
        del g
    q.put({"rank": rank, "elements": mine, "step_s": times})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n", type=int, default=N_RFD4)
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    load0 = os.getloadavg()
    ps = [ctx.Process(target=rank_main, args=(r, a.world, a.n, a.steps, a.shard, q)) for r in range(a.world)]
    for p in ps: p.start()
    res = sorted((q.get(timeout=3600) for _ in ps), key=lambda r: r["rank"])
    for p in ps: p.join()
    warm = [t for r in res for t in r["step_s"][1:]]
    out = {"host": socket.gethostname(), "world": a.world, "shard": a.shard, "n": a.n,
           "median_opt_s": float(np.median(warm)), "max_opt_s": float(max(warm)),
           "loadavg_before": load0, "per_rank": res}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rank"}))


if __name__ == "__main__":
    main()
