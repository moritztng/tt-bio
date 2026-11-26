"""One sharded AdamW step (``AdamW(shard=True).step``) per rank at RFD4-Proteina size, world ranks on one host.

The optimizer's real code path: fill the flat gradient, ``reduce_scatter``, Adam on the owned
tensors, gather the stepped masters back into every rank's masters, bf16 write-back control.
No card: to_host/to_device are numpy stand-ins (the PCIe readback/write is priced elsewhere).
Times the whole step and, inside it, the two collectives (``rs_s``: reduce_scatter;
``gather_s``: all_gather / all_gather_parts, excluding the copy into the masters, which the
step wall includes). Every rank's masters must be bit-identical after every step (digest), and after step 1 rank
0's must equal an unsharded AdamW stepped on the rank-order sum (rtol 1e-5: the clip norm is
summed in another order). --broken: the last rank perturbs its step-1 gradient, which must
fail that reference. Step 1 (reference, page faults) is outside the medians.

    python3 perf/r4t_scale/sharded_step_cost.py --world 16 --steps 5 --out x.json
"""
import argparse, hashlib, json, multiprocessing as mp, os, shutil, socket, time
from pathlib import Path
import numpy as np

N_RFD4 = 178_448_304
TENSOR = 768 * 1536


class _DType:
    name = "bfloat16"


class _Value:
    dtype = _DType()

    def __init__(self, arr):
        self.arr = arr

    def device(self):
        return None


class _Param:
    def __init__(self, arr):
        self.value, self.grad = _Value(arr), None


def rank_main(rank, world, n, steps, run, broken, q):
    from tt_bio.train import optim, launcher as L
    import torch
    torch.set_num_threads(L.host_threads(world))
    optim.to_host = lambda t, dtype=None: t.arr if isinstance(t, _Value) else t
    optim.to_device = lambda arr, dev, dtype=None: _Value(arr)
    sizes = [TENSOR] * (n // TENSOR) + ([n % TENSOR] if n % TENSOR else [])
    names = [f"p{i:04d}" for i in range(len(sizes))]
    params = {nm: _Param(np.full(s, 0.02, np.float32)) for nm, s in zip(names, sizes)}
    ax = L.ProcessAxis(name="dp", device_ids=tuple(range(world)), dp_rank=rank, run=str(run))
    tm = {"rs": 0.0, "ga": 0.0}

    def timed(fn, key):
        def w(*a, **k):
            t0 = time.perf_counter(); r = fn(*a, **k); tm[key] += time.perf_counter() - t0
            return r
        return w
    object.__setattr__(ax, "reduce_scatter", timed(ax.reduce_scatter, "rs"))
    object.__setattr__(ax, "all_gather", timed(ax.all_gather, "ga"))
    if hasattr(ax, "all_gather_parts"):
        object.__setattr__(ax, "all_gather_parts", timed(ax.all_gather_parts, "ga"))
    opt = optim.AdamW(params, lr=1e-4, clip_norm=1.0, weight_decay=0.0,
                      data_parallel=ax, shard=True)
    grads = {nm: np.empty(s, np.float32) for nm, s in zip(names, sizes)}
    rows = []
    for s in range(1, steps + 1):
        for i, nm in enumerate(names):
            grads[nm].fill(np.float32(1e-3 * ((i + s + rank) % 7 - 3)))
        if broken and rank == world - 1 and s == 1:
            grads[names[0]][0] += np.float32(1.0)
        tm["rs"] = tm["ga"] = 0.0
        t0 = time.perf_counter()
        opt.step(replicas={nm: [g] for nm, g in grads.items()})
        wall = time.perf_counter() - t0
        h = hashlib.blake2b(digest_size=8)
        for nm in names[::17]:
            h.update(opt.master[nm].tobytes())
        row = {"step": s, "step_s": wall, "rs_s": tm["rs"], "gather_s": tm["ga"],
               "digest": h.hexdigest()}
        if rank == 0 and s == 1:
            ref = optim.AdamW({nm: _Param(np.full(z, 0.02, np.float32)) for nm, z in zip(names, sizes)},
                              lr=1e-4, clip_norm=1.0, weight_decay=0.0)
            for i, nm in enumerate(names):
                g = np.zeros(sizes[i], np.float32)
                for r in range(world):
                    g += np.float32(1e-3 * ((i + s + r) % 7 - 3))
                ref.params[nm].grad = g
            ref.step()
            row["ref_ok"] = all(np.allclose(opt.master[nm], ref.master[nm], rtol=1e-5, atol=1e-6)
                                for nm in names)
            del ref
        rows.append(row)
    q.put({"rank": rank, "rows": rows, "threads": torch.get_num_threads()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--n", type=int, default=N_RFD4)
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path("/dev/shm") / f"r4t-scale-opt-{os.getpid()}"
    run.mkdir()
    ctx = mp.get_context("spawn"); q = ctx.Queue()
    load0 = os.getloadavg()
    ps = [ctx.Process(target=rank_main, args=(r, a.world, a.n, a.steps, run, a.broken, q))
          for r in range(a.world)]
    try:
        for p in ps: p.start()
        res = sorted((q.get(timeout=1800) for _ in ps), key=lambda r: r["rank"])
        for p in ps: p.join()
    finally:
        shutil.rmtree(run, ignore_errors=True)
    agree = all(len({r["rows"][i]["digest"] for r in res}) == 1 for i in range(a.steps))
    ref_ok = bool(res[0]["rows"][0]["ref_ok"])
    med = lambda k: float(np.median([row[k] for r in res for row in r["rows"][1:]]))
    slowest = [max(r["rows"][i]["step_s"] for r in res) for i in range(1, a.steps)]
    out = {"host": socket.gethostname(), "world": a.world, "n": a.n, "steps": a.steps,
           "broken": a.broken, "ranks_agree": agree, "ref_ok": ref_ok, "PASS": agree and ref_ok,
           "median_step_s": med("step_s"), "median_slowest_rank_step_s": float(np.median(slowest)),
           "median_rs_s": med("rs_s"), "median_gather_s": med("gather_s"),
           "torch_threads": res[0]["threads"], "loadavg_before": load0, "per_rank": res}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_rank"}))
    raise SystemExit(0 if out["PASS"] != a.broken else 1)


if __name__ == "__main__":
    main()
