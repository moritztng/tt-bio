"""r4t-scale: data-parallel training steps on RFD4's taped pair stack, one rank per chip.

K production PairUpdate blocks (default 7, the whole pair track, 91-98 % of RFD4's FLOPs) taped
in place, one example per rank per step, gradients summed across ranks by the launcher's
collective and stepped by AdamW, replicated or shard=True (ZeRO-1). Synthetic release-shaped
weights (dim 768, pair 384, tri 256), loss <out, C>/numel. Ranks are processes, each pinned to
its own card; rank r's example at step s is seed 1000 + s*world + r, so a world-k step consumes
k distinct examples. Per step it records fwd / bwd / read-back / optimizer (exchange inside) /
total, AICLK sampled every 50 ms during the step, and the master digest, which must be equal
across ranks.

  python3 perf/r4t_scale/dp_pairstack.py --cards 0 --steps 6 --out ~/r4t/scale/dp1
  python3 perf/r4t_scale/dp_pairstack.py --cards 0,1 --steps 6 --shard --out ~/r4t/scale/dp2
"""
import argparse, hashlib, json, os, socket, subprocess, sys, threading, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def rank_main(a, rank, world, run):
    sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import numpy as np, torch, ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.train import lora
    from tt_bio.train.launcher import ProcessAxis
    from tt_bio.train.mesh import Axis
    from tt_bio.train.optim import AdamW
    import rfd4_reference as REF
    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    clk, stop = [], threading.Event()

    def sample():
        path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
        while not stop.is_set():
            try:
                clk.append(int(open(path).read().split()[0]))
            except Exception:
                pass
            time.sleep(0.05)

    def put(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
    blocks = []
    for k in range(a.blocks):
        w = REF.pair_update_weights(768, 384, 256, seed=100 + k, prefix="pu")
        blocks.append(TR.PairUpdate(T.WeightScope({n: v.to(torch.float32) for n, v in w.items()})
                                    .child("pu"), TR.compute_kernel_config()))

    class Stack:
        def __init__(self, blocks):
            self.blocks = blocks

        def __call__(self, x, z):
            for b in self.blocks:
                z = b(x, z, None, None, None)
            return z
    model = Stack(blocks)
    x0, z0, _, _ = REF.block_inputs(1, a.n, 768, 384, 256, seed=1)
    xt = put(x0)
    C = torch.randn(1, a.n, a.n, 384, generator=torch.Generator().manual_seed(7))
    numel = C.numel()
    seed_dev = put(C / numel)
    ag.install()
    params = lora.walked_weights(model, None, model, xt, put(z0))
    ax = (ProcessAxis(name="dp", device_ids=tuple(range(world)), dp_rank=rank, run=str(run))
          if world > 1 else Axis(name="dp", device_ids=(card,)))
    opt = AdamW(dict(params), lr=1e-4, clip_norm=1.0, weight_decay=0.0, data_parallel=ax,
                shard=a.shard)
    threading.Thread(target=sample, daemon=True).start()
    rows = []
    for step in range(a.steps):
        zin = put(REF.block_inputs(1, a.n, 768, 384, 256, seed=1000 + step * world + rank)[1])
        ttnn.synchronize_device(dev)
        c0 = len(clk)
        t0 = time.time()
        with ag.tape():
            out = model(xt, zin)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        out_t = out if isinstance(out, ag.Tensor) else ag.Tensor(out)
        loss = float((ttnn.to_torch(out_t.value).float() * C).sum() / numel)
        ag.backward([out_t], [seed_dev])
        ttnn.synchronize_device(dev)
        t2 = time.time()
        reps = ({n: [np.ascontiguousarray(ttnn.to_torch(p.grad).float().numpy())]
                 for n, p in params.items() if p.grad is not None} if world > 1 else None)
        t3 = time.time()
        opt.step(replicas=reps) if world > 1 else opt.step()
        params.rebind()
        ttnn.synchronize_device(dev)
        t4 = time.time()
        for p in params.values():
            p.grad = None
        ag.release_pins()
        del out, out_t
        md = hashlib.sha256(b"".join(opt.master[n].tobytes() for n in sorted(opt.master))).hexdigest()[:16]
        s = clk[c0:] or clk[-1:]
        ex = getattr(ax, "state", {})
        rows.append(dict(rank=rank, step=step + 1, loss=loss, fwd_s=t1 - t0, bwd_s=t2 - t1,
                         readback_s=t3 - t2, opt_s=t4 - t3, step_s=t4 - t0, master_digest=md,
                         aiclk=[min(s or [0]), max(s or [0])]))
        print(json.dumps(rows[-1]), flush=True)
    stop.set()
    Path(a.out, f"rank{rank}.json").write_text(json.dumps(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--run", default="")
    a = ap.parse_args()
    cards = [int(c) for c in a.cards.split(",")]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.rank >= 0:
        return rank_main(a, a.rank, len(cards), a.run)
    run = Path("/dev/shm") / f"r4t-scale-dp-{os.getpid()}"
    run.mkdir()
    holder = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:r4t-scale")
    procs = []
    for r, c in enumerate(cards):
        env = dict(os.environ, TT_VISIBLE_DEVICES=str(c), TT_BIO_LEASE_CARDS=str(c),
                   TT_BIO_LEASE_HOLDER=holder)
        argv = [sys.executable, __file__, "--cards", a.cards, "--n", str(a.n), "--blocks",
                str(a.blocks), "--steps", str(a.steps), "--out", a.out, "--rank", str(r),
                "--run", str(run)] + (["--shard"] if a.shard else [])
        procs.append(subprocess.Popen(argv, env=env, stdout=open(Path(a.out, f"rank{r}.log"), "w"),
                                      stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    subprocess.run(["rm", "-rf", str(run)])
    rows = [json.loads(Path(a.out, f"rank{r}.json").read_text()) for r in range(len(cards))
            if Path(a.out, f"rank{r}.json").exists()]
    summ = {"host": socket.gethostname(), "cards": cards, "world": len(cards), "shard": a.shard,
            "n": a.n, "blocks": a.blocks, "rcs": rcs}
    if len(rows) == len(cards) and all(rc == 0 for rc in rcs):
        import statistics as st
        warm = [r for rk in rows for r in rk[1:]]
        summ.update(
            digests_agree=all(len({rk[i]["master_digest"] for rk in rows}) == 1 for i in range(a.steps)),
            median_step_s=st.median(max(rk[i]["step_s"] for rk in rows) for i in range(1, a.steps)),
            median_fwd_s=st.median(r["fwd_s"] for r in warm), median_bwd_s=st.median(r["bwd_s"] for r in warm),
            median_readback_s=st.median(r["readback_s"] for r in warm),
            median_opt_s=st.median(r["opt_s"] for r in warm),
            aiclk=[min(r["aiclk"][0] for r in warm), max(r["aiclk"][1] for r in warm)],
            samples_per_s=len(cards) / st.median(max(rk[i]["step_s"] for rk in rows) for i in range(1, a.steps)))
    Path(a.out, "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    sys.exit(0 if all(rc == 0 for rc in rcs) else 1)


if __name__ == "__main__":
    main()
