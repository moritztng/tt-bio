"""r4t-scale (b): data-parallel gradient equivalence on one taped production PairUpdate block.

Claim graded: DP over k chips, one example each, summed by the launcher's collective, equals one
chip taking the same k examples (gradient accumulated on the tape), within the G5 bar (PLAN.md:
bf16 reduction rel_L2 <= 1.0e-2 and cos >= 0.9999, per parameter and global).

  --role acc  --k K               one chip, examples 0..K-1 accumulated -> OUT/acc.npz
  --role rank --rank r            one chip, example r alone              -> OUT/local_r.npz
  --role reduce --k K [--broken]  host only: the K local grads through ProcessAxis._exchange
                                  (K threads, rank order), -> OUT/dp.npz; --broken leaves the
                                  last rank unreduced, which must FAIL
  --role grade                    OUT/acc.npz vs OUT/dp.npz -> verdict JSON, rc 0 PASS / 1 FAIL

Ranks run one at a time on one card (numeric arm) or concurrently on k cards (true DP arm, same
files). Synthetic release-shaped weights (dim 768, pair 384, tri 256), loss <out, C>/numel.
"""
import argparse, hashlib, json, os, sys, threading, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
BAR_REL, BAR_COS = 1.0e-2, 0.9999


def device_grads(examples, n):
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import numpy as np, torch, ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import trunk as TR
    from tt_bio.train import lora
    import rfd4_reference as REF
    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])

    def put(t):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w = REF.pair_update_weights(768, 384, 256, seed=100, prefix="pu")
    blk = TR.PairUpdate(T.WeightScope({k: v.to(torch.float32) for k, v in w.items()}).child("pu"),
                        TR.compute_kernel_config())
    x0, z0, _, _ = REF.block_inputs(1, n, 768, 384, 256, seed=1)
    xt = put(x0)
    C = torch.randn(1, n, n, 384, generator=torch.Generator().manual_seed(7))
    seed_dev = put(C / C.numel())
    ag.install()
    params = lora.walked_weights(lambda x, z: blk(x, z, None, None, None), None, blk, xt,
                                  put(z0))
    clk = []
    for e in examples:
        zin = put(REF.block_inputs(1, n, 768, 384, 256, seed=1000 + e)[1])
        with ag.tape():
            out = blk(xt, zin, None, None, None)
        out_t = out if isinstance(out, ag.Tensor) else ag.Tensor(out)
        ag.backward([out_t], [seed_dev])   # p.grad accumulates (fp32) across examples
        ttnn.synchronize_device(dev)
        clk.append(int(open(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read().split()[0]))
        ag.release_pins()
        del out, out_t
    g = {k: ttnn.to_torch(p.grad).float().numpy() for k, p in params.items() if p.grad is not None}
    return g, {"card": card, "aiclk_after_each_bwd": clk, "params": len(params), "with_grad": len(g)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["acc", "rank", "reduce", "grade"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.role in ("acc", "rank"):
        ex = list(range(a.k)) if a.role == "acc" else [a.rank]
        t0 = time.time()
        g, meta = device_grads(ex, a.n)
        name = "acc" if a.role == "acc" else f"local_{a.rank}"
        np.savez(out / f"{name}.npz", **g)
        meta.update(role=a.role, examples=ex, n=a.n, wall_s=round(time.time() - t0, 2))
        (out / f"{name}.json").write_text(json.dumps(meta))
        print(json.dumps(meta))
        return
    if a.role == "reduce":
        from tt_bio.train.launcher import ProcessAxis
        run = out / "rdv"; run.mkdir(exist_ok=True)
        locs = [dict(np.load(out / f"local_{r}.npz")) for r in range(a.k)]
        res = [None] * a.k

        def go(r):
            if a.broken and r == a.k - 1:
                res[r] = locs[r]     # the unreduced rank
                return
            ax = ProcessAxis(name="dp", device_ids=tuple(range(a.k)), dp_rank=r, run=str(run))
            res[r] = ax.reduce_all({n: [v] for n, v in locs[r].items()})
        ts = [threading.Thread(target=go, args=(r,)) for r in range(a.k - (1 if a.broken else 0))]
        [t.start() for t in ts]; [t.join() for t in ts]
        if a.broken:
            go(a.k - 1)
        np.savez(out / "dp.npz", **res[a.k - 1])   # the last rank's view: the broken one if broken
        print(json.dumps({"reduced": a.k, "broken": a.broken}))
        return
    acc, dp = dict(np.load(out / "acc.npz")), dict(np.load(out / "dp.npz"))
    assert sorted(acc) == sorted(dp), "parameter sets differ"
    rows, num, den, dot, na, nb = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for k in sorted(acc):
        x, y = acc[k].astype(np.float64).ravel(), dp[k].astype(np.float64).ravel()
        rel = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
        cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))
        rows.append((k, rel, cos))
        num += float(((y - x) ** 2).sum()); den += float((x ** 2).sum())
        dot += float(x @ y); na += float(x @ x); nb += float(y @ y)
    grel, gcos = (num / den) ** 0.5, dot / (na * nb) ** 0.5
    worst = max(rows, key=lambda r: r[1])
    ok = all(r[1] <= BAR_REL and r[2] >= BAR_COS for r in rows)
    v = {"PASS": ok, "params": len(rows), "global_rel_L2": grel, "global_cos": gcos,
         "worst": {"param": worst[0], "rel_L2": worst[1], "cos": worst[2]},
         "bar": {"rel_L2": BAR_REL, "cos": BAR_COS},
         "acc_digest": hashlib.sha256(b"".join(acc[k].tobytes() for k in sorted(acc))).hexdigest()[:16]}
    (out / "verdict.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
