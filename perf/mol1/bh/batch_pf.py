"""Does Mol-1's trunk pair stack pay for K inputs on the leading dim on one Blackhole chip?

mol1-bh-batch, step 1. Real Mol-1 weights and levers, the 48-block trunk stack (Pairformer at c_z 320, PairMixer at
448, the width adapters), random (s, z) at the trunk's shapes, no padding mask. Times one stack call per K at --n and
one solo call at --n2, warm median of --reps, AICLK sampled during. For K > 1 it checks each member's z against the
same input run solo (does a prediction depend on its batch neighbours?). A K that raises is recorded with its
traceback: that is the list of ops the batch dim still has to reach.

Run: TT_VISIBLE_DEVICES=<card> python perf/mol1/bh/batch_pf.py --out batch_pf.json [--n 256 --ks 1,2,4 --n2 512]
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clocksample import during  # noqa: E402


def inputs(B, N, C, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, 384, generator=g), torch.randn(B, N, N, C, generator=g)


def run(pf, dev, s_h, z_h):
    s = ttnn.from_torch(s_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    z = ttnn.from_torch(z_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    so, zo = pf(s, z)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    zh = ttnn.to_torch(zo).float()
    for t in (so, zo):
        ttnn.deallocate(t)
    return dt, zh


def timed(pf, dev, s_h, z_h, reps):
    run(pf, dev, s_h, z_h)
    ts, z = [], None
    for _ in range(reps):
        dt, z = run(pf, dev, s_h, z_h)
        ts.append(dt)
    return sorted(ts)[len(ts) // 2], ts, z


def cmp(a, b):
    a, b = a.double(), b.double()
    return {"equal": bool(torch.equal(a, b)), "max_abs": (a - b).abs().max().item(),
            "rel_rms": ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--n2", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    res = {"n": a.n, "n2": a.n2, "reps": a.reps, "k": {}}

    def save():
        a.out.write_text(json.dumps(res, indent=1))

    import tt_bio.tenstorrent as T
    from tt_bio import weights
    from tt_bio.protenix import Protenix
    model = Protenix.load_from_checkpoint(str(weights.fetch("mol1")))
    trunk = model.trunk
    dev, pf, C = trunk.dev, trunk.PF, trunk.C_Z
    res["c_z"] = C
    ks = [int(k) for k in a.ks.split(",")]
    xs = [inputs(1, a.n, C, 10 + i) for i in range(max(ks))]
    with T.levers(model._levers), during() as clk:
        med, ts, solo = timed(pf, dev, *xs[0], a.reps)
        res["k"]["1"] = {"s": med, "all": ts, "s_per_pred": med}
        _, z2 = run(pf, dev, *xs[0])
        res["solo_repeat"] = cmp(z2, solo)
        save()
        for k in ks:
            if k == 1:
                continue
            try:
                s_h = torch.cat([x[0] for x in xs[:k]])
                z_h = torch.cat([x[1] for x in xs[:k]])
                med, ts, zb = timed(pf, dev, s_h, z_h, a.reps)
                res["k"][str(k)] = {"s": med, "all": ts, "s_per_pred": med / k,
                                    "speedup_per_pred": res["k"]["1"]["s"] / (med / k),
                                    "member0_vs_solo": cmp(zb[0:1], solo)}
            except Exception as e:  # the ops that do not take a batch yet are the result
                res["k"][str(k)] = {"error": type(e).__name__, "msg": str(e)[:1500],
                                    "tb": traceback.format_exc()[-5000:]}
            save()
        if a.n2:
            med, ts, _ = timed(pf, dev, *inputs(1, a.n2, C, 99), a.reps)
            res["solo_n2"] = {"s": med, "all": ts}
            save()
    res["aiclk"] = clk.summary()
    save()
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
