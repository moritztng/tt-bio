"""Does OpenDDE's 48-block trunk pairformer run two inputs on the leading dim, and what does it cost?

The spd-bet-foldbatch falsifier, trunk half. Random (s, z) at the trunk's shapes (c_s 384, c_z 384),
real OpenDDE weights, one Wormhole chip:
  solo  N=288 (inputs A, B, C), solo N=576  -> warm ms per stack call
  batch [A; B] at N=288 (leading dim 2)     -> warm ms, member z vs its solo z, and member A's z
                                               with partner C instead of B (partner dependence)
A batch that raises is recorded with its traceback: that is the list of ops the bet has to fix.
Kill (BETS.md idea 2): batch >= 1.75x one solo, or a member's output depends on its partner.

Run: TT_VISIBLE_DEVICES=<chip> python perf/foldbatch/trunk_b2.py --out trunk_b2.json [--reps 3]
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

from perf.clocksample import during  # noqa: E402
from tt_bio.opendde import OpenDDE  # noqa: E402

C = 384


def inputs(B, N, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, C, generator=g), torch.randn(B, N, N, C, generator=g)


def run(pf, dev, s_h, z_h):
    """One stack call on fresh device copies (the stack adds into s and z in place)."""
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
    run(pf, dev, s_h, z_h)  # warm: program cache, weight caches
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
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=288)
    a = ap.parse_args()
    res = {"n": a.n, "reps": a.reps}

    def save():
        a.out.write_text(json.dumps(res, indent=1))

    model = OpenDDE.load_from_checkpoint()
    dev, pf = model.dev, model._protenix.trunk.PF
    N = a.n
    sA, zA = inputs(1, N, 1)
    sB, zB = inputs(1, N, 2)
    sC, zC = inputs(1, N, 3)
    with model_levers(model), during() as clk:
        res["solo"] = {}
        outs = {}
        for name, (s, z) in {"A": (sA, zA), "B": (sB, zB), "C": (sC, zC)}.items():
            med, ts, outs[name] = timed(pf, dev, s, z, a.reps)
            res["solo"][name] = {"s": med, "all": ts}
            save()
        # Same input twice: is the solo run itself deterministic? This is the noise floor.
        _, z2 = run(pf, dev, sA, zA)
        res["solo_repeat_A"] = cmp(z2, outs["A"])
        save()
        s2, z2_ = inputs(1, 2 * N, 4)
        med, ts, _ = timed(pf, dev, s2, z2_, a.reps)
        res["solo_2N"] = {"n": 2 * N, "s": med, "all": ts}
        save()
        try:
            med, ts, zAB = timed(pf, dev, torch.cat([sA, sB]), torch.cat([zA, zB]), a.reps)
            res["batch_AB"] = {"s": med, "all": ts,
                               "x_one_solo": med / res["solo"]["A"]["s"],
                               "A_vs_solo": cmp(zAB[0:1], outs["A"]), "B_vs_solo": cmp(zAB[1:2], outs["B"])}
            save()
            _, zAC = run(pf, dev, torch.cat([sA, sC]), torch.cat([zA, zC]))
            res["batch_AC"] = {"A_vs_AB": cmp(zAC[0:1], zAB[0:1]), "C_vs_solo": cmp(zAC[1:2], outs["C"])}
        except Exception as e:  # the ops that do not take B=2 yet are the result
            res["batch_error"] = {"type": type(e).__name__, "msg": str(e)[:2000],
                                  "tb": traceback.format_exc()[-6000:]}
    res["aiclk"] = clk.summary()
    save()
    print(json.dumps({k: v for k, v in res.items() if k != "batch_error"}, indent=1))
    if "batch_error" in res:
        print(res["batch_error"]["tb"])


def model_levers(model):
    import tt_bio.tenstorrent as T
    return T.levers(model._levers)


if __name__ == "__main__":
    main()
