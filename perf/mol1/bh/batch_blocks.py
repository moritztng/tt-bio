"""Where in Mol-1's trunk stack does a batch member first leave its solo run?

mol1-bh-batch. Every op of every block on the z the stack itself carries (not fresh random input, which
batch_bisect.py showed is bit-exact per op): solo input A and batch [A; B] advance in lockstep through the 48 blocks,
op by op, and member 0 is compared with the solo z after each op. Stops at the first op that differs.

Run: TT_VISIBLE_DEVICES=<card> python perf/mol1/bh/batch_blocks.py --out blocks.json [--k 2 --extra pair_batch]
"""
import argparse
import json
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402

OPS = ("triangle_multiplication_start", "triangle_multiplication_end", "triangle_attention_start",
       "triangle_attention_end", "transition_z")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    import tt_bio.tenstorrent as T
    from tt_bio import weights
    from tt_bio.protenix import Protenix
    m = Protenix.load_from_checkpoint(str(weights.fetch("mol1")))
    pf, dev, C = m.trunk.PF, m.trunk.dev, m.trunk.C_Z
    g = torch.Generator().manual_seed(10)
    zs = [torch.randn(1, a.n, a.n, C, generator=g) for _ in range(a.k)]
    up = lambda h: ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    host = lambda t: ttnn.to_torch(t).float()  # noqa: E731
    zs1, zb = up(zs[0]), up(torch.cat(zs))
    res = {"n": a.n, "k": a.k, "extra": a.extra, "steps": []}

    def check(where):
        h1, hb = host(zs1), host(zb)[0:1]
        d = (hb.double() - h1.double())
        rec = {"at": where, "equal": bool(torch.equal(hb, h1)), "max_abs": d.abs().max().item(),
               "rel_rms": (d.pow(2).mean().sqrt() / h1.double().pow(2).mean().sqrt()).item(),
               "mc": str(zb.memory_config().buffer_type), "dtype": str(zb.dtype)}
        res["steps"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        return rec["equal"]

    with T.levers(m._levers | T.parse_levers([x for x in a.extra.split(",") if x])):
        check("input")
        for i, blk in enumerate(pf.blocks):
            if i in pf.adapters:
                zs1, zb = pf._adapt(zs1, i), pf._adapt(zb, i)
                if not check(f"adapter {i}"):
                    return
            for op in OPS:
                f = getattr(blk, op, None)
                if f is None:
                    continue
                kw = {"memory_config": None} if op == "transition_z" else {}
                zs1, zb = f(zs1, add_to_input=True, **kw), f(zb, add_to_input=True, **kw)
                if not check(f"block {i} {op}"):
                    return
        res["all_equal"] = True
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
