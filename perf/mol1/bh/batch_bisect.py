"""Which pair op of Mol-1's trunk block gives a batch member a different answer than its solo call?

mol1-bh-batch. Real Mol-1 weights and levers (plus --extra levers), random z at --n. For a Pairformer block (c_z 320)
and a PairMixer block (448), each z op runs once on member A alone and once on [A; B], and member 0 of the batch is
compared with the solo output. Timing per op is the warm second call.

Run: TT_VISIBLE_DEVICES=<card> python perf/mol1/bh/batch_bisect.py --out bisect.json [--n 256 --k 2 --extra pair_batch]
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

OPS = ("triangle_multiplication_start", "triangle_multiplication_end", "triangle_attention_start",
       "triangle_attention_end", "transition_z")


def call(op, dev, z_h):
    z = ttnn.from_torch(z_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    out = op(z, add_to_input=True)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    h = ttnn.to_torch(out).float()
    ttnn.deallocate(out)
    return dt, h


def cmp(a, b):
    a, b = a.double(), b.double()
    return {"equal": bool(torch.equal(a, b)), "max_abs": (a - b).abs().max().item(),
            "rel_rms": ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()}


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
    model = Protenix.load_from_checkpoint(str(weights.fetch("mol1")))
    pf, dev = model.trunk.PF, model.trunk.dev
    lv = model._levers | T.parse_levers([x for x in a.extra.split(",") if x])
    pick = {"pairformer": next(i for i, b in enumerate(pf.blocks) if b.triangle_attention_start is not None),
            "pairmixer": next((i for i, b in enumerate(pf.blocks) if b.triangle_attention_start is None), None)}
    res = {"n": a.n, "k": a.k, "extra": a.extra, "blocks": {}}
    with T.levers(lv), during() as clk:
        for name, i in pick.items():
            if i is None:
                continue
            blk = pf.blocks[i]
            # the width this block runs at: adapters before it set z's channels
            C = int(blk.transition_z.__dict__.get("dim", 0)) or None
            rec = res["blocks"][name] = {"index": i, "ops": {}}
            for c in (448, 320):
                try:
                    g = torch.Generator().manual_seed(7)
                    zs = [torch.randn(1, a.n, a.n, c, generator=g) for _ in range(a.k)]
                    call(blk.transition_z, dev, zs[0])
                    rec["c_z"] = c
                    break
                except Exception:
                    continue
            for op in OPS:
                f = getattr(blk, op, None)
                if f is None:
                    continue
                try:
                    call(f, dev, zs[0])
                    t1, solo = call(f, dev, zs[0])
                    call(f, dev, torch.cat(zs))
                    tk, bat = call(f, dev, torch.cat(zs))
                    rec["ops"][op] = {"solo_s": t1, "batch_s": tk, "per_pred_x": t1 * a.k / tk,
                                      "member0": cmp(bat[0:1], solo)}
                except Exception as e:
                    rec["ops"][op] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-3000:]}
                a.out.write_text(json.dumps(res, indent=1))
    res["aiclk"] = clk.summary()
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
