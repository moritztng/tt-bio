"""Does design k's arithmetic depend on what else is in the batch? Sublayer by sublayer, on the card.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/batch/independence.py --weights <bundle> --out <json> \
        [--batch 4] [--n 128]

Each trunk sublayer gets B different inputs stacked on the sample axis, then the same B inputs one
at a time, and the outputs are compared bit for bit. Every sublayer sees identical inputs on both
sides, so a mismatch names the sublayer whose kernels reduce differently at width B, not one that
merely inherited a difference. Design 0 also runs alone a second time: a sublayer that is not
repeatable at width 1 is nondeterministic, which is a different finding from batch dependence.
The last line runs a whole trunk pass the same way.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()

    import ttnn

    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.trunk import _per_design, trunk_masks

    eng = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, tr = eng._device, eng.trunk
    B, n = a.batch, a.n
    cfg = eng.config["nn"]
    D, P, C = cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(0)
    s = torch.randn(B, n, D, generator=g)
    z = torch.randn(B, n, n, P, generator=g)
    c = torch.randn(B, n, C, generator=g)
    m = torch.ones(B, n)
    m[:, n - 7:] = 0  # a few masked tokens, as the bucket pads them

    def up(t, dtype):
        return ttnn.from_torch(t.float().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def host(t):
        """An output back on the host; the device copy is freed."""
        out = ttnn.to_torch(t).float()
        ttnn.deallocate(t)
        return out

    f32, b16 = ttnn.float32, ttnn.bfloat16
    blk, pu = tr.blocks[0], tr.pair_updates[0]

    def inputs(rows):
        mk = trunk_masks(m[rows], dev, dtype=b16, residual_dtype=f32)
        return {"s32": up(s[rows], f32), "s16": up(s[rows], b16), "z32": up(z[rows], f32),
                "z16": up(z[rows], b16), "c": up(c[rows], b16), "masks": mk}

    subs = {
        "attn.adaln": lambda x: blk.mhba.adaln(x["s16"], x["c"], x["masks"][0]),
        "attn.mha": lambda x: blk.mhba.mha(x["s16"], x["z16"], x["masks"][1]),
        "attn.scale_output": lambda x: blk.mhba.scale_output(x["s16"], x["c"], x["masks"][0]),
        "transition_adaln": lambda x: blk.transition(x["s16"], x["c"], x["masks"][0]),
        "trunk_block": lambda x: blk(x["s32"], x["z16"], x["c"], x["masks"][0], x["masks"][1], x["masks"][4]),
        "pair.write": lambda x: pu._write(x["s16"], x["masks"][0])[0],
        "pair.tri_mult_out": lambda x: _per_design(pu.tri_out, x["z16"], x["masks"][2]),
        "pair.tri_mult_in": lambda x: _per_design(pu.tri_in, x["z16"], x["masks"][2]),
        "pair.transition": lambda x: pu._transition(x["z16"], x["masks"][3]),
        "pair_update": lambda x: pu(x["s32"], x["z32"], x["masks"][0], x["masks"][2], x["masks"][3], x["masks"][5]),
        "recycle.s": lambda x: tr._recycle(x["s32"], x["s32"], tr.s_norm, tr.s_proj),
        "recycle.z": lambda x: tr._recycle(x["z32"], x["z32"], tr.z_norm, tr.z_proj),
        "trunk_pass": lambda x: tr.run_pass(ttnn.clone(x["s32"]), ttnn.clone(x["z32"]), x["c"], x["masks"]),
    }

    def free(x):
        for t in (*[v for k, v in x.items() if k != "masks"], *x["masks"]):
            if t.is_allocated():
                ttnn.deallocate(t)

    rows = []
    for name, fn in subs.items():
        # Fresh inputs per sublayer: some sublayers free what they read.
        batched, single, again = inputs(list(range(B))), [inputs([k]) for k in range(B)], inputs([0])
        got = fn(batched)
        got = [host(t) for t in got] if isinstance(got, tuple) else [host(got)]
        ref = [fn(x) for x in single]
        ref = [[host(t) for t in r] if isinstance(r, tuple) else [host(r)] for r in ref]
        rep = fn(again)
        rep = [host(t) for t in rep] if isinstance(rep, tuple) else [host(rep)]
        repeatable = all(torch.equal(x, y) for x, y in zip(rep, ref[0]))
        exact, worst = True, 0.0
        for k in range(B):
            for o, gk in enumerate(got):
                rk = ref[k][o][0]
                exact &= bool(torch.equal(gk[k], rk))
                worst = max(worst, float((gk[k] - rk).abs().max() / rk.abs().max().clamp_min(1e-30)))
        for x in (batched, *single, again):
            free(x)
        row = {"sublayer": name, "batch": B, "bit_exact": exact, "max_rel_dev": worst,
               "single_repeatable": repeatable}
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps({"n": n, "batch": B, "rows": rows}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
