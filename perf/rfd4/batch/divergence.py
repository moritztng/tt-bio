"""Which op makes design k's numbers depend on its batch? One sublayer, op by op, on the card.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/batch/divergence.py --weights <bundle> --out <json> \
        [--sublayer transition_adaln] [--batch 4] [--n 128]

`independence.py` names the sublayer; this names the op inside it. Every ttnn call the sublayer makes
is recorded in order, once for the batch of B and once for design 0 alone, and design 0's slice of
each batched output is compared bit for bit with the single run's output. The first op whose inputs
agreed but whose output does not is the one that reduces differently at width B. A control runs
design 0 alone twice, so run-to-run nondeterminism cannot pass for batch dependence.
"""
import argparse
import functools
import json
import sys

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

TRACED = ("linear", "matmul", "layer_norm", "rms_norm", "multiply", "add", "subtract", "sigmoid", "silu",
          "permute", "transpose", "typecast", "reshape", "concat", "slice", "softmax", "to_layout", "mul",
          "sum", "exp", "clone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sublayer", default="transition_adaln", choices=("transition_adaln", "tri_mult_out", "tri_mult_in"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()

    import ttnn

    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.trunk import trunk_masks

    eng = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev, tr = eng._device, eng.trunk
    B, n = a.batch, a.n
    cfg = eng.config["nn"]
    D, P, C = cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(0)
    s, z, c = torch.randn(B, n, D, generator=g), torch.randn(B, n, n, P, generator=g), torch.randn(B, n, C, generator=g)
    m = torch.ones(B, n)
    m[:, n - 7:] = 0

    def up(t, dtype):
        return ttnn.from_torch(t.float().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    blk, pu = tr.blocks[0], tr.pair_updates[0]
    if a.sublayer == "transition_adaln":
        def run(rows):
            mk = trunk_masks(m[rows], dev, dtype=ttnn.bfloat16, residual_dtype=ttnn.float32)
            return blk.transition(up(s[rows], ttnn.bfloat16), up(c[rows], ttnn.bfloat16), mk[0])
    else:
        mod = pu.tri_out if a.sublayer == "tri_mult_out" else pu.tri_in

        def run(rows):
            mk = trunk_masks(m[rows], dev, dtype=ttnn.bfloat16, residual_dtype=ttnn.float32)
            return mod(up(z[rows], ttnn.bfloat16), mk[2])

    tape = []

    def record(name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kw):
            out = fn(*args, **kw)
            if isinstance(out, ttnn.Tensor) and out.is_allocated():
                ins = [t for t in args if isinstance(t, ttnn.Tensor)][:2]
                tape.append((name, ttnn.to_torch(out).float(), [tuple(t.shape) for t in ins],
                             [ttnn.to_torch(t).float() if t.is_allocated() else None for t in ins]))
            return out
        return wrapped

    originals = {k: getattr(ttnn, k) for k in TRACED if hasattr(ttnn, k)}

    def traced(rows):
        tape.clear()
        for k, f in originals.items():
            setattr(ttnn, k, record(k, f))
        try:
            run(rows)
        finally:
            for k, f in originals.items():
                setattr(ttnn, k, f)
        return list(tape)

    def head(t, shape0):
        """Design 0's part of a tensor whose leading axis carries the batch (B or B*something)."""
        if t.shape[0] % B == 0 and t.shape[0] // B == shape0:
            return t[:shape0]
        return None

    batched = traced(list(range(B)))
    single = traced([0])
    again = traced([0])
    control = len(single) == len(again) and all(torch.equal(x[1], y[1]) for x, y in zip(single, again))

    ops, first = [], None
    for i, (sb, ss) in enumerate(zip(batched, single)):
        name = ss[0]
        sl = head(sb[1], ss[1].shape[0]) if sb[0] == name else None
        if sl is None or sl.shape != ss[1].shape:
            ops.append({"i": i, "op": name, "batched_op": sb[0], "comparable": False,
                        "single_shape": list(ss[1].shape), "batched_shape": list(sb[1].shape)})
            continue
        ins_equal = all(xb is None or xs is None or (head(xb, xs.shape[0]) is not None
                        and torch.equal(head(xb, xs.shape[0]), xs)) for xb, xs in zip(sb[3], ss[3]))
        equal = torch.equal(sl, ss[1])
        dev_ = float((sl - ss[1]).abs().max() / ss[1].abs().max().clamp_min(1e-30))
        row = {"i": i, "op": name, "comparable": True, "inputs_equal": ins_equal, "bit_exact": equal,
               "max_rel_dev": dev_, "single_shape": list(ss[1].shape), "input_shapes": ss[2],
               "batched_input_shapes": sb[2]}
        ops.append(row)
        if first is None and ins_equal and not equal:
            first = row
    res = {"sublayer": a.sublayer, "batch": B, "n": n, "ops_batched": len(batched), "ops_single": len(single),
           "same_op_sequence": [x[0] for x in batched] == [x[0] for x in single],
           "single_repeatable": control, "first_divergent_op": first, "ops": ops}
    open(a.out, "w").write(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "ops"}, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
