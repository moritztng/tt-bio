#!/usr/bin/env python3
"""bcx-reduce: this row's three backward levers as switches inside one process, on the lever arm
`bwd` (bcx-bwdplan's levers, bcx-bytes's tri fix in the tree).

An arm is `+`-joined switches; `base` is all three off:
  reduce  `autograd.TREE_REDUCE`: fp32 leading-axis and bias reduces as a pairwise add tree
  perm    `taped_ttnn.REBLOCK_PERMUTE_BW`: a permute's backward through the reblock kernels
  fanin   `autograd.FANIN_MIXED`: `add_grad` without the widening casts

  arms   bcx-bytes's block driver (`bytes.py arms`): one checkpointed block forward and backward
         per arm, interleaved, walls, CPU, AICLK, bits against the first arm, `--f64` grade
  calls  one block per arm with every `_reduce_to` and `_permute_back` call of the backward graded
         where it happens: the reduce against a float64 sum of its own device input, the permute
         against torch's permute of its own device input, bit for bit
  whole  bcx-stack's whole 4+48 gradient step, arms interleaved; --f64 grades the logit gradient
         against float64 (~49 min at n=256), without it the arms are graded against base only
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "bcx_reduce"
MINE = ("reduce", "perm", "fanin")


class Arms:
    def __init__(self):
        from tt_bio import autograd as ag, taped_ttnn as T, reblock_permute as R
        self.ag, self.T, self.R = ag, T, R
        self.served = collections.Counter()
        tree = ag._tree_sum

        def counted(t, ax):
            self.served[f"tree {[int(d) for d in t.shape]} ax={ax}"] += 1
            return tree(t, ax)

        ag._tree_sum = counted

    def set(self, name):
        sw = set(name.split("+")) - {"base"}
        assert sw <= set(MINE), f"unknown switch in {name}"
        self.ag.TREE_REDUCE = "reduce" in sw
        self.T.REBLOCK_PERMUTE_BW = "perm" in sw
        self.ag.FANIN_MIXED = "fanin" in sw
        self.served["reblock fwd+back (cumulative)"] = self.R.STATS[0] + self.R.STATS_BACK[0]


def cmd_calls(args):
    import torch
    import ttnn
    from perf.bcx_bytes import bytes as B
    from perf.bcx_stack import stack as S
    arms_ = Arms()
    ag, T = arms_.ag, arms_.T
    log = []
    real_reduce, real_perm = ag._reduce_to, T._permute_back

    def reduce_to(g, shape):
        out = real_reduce(g, shape)
        if g.dtype == ttnn.float32 and [int(d) for d in g.shape] != [int(d) for d in shape]:
            x = ttnn.to_torch(g).double()
            want = [int(d) for d in shape]
            if x.numel() <= torch.Size(want).numel():
                return out                  # a reshape, or an operand wider than its gradient
            pad = [1] * (x.dim() - len(want)) + want
            axes = [i for i in range(x.dim()) if pad[i] == 1 and x.shape[i] != 1]
            ref = x.sum(dim=axes, keepdim=True).reshape(want)
            y = ttnn.to_torch(out).double().reshape(want)
            log.append({"call": "reduce_to", "shape": list(x.shape), "axes": axes,
                        "rel_l2_vs_f64": float((y - ref).norm() / ref.norm()),
                        "torch_f32_rel_l2_vs_f64": float(
                            (x.float().sum(dim=axes, keepdim=True).reshape(want).double() - ref).norm()
                            / ref.norm())})
        return out

    def permute_back(g, inv):
        out = real_perm(g, inv)
        x = ttnn.to_torch(g)
        log.append({"call": "permute_back", "shape": [int(d) for d in g.shape], "inv": inv,
                    "dtype": str(g.dtype),
                    "bit_identical_to_torch": bool(torch.equal(ttnn.to_torch(out), x.permute(inv)))})
        return out

    ag._reduce_to = reduce_to
    T._reduce_to = reduce_to          # taped_ttnn imported it by name
    T._permute_back = permute_back
    lv, dev, ref = S.open_all(args)
    lv.arm(args.arm)
    blob = {"stamp": S.stamp(args), "lever_arm": args.arm, "points": []}
    for n in [int(x) for x in args.ns.split(",")]:
        m0, z0, wm, wz = S.inputs(ref, n, args.seed)
        for stack_name in args.stacks.split(","):
            for a in args.arms.split(","):
                arms_.set(a)
                log.clear()
                B._run_block(dev, lv, m0, z0, wm, wz, stack_name)
                by = collections.defaultdict(list)
                for r in log:
                    key = (r["call"], str(r["shape"]), str(r.get("inv", r.get("axes"))))
                    by[key].append(r)
                rows = []
                for (call, shape, what), rs in sorted(by.items()):
                    row = {"call": call, "shape": shape, "arg": what, "calls": len(rs)}
                    if call == "reduce_to":
                        v = [r["rel_l2_vs_f64"] for r in rs]
                        row.update(rel_l2_vs_f64_max=max(v), rel_l2_vs_f64_min=min(v),
                                   torch_f32_max=max(r["torch_f32_rel_l2_vs_f64"] for r in rs))
                    else:
                        row["bit_identical_all"] = all(r["bit_identical_to_torch"] for r in rs)
                    rows.append(row)
                pt = {"n": n, "stack": stack_name, "arm": a, "calls": rows}
                print(json.dumps(pt), flush=True)
                blob["points"].append(pt)
    blob["served"] = dict(arms_.served)
    (OUT / (args.out or "calls.json")).write_text(json.dumps(blob, indent=1, default=str))


def cmd_arms(args):
    from perf.bcx_bytes import bytes as B
    B.Arms, B.OUT = Arms, OUT
    _REAL["arms"](args)


def cmd_whole(args):
    from perf.bcx_stack import stack as S
    arms_ = Arms()
    real = S.open_all

    def open_all(a):
        lv, dev, ref = real(a)
        lever = lv.arm

        def arm(name):
            lever(args.arm)
            arms_.set(name)

        lv.arm = arm
        return lv, dev, ref

    S.open_all, S.OUT = open_all, OUT
    S.cmd_whole(args)
    print("served", dict(arms_.served), flush=True)


_REAL = {}


def main():
    """bytes.py's own argument parser, with its `arms`/`whole` routed here."""
    from perf.bcx_bytes import bytes as B
    sub = sys.argv[1]
    sys.argv[1] = {"calls": "arms"}.get(sub, sub)
    _REAL["arms"] = B.cmd_arms
    B.cmd_arms = B.cmd_whole = {"arms": cmd_arms, "calls": cmd_calls, "whole": cmd_whole}[sub]
    B.main()


if __name__ == "__main__":
    main()
