#!/usr/bin/env python3
"""How many nodes of each J4 op the crop-384 OF3T step actually puts on the tape.

Seconds off the step are (per-node delta from `speed.py`) x (this count). The count is the
half that is cheap to get wrong, so it is READ OFF THE TAPE rather than modelled: build the
step's forward exactly as `perf/of3t_stepfloor/fullstep.py` does, walk `_reverse_topo` over
the same roots, and classify each node by its backward closure's qualname. No backward runs
-- the forward is seconds and the backward is 456 of them, and the node count is settled
before the first closure fires.

Each node also records the requires_grad pattern of its parents, because `mul` and `concat`
delete a verb only when both sides want a gradient and the count of those is the number that
matters.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_perf import step as S                                  # noqa: E402
from perf.of3t_stepfloor.fullstep import (declare_all, diffusion_train,  # noqa: E402
                                          trunk_forward)

SEED = 20260921

# The nine J4 ops, and the verbs their backward closure issues per node BEFORE and AFTER
# this row. "both" is the two-parents-want-a-gradient branch. Counted from the closure
# source, which is why the census below also reports the branch each node would take.
# The nine J4 ops and the verbs their backward closure issues per taped node, BEFORE and
# AFTER this row. Counted from the closure source. "both" is the two-parents-want-a-gradient
# branch, which is the only one `mul` and `concat` shorten.
#
# `narrow` is the conservative figure: the composed form is one `ttnn.zeros` per padded side
# plus a `concat`, so two verbs when the slice touches an end of the axis and three when it
# does not. 2 -> 1 is the floor, and the floor is what gets quoted.
VERBS = {
    "mul":     {"both": (2, 1), "one": (1, 1)},
    "scale":   {"one": (1, 1)},
    "add":     {"both": (0, 0), "one": (0, 0)},
    "relu":    {"one": (2, 1)},
    "sigmoid": {"one": (3, 1)},
    "silu":    {"one": (6, 1)},
    "reshape": {"one": (1, 1)},
    "narrow":  {"one": (2, 1)},
    "concat":  {"both": (2, 1), "one": (None, None)},
}

# The same three activations again, in the copy a SHIPPED module's tape actually reaches.
# `taped_ttnn._VERBS` intercepts `ttnn.silu`/`sigmoid`/`relu` calls; `tt_bio.autograd.silu`
# and friends are only called by code that names them, which the OF3T modules do not.
TAPED_VERBS = {"silu": (6, 1), "sigmoid": (3, 1), "relu": (2, 1)}

# `multiply` is counted separately because only the nodes that meet `_binary`'s `both`
# condition shorten: same-shape operands, no fused unary, both sides wanting a gradient.
# A broadcast multiply still needs `_reduce_to` and a fused one still needs its correction.
MUL_FASTPATH = (2, 1)

# THE THIRD TABLE, and the reason this script missed the gate copy for two passes: it counted
# `autograd` and `_VERBS` and stopped. `taped_ttnn._FUSED_UNARY` holds the same three
# derivatives once more, for an activation that RIDES another verb --
# `multiply(o, g, input_tensor_b_activations=[SIGMOID])`, which is how every OpenFold3 gate is
# written. A gate never reaches `_VERBS["sigmoid"]` at all, so the two tables above cannot see
# it. Counted per ACTIVATED OPERAND THAT WANTS A GRADIENT, because that is the condition under
# which `_chain_fused` runs at all.
FUSED_UNARY_VERBS = {"SIGMOID": (3, 1), "SILU": (6, 1), "RELU": (2, 1)}
_ACT_KEYS = (("input_tensor_a_activations", 0), ("input_tensor_b_activations", 1))


def classify(t):
    """`_reverse_topo` hands back TENSORS, so the op is on `t.node.fn`; a leaf has no node."""
    n = t.node
    if n is None:
        return "<leaf>"
    q = getattr(n.fn, "__qualname__", "") or ""
    return q.split(".")[0] if q else "<unknown>"


def instrument(ag, tw, calls, taped, shapes, branch):
    """Count INVOCATIONS, not tape nodes.

    A tape walk cannot separate `silu` from `relu`: both are `_unary`, so both closures carry
    the qualname `_unary.<locals>.impl.<locals>.make.<locals>.bw`. Wrapping the two dispatch
    tables gives the op by name, the output shape, and -- for `mul` and `concat`, the only two
    whose saving depends on it -- whether both parents wanted a gradient.
    """
    def note(key, out, parents):
        calls[key] += 1
        if getattr(out, "node", None) is None:
            return
        taped[key] += 1
        want = sum(1 for p in parents if getattr(p, "requires_grad", False))
        branch[f"{key}|{'both' if want >= 2 else 'one'}|{len(parents)}p"] += 1
        try:
            shapes[key][str([int(d) for d in out.value.shape])] += 1
        except Exception:
            shapes[key]["<unreadable>"] += 1

    import ttnn as _ttnn

    def mul_fastpath(args, kwargs):
        """The condition `_binary`'s `both` branch tests, evaluated at forward time.

        `mul_bw` replaces the two multiplies only when no fused unary sits on either operand,
        both operands are tensors, and no operand broadcasts -- everything else still needs
        `_reduce_to` or an activation correction. requires_grad is known here too, so the
        forward-time answer is the backward-time answer.
        """
        if len(args) < 2:
            return False
        if kwargs.get("input_tensor_a_activations") or kwargs.get(
                "input_tensor_b_activations"):
            return False
        a, b = args[0], args[1]
        if not isinstance(b, (ag.Tensor, _ttnn.Tensor)):
            return False
        try:
            sa = tuple(int(d) for d in (a.value if isinstance(a, ag.Tensor) else a).shape)
            sb = tuple(int(d) for d in (b.value if isinstance(b, ag.Tensor) else b).shape)
        except Exception:
            return False
        return sa == sb

    for name, impl in list(tw._VERBS.items()):
        def w(shipped, args, kwargs, _n=name, _i=impl):
            out = _i(shipped, args, kwargs)
            key = f"taped_ttnn:{_n}"
            note(key, out, [a for a in args if isinstance(a, ag.Tensor)])
            if _n in ("multiply", "multiply_") and getattr(out, "node", None) is not None \
                    and mul_fastpath(args, kwargs):
                parents = [a for a in args if isinstance(a, ag.Tensor)]
                if sum(1 for p in parents if p.requires_grad) >= 2:
                    calls["taped_ttnn:multiply|fastpath"] += 1
            if getattr(out, "node", None) is not None:
                for _k, _idx in _ACT_KEYS:
                    acts = list(kwargs.get(_k) or ())
                    if len(acts) != 1 or _idx >= len(args):
                        continue
                    operand = args[_idx]
                    if not isinstance(operand, ag.Tensor) or not operand.requires_grad:
                        continue
                    # A bare UnaryOpType only; a parameterised one carries `op_type` and its
                    # derivative is the constant, which has no wheel backward to take.
                    act = acts[0]
                    if hasattr(act, "op_type"):
                        continue
                    nm = getattr(act, "name", None) or str(act).rsplit(".", 1)[-1]
                    if nm in FUSED_UNARY_VERBS:
                        calls[f"fused_unary:{nm}"] += 1
            return out
        tw._VERBS[name] = w

    for name in ("mul", "scale", "add", "relu", "sigmoid", "silu", "reshape",
                 "narrow", "concat"):
        orig = getattr(ag, name)
        def w2(*args, _n=name, _o=orig, **kw):
            out = _o(*args, **kw)
            first = args[0] if args else None
            parents = list(first) if _n == "concat" and isinstance(first, (list, tuple)) \
                else [a for a in args if isinstance(a, ag.Tensor)]
            note(f"autograd:{_n}", out, parents)
            return out
        setattr(ag, name, w2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "config": {"crop": a.tokens, "diffusion_samples": a.samples, "cycles": a.cycles,
                   "stage": a.stage, "backward_run": False}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        import numpy as np
        import ttnn
        from tt_bio import autograd as ag
        from tt_bio.tenstorrent import get_device

        # READ, not asserted. The sprint grades on `exact_training(False)` (pass 450), and an
        # arm whose setting is not written down is comparable to no other arm.
        out["env"]["exact_training_ops"] = list(ag.exact_training_ops())

        calls = collections.Counter(); taped = collections.Counter()
        branch = collections.Counter()
        shapes = collections.defaultdict(collections.Counter)
        import tt_bio.taped_ttnn as tw
        instrument(ag, tw, calls, taped, shapes, branch)

        # `S.capture` runs a real untaped `predict_one` and intercepts it at the trunk, so
        # the counters after it are an INFERENCE fold's reach -- which is A46 clause 4's
        # question, measured rather than argued. A taped count of zero here says no closure
        # this row touched exists on an inference path at all.
        held, _meta = S.capture(a.tokens, out)
        out["inference_fold"] = {
            "calls": dict(calls.most_common()),
            "taped_calls": dict(taped.most_common()),
            "note": "counters after S.capture's untaped predict_one, before any tape opens",
        }
        trunk = held["trunk"][0]
        sampler, sargs, _skw = held["sampler"]
        dev = get_device()
        declare_all(trunk, sampler, out)
        rng = np.random.default_rng(SEED)

        t0 = time.perf_counter()
        with ag.tape():
            s_tr, z_tr = trunk_forward(trunk, held, 1, taped=True)
            ttnn.synchronize_device(dev)
            d_out = {}
            roots, _keep, _rep = diffusion_train(sampler, sargs, s_tr, z_tr,
                                                 a.samples, rng, d_out)
        out["forward_s"] = round(time.perf_counter() - t0, 3)

        # No loss heads here on purpose. The tape's topology is fixed the moment the forward
        # closes, so the node census is complete without them, and `host_losses` is a PCIe
        # download plus ~0.8 s of numpy per root that this row never reads. The call that used
        # to sit here passed `0` where it wants the token-scope INDEX ARRAY `diffusion_train`
        # returns, and died in its own banner on `len(rep)` -- deterministically, every run.
        nodes = ag._reverse_topo(list(roots))
        out["tape_nodes"] = len(nodes)
        out["nodes_by_closure"] = dict(collections.Counter(
            classify(t) for t in nodes).most_common())
        out["calls"] = dict(calls.most_common())
        out["taped_calls"] = dict(taped.most_common())
        out["branch"] = dict(sorted(branch.items()))
        out["shapes"] = {k: dict(v.most_common(8)) for k, v in shapes.items()}

        saved = {}
        for op, spec in VERBS.items():
            tot = 0
            for key, (before, after) in spec.items():
                if before is None:
                    continue
                tot += sum(v for k, v in branch.items()
                           if k.startswith(f"autograd:{op}|{key}|")) * (before - after)
            saved[f"autograd:{op}"] = tot
        for op, (before, after) in TAPED_VERBS.items():
            saved[f"taped_ttnn:{op}"] = taped.get(f"taped_ttnn:{op}", 0) * (before - after)
        b4, aft = MUL_FASTPATH
        saved["taped_ttnn:multiply(fastpath)"] = \
            calls.get("taped_ttnn:multiply|fastpath", 0) * (b4 - aft)
        for nm, (before, after) in FUSED_UNARY_VERBS.items():
            saved[f"fused_unary:{nm}"] = \
                calls.get(f"fused_unary:{nm}", 0) * (before - after)
        out["verbs_deleted_by_op"] = {k: v for k, v in
                                      sorted(saved.items(), key=lambda kv: -kv[1])}
        out["verbs_deleted_total"] = sum(saved.values())
        print(json.dumps({"tape_nodes": out["tape_nodes"],
                          "verbs_deleted_total": out["verbs_deleted_total"],
                          "by_op": out["verbs_deleted_by_op"]}, indent=1), flush=True)
    out["aiclk_during"] = clk.summary()
    out["aiclk_line"] = clk.line()
    print(out["aiclk_line"], flush=True)
    dump()
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
