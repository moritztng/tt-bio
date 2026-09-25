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
                                          host_losses, trunk_forward)

SEED = 20260921

# The nine J4 ops, and the verbs their backward closure issues per node BEFORE and AFTER
# this row. "both" is the two-parents-want-a-gradient branch. Counted from the closure
# source, which is why the census below also reports the branch each node would take.
VERBS = {
    "mul":     {"both": (2, 1), "one": (1, 1)},
    "scale":   {"one": (1, 1)},
    "add":     {"both": (0, 0), "one": (0, 0)},
    "relu":    {"one": (2, 1)},
    "sigmoid": {"one": (3, 1)},
    "silu":    {"one": (6, 1)},
    "reshape": {"one": (1, 1)},
    "narrow":  {"one": (3, 1)},       # two zero blocks + concat -> one pad
    "concat":  {"both": (2, 1), "one": (None, None)},   # N-way falls through to the slices
}


def classify(t):
    """`_reverse_topo` hands back TENSORS, so the op is on `t.node.fn`; a leaf has no node."""
    n = t.node
    if n is None:
        return "<leaf>"
    q = getattr(n.fn, "__qualname__", "") or ""
    return q.split(".")[0] if q else "<unknown>"


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
        from tt_bio.train.losses import of3_loss_weights

        held, _meta = S.capture(a.tokens, out)
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
        host_losses(roots, 0, of3_loss_weights(a.stage), rng, {})

        nodes = ag._reverse_topo(list(roots))
        out["tape_nodes"] = len(nodes)
        per = collections.Counter()
        branch = collections.Counter()
        shapes = collections.defaultdict(collections.Counter)
        for t in nodes:
            op = classify(t)
            per[op] += 1
            if op in VERBS:
                parents = t.node.parents
                k = "both" if sum(1 for p in parents if p.requires_grad) >= 2 else "one"
                branch[f"{op}|{k}|{len(parents)}p"] += 1
                try:
                    shapes[op][str([int(d) for d in t.value.shape])] += 1
                except Exception:
                    shapes[op]["<freed>"] += 1
        out["nodes_by_op"] = dict(per.most_common())
        out["j4_branch"] = dict(sorted(branch.items()))
        out["j4_parent_shapes"] = {k: dict(v.most_common(8)) for k, v in shapes.items()}

        saved = {}
        for op, spec in VERBS.items():
            tot = 0
            for key, (before, after) in spec.items():
                if before is None:
                    continue
                n = sum(v for k, v in branch.items()
                        if k.startswith(f"{op}|{key}|"))
                tot += n * (before - after)
            saved[op] = tot
        out["verbs_deleted_by_op"] = saved
        out["verbs_deleted_total"] = sum(saved.values())
        print(json.dumps({"tape_nodes": out["tape_nodes"],
                          "verbs_deleted_total": out["verbs_deleted_total"],
                          "by_op": saved}, indent=1), flush=True)
    out["aiclk_during"] = clk.summary()
    out["aiclk_line"] = clk.line()
    print(out["aiclk_line"], flush=True)
    dump()
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
