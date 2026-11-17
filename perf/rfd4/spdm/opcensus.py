#!/usr/bin/env python3
"""Measured per-op census of one RFD4 trunk call, against the analytic byte model.

`state/rfd4/census/rfd4_census.py` prices the trunk op by op and says 59 % of the DRAM traffic at 336 tokens is
pair norm + elementwise carrying 0.5 % of the FLOPs. That model is UNFUSED, and main already ships several pair
fusion levers, so it cannot say how much of that traffic a shipped tree still moves. This does: every ttnn call in
one depth-3 trunk call is timed between two device syncs and charged the bytes of its tensor arguments and its
result, then rolled up by op name and by the census families.

    TT_VISIBLE_DEVICES=<chip> PYTHONPATH=<tree> python perf/rfd4/spdm/opcensus.py \\
        --weights <rfd4 bundle> --n 352 --real 336 --out <dir>/opcensus.json

Timing between syncs charges each op its own dispatch latency, so the total overshoots the untimed call; the
SHARES are the reading, and the untimed call time is recorded beside them as `call_ms_untimed` so the overshoot is
visible. Levers come from TT_BIO_LEVERS or the mode, as everywhere else.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ttnn  # noqa: E402

#: op name -> rfd4_census.py family. A name not here lands in `other` and is listed, never silently dropped.
FAMILY = {
    "linear": "matmul", "matmul": "matmul", "experimental.minimal_matmul": "matmul",
    "rms_norm": "norm", "layer_norm": "norm",
    "multiply": "elementwise", "multiply_": "elementwise", "add": "elementwise", "add_": "elementwise",
    "subtract": "elementwise", "silu": "elementwise", "sigmoid": "elementwise", "typecast": "elementwise",
    "concat": "data_movement", "slice": "data_movement", "permute": "data_movement",
    "reshape": "data_movement", "squeeze": "data_movement", "unsqueeze": "data_movement",
    "clone": "data_movement", "to_memory_config": "data_movement", "to_layout": "data_movement",
    "allocate_tensor_on_device": "alloc", "deallocate": "alloc",
    "transformer.scaled_dot_product_attention": "attention",
    "experimental.nlp_create_qkv_heads": "attention", "experimental.nlp_concat_heads": "attention",
    "generic_op": "generic_op",
}


def tensor_bytes(x) -> int:
    if not isinstance(x, ttnn.Tensor):
        return 0
    try:
        n = 1
        for d in x.shape:
            n *= int(d)
        return n * x.element_size()
    except Exception:
        return 0


def walk_bytes(obj, seen=None) -> int:
    if isinstance(obj, ttnn.Tensor):
        return tensor_bytes(obj)
    if isinstance(obj, (list, tuple)):
        return sum(walk_bytes(o) for o in obj)
    return 0


def caller() -> str:
    """The tt-bio function that dispatched a `generic_op`: every fused kernel goes through it, so the op name alone
    lumps the trunk's whole custom-kernel time into one row (oc1: 278 calls, 74 % of the call)."""
    f = sys._getframe(2)
    while f and (f.f_code.co_filename == __file__ or "/ttnn/" in f.f_code.co_filename):
        f = f.f_back
    names = []
    while f and len(names) < 2:   # the dispatching helper and its caller: a shared runner names nothing alone
        names.append(f"{f.f_code.co_name}@{Path(f.f_code.co_filename).stem}")
        f = f.f_back
    return "<".join(names) or "?"


class Census:
    """Wraps the ttnn callables the trunk uses and records only the outermost call."""

    def __init__(self, device):
        self.device, self.depth, self.active = device, 0, False
        self.rows = collections.defaultdict(lambda: {"n": 0, "ms": 0.0, "bytes": 0})
        self.orig = []

    def _wrap(self, holder, name, full):
        fn = getattr(holder, name)
        if not callable(fn):
            return
        census = self

        def w(*args, **kwargs):
            if not census.active or census.depth:
                return fn(*args, **kwargs)
            census.depth += 1
            try:
                ttnn.synchronize_device(census.device)
                t = time.perf_counter()
                out = fn(*args, **kwargs)
                ttnn.synchronize_device(census.device)
                dt = 1e3 * (time.perf_counter() - t)
                r = census.rows[full if full != "generic_op" else f"generic_op:{caller()}"]
                r["n"] += 1
                r["ms"] += dt
                r["bytes"] += walk_bytes(args) + walk_bytes(list(kwargs.values())) + walk_bytes(out)
                return out
            finally:
                census.depth -= 1

        setattr(holder, name, w)
        self.orig.append((holder, name, fn))

    def install(self, names):
        for full in names:
            holder, _, leaf = (ttnn, None, full) if "." not in full else \
                (getattr(ttnn, full.split(".")[0], None), None, full.split(".", 1)[1])
            if holder is not None and hasattr(holder, leaf):
                self._wrap(holder, leaf, full)

    def remove(self):
        for holder, name, fn in self.orig:
            setattr(holder, name, fn)
        self.orig.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=352)
    ap.add_argument("--real", type=int, default=336)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--depth", type=int, default=3, help="recycles in the call (thinking_medium runs 1..6)")
    a = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import network as N
    from tt_bio.rfd4 import trunk as TR

    dev = T.get_device(trace=None)
    state, config = N.load_checkpoint(a.weights)
    cfg = config["nn"]
    trunk = TR.Trunk(T.WeightScope({k: v.float() for k, v in N._sub(state, "nn.").items()}),
                     TR.compute_kernel_config(), nlayers=cfg["nlayers"], n_heads=cfg["nheads"],
                     every_n=cfg["update_pair_repr_every_n"], residual_dtype=ttnn.float32)
    n, b = a.n, a.batch
    tok, pair, dc = cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(0)
    s0, z0 = torch.randn(b, n, tok, generator=g), torch.randn(b, n, n, pair, generator=g)
    c0 = [torch.randn(b, n, dc, generator=g) for _ in range(a.depth)]
    mask = torch.ones(b, n)
    mask[:, a.real:] = 0

    def up(t, dt):
        return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    masks = TR.trunk_masks(mask, dev, residual_dtype=ttnn.float32)
    s, z = up(s0, ttnn.float32), up(z0, trunk.pair_dtype(masks))
    conds = [up(c, ttnn.bfloat16) for c in c0]

    def call():
        return trunk(s, z, conds, masks)

    # warm, then the untimed call, then the same call with every op timed
    for _ in range(2):
        call()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    call()
    ttnn.synchronize_device(dev)
    untimed_ms = 1e3 * (time.perf_counter() - t)

    census = Census(dev)
    census.install(sorted(FAMILY))
    census.active = True
    call()
    census.active = False
    census.remove()

    rows = {k: {**v, "ms": round(v["ms"], 3), "gb": round(v["bytes"] / 1e9, 3),
                "family": FAMILY.get(k.split(":")[0], "other")} for k, v in census.rows.items()}
    tot_ms = sum(r["ms"] for r in rows.values())
    fam = collections.defaultdict(lambda: {"n": 0, "ms": 0.0, "gb": 0.0})
    for k, r in rows.items():
        f = fam[r["family"]]
        f["n"] += r["n"]
        f["ms"] += r["ms"]
        f["gb"] += r["gb"]
    out = {"n": n, "real": a.real, "batch": b, "depth": a.depth, "nlayers": cfg["nlayers"],
           "levers": sorted(getattr(trunk, "_levers", ())),
           "call_ms_untimed": round(untimed_ms, 3), "timed_total_ms": round(tot_ms, 3),
           "ops": dict(sorted(rows.items(), key=lambda kv: -kv[1]["ms"])),
           "by_family": {k: {"n": v["n"], "ms": round(v["ms"], 3), "gb": round(v["gb"], 3),
                             "share_pct": round(100 * v["ms"] / tot_ms, 1) if tot_ms else 0.0}
                         for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["ms"])}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["by_family"], indent=1))
    print(f"untimed call {untimed_ms:.1f} ms, timed total {tot_ms:.1f} ms "
          f"(sync-per-op overshoot {tot_ms / untimed_ms:.2f}x)")


if __name__ == "__main__":
    main()
