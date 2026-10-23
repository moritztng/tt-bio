"""One RFD4 trunk pass on a card: ops per pass, eager vs traced time, and its output for a bit-compare.

    TT_VISIBLE_DEVICES=<card> TT_BIO_TRACE_REGION_MB=512 python perf/rfd4/e2e/pass_bench.py \
        --weights <bundle> --out <file.json> [--dump <file.pt>] [--n 128] [--batch 1] [--reps 20]

Runs against whatever `tt_bio` is on PYTHONPATH, so the same script times the code before and
after a change. Inputs are seeded normals at the released widths with every token real: the time of
a pass does not depend on the values, and the dump compares two trees on identical inputs.

* `ops_per_pass`: ttnn calls during one warm pass, by name. `deallocate` is not a device op and is
  left out of the total.
* `eager_ms`: one pass, synced, median of `--reps`.
* `traced_ms`: the same pass captured once and replayed. The body clones s and z first, because a
  pass frees its inputs and a replay must not; the clones are in the eager time too.
* `call_*_ms`: a whole network call at depth 3 (three passes and two recycles), eager and traced.

AICLK is sampled for the life of the run (`perf/clocksample.py`).
"""
import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402
import ttnn  # noqa: E402


def count_ops(fn):
    """Run `fn()` with every ttnn operation wrapped by a counter; returns (result, Counter)."""
    counts = collections.Counter()
    patched = []
    for mod, prefix in ((ttnn, ""), (ttnn.transformer, "transformer."), (ttnn.experimental, "experimental.")):
        for name in dir(mod):
            obj = getattr(mod, name)
            if type(obj).__name__ not in ("FastOperation", "Operation") or name == "deallocate":
                continue

            def wrapped(*a, _o=obj, _n=prefix + name, **k):
                counts[_n] += 1
                return _o(*a, **k)

            setattr(mod, name, wrapped)
            patched.append((mod, name, obj))
    try:
        out = fn()
    finally:
        for mod, name, obj in patched:
            setattr(mod, name, obj)
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1, help="designs a trunk call (the sample width)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import network as N
    from tt_bio.rfd4 import trunk as TR

    dev = T.get_device(trace=None if a.no_trace else "rf3")
    state, config = N.load_checkpoint(a.weights)
    cfg = config["nn"]
    trunk = TR.Trunk(T.WeightScope({k: v.float() for k, v in N._sub(state, "nn.").items()}),
                     TR.compute_kernel_config(), nlayers=cfg["nlayers"], n_heads=cfg["nheads"],
                     every_n=cfg["update_pair_repr_every_n"], residual_dtype=ttnn.float32)
    n, b, tok, pair, dc = a.n, a.batch, cfg["token_dim"], cfg["pair_repr_dim"], cfg["dim_cond"]
    g = torch.Generator().manual_seed(0)
    s0, z0 = torch.randn(b, n, tok, generator=g), torch.randn(b, n, n, pair, generator=g)
    c0 = [torch.randn(b, n, dc, generator=g) for _ in range(3)]
    mask = torch.ones(b, n)

    up = lambda t, dt: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    masks = TR.trunk_masks(mask, dev, residual_dtype=ttnn.float32)
    s, z = up(s0, ttnn.float32), up(z0, trunk.pair_residual_dtype)
    conds = [up(c, ttnn.bfloat16) for c in c0]
    sync = lambda: ttnn.synchronize_device(dev)  # noqa: E731

    def one_pass():
        return trunk.run_pass(ttnn.clone(s), ttnn.clone(z), conds[0], masks)

    def one_call():
        return trunk(s, z, conds, masks)

    def free(ts):
        for t in ts:
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)

    def timed(fn, reps):
        ms = []
        for _ in range(reps):
            sync()
            t = time.perf_counter()
            free(fn())
            sync()
            ms.append(1e3 * (time.perf_counter() - t))
        return ms

    rec = {"n": n, "batch": b, "nlayers": cfg["nlayers"], "tt_bio": str(Path(TR.__file__).resolve().parents[2])}
    # Is a mixed fp32 + bf16 add the fp32 add of the widened bf16? If so the residual's typecast
    # can go; if the unpacker narrows the fp32 operand instead, it cannot.
    zf = up(z0, ttnn.float32)
    zb = ttnn.typecast(zf, ttnn.bfloat16)
    wide = ttnn.typecast(zb, ttnn.float32)
    ref, mixed = ttnn.add(zf, wide), ttnn.add(zf, zb, dtype=ttnn.float32)
    rec["mixed_add_exact"] = bool(torch.equal(ttnn.to_torch(ref), ttnn.to_torch(mixed)))
    rec["mixed_add_max_abs"] = float((ttnn.to_torch(ref) - ttnn.to_torch(mixed)).abs().max())
    free((zf, zb, wide, ref, mixed))
    with clocksample.during(period=0.5) as clk:
        free(one_pass())  # compile
        free(one_call())
        sync()
        out, counts = count_ops(one_pass)
        sync()
        if a.dump:
            call = one_call()
            torch.save({"s": ttnn.to_torch(out[0]), "z": ttnn.to_torch(out[1]),
                        "call_s": ttnn.to_torch(call[0]), "call_z": ttnn.to_torch(call[1])}, a.dump)
            free(call)
        free(out)
        rec["ops_per_pass"] = sum(counts.values())
        rec["ops_by_name"] = dict(counts.most_common())
        _, call_counts = count_ops(lambda: free(one_call()))
        rec["ops_per_call_depth3"] = sum(call_counts.values())
        ms = timed(one_pass, a.reps)
        rec["eager_ms"] = {"median": round(statistics.median(ms), 3), "min": round(min(ms), 3)}
        ms = timed(one_call, max(3, a.reps // 3))
        rec["call_eager_ms"] = {"median": round(statistics.median(ms), 3), "min": round(min(ms), 3)}

        if not a.no_trace:
            for key, fn, reps in (("traced_ms", one_pass, a.reps), ("call_traced_ms", one_call, max(3, a.reps // 3))):
                sync()
                t = time.perf_counter()
                tid, res = T.capture_trace(dev, fn)
                sync()
                capture_s = time.perf_counter() - t
                ms = []
                for _ in range(reps):
                    t = time.perf_counter()
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
                    ms.append(1e3 * (time.perf_counter() - t))
                if key == "traced_ms":
                    ref = one_pass()
                    rec["traced_equals_eager"] = bool(torch.equal(ttnn.to_torch(res[0]), ttnn.to_torch(ref[0]))
                                                      and torch.equal(ttnn.to_torch(res[1]), ttnn.to_torch(ref[1])))
                    free(ref)
                ttnn.release_trace(dev, tid)
                free(res)
                rec[key] = {"median": round(statistics.median(ms), 3), "min": round(min(ms), 3),
                            "capture_s": round(capture_s, 2)}
    rec["aiclk"] = clk.summary()
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
