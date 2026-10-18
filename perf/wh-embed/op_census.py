"""Per-op device time of one warm eager embed forward, on the perf-page protocol (prot512.seq x N, one batch).

Every ttnn op the ESMC/SaProt module reaches is wrapped: synchronize, call, synchronize, so each row is that op's
dispatch + device time with nothing overlapped. Rows are keyed by op name and first-input shape and summed over the
layers. The sum is >= the traced forward (dispatch is not hidden here); the split between op kinds is the point.

    python3 perf/wh-embed/op_census.py --model esmc-600m --n-seqs 7 --out census_ops.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import ttnn

OPS = ["layer_norm", "chunk", "concat", "multiply", "add", "silu", "linear", "matmul", "typecast", "reshape",
       "to_layout", "permute", "softmax", "slice", "unsqueeze", "squeeze", "rms_norm", "gelu", "pad", "where", "generic_op", "repeat"]
EXP = ["nlp_create_qkv_heads", "nlp_concat_heads", "rotary_embedding", "minimal_matmul", "view",
       "dit_minimal_matmul_addcmul_fused"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seqs", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = Path(args.seq_file).read_text().strip()
    if args.model.startswith("saprot"):
        from tt_bio import saprot as mod
        model = mod.load_saprot(args.model, fast=args.fast)
        seqs = {f"s{i}": (seq, "#" * len(seq)) for i in range(args.n_seqs)}
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(args.model, fast=args.fast, trace=False)
        seqs = {f"s{i}": seq for i in range(args.n_seqs)}
    embed = lambda: mod.embed_sequences(model, seqs, batch_size=args.batch_size)
    dev = model.tt_device
    from tt_bio import tenstorrent as T

    for _ in range(2):
        embed()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    embed()
    ttnn.synchronize_device(dev)
    eager_ms = (time.perf_counter() - t0) * 1000

    acc, depth = {}, [0]

    def wrap(name, fn):
        def w(*a, **k):
            if depth[0]:
                return fn(*a, **k)
            depth[0] += 1
            try:
                ttnn.synchronize_device(dev)
                t = time.perf_counter()
                r = fn(*a, **k)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t
            finally:
                depth[0] -= 1
            x = a[0] if a else next(iter(k.values()), None)
            if name == "generic_op":  # hand-written programs (qk_rope_heads, shsdpa, ...): key by the launching module
                x = x[0] if isinstance(x, (list, tuple)) and x else x
                name_k = "generic_op:" + sys._getframe(1).f_globals.get("__name__", "?").rsplit(".", 1)[-1]
            else:
                name_k = name
            shp = tuple(x.shape) if hasattr(x, "shape") else None
            w2 = a[1] if len(a) > 1 and hasattr(a[1], "shape") else k.get("weight_tensor")
            key = f"{name_k} {shp}" + (f" x {tuple(w2.shape)}" if hasattr(w2, "shape") and name in
                                     ("linear", "matmul", "minimal_matmul", "dit_minimal_matmul_addcmul_fused") else "")
            n, s = acc.get(key, (0, 0.0))
            acc[key] = (n + 1, s + dt)
            return r
        return w

    for o in OPS:
        if hasattr(ttnn, o):
            setattr(ttnn, o, wrap(o, getattr(ttnn, o)))
    for o in EXP:
        if hasattr(ttnn.experimental, o):
            setattr(ttnn.experimental, o, wrap(o, getattr(ttnn.experimental, o)))
    T.ops.linear = wrap("linear", T.ops.linear)
    for m in (mod, T):
        if hasattr(m, "fused_sdpa"):
            setattr(m, "fused_sdpa", wrap("sdpa", getattr(m, "fused_sdpa")))

    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    embed()
    ttnn.synchronize_device(dev)
    wrapped_ms = (time.perf_counter() - t0) * 1000

    rows = sorted(({"op": k, "n": n, "ms": round(s * 1000, 3)} for k, (n, s) in acc.items()),
                  key=lambda r: -r["ms"])
    kinds = {}
    for r in rows:
        kinds[r["op"].split()[0]] = round(kinds.get(r["op"].split()[0], 0) + r["ms"], 3)
    out = dict(model=args.model, fast=args.fast, n_seqs=args.n_seqs, eager_ms=round(eager_ms, 2),
               wrapped_ms=round(wrapped_ms, 2), op_sum_ms=round(sum(r["ms"] for r in rows), 2),
               by_kind=dict(sorted(kinds.items(), key=lambda kv: -kv[1])), rows=rows)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows[:30]:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
