"""Where one warm embed call spends its time, on the perf-page protocol (prot512.seq x N, one batch).

Splits embed_sequences into: uploads (ttnn.from_torch to device), the device module call (dispatch time
= when Python returns, device time = after synchronize), readbacks (ttnn.to_torch) and host remainder
(tokenise, masks, numpy, pooling). Synchronizing around the module call costs the overlap of dispatch
with readback, so the total here is >= the timed harness number; the split is what this tool is for.

    python3 perf/wh-embed/embed_census.py --model saprot-35m --n-seqs 7 --out census.json
"""
import argparse
import json
import time
from pathlib import Path

import ttnn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seqs", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = Path(args.seq_file).read_text().strip()
    if args.model.startswith("saprot"):
        from tt_bio import saprot as mod
        model = mod.load_saprot(args.model, fast=args.fast)
        seqs = {f"s{i}": (seq, "#" * len(seq)) for i in range(args.n_seqs)}
        embed = lambda: mod.embed_sequences(model, seqs, batch_size=args.batch_size)
    else:
        from tt_bio import esmc as mod
        model = mod.load_esmc(args.model, fast=args.fast)
        seqs = {f"s{i}": seq for i in range(args.n_seqs)}
        embed = lambda: mod.embed_sequences(model, seqs, batch_size=args.batch_size)

    dev = model.tt_device
    acc = {}

    def add(k, dt):
        acc[k] = acc.get(k, 0.0) + dt

    def timed(name, fn, sync=False):
        def w(*a, **k):
            if sync:
                ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = fn(*a, **k)
            if sync:
                add(name + "_dispatch", time.perf_counter() - t0)
                ttnn.synchronize_device(dev)
            add(name, time.perf_counter() - t0)
            return r
        return w

    orig_from, orig_to = ttnn.from_torch, ttnn.to_torch

    def from_torch(*a, **k):
        name = "upload" if k.get("device") is not None else "host_tilize"
        return timed(name, orig_from)(*a, **k)

    ttnn.from_torch = from_torch
    ttnn.to_torch = timed("readback", orig_to, sync=False)
    mcls = type(model.module)
    orig_call = mcls.__call__
    mcls.__call__ = timed("module", orig_call, sync=True)

    rows = []
    for i in range(2 + args.repeat):
        acc.clear()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        embed()
        total = time.perf_counter() - t0
        if i >= 2:
            r = {k: round(v * 1000, 2) for k, v in acc.items()}
            r["total"] = round(total * 1000, 2)
            r["host_other"] = round((total - sum(v for k, v in acc.items() if not k.endswith("_dispatch"))) * 1000, 2)
            rows.append(r)
            print(r, flush=True)
    med = {k: sorted(r[k] for r in rows)[len(rows) // 2] for k in rows[0]}
    Path(args.out).write_text(json.dumps(dict(model=args.model, n_seqs=args.n_seqs, fast=args.fast,
                                              median_ms=med, rows=rows), indent=1))
    print("median", med)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
