"""Where a warm embed batch spends its wall time: host enqueue of the forward, device tail, readback, the rest.

One embed_sequences call on the perf-page protocol (prot512.seq x N), split by wrapping the model's device module
(time to return = host enqueue; then synchronize = device work still queued) and its _to_torch (readback: device
untilize, host copy, fp32 cast). The ESMC-6B module reads its result back itself, so there the enqueue clock stops at
its first _to_host, and that readback is one "readback" lap. "rest" is tokenising, batching, uploads and pooling. If enqueue ~ total device time the forward is host-bound and a trace pays.

    python3 perf/spd_gap_whlm/phase_prof.py --model saprot-35m --n-seqs 7 --out phase.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import _device_tilize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-seqs", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=7)
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
    dev = model.tt_device
    embed = lambda: mod.embed_sequences(model, seqs, batch_size=8)
    for _ in range(2):
        embed()
    ttnn.synchronize_device(dev)

    t = {}
    module, to_torch = model.module, model._to_torch

    inner_rb = isinstance(model, mod.ESMCLanguageModel) if hasattr(mod, "ESMCLanguageModel") else False
    to_host = module._to_host if inner_rb else None

    def timed_module(*a, **k):
        t0 = time.perf_counter()
        if inner_rb:
            def timed_to_host(x):
                t1 = time.perf_counter()
                ttnn.synchronize_device(dev)
                t2 = time.perf_counter()
                r = to_host(x)
                t["enqueue"] = t.get("enqueue", 0.0) + t1 - t0
                t["device_tail"] = t.get("device_tail", 0.0) + t2 - t1
                t["readback"] = t.get("readback", 0.0) + time.perf_counter() - t2
                return r
            module._to_host = timed_to_host
            try:
                return module(*a, **k)
            finally:
                module._to_host = to_host
        r = module(*a, **k)
        t1 = time.perf_counter()
        ttnn.synchronize_device(dev)
        t["enqueue"] = t.get("enqueue", 0.0) + t1 - t0
        t["device_tail"] = t.get("device_tail", 0.0) + time.perf_counter() - t1
        return r

    def lap(key, t0):
        t1 = time.perf_counter()
        t[key] = t.get(key, 0.0) + t1 - t0
        return t1

    def timed_to_torch(x):   # tenstorrent.download + the wrapper's float cast, one lap per step
        t0 = time.perf_counter()
        t["readback_mb"] = t.get("readback_mb", 0.0) + x.volume() * (4 if x.dtype == ttnn.float32 else 2) / 1e9   # MB once the row scales by 1000
        if x.layout == ttnn.TILE_LAYOUT and _device_tilize(x.volume() * 2):
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.synchronize_device(dev)
            t0 = lap("rb_untilize", t0)
        h = ttnn.to_torch(x)
        t0 = lap("rb_to_torch", t0)
        r = torch.Tensor(h).to(torch.float32)
        lap("rb_cast", t0)
        return r

    plain, split = [], []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        embed()
        ttnn.synchronize_device(dev)
        plain.append((time.perf_counter() - t0) * 1000)
    model.module, model._to_torch = timed_module, timed_to_torch
    for _ in range(args.rounds):
        t.clear()
        t0 = time.perf_counter()
        embed()
        ttnn.synchronize_device(dev)
        total = time.perf_counter() - t0
        row = {k: v * 1000 for k, v in t.items()}
        row["total"] = total * 1000
        row["rest"] = row["total"] - sum(v for k, v in row.items() if k not in ("total", "readback_mb"))
        split.append(row)
    model.module, model._to_torch = module, to_torch

    med = {k: round(statistics.median(r[k] for r in split), 2) for k in split[0]}
    out = dict(model=args.model, fast=args.fast, n_seqs=args.n_seqs, plain_ms=round(statistics.median(plain), 2),
               plain_all=[round(x, 2) for x in plain], split_median_ms=med)
    Path(args.out).write_text(json.dumps(out) + "\n")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
