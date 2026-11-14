"""Where a slow ESMC page batch spends its extra 40-95 ms.

The gapboard's ESMC p150a cell (batch of 7 x 512 aa, perf/wh-embed/embed_split_screen.py) is flat ~75 ms in some
processes and bimodal 75 / 115-170 ms in others, at a flat 1350 MHz AICLK. Batched trace replay keeps the outliers
(grid etb), so per-op host dispatch is not the cause. This times each batch by phase: uploads (ttnn.from_torch),
the forward call, downloads (ttnn.to_torch), Python GC pauses, and the rest of embed_sequences. With --writes the
same npz writes the gapboard harness does run between batches, so the two processes see the same host churn.

    python3 perf/spd_gap_esmf/esmc_jitter.py --model esmc-300m [--fast] [--writes] --repeat 30 --out j.json
"""
import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import ttnn

from tt_bio import esmc as E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--writes", action="store_true", help="write the npz outputs between batches, as the gapboard does")
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    seq = Path(a.seq_file).read_text().strip()
    seqs = {f"seq{i}": seq for i in range(7)}
    model = E.load_esmc(a.model, fast=a.fast)

    acc = {}

    def timed(name, fn):
        def w(*x, **k):
            t0 = time.perf_counter()
            try:
                return fn(*x, **k)
            finally:
                acc[name] = acc.get(name, 0.0) + time.perf_counter() - t0
        return w

    ttnn.from_torch = timed("upload", ttnn.from_torch)
    ttnn.to_torch = timed("download", ttnn.to_torch)
    cls = type(model)
    cls.__call__ = timed("forward", cls.__call__)
    gc_t0 = [0.0]

    def on_gc(phase, info):
        if phase == "start":
            gc_t0[0] = time.perf_counter()
        else:
            acc["gc"] = acc.get("gc", 0.0) + time.perf_counter() - gc_t0[0]
            acc["gc_gen2"] = acc.get("gc_gen2", 0) + (info.get("generation") == 2)
    gc.callbacks.append(on_gc)

    out_dir = Path(tempfile.mkdtemp(prefix="esmc-jitter-"))
    rows = []
    for i in range(a.warmup + a.repeat):
        acc.clear()
        t0 = time.perf_counter()
        res = E.embed_sequences(model, seqs, batch_size=8)
        total = time.perf_counter() - t0
        row = {k: round(v * 1000, 2) if isinstance(v, float) else v for k, v in acc.items()}
        # forward includes the downloads it issues itself; "other" is embed_sequences outside the forward
        row.update(i=i, total_ms=round(total * 1000, 2), load1=float(open("/proc/loadavg").read().split()[0]),
                   other_ms=round((total - acc.get("forward", 0.0) - acc.get("upload", 0.0)) * 1000, 2))
        if a.writes:
            t1 = time.perf_counter()
            for e in res:
                E.write_npz(e, out_dir / f"{e.id}.npz")
            E.write_npz_many(res, out_dir)
            row["writes_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        print(json.dumps(row), flush=True)
        if i >= a.warmup:
            rows.append(row)
    tot = sorted(r["total_ms"] for r in rows)
    med = tot[len(tot) // 2]
    slow = [r for r in rows if r["total_ms"] > 1.25 * med]
    phases = sorted({k for r in rows for k in r if k.endswith(("upload", "download", "forward", "gc")) or k == "other_ms"})
    mean = lambda rs, k: round(sum(r.get(k, 0) for r in rs) / max(1, len(rs)), 2)  # noqa: E731
    fast_rows = [r for r in rows if r not in slow]
    summary = dict(model=a.model, fast=a.fast, writes=a.writes, n=len(rows), median_ms=med, n_slow=len(slow),
                   slow_mean={k: mean(slow, k) for k in phases + ["total_ms"]},
                   quiet_mean={k: mean(fast_rows, k) for k in phases + ["total_ms"]},
                   pid=os.getpid(), rows=rows)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
