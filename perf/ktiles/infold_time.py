#!/usr/bin/env python3
"""Time both paths on the real operands, inside a real fold.

protenix-v2 banks 187 ms/fold where its op-isolated A/B projected 1172. Either the ops do not get
as much faster on live operands as they do on a synthetic harness, or they do and something else
absorbs it. This times plain vs config per call, in the fold, device-synchronised, on the first few
calls of each class, and multiplies by the fold's call count.
"""
import argparse, json, statistics, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
SAMPLES, REPS = 3, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["protenix-v2", "opendde"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    dev = T.get_device()
    acc = defaultdict(lambda: dict(calls=0, plain=[], cfg=[]))

    def bench(fn):
        fn(); ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        return (time.perf_counter() - t0) / REPS * 1e3

    def spy(x, y, compute_kernel_config=None):
        sa, sb = tuple(int(d) for d in x.shape), tuple(int(d) for d in y.shape)
        cfg = None
        if len(sa) == 4 and len(sb) == 4 and x.dtype == y.dtype:
            cfg = T._batched_matmul_config(sa[0] * sa[1], -(-sa[2] // 32), -(-sa[3] // 32),
                                          -(-sb[3] // 32), 4 if x.dtype == ttnn.float32 else 2)
        k = f"{sa}x{sb}" + ("" if cfg is None else f" pcm={cfg.per_core_M} bw={cfg.in0_block_w}")
        e = acc[k]
        e["calls"] += 1
        if e["calls"] <= SAMPLES:
            e["plain"].append(bench(lambda: ttnn.matmul(
                x, y, compute_kernel_config=compute_kernel_config)))
            if cfg is not None:
                e["cfg"].append(bench(lambda: ttnn.matmul(
                    x, y, compute_kernel_config=compute_kernel_config, program_config=cfg)))
        return ttnn.matmul(x, y, compute_kernel_config=compute_kernel_config, program_config=cfg)

    T.batched_matmul = spy
    P.batched_matmul = spy
    import tt_baseline as B
    msa_dir = Path(tempfile.mkdtemp(prefix="ktiles-time-"))
    one_fold, _meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / "examples" / "prot300.yaml", Path(B.FIXTURES) / "prot300.a3m")
    one_fold()
    rows, total = [], 0.0
    for k, e in sorted(acc.items(), key=lambda kv: -kv[1]["calls"]):
        pl = statistics.median(e["plain"])
        cf = statistics.median(e["cfg"]) if e["cfg"] else None
        saved = (pl - cf) * e["calls"] if cf else 0.0
        total += saved
        print(f"  {k}\n     calls={e['calls']:5d}  plain {pl:8.4f} ms  " +
              (f"cfg {cf:8.4f} ms  {pl / cf:5.2f}x  in-fold saving {saved:8.1f} ms/fold"
               if cf else "declined"), flush=True)
        rows.append(dict(key=k, calls=e["calls"], plain_ms=pl, cfg_ms=cf,
                         ratio=None if not cf else pl / cf, ms_per_fold_saved=saved))
    print(f"\n  in-fold per-call projection: {total:.0f} ms/fold")
    a.out.write_text(json.dumps({"model": a.model, "rows": rows,
                                 "in_fold_projection_ms": total}, indent=1))
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()
