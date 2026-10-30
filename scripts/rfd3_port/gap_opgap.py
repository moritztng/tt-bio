#!/usr/bin/env python3
"""gap_opgap: what one more op costs the card beyond its kernel, at the DiT's shapes.

Chains of N back-to-back calls of one op (rms_norm, multiply, the pinned 768->1024 linear, and a mixed
rms_norm -> multiply -> linear chain that switches program every call) timed two ways: eager (host enqueue,
then one sync) and as a replayed trace (no host). ms/op from the trace is kernel + device dispatch; set it
beside the kernel time Tracy reports for the same op in the census to read the per-op device overhead.
A single-op N ladder (N = 8, 32, 128) separates a per-chain constant from the per-op slope.
    python scripts/rfd3_port/gap_opgap.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio import tenstorrent as T                                         # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402

OUT = pathlib.Path(sys.argv[1])
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 9
g = torch.Generator().manual_seed(0)


def tt(shape, dev):
    return ttnn.from_torch(torch.randn(*shape, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def wall(fn, dev):
    fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    dev = T.get_device(trace="rf3")
    ckc = M._default_compute_kernel_config()
    x768, x1024, w = tt((1, 704, 768), dev), tt((1, 704, 1024), dev), tt((768, 1024), dev)
    ops = {
        "rms_norm": lambda: ttnn.rms_norm(x1024, epsilon=1e-6, compute_kernel_config=ckc),
        "multiply": lambda: ttnn.multiply(x1024, x1024),
        "linear": lambda: M._tuned_linear(x768, w, ckc=ckc, dtype=ttnn.bfloat16, core_grid=M.BATCH_INVARIANT_GRID),
    }
    rec = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch()),
           "reps": N_REP, "mm2d": M._mm2d_on(), "ops": {}}
    for name in ("rms_norm", "multiply", "linear", "mixed"):
        r = rec["ops"][name] = {}
        for n in (8, 32, 128):
            if name == "mixed":
                seq = [ops[k] for k in ("rms_norm", "multiply", "linear")] * (n // 3 + 1)
                seq = seq[:n]
            else:
                seq = [ops[name]] * n

            def chain():
                for f in seq:
                    ttnn.deallocate(f())

            eager = wall(chain, dev)
            tid, _ = T.capture_trace(dev, chain)
            replay = wall(lambda: ttnn.execute_trace(dev, tid, cq_id=0, blocking=False), dev)
            ttnn.release_trace(dev, tid)
            r[n] = {"eager_ms": eager, "trace_ms": replay, "eager_us_op": 1e3 * eager / n,
                    "trace_us_op": 1e3 * replay / n}
            print(f"[opgap] {name:9s} N={n:4d} eager {1e3 * eager / n:7.1f} us/op  trace {1e3 * replay / n:7.1f} us/op",
                  flush=True)
        n0, n1 = 32, 128
        r["trace_slope_us"] = 1e3 * (r[n1]["trace_ms"] - r[n0]["trace_ms"]) / (n1 - n0)
        r["eager_slope_us"] = 1e3 * (r[n1]["eager_ms"] - r[n0]["eager_ms"]) / (n1 - n0)
        print(f"[opgap] {name:9s} slope eager {r['eager_slope_us']:.1f} trace {r['trace_slope_us']:.1f} us/op", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, default=str))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
