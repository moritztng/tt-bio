#!/usr/bin/env python3
"""gap_pt_shard: RFD3's pair Transition on the engine's shared block-sharded SwiGLU, against the shipped path.

`ws` is tt_bio.pair_swiglu_ws, the weight-stationary kernel (z read once, weights in L1).
tt_bio.tenstorrent already carries a pair SwiGLU with block-sharded intermediates (`transition_shard`) and an
optional fused fc1|fc2+silu+gate kernel (`swiglu_fc12g`), graded and shipped for ESMFold2. RFD3 runs its own
chunked L1 path (`Transition.__call__`). This times both at the R4 pair shape [1, 685, 704, 128], hidden 256 and
512 (4 calls each per step), bf16 and bfp8 hidden, and scores every arm against a float64 reference.

    python scripts/rfd3_port/gap_pt_shard.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio import tenstorrent as T                                      # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402
from tt_bio import pair_swiglu_ws as WS                                  # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_pt_shard.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
I, IP, C = 685, 704, 128
CALLS, STEPS = 4, 200


def timeit(fn, dev):
    o = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N):
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), min(ts), max(ts)


def rel(o, ref):
    d = ttnn.to_torch(o).double()[:, :I, :I]
    assert torch.isfinite(d).all(), "non-finite output"
    return float((d - ref).norm() / ref.norm())


def main():
    dev = get_device()
    torch.manual_seed(42)
    ckc = M._default_compute_kernel_config()
    x_t = torch.randn(1, I, IP, C)
    x_t[:, :, I:] = 0
    x = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xr = x_t.to(torch.bfloat16).double()[:, :, :I]
    rows_out, step = [], {}
    for hidden in (256, 512):
        g = torch.Generator().manual_seed(hidden)
        nw = 1 + 0.1 * torch.randn(C, generator=g)
        w1, w2 = (torch.randn(C, hidden, generator=g) * C ** -0.5 for _ in range(2))
        w3 = torch.randn(hidden, C, generator=g) * hidden ** -0.5
        q = lambda t: t.to(torch.bfloat16).double()                     # noqa: E731
        xn = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6) * q(nw)
        ref = (torch.nn.functional.silu(xn @ q(w1)) * (xn @ q(w2))) @ q(w3)

        def up(t, dt=ttnn.bfloat16):
            return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

        t = object.__new__(M.Transition)
        t.dtype, t.compute_kernel_config = ttnn.bfloat16, ckc
        t.norm_w, t.fc1_w, t.fc2_w, t.fc3_w = up(nw), up(w1), up(w2), up(w3)
        rows = T._transition_shard_rows(IP, C, hidden)
        wt = {}
        for name, dt in (("bf16", ttnn.bfloat16), ("b8", ttnn.bfloat8_b)):
            wt[name] = (up(w1, dt), up(w2, dt), up(w3, dt),
                        up(T._fc12_interleave(w1, w2, 8), dt))

        def shard(hid, fc12g):
            a1, a2, a3, a12 = wt["b8" if hid == ttnn.bfloat8_b else "bf16"]
            parts = []
            for s in range(0, I, rows):
                c = x[:, s:min(s + rows, I)]
                n = ttnn.rms_norm(c, weight=t.norm_w, epsilon=1e-6, compute_kernel_config=ckc)
                o = T._transition_swiglu_sharded(n, a1, a2, a3, ckc, T.silu_ckc(ckc), hid, ttnn.bfloat16,
                                                 w12=a12 if fc12g else None)
                if o is None:                                            # ragged tail: the shipped ops
                    o = t._swiglu(c, None)
                ttnn.deallocate(n)
                ttnn.deallocate(c)
                parts.append(o)
            o = ttnn.concat(parts, dim=1)
            for p in parts:
                ttnn.deallocate(p)
            return o

        ws12, ws3 = WS.weights(w1, w2, w3, dev)

        def ws():
            n = ttnn.rms_norm(x, weight=t.norm_w, epsilon=1e-6, compute_kernel_config=ckc)
            o = WS.pair_swiglu(n, ws12, ws3)
            ttnn.deallocate(n)
            return o

        arms = {
            "rfd3": lambda: t(x),
            "ws": ws,
            "shard": lambda: shard(ttnn.bfloat16, False),
            "shard+fc12g": lambda: shard(ttnn.bfloat16, True),
            "shard_b8": lambda: shard(ttnn.bfloat8_b, False),
            "shard_b8+fc12g": lambda: shard(ttnn.bfloat8_b, True),
        }
        for arm, fn in list(arms.items()) + [("rfd3_b8", lambda: t(x))]:
            M._TRANSITION_B8 = arm == "rfd3_b8"
            with T.lever_scope(["silu_f32"] if "fc12g" in arm else []):
                try:
                    o, med, lo, hi = timeit(fn, dev)
                    err = rel(o, ref)
                    ttnn.deallocate(o)
                except Exception as e:                                   # noqa: BLE001
                    print("hidden=%d %-15s EXC %s" % (hidden, arm, str(e)[:200]), flush=True)
                    rows_out.append(dict(hidden=hidden, arm=arm, error=str(e)[:400]))
                    continue
            M._TRANSITION_B8 = False
            print("hidden=%d %-15s %8.3f ms/call [%.3f, %.3f]  rel err vs f64 %.5f  rows/block %d"
                  % (hidden, arm, med, lo, hi, err, rows), flush=True)
            rows_out.append(dict(hidden=hidden, arm=arm, ms=round(med, 4), lo=round(lo, 4), hi=round(hi, 4),
                                 rel_err_f64=round(err, 6), shard_rows=rows))
            step[arm] = step.get(arm, 0.0) + CALLS * med
    print("\nper step (4 x hidden 256 + 4 x hidden 512), s/design at 200 steps:")
    for arm, ms in step.items():
        print("  %-15s %8.2f ms/step  %6.2f s/design" % (arm, ms, ms * STEPS / 1e3))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(shape=[1, I, IP, C], reps=N, rows=rows_out,
                                   ms_per_step={k: round(v, 3) for k, v in step.items()}), indent=1))


if __name__ == "__main__":
    main()
