#!/usr/bin/env python3
"""gap_pws_rms: RFD3's token pair Transition with its RMSNorm as pair_swiglu_ws's prologue, against shipped.

  shipped  ttnn.rms_norm -> pair_swiglu_ws (what RFD3_PAIR_WS runs)
  fused    pair_swiglu_ws(ln=(gamma, None, eps)), the norm inside the program (what --fast ships)
  bare     pair_swiglu_ws alone, no norm; *_shift: the output at a new address every call, as in a fold
Both against a float64 reference of rmsnorm -> fc3(silu(fc1) * fc2) on the same bf16 inputs, at RFD3's fidelity,
for hidden 256 and 512. Synced ms per call, median of `reps`.

    python scripts/rfd3_port/gap_pws_rms.py OUT.json [reps] [I]
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                      # noqa: E402

ensure_p300_mesh_descriptor()
from tt_bio import pair_swiglu_ws as WS                                  # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_pws_rms.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
I = int(sys.argv[3]) if len(sys.argv) > 3 else 685
C, EPS = 128, 1e-6


def timeit(fn, dev, shift=False):
    """Synced ms per call. `shift` parks a one-tile tensor before every call so the output lands at a new
    address each time, as it does inside a fold; a program cache keyed on buffer addresses then misses."""
    o = fn()
    ttnn.synchronize_device(dev)
    ts, park, host = [], [], []
    for _ in range(N):
        ttnn.deallocate(o)
        if shift:
            park.append(ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))
        t0 = time.perf_counter()
        o = fn()
        host.append((time.perf_counter() - t0) * 1e3)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    for t in park:
        ttnn.deallocate(t)
    timeit.host_ms = statistics.median(host)
    return o, statistics.median(ts), min(ts), max(ts)


def err(o, ref):
    d = ttnn.to_torch(o).double()[..., :C]
    assert torch.isfinite(d).all(), "non-finite device output"
    return {"max_abs_f64": (d - ref).abs().max().item(),
            "rel_err_f64": ((d - ref).norm() / ref.norm()).item()}


def main():
    dev = get_device()
    torch.manual_seed(42)
    fid = M._default_compute_kernel_config().math_fidelity
    ckc = M._default_compute_kernel_config()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    # Channel scales spread over ~2 decades, as a trained pair tensor's are, so the norm is not a no-op.
    x_t = (torch.randn(1, I, I, C) * torch.logspace(-1, 1, C)).bfloat16()
    g_t = (1 + 0.3 * torch.randn(C)).bfloat16()
    x, g = up(x_t), up(g_t.reshape(1, C))
    xd = x_t.double()
    xn = xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + EPS) * g_t.double()
    rec = {"I": I, "C": C, "reps": N, "fidelity": str(fid), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "host": os.uname().nodename, "arch": str(dev.arch()), "hidden": {}}
    for hid in (256, 512):
        w1, w2, w3 = ((torch.randn(a, b) / a ** 0.5).bfloat16() for a, b in ((C, hid), (C, hid), (hid, C)))
        ref = (torch.nn.functional.silu(xn @ w1.double()) * (xn @ w2.double())) @ w3.double()
        w12, w3d = up(torch.cat([w1, w2], 1)), up(w3)
        r = {"eligible_fused": WS.eligible(x, w12, w3d, ln=(g, None))}

        def shipped():
            n = ttnn.rms_norm(x, weight=g, epsilon=EPS, compute_kernel_config=ckc)
            o = WS.pair_swiglu(n, w12, w3d, fid)
            ttnn.deallocate(n)
            return o
        fused = lambda: WS.pair_swiglu(x, w12, w3d, fid, ln=(g, None, EPS))  # noqa: E731
        bare = lambda: WS.pair_swiglu(x, w12, w3d, fid)                          # noqa: E731
        for name, fn, shift in (("shipped", shipped, False), ("fused", fused, False),
                                ("bare", bare, False), ("bare_shift", bare, True), ("fused_shift", fused, True)):
            try:
                o, med, lo, hi = timeit(fn, dev, shift)
                r[name] = {"ms_median": med, "ms_min": lo, "ms_max": hi, "host_ms": timeit.host_ms,
                           **({} if name.startswith("bare") else err(o, ref))}
                ttnn.deallocate(o)
            except Exception as e:                                       # noqa: BLE001
                r[name] = {"error": repr(e)[:400]}
        rec["hidden"][hid] = r
        print(hid, json.dumps(r), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
