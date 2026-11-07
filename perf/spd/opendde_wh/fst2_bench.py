"""Fused fp32 single Transition (tt_bio.fp32_transition) against the composed tenstorrent.fp32_single_transition on one
Wormhole chip, on trunk block 43's captured input (8fg6 tap.pt, rows tiled to N tokens with the captured spread, as
perf/mol1_whl3/st_bench.py): warm per-call ms, small-channel token rel err vs float64, max |new - composed|.
TT_VISIBLE_DEVICES=C python fst2_bench.py TAP.pt LNW.pt N[,N..] OUT.json"""
import json, sys, time, traceback
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import fp32_transition as F32T

tap = torch.load(sys.argv[1], weights_only=False); lnw = torch.load(sys.argv[2], weights_only=False)
x0 = tap["in"][0].double().reshape(-1, 384)
bf = lambda t: t.to(torch.bfloat16).double()
dev = T.get_device()
up = lambda t, d=ttnn.bfloat16: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=d)
rows = []
for N in map(int, sys.argv[3].split(",")):
    torch.manual_seed(0)
    x = x0[torch.arange(N) % x0.shape[0]] + (x0 - x0.mean(0)).std(0) * 0.3 * torch.randn(N, 384, dtype=torch.float64)
    lw, lb = bf(lnw["w"].double()), bf(lnw["b"].double())
    w1, w2 = (bf(torch.randn(384, 1536, dtype=torch.float64) / 384 ** .5 * 8) for _ in range(2))
    w3 = bf(torch.randn(1536, 384, dtype=torch.float64) / 1536 ** .5)
    xn = torch.nn.functional.layer_norm(bf(x), (384,), lw, lb, 1e-5)
    ref = (torch.nn.functional.silu(xn @ w1) * (xn @ w2)) @ w3
    small = ref.abs().amax(0) < ref.abs().amax(0).median()
    tok = lambda t: t - t.mean(0)
    err = lambda o: float((tok(o[:, small]) - tok(ref[:, small])).norm() / tok(ref[:, small]).norm())
    X = up(x.reshape(1, N, 384)); LW, LB = up(lw), up(lb); W1, W2, W3 = up(w1), up(w2), up(w3)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)
    row = dict(N=N)
    outs = {}
    for name, f in (("composed", lambda: T.fp32_single_transition(X, LW, LB, W1, W2, W3, ckc, ttnn.bfloat16)),
                    ("fused", lambda: F32T.transition(X, LW, LB, W1, W2, W3, ckc, ttnn.bfloat16))):
        try:
            y = f()
            outs[name] = ttnn.to_torch(y).double().reshape(N, 384)
            for _ in range(10):
                ttnn.deallocate(f())
            ttnn.synchronize_device(dev)
            ms = []
            for _ in range(5):
                t = time.perf_counter()
                for _ in range(50):
                    ttnn.deallocate(f())
                ttnn.synchronize_device(dev)
                ms.append(round((time.perf_counter() - t) / 50 * 1e3, 4))
            row[name] = dict(ms=ms, err=err(outs[name]), finite=bool(torch.isfinite(outs[name]).all()))
        except Exception as e:
            row[name] = dict(error=f"{type(e).__name__}: {e}"[:2000], tb=traceback.format_exc()[-3000:])
    if len(outs) == 2:
        d = (outs["fused"] - outs["composed"]).abs()
        row["max_abs_diff"] = float(d.max()); row["rel_rms_diff"] = float(d.norm() / outs["composed"].norm())
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(rows, open(sys.argv[4], "w"), indent=1)
