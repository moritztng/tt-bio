"""Bisect transition_ws's c_z 256 bf16 fault (9ce886ab7): sparse errors of exactly 1, 2 and 4 on O(1) outputs.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ws_bisect.py

Outputs here are O(1) (weights scaled 1/sqrt(fan_in)), so a power-of-two error is a kernel defect, not bf16
rounding. Each case is one hidden chunk against its own float64 partial, varying one thing at a time from the
c_z 128 bf16 path that is clean: channels (kt / ct 4 -> 8), hidden tiles per chunk, weight dtype, fidelity, silu.
Each case runs twice, so a race reads as two different outputs. Per case: max_abs, count over 0.1, how many bad
elements sit in each output tile column and in each 32-row tile modulo the core count, and the worst element's
got / want.
"""
import json

import torch

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

N = 512
dev = T.get_device()
CORES = dev.compute_with_storage_grid_size().x * dev.compute_with_storage_grid_size().y


def case(tag, c, h, wdt, fid, silu=2, seed=0, defines=()):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = torch.randn(h, c, generator=g) / c ** 0.5, torch.randn(h, c, generator=g) / c ** 0.5
    w3 = torch.randn(c, h, generator=g) / h ** 0.5
    x = torch.nn.functional.layer_norm(torch.randn(1, N, N, c, generator=g), (c,)).to(torch.bfloat16)
    xd = x.double()
    if wdt == ttnn.bfloat8_b:
        # the reference sees the weights the kernel sees
        q = lambda t: ttnn.to_torch(ttnn.from_torch(t, dtype=wdt, layout=ttnn.TILE_LAYOUT)).double()  # noqa: E731
        w1d, w2d, w3d = q(w1.t().contiguous()).t(), q(w2.t().contiguous()).t(), q(w3.t().contiguous()).t()
    else:
        w1d, w2d, w3d = w1.to(torch.bfloat16).double(), w2.to(torch.bfloat16).double(), w3.to(torch.bfloat16).double()
    ref = (torch.nn.functional.silu(xd @ w1d.t()) * (xd @ w2d.t())) @ w3d.t()
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=wdt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    w12, w3t = up(torch.cat([w1.t(), w2.t()], 1)), up(w3.t())
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    if not W.eligible(xt, w12, w3t):
        print(json.dumps(dict(tag=tag, eligible=False, l1=W._l1_bytes(c // 32, h // 32, c // 32, wdt))), flush=True)
        return
    outs = [ttnn.to_torch(W.pair_swiglu(xt, w12, w3t, fid, defines=defines, silu_poly=silu)).double() for _ in range(2)]
    d = (outs[0] - ref).abs()
    bad = (d > 0.1).reshape(N * N // 32, 32, c // 32, 32)
    i = int(d.argmax())
    print(json.dumps(dict(
        tag=tag, c=c, h=h, wdt=str(wdt), fid=str(fid), silu=silu, l1=W._l1_bytes(c // 32, h // 32, c // 32, wdt),
        max_abs=round(float(d.max()), 4), rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
        n_bad=int(bad.sum()), repeat_identical=bool(torch.equal(outs[0], outs[1])),
        repeat_diff=int((outs[0] != outs[1]).sum()),
        bad_by_out_tile=[int(v) for v in bad.sum((0, 1, 3))],
        bad_rowtiles_mod_core=sorted({int(v) % CORES for v in bad.flatten(1).any(1).nonzero().flatten()})[:16],
        bad_rows_in_tile=[int(v) for v in bad.sum((0, 2, 3)).nonzero().flatten()][:32],
        got=float(outs[0].flatten()[i]), want=float(ref.flatten()[i]),
        ref_absmax=float(ref.abs().max()))), flush=True)


H4, H2 = ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2
B16, B8 = ttnn.bfloat16, ttnn.bfloat8_b
import sys  # noqa: E402
if sys.argv[1:] == ["2"]:
    # wsb2: wsb1 put the fault on HiFi4 at every shape (HiFi2 clean). A full matmul init per phase (a WS_FULL_INIT
    # define, since removed) left it unchanged: it is dest_guard.py's Wormhole fp32-dest cancellation fault.
    CASES = (("c128 h512 bf16 HiFi4", 128, 512, B16, H4),
             ("c128 h512 bf16 HiFi3", 128, 512, B16, ttnn.MathFidelity.HiFi3),
             ("c128 h512 bf16 LoFi", 128, 512, B16, ttnn.MathFidelity.LoFi),
             ("c128 h512 bf16 HiFi2", 128, 512, B16, H2),
             ("c256 h512 bf16 HiFi3", 256, 512, B16, ttnn.MathFidelity.HiFi3))
else:
    CASES = (("ctrl c128 h512 bf16 HiFi4", 128, 512, B16, H4),
             ("c256 h512 bf16 HiFi4 (= one chunk of the faulty path)", 256, 512, B16, H4),
             ("c256 h512 bf16 HiFi2", 256, 512, B16, H2),
             ("c256 h512 bf16 HiFi4 silu metal", 256, 512, B16, H4, 0),
             ("c256 h256 bf16 HiFi4", 256, 256, B16, H4),
             ("c256 h512 bfp8 HiFi4", 256, 512, B8, H4),
             ("c256 h1024 bfp8 HiFi2 (shipped fast, one chunk)", 256, 1024, B8, H2),
             ("c128 h1024 bf16 HiFi4", 128, 1024, B16, H4),
             ("c256 h512 bf16 HiFi4 seed1", 256, 512, B16, H4, 2, 1))
for args in CASES:
    case(*args)
