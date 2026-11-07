"""Where does the fused fp32 Transition (tt_bio.fp32_transition) go wrong on Wormhole when Blackhole is right?
Per N and per forced core split G ("auto" = plan()): the fused output vs the composed one per 32-token row tile
(rel rms), and the max |diff| between three fused calls on the same input (non-zero = a race, not a precision fault).
Input as fst2_bench. TT_VISIBLE_DEVICES=C python fst2_diag.py TAP.pt LNW.pt N[,N..] G[,G..] OUT.json"""
import json, sys
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
auto_plan = F32T.plan
rows = []
for N in map(int, sys.argv[3].split(",")):
    torch.manual_seed(0)
    x = x0[torch.arange(N) % x0.shape[0]] + (x0 - x0.mean(0)).std(0) * 0.3 * torch.randn(N, 384, dtype=torch.float64)
    lw, lb = bf(lnw["w"].double()), bf(lnw["b"].double())
    w1, w2 = (bf(torch.randn(384, 1536, dtype=torch.float64) / 384 ** .5 * 8) for _ in range(2))
    w3 = bf(torch.randn(1536, 384, dtype=torch.float64) / 1536 ** .5)
    X = up(x.reshape(1, N, 384)); LW, LB = up(lw), up(lb); W1, W2, W3 = up(w1), up(w2), up(w3)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)
    get = lambda f: ttnn.to_torch(f()).double().reshape(N, 384)
    comp = get(lambda: T.fp32_single_transition(X, LW, LB, W1, W2, W3, ckc, ttnn.bfloat16))
    for gs in sys.argv[4].split(","):
        F32T._CACHE.clear()
        F32T.plan = auto_plan if gs == "auto" else (lambda mt, kt, nh, nc, g=int(gs): g)
        row = dict(N=N, G=gs, G_used=auto_plan(-(-N // 32), 12, 48, dev.compute_with_storage_grid_size().x
                                                * dev.compute_with_storage_grid_size().y) if gs == "auto" else int(gs))
        try:
            outs = [get(lambda: F32T.transition(X, LW, LB, W1, W2, W3, ckc, ttnn.bfloat16)) for _ in range(3)]
            row["repeat_max_abs"] = max(float((o - outs[0]).abs().max()) for o in outs[1:])
            d = outs[0] - comp
            per = [float(d[i:i + 32].norm() / comp[i:i + 32].norm()) for i in range(0, N, 32)]
            row["tile_rel"] = [round(p, 4) for p in per]
            row["bad_tiles"] = [i for i, p in enumerate(per) if p > 0.02]
            # within a bad tile: which rows / which 32-col output tiles
            if row["bad_tiles"]:
                t = row["bad_tiles"][0] * 32
                blk = d[t:t + 32]
                row["bad_rows"] = [int(r) for r in (blk.norm(dim=1) / comp[t:t + 32].norm(dim=1) > 0.02).nonzero().flatten()]
                row["bad_cols"] = [c for c in range(12) if float(blk[:, c*32:c*32+32].norm() / comp[t:t+32, c*32:c*32+32].norm()) > 0.02]
            row["finite"] = all(bool(torch.isfinite(o).all()) for o in outs)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"[:1500]
        print(json.dumps(row), flush=True)
        rows.append(row)
F32T.plan = auto_plan
json.dump(rows, open(sys.argv[5], "w"), indent=1)
