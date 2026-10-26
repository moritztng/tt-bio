"""DiT _hoist linear [1,704,384] x [384,27648] (+bias, sigmoid) on one chip: ttnn's default program vs explicit 2D
multicast configs and N-chunked calls. Prints ms/call (warm median of 20, synchronised) and max abs error against a
float64 reference, and whether each output equals the default bit for bit. JSON to argv[1]."""
import json, sys, time
import torch
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

dev = get_device()
M, K, N = 704, 384, 27648
torch.manual_seed(0)
x = torch.randn(1, M, K) * 0.5
w = torch.randn(K, N) / K ** 0.5
b = torch.randn(1, N) * 0.1
xb, wb, bb = (t.to(torch.bfloat16) for t in (x, w, b))
ref = torch.sigmoid(xb.double() @ wb.double() + bb.double())
T = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
tx, tw, tb = T(xb), T(wb), T(bb)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
grid = dev.compute_with_storage_grid_size()


def timeit(fn, n=20):
    out = fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        ttnn.deallocate(out)
        t0 = time.perf_counter(); out = fn(); ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
    ts.sort()
    return out, 1e3 * ts[len(ts) // 2]


res = {"grid": [grid.x, grid.y]}
base, res["default_ms"] = timeit(lambda: ttnn.linear(tx, tw, bias=tb, activation="sigmoid", compute_kernel_config=ckc,
                                                     dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN))
bt = ttnn.to_torch(base).double()
res["default_err"] = float((bt - ref).abs().max())
print("default", res["default_ms"], res["default_err"], flush=True)


def sub(h, w_):
    for sw in range(min(w_, 8), 0, -1):
        if w_ % sw == 0:
            for sh in range(min(h, 8 // sw), 0, -1):
                if h % sh == 0 and sh * sw <= 4:
                    return sh, sw
    return 1, 1


cands = []
for gx, gy in ((grid.x, grid.y), (8, 8)):
    mt, nt = M // 32, N // 32
    pm, pn = -(-mt // gy), -(-nt // gx)
    for kb in (1, 2, 3, 4, 6):
        for ow in (pn, pn // 2, pn // 3, pn // 4, pn // 6, pn // 9, pn // 12):
            if ow < 1 or pn % ow:
                continue
            sh, sw = sub(pm, ow)
            cands.append(dict(gx=gx, gy=gy, kb=kb, pm=pm, pn=pn, oh=pm, ow=ow, sh=sh, sw=sw))
rows = []
for c in cands:
    try:
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(c["gx"], c["gy"]), in0_block_w=c["kb"], out_subblock_h=c["sh"],
            out_subblock_w=c["sw"], out_block_h=c["oh"], out_block_w=c["ow"], per_core_M=c["pm"], per_core_N=c["pn"],
            transpose_mcast=False, fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID), fuse_batch=True)
        o, ms = timeit(lambda: ttnn.linear(tx, tw, bias=tb, program_config=pc, compute_kernel_config=ckc,
                                           dtype=ttnn.bfloat16))
        ot = ttnn.to_torch(o).double(); ttnn.deallocate(o)
        c.update(ms=ms, err=float((ot - ref).abs().max()), equal=bool(torch.equal(ot, bt)))
    except Exception as e:  # noqa: BLE001  (L1 refusals are expected for big blocks)
        c.update(fail=str(e).splitlines()[0][:160])
    rows.append(c)
    print(c, flush=True)
res["cands"] = rows
json.dump(res, open(sys.argv[1], "w"), indent=1)
