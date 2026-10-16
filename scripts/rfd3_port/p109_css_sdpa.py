#!/usr/bin/env python3
"""p109 -- atom attention SDPA that visits only the key chunks each query chunk touches (`css`).

    p109_css_sdpa.py OUT.json [L] [CHUNK ...]

p108 measured the share of (query chunk, key chunk) score tiles that hold a neighbour on the real
atom index in sequence order: 0.17 at 32/32, 0.24 at 64/64, 0.37 at 128/128 (mean over a 200-step
design). Every other tile is the -1e4 mask, whose exp is exactly 0 in the online softmax. This
runs, on the same bf16 inputs and the same bias the model builds (`rfd3_bias.sparse_mask_bf16`):

  * ship   the shipped call, `fused_sdpa` at q160/k320;
  * dense  `sdpa_generic.sdpa` at q=k=CHUNK, every key chunk (same kernels as ttnn's op);
  * css    the same program visiting only the listed chunks (`sdpa_generic.css_kernel_dir`).

css against dense at the same chunk is expected bit-identical: a skipped chunk adds exact zeros.
All three are graded against float64. The index is protein-shaped rather than random: a 3D random
walk with 1.5 A steps (atom i's sequence neighbours are its spatial neighbours too), 128 keys per
row = the 32 nearest in sequence plus the 96 nearest in space, so the tile density is comparable
to p108's (reported). ms/call is the median of 20 synced calls; `css_lists_ms` is the host build
plus upload of the chunk lists, paid once a step for the step's nine calls.
"""
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio import rfd3_bias, sdpa_generic as SG                           # noqa: E402
from tt_bio.tenstorrent import get_device, fused_sdpa, _sdpa_program_config  # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402
from tt_bio.rfd3.tiles import align_tile, pad_axis                         # noqa: E402

out = pathlib.Path(sys.argv[1])
L = int(sys.argv[2]) if len(sys.argv) > 2 else 6051
CHUNKS = [int(c) for c in sys.argv[3:]] or [64, 32, 128]
H, D, K = 4, 32, 128
N = align_tile(L)
scale = D ** -0.5
g = torch.Generator().manual_seed(0)

dev = get_device()
ckc = M._default_compute_kernel_config()
ckc_t = (ckc.math_fidelity, ckc.math_approx_mode, ckc.fp32_dest_acc_en, False)
grid = dev.compute_with_storage_grid_size()
grid = (grid.x, grid.y)
dt = ttnn.bfloat16


def rms(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt()


def protein_index():
    step = torch.randn(L, 3, generator=g)
    x = (step / step.norm(dim=1, keepdim=True) * 1.5).cumsum(0)
    i = torch.arange(L)
    seq = (i[:, None] - i[None, :]).abs().float()
    seq_nb = seq.topk(32, largest=False).indices
    d = torch.cdist(x, x)
    d.scatter_(1, seq_nb, float("inf"))
    sp_nb = d.topk(K - 32, largest=False).indices
    return torch.cat([seq_nb, sp_nb], 1).sort(1).values.unsqueeze(0)       # [1, L, K]


q = rms(torch.randn(1, H, L, D, generator=g)).bfloat16()
k = rms(torch.randn(1, H, L, D, generator=g)).bfloat16()
v = torch.randn(1, H, L, D, generator=g).bfloat16()
idx = protein_index()
pb = torch.randn(1, H, L, K, generator=g).bfloat16()

bias = torch.full((1, H, L, L), -1e4, dtype=torch.float64)
bias.scatter_(3, idx.unsqueeze(1).expand(1, H, L, K), pb.double())
ref = torch.softmax(q.double() @ k.double().transpose(-1, -2) * scale + bias, -1) @ v.double()
del bias


def up(x):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)


qd, kd, vd, pbd = up(q), up(k), up(v), up(pb)
qd, kd, vd = (pad_axis(t, N, 2, 0.0) for t in (qd, kd, vd))
idx_rm = M._sparse_attn_index_rm(idx, dev)
mask = rfd3_bias.sparse_mask_bf16(ttnn.multiply(pbd, D ** 0.5), idx_rm)


def chunk_lists(c):
    """Sorted key chunks per q chunk of the [1, L, K] index; padded q rows inherit nothing."""
    ki = idx[0] // c
    lists = []
    for qc in range(N // c):
        rows = ki[qc * c:min((qc + 1) * c, L)]
        lists.append(sorted(set(rows.flatten().tolist())) if rows.numel() else [0])
    return lists


def timed(fn, n=20):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
    return round(statistics.median(ts), 4)


def host(o):
    return ttnn.to_torch(o)[:, :, :L, :D]


def grade(o):
    o = host(o).double()
    assert torch.isfinite(o).all(), "non-finite output"
    e = (o - ref).abs()
    pcc = torch.corrcoef(torch.stack([o.flatten(), ref.flatten()]))[0, 1].item()
    return {"maxabs": e.max().item(), "meanabs": e.mean().item(), "pcc": pcc}


def ship():
    return fused_sdpa(qd, kd, vd, attn_mask=mask, scale=scale,
                      program_config=_sdpa_program_config(160, 320), compute_kernel_config=ckc)


# Dirty the allocator so unwritten mask tiles hold garbage rather than a previous mask's zeros.
_junk = [ttnn.from_torch(torch.full((1, H, N, N), float("nan")).bfloat16(), dtype=dt,
                         layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(1)]
del _junk
res = {"L": L, "N": N, "arch": str(dev.arch()), "grid": grid, "ship_160_320": {"ms": timed(ship), **grade(ship())}}
print(json.dumps(res), flush=True)
for c in CHUNKS:
    if N % c:
        res[f"c{c}"] = {"skip": "chunk does not divide %d" % N}
        continue
    o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, H, N, D]), dt, ttnn.TILE_LAYOUT, dev,
                                       ttnn.DRAM_MEMORY_CONFIG)
    lists = chunk_lists(c)

    def css_up():
        occ = torch.zeros(N // c, N // c, dtype=torch.bool)
        for i, ks in enumerate(chunk_lists(c)):
            occ[i, ks] = True
        rows, counts = SG.css_lists(occ)
        return ttnn.from_torch(rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev), counts

    css = css_up()
    r = {"density": round(sum(map(len, lists)) / (len(lists) * (N // c)), 4),
         "max_list": max(map(len, lists))}
    try:
        def dense():
            return SG.sdpa(dev, qd, kd, vd, mask, o, c, c, grid, ckc_t, scale)

        def sparse():
            return SG.sdpa(dev, qd, kd, vd, mask, o, c, c, grid, ckc_t, scale, css=css)

        r["dense"] = {"ms": timed(dense), **grade(dense())}
        od = host(dense()).clone()
        r["css"] = {"ms": timed(sparse), **grade(sparse())}
        oc = host(sparse())
        r["css_vs_dense_bit_equal"] = bool(torch.equal(od, oc))
        r["css_vs_dense_maxabs"] = (od.float() - oc.float()).abs().max().item()
        # The mask writer with the same lists leaves every tile outside them unwritten: css on that
        # mask must equal css on the full one. Fill the buffer with garbage first so a tile the
        # SDPA should not read but does shows up.
        pbs = ttnn.multiply(pbd, D ** 0.5)

        def mask_css():
            return rfd3_bias.sparse_mask_bf16(pbs, idx_rm, css=(css[0], c))

        def mask_full():
            return rfd3_bias.sparse_mask_bf16(pbs, idx_rm)

        r["mask_full_ms"], r["mask_css_ms"] = timed(mask_full), timed(mask_css)
        mc = mask_css()
        oc2 = host(SG.sdpa(dev, qd, kd, vd, mc, o, c, c, grid, ckc_t, scale, css=css))
        r["css_mask_vs_full_mask_bit_equal"] = bool(torch.equal(oc, oc2))
        t0 = time.perf_counter()
        for _ in range(5):
            css_up()
        r["css_lists_ms"] = round((time.perf_counter() - t0) / 5 * 1e3, 3)
    except Exception as e:                                                   # noqa: BLE001
        r["error"] = repr(e)[:600]
    res[f"c{c}"] = r
    print(json.dumps(res), flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
