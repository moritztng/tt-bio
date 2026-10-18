"""OpenDDE's triangle-attention LayerNorm folded into the qkvgb projection (triatt_ln column groups).

At c_z 384 the [12 x 49]-tile qkvgb weight does not fit one Wormhole core, so `triatt_ln` used to
decline and the fold ran (pair_transpose for the ending node) + ttnn.layer_norm + qkvgb_heads. The
column-group split holds 1/G of the weight per core. This times both paths at the served shape and
grades every output against a float64 reference on every 23rd pair row.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ln_bench.py OUT.json [--seq 512] [--fid HiFi4]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--fid", default="HiFi4"); ap.add_argument("--ncall", type=int, default=6)
ap.add_argument("--reps", type=int, default=4)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import mm_generic as G, triatt_qkv as Q, triatt_ln as L

dev = T.get_device()
S, H, D = a.seq, a.heads, 32
C = H * D
NW = 4 * C + 32
torch.manual_seed(0)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
xt = torch.randn(S, S, C) * 2 + 0.5
wt = torch.randn(C, NW) * C ** -0.5
gt, bt = 1 + 0.1 * torch.randn(C), 0.1 * torch.randn(C)
x, w, g, b = up(xt), up(wt), up(gt[None]), up(bt[None])
w_o = up(torch.randn(C, C) * C ** -0.5)   # only read by qkvgb_heads' guards
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, a.fid), math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
grid = tuple(T.COMPUTE_GRID_MAIN)


def ref(end):
    xi = T._pair_transpose(x, T._transpose_memory_config(x)) if end else x
    xn = ttnn.layer_norm(xi, weight=g, bias=b, epsilon=1e-5,
                         compute_kernel_config=T.ln_compute_kernel_config(ckc),
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if end:
        ttnn.deallocate(xi)
    out = Q.qkvgb_heads(xn, w, w_o, ckc, H, D, ttnn.bfloat16, T._qkv_mm_config(xn, w, None), 1)
    ttnn.deallocate(xn)
    return out


def fused(end):
    with T.levers(["triatt_ln"]):
        if not end:
            return L.ln_qkvgb_heads(x, g, b, w, G.ckc_args(ckc), H, D, 1, grid)
        # the fold's own route: TR when the kernel takes it, else transpose and the plain call
        out = L.ln_qkvgb_heads(x, g, b, w, G.ckc_args(ckc), H, D, 1, grid, transposed=True)
        if out is None:
            xt_ = T._pair_transpose(x, T._transpose_memory_config(x))
            out = L.ln_qkvgb_heads(xt_, g, b, w, G.ckc_args(ckc), H, D, 1, grid)
            ttnn.deallocate(xt_)
        return out


def flat(out):
    (q, k, v), gate, bias = out
    return [ttnn.to_torch(t).float() for t in (q, k, v, gate, bias)]


def drop(out):
    (q, k, v), gate, bias = out
    for t in (q, k, v, gate, bias):
        ttnn.deallocate(t)


ROWS = list(range(0, S, 23))


def gold(end):
    xr = (xt.transpose(0, 1) if end else xt)[ROWS].double()
    xr = torch.nn.functional.layer_norm(xr, (C,), gt.double(), bt.double(), 1e-5)
    y = xr @ wt.to(torch.bfloat16).double()
    return y


def grade(out, end):
    q, k, v, gate, bias = flat(out)
    got = torch.cat([t[ROWS].permute(0, 2, 1, 3).reshape(len(ROWS), S, C) for t in (q, k, v, gate)]
                    + [bias[ROWS][..., :1]], -1).double()
    ref_ = gold(end)[..., :4 * C + 1]
    e = got - ref_
    return float(e.pow(2).mean().sqrt() / ref_.pow(2).mean().sqrt()), float(e.abs().max())


res = {"seq": S, "heads": H, "fid": a.fid, "plan": {}, "arms": {}}
arms = {"ref_start": lambda: ref(False), "fused_start": lambda: fused(False),
        "ref_end": lambda: ref(True), "fused_end": lambda: fused(True)}
for name, fn in arms.items():
    out = fn()
    if out is None:
        res["arms"][name] = {"declined": {str(k): v for k, v in L.REJECTS.items()}}
        continue
    rr, mx = grade(out, name.endswith("end"))
    res["arms"][name] = {"rel_rms": rr, "max_abs": mx, "ms": []}
    drop(out)
    print(name, "rel_rms", rr, "max", mx, flush=True)
for _ in range(a.reps):
    for name, fn in arms.items():
        if "ms" not in res["arms"][name]:
            continue
        drop(fn())
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            drop(fn())
        ttnn.synchronize_device(dev)
        res["arms"][name]["ms"].append((time.perf_counter() - t0) / a.ncall * 1e3)
for name, r in res["arms"].items():
    if r.get("ms"):
        r["median_ms"] = statistics.median(r["ms"])
        print(name, "median ms", round(r["median_ms"], 3), flush=True)
res["l1_cache_keys"] = [str(k) for k in L._CACHE]
json.dump(res, open(a.out, "w"), indent=1)
