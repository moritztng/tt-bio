"""OpenDDE triangle attention at a prime-tile length: today's route vs padding the axis up a tile or two.

608 tokens is 19 tiles, prime, and it is the partner's main bucket. Its only 32-aligned chunk sizes are 32 and 608,
so the native fused pair declines (`pm_over_l1`, ~1050 calls a fold in kw2ab) and the call takes whatever the
ladder serves next. 640 = 20 tiles has the (q320, k160) pair in normal and (q320, k640) under triatt_k1 in fast.
This times, per lever set, the gated call the fold makes (`_tri_att_gated_sdpa`, falling back to `_tri_att_sdpa`
like the block does) at S, and the same call padded to P (keys masked through `_sdpa_pad_ragged(to=P)`, gate
padded with 0, rows sliced back), pad and slice included in the time.

Accuracy: rel_rms of each arm's gated output against float64 sigmoid(g) * softmax((qk + b) * scale) v on every
23rd batch row.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ta_padup.py OUT.json [--seq 608] [--to 640 672]
"""
import argparse, json, os, statistics, time, types

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=608)
ap.add_argument("--to", type=int, nargs="+", default=[640, 672]); ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--ncall", type=int, default=6); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import opendde as OD

dev = T.get_device()
S, H, D = a.seq, a.heads, 32
torch.manual_seed(S)
hq, hk, hv, hg = (torch.randn(S, H, S, D) for _ in range(4))
hb = torch.randn(1, H, S, S) * 2
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
q, k, v, g, b = (up(t) for t in (hq, hk, hv, hg, hb))
sc = D ** -0.5
sel = slice(0, S, 23)
q64, k64, v64, g64 = (ttnn.to_torch(t)[sel].double() for t in (q, k, v, g))
# the fused kernel adds the bias before the scale: exp((qk + b - max) * scale)
ref = torch.sigmoid(g64) * (torch.softmax((q64 @ k64.transpose(-1, -2) + ttnn.to_torch(b).double()) * sc, -1) @ v64)
att = types.SimpleNamespace(subtile=False, biased=False, fp32_softmax=False)
MODES = {"normal": OD.NORMAL_LEVERS, "fast": OD.WH_FAST_LEVERS}


def gated(q_, k_, v_, b_, g_):
    o = T._tri_att_gated_sdpa(att, q_, k_, v_, b_, sc, g_)
    if o is None:   # the block's own fallback: ungated ladder, then the gate multiply
        u = T._tri_att_sdpa(q_, k_, v_, b_, sc)
        o = ttnn.multiply(u, ttnn.sigmoid(g_))
        u.deallocate()
    return o


def call(to):
    if to == S:
        return gated(q, k, v, b, g)
    qp, kp, vp, bp, _ = T._sdpa_pad_ragged(q, k, v, b, to=to)
    gp = ttnn.pad(g, [(0, 0), (0, 0), (0, to - S), (0, 0)], value=0.0)
    o = gated(qp, kp, vp, bp, gp)
    for t in (qp, kp, vp, bp, gp):
        t.deallocate()
    sl = ttnn.slice(o, [0, 0, 0, 0], [S, H, S, D])
    o.deallocate()
    return sl


res = {"seq": S, "heads": H, "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN), "modes": {}}
arms = [S] + a.to
for mode, lv in MODES.items():
    with T.levers(lv):
        out = res["modes"][mode] = {}
        for to in arms:
            T.SDPA_CHUNK_PICKS.clear()
            o = call(to)
            got = ttnn.to_torch(o)[sel].double()
            o.deallocate()
            out[str(to)] = {"rel_rms": float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                            "picks": {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}, "ms": []}
        for _ in range(a.reps):
            for to in arms:
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(a.ncall):
                    call(to).deallocate()
                ttnn.synchronize_device(dev)
                out[str(to)]["ms"].append((time.perf_counter() - t0) / a.ncall * 1e3)
        for to in arms:
            r = out[str(to)]
            r["ms_median"] = statistics.median(r["ms"])
            print(f"{mode:6s} S{S}->{to}: {r['ms_median']:7.3f} ms  rel_rms {r['rel_rms']:.4e}  picks {r['picks']}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
