"""OpenDDE triangle-attention SDPA at its served shape, one call, every config the fused kernel can take.

The op is [S, H, S, D] q/k/v + [1, H, S, S] bias, S=512 H=12 D=32 on a Wormhole chip: 31.7 % of the
fold's device time (cs1 census, 8.5 ms x 2 per block). At head_dim 32 the score matrix is 16x the
operands, so the matmul roof does not price it; this bench splits the time by arm instead.

Arms (each times NCALL back-to-back calls after one sync, REPS times, arms interleaved):
  ship    q512 k256 HiFi2,approx,acc0      today's fold
  k1      q256 k512, same ckc              one k chunk: no second online-softmax rescale
  lofi    q512 k256 LoFi                   matmul fidelity
  k1lofi  q256 k512 LoFi
Accuracy: rel_rms of each arm against float64 softmax(qk*scale + bias) v on every 23rd batch row.
TT_BIO_TRIATT_ABLATE=EXP|MASKADD (an instrument, wrong output on purpose) is read at import, so an
ablation is its own process: run this script with the env set and --tag ablate-exp.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ta_bench.py OUT.json [--seq 512] [--heads 12] [--tag X]
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=512); ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--ncall", type=int, default=8); ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--tag", default="")
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import triatt_sdpa as TS

dev = T.get_device()
S, H, D = a.seq, a.heads, 32
torch.manual_seed(S)
hq, hk, hv = (torch.randn(S, H, S, D) for _ in range(3))
hb = torch.randn(1, H, S, S) * 2
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
q, k, v, b = (up(t) for t in (hq, hk, hv, hb))
sc = D ** -0.5
sel = slice(0, S, 23)
q64, k64, v64 = (ttnn.to_torch(t)[sel].double() for t in (q, k, v))
ref = torch.softmax(q64 @ k64.transpose(-1, -2) * sc + ttnn.to_torch(b).double() * sc, -1) @ v64
# the fused kernel adds the bias before the scale: exp((qk + b - max) * scale), so b is scaled above too.

HIFI2 = (ttnn.MathFidelity.HiFi2, True, False, False)
LOFI = (ttnn.MathFidelity.LoFi, True, False, False)
ARMS = {"ship": (512, 256, HIFI2), "k1": (256, 512, HIFI2), "lofi": (512, 256, LOFI), "k1lofi": (256, 512, LOFI)}


def call(arm):
    qc, kc, ckc = ARMS[arm]
    o = TS.sdpa(q, k, v, b, sc, qc, kc, ckc_default=ckc, q_split_cap=0, padded_mask=True)
    assert o is not None, (arm, TS.REJECTS if hasattr(TS, "REJECTS") else "rejected")
    return o


res = {"seq": S, "heads": H, "head_dim": D, "tag": a.tag, "ablate": os.environ.get("TT_BIO_TRIATT_ABLATE", ""),
       "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN), "arms": {}}
for arm in ARMS:
    o = call(arm)
    res["arms"][arm] = {"rel_rms": float(((ttnn.to_torch(o)[sel].double() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())}
    o.deallocate()
times = {arm: [] for arm in ARMS}
for _ in range(a.reps):
    for arm in ARMS:
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            call(arm).deallocate()
        ttnn.synchronize_device(dev)
        times[arm].append((time.perf_counter() - t0) / a.ncall * 1e3)
for arm, t in times.items():
    res["arms"][arm].update(ms_median=statistics.median(t), ms_min=min(t), ms=t)
    print(f"{a.tag or 'normal':12s} {arm:8s} {statistics.median(t):7.3f} ms  rel_rms {res['arms'][arm]['rel_rms']:.2e}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
