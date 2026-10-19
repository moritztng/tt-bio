"""OpenDDE triangle-attention SDPA at the production lengths: the fold's ladder pick against every (q, k) pair.

cs5 (9LLG, 640 bucket, main + t1big, 1000 MHz) puts this op first at JapanFold's shape: 17.7 ms x 1204 calls =
21.3 s of 121 s device, in normal and fast alike. At 512 the one-k-chunk pair (k1) was 1.14x; this asks what
the ladder serves at 608/640/672 and whether a pair it does not offer runs faster.

Arms per seq: `fold` = tenstorrent._tri_att_sdpa_at (the route the Pairformer calls, at its default ckc, with
TT_BIO_LEVERS from the environment), then every (q_chunk, k_chunk) in --qs x --ks through triatt_sdpa.sdpa at
HiFi2 with padded_mask, plus fused_pairs()'s first --top picks. A pair the kernel refuses is recorded, not timed.
Accuracy: rel_rms against float64 softmax((qk + b) * scale) v on every 41st batch row.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ta_sweep.py OUT.jsonl --seqs 608,640,672
"""
import argparse, json, os, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seqs", default="608,640,672"); ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--qs", default="64,96,128,160,192,224,256,320,640")
ap.add_argument("--ks", default="128,160,256,320,640")
ap.add_argument("--top", type=int, default=6)
ap.add_argument("--ncall", type=int, default=4); ap.add_argument("--reps", type=int, default=3)
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
H, D = a.heads, 32
HIFI2 = (ttnn.MathFidelity.HiFi2, True, False, False)
cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
out = open(a.out, "a")


def timed(fn):
    o = fn()
    if o is None:
        return None, None
    ts = []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            fn().deallocate()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / a.ncall * 1e3)
    return o, ts


for S in map(int, a.seqs.split(",")):
    torch.manual_seed(S)
    hq, hk, hv = (torch.randn(S, H, S, D) for _ in range(3))
    hb = torch.randn(1, H, S, S) * 2
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q, k, v, b = (up(t) for t in (hq, hk, hv, hb))
    sc = D ** -0.5
    sel = slice(0, S, 41)
    q64, k64, v64 = (ttnn.to_torch(t)[sel].double() for t in (q, k, v))
    ref = torch.softmax((q64 @ k64.transpose(-1, -2) + ttnn.to_torch(b).double()) * sc, -1) @ v64
    rel = lambda o: float(((ttnn.to_torch(o)[sel].double() - ref).pow(2).mean() / ref.pow(2).mean()).sqrt())

    arms = [("fold", None, None)]
    ranked = TS.fused_pairs(S, H, D, cores, b.dtype, padded=True)
    arms += [("ranked", qc, kc) for qc, kc in ranked[:a.top]]
    arms += [("grid", qc, kc) for qc in map(int, a.qs.split(",")) for kc in map(int, a.ks.split(","))
             if qc <= S + 31 and kc <= S + 31 and ("ranked", qc, kc) not in arms]
    for kind, qc, kc in arms:
        if kind == "fold":
            T.SDPA_CHUNK_PICKS.clear()
            fn = lambda: T._tri_att_sdpa_at(q, k, v, b, sc)
        else:
            fn = lambda qc=qc, kc=kc: TS.sdpa(q, k, v, b, sc, qc, kc, ckc_default=HIFI2, q_split_cap=0, padded_mask=True)
        row = {"seq": S, "kind": kind, "q": qc, "k": kc, "cores": cores,
               "levers": os.environ.get("TT_BIO_LEVERS", "")}
        try:
            o, ts = timed(fn)
        except Exception as exc:  # noqa: BLE001 -- a refused config is a row, not a crash
            o, ts = None, None
            row["error"] = repr(exc)[:200]
        if kind == "fold":
            row["pick"] = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}
        if o is not None:
            row.update(ms=statistics.median(ts), ms_all=ts, rel_rms=rel(o))
            o.deallocate()
        else:
            row["rejected"] = True
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n"); out.flush()
    for t in (q, k, v, b):
        t.deallocate()
