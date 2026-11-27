"""Device all-reduce of the RFD4 gradient (178,448,304 fp32/rank) over the Galaxy fabric.

One process owns an N-chip fabric mesh; each chip holds its own rank's gradient on device.
Arms: builtin (ttnn.all_reduce), agadd (all_gather + rank-order fp32 add on device),
a2a (all_to_all_async column slices in rank order + rank-order fp32 add + all_gather: optimal
traffic, needs a Ring), broken / a2abroken (last rank's gradient dropped; must FAIL).
FABRIC=FABRIC_1D_RING SUB=RxC opens the full mesh SHAPE and benches a RxC submesh of it.
Gate: every chip's result is bit-identical across chips, and max|dev - serial fp32 sum|
<= max|order-perturbed fp32 sum - serial| (the fp32 reduction-order floor: reversed and
pairwise-tree orders). Run: TT_VISIBLE_DEVICES=0..N-1 python ccl_allreduce.py N RxC out.json
"""
import json, sys, time, hashlib
import numpy as np, torch

N = int(sys.argv[1]); SHAPE = tuple(int(v) for v in sys.argv[2].split("x")); OUT = sys.argv[3]
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["builtin", "agadd", "broken"]
REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
P, COLS, CR = 178_448_304, 1024, 16384           # params, columns, rows per chunk (64 MB fp32)
NCH = -(-(-(-P // COLS)) // CR)                    # chunks; tail rows are zero padding
import os, ttnn
ttnn.set_fabric_config(getattr(ttnn.FabricConfig, os.environ.get("FABRIC", "FABRIC_1D")))
full = ttnn.open_mesh_device(ttnn.MeshShape(*SHAPE))
SUB = os.environ.get("SUB")
m = full.create_submesh(ttnn.MeshShape(*(int(v) for v in SUB.split("x")))) if SUB else full
TOPO = ttnn.Topology.Ring if "RING" in os.environ.get("FABRIC", "") else ttnn.Topology.Linear
assert m.get_num_devices() == N, (m.get_num_devices(), N)

def chunk_data(c):
    g = torch.Generator().manual_seed(1000 + c)
    x = torch.randn(N, 1, CR, COLS, generator=g)
    x *= torch.pow(10.0, -6 * torch.rand(N, 1, CR, 1, generator=g))   # gradient-like spread of magnitudes
    lo = c * CR * COLS
    if lo + CR * COLS > P:                                             # zero the padding tail
        x.view(N, -1)[:, max(P - lo, 0):] = 0
    return x

def sums(x):
    ser = x[0].clone()
    for i in range(1, N): ser += x[i]
    rev = x[N - 1].clone()
    for i in range(N - 2, -1, -1): rev += x[i]
    t = list(x)
    while len(t) > 1: t = [t[i] + t[i + 1] if i + 1 < len(t) else t[i] for i in range(0, len(t), 2)]
    return ser, max(float((rev - ser).abs().max()), float((t[0] - ser).abs().max()))

up = lambda x: ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=m,
                                mesh_mapper=ttnn.ShardTensorToMesh(m, dim=0))
grads, sers, floor, broken_in = [], [], 0.0, []
t0 = time.time()
for c in range(NCH):
    x = chunk_data(c); s, f = sums(x); sers.append(s); floor = max(floor, f)
    grads.append(up(x))
    if "broken" in ARMS or "a2abroken" in ARMS:
        xb = x.clone(); xb[N - 1] = 0; broken_in.append(up(xb))
print(f"uploaded {NCH} chunks x {N} ranks in {time.time()-t0:.1f}s, floor {floor:.3e}", flush=True)

def agadd(t):
    g = ttnn.all_gather(t, dim=0, topology=TOPO)
    s = ttnn.slice(g, [0, 0, 0, 0], [1, 1, CR, COLS])
    for i in range(1, N): s = ttnn.add(s, ttnn.slice(g, [i, 0, 0, 0], [i + 1, 1, CR, COLS]))
    g.deallocate(); return s
_a2a = {}
def a2a(t):
    if not _a2a:
        z = lambda shape: ttnn.from_torch(torch.zeros(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=m,
                                          mesh_mapper=ttnn.ShardTensorToMesh(m, dim=0))
        g = m.compute_with_storage_grid_size()
        _a2a.update(inter=z((N, 1, CR, COLS)), out=z((N, 1, N * CR, COLS // N)), sem=ttnn.create_global_semaphore(
            m, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))}), 0))
    y = ttnn.experimental.all_to_all_async(t, _a2a["inter"], _a2a["out"], in_dim=2, out_dim=3,
                                           multi_device_global_semaphore=_a2a["sem"], topology=ttnn.Topology.Ring)
    s = ttnn.slice(y, [0, 0, 0, 0], [1, 1, CR, COLS // N])
    for i in range(1, N): s = ttnn.add(s, ttnn.slice(y, [0, 0, i * CR, 0], [1, 1, (i + 1) * CR, COLS // N]))
    o = ttnn.all_gather(s, dim=3, topology=TOPO); s.deallocate(); return o
OPS = {"builtin": lambda t: ttnn.all_reduce(t, topology=TOPO), "agadd": agadd, "broken": agadd, "a2a": a2a, "a2abroken": a2a}

res = {"n": N, "shape": SHAPE, "params": P, "padded": NCH * CR * COLS, "chunks": NCH, "floor": floor, "arms": {}}
for arm in ARMS:
    src = broken_in if "broken" in arm else grads
    times, outs = [], None
    for r in range(REPS):
        ttnn.synchronize_device(m); t = time.time()
        o = [OPS[arm](g) for g in src]
        ttnn.synchronize_device(m); times.append(time.time() - t)
        if r < REPS - 1:
            for z in o: z.deallocate()
        else: outs = o
    err, agree, h = 0.0, True, hashlib.sha256()
    for c, o in enumerate(outs):
        b = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(m, dim=0))
        agree &= all(torch.equal(b[0], b[i]) for i in range(N))
        err = max(err, float((b[0] - sers[c]).abs().max())); h.update(b[0].numpy().tobytes()); o.deallocate()
    ok = agree and err <= floor
    res["arms"][arm] = {"s": times, "s_med": float(np.median(times)), "GBps_per_chip_payload": P * 4 / float(np.median(times)) / 1e9,
                        "max_err_vs_serial": err, "ranks_agree": agree, "verdict": "PASS" if ok else "FAIL", "digest": h.hexdigest()[:16]}
    print(arm, json.dumps(res["arms"][arm]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
ttnn.close_mesh_device(full); print("DONE", OUT)
