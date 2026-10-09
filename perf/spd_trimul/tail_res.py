"""The weights-resident trimul tail against the 2D one: torch.equal, float64 error, ms per call.

`trimul_tail.RES` swaps the 2D minimal_matmul transcription (in0 multicast chain) for
`kernels/trimul_tail_res`: every core keeps both weights in L1 and streams its own activation rows.
The epilogue is EPI 1 / 2 op for op, so the two must agree bit for bit.

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_trimul/tail_res.py [--n 736] [--calls 20] [--reps 3]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=736)
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--epi", default="1,2")
ap.add_argument("--abl", default="", help="time only: resident stage ablations, e.g. 1,2,4,7")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, trimul_tail as TT, mm_generic as MG
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
grid = tuple(T.COMPUTE_GRID_MAIN)
g = torch.Generator().manual_seed(0)
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
h = {k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in
     (("xa", (1, A.n * A.n, 256)), ("xb", (1, A.n * A.n, 256)), ("z", (1, A.n * A.n, 256)))}
h["wa"] = (torch.randn(256, 256, generator=g) / 16).to(torch.bfloat16)
h["wb"] = (torch.randn(256, 256, generator=g) / 16).to(torch.bfloat16)
d = {k: up(v) for k, v in h.items()}
P = A.n * A.n * 256 * 2
print(json.dumps({"n": A.n, "grid": grid, "ckc": [str(c) for c in ckc]}), flush=True)


def ref64(shared, resid):
    x64 = {k: v.double() for k, v in h.items()}
    p = x64["xa"] @ x64["wa"]
    q = (x64["xa"] if shared else x64["xb"]) @ x64["wb"]
    y = p * torch.sigmoid(q)
    return y + x64["z"] if resid else y


for epi in (int(e) for e in A.epi.split(",")):
    TT.set_epi(epi)
    for shared in (True, False):
        xb = d["xa"] if shared else d["xb"]
        got, ms_of = {}, {}
        for res in (False, True):
            TT.set_res(res)
            rec = {"epi": epi, "shared": shared, "res": res}
            try:
                if epi == 2:
                    z = up(h["z"])
                    f = lambda: TT.fused_tail(d["xa"], xb, d["wa"], d["wb"], ckc, grid, resid=z)
                    y = f(); ttnn.synchronize_device(dev)
                    got[res] = ttnn.to_torch(y).reshape(h["z"].shape)
                else:
                    f = lambda: TT.fused_tail(d["xa"], xb, d["wa"], d["wb"], ckc, grid)
                    y = f(); ttnn.synchronize_device(dev)
                    got[res] = ttnn.to_torch(y).reshape(h["z"].shape)
                    ttnn.deallocate(y)
                rec["res_served"] = TT.RES_STATS[0]
                ms = []
                for _ in range(A.reps):
                    t0 = time.perf_counter()
                    outs = [f() for _ in range(A.calls)]
                    ttnn.synchronize_device(dev)
                    ms.append((time.perf_counter() - t0) * 1e3 / A.calls)
                    if epi != 2:
                        for o in outs:
                            ttnn.deallocate(o)
                rec["ms"] = round(min(ms), 4)
                rec["spread_ms"] = round(max(ms) - min(ms), 4)
                rec["GBps"] = round(((2 if shared else 3) + (2 if epi == 2 else 1) - 1) * P
                                    / rec["ms"] / 1e6, 1)
                r = ref64(shared, epi == 2)
                rec["rel_rms_f64"] = float(((got[res].double() - r).pow(2).mean().sqrt()
                                            / r.pow(2).mean().sqrt()))
                ms_of[res] = rec["ms"]
            except Exception as e:                                            # noqa: BLE001
                rec["err"] = str(e).splitlines()[0][:300]
            if res and False in got and True in got:
                rec["equal"] = bool(torch.equal(got[False], got[True]))
                rec["max_abs"] = float((got[False].float() - got[True].float()).abs().max())
                if False in ms_of and True in ms_of:
                    rec["speedup"] = round(ms_of[False] / ms_of[True], 3)
            print(json.dumps(rec), flush=True)
TT.set_res(False)
TT.set_epi(2)
for abl in [int(a) for a in A.abl.split(",") if a]:
    TT.set_res(True); TT.RES_ABL = abl
    z = up(h["z"])
    f = lambda: TT.fused_tail(d["xa"], d["xa"], d["wa"], d["wb"], ckc, grid, resid=z)
    f(); ttnn.synchronize_device(dev)
    ms = []
    for _ in range(A.reps):
        t0 = time.perf_counter(); [f() for _ in range(A.calls)]; ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / A.calls)
    print(json.dumps({"epi": 2, "shared": True, "res": True, "abl": abl, "ms": round(min(ms), 4)}), flush=True)
TT.RES_ABL = 0
TT.set_res(False)
