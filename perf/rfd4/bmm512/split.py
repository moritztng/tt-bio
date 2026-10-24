"""bmm512's split of bmm_res at RFD4's shapes: per channel count C (220 = 2 channels on every one of p300c's 110 cores,
256 = served, 330 = 3 on every core), transpose (none / a) and stage mask (`abl` 1 no matmul, 2 no write, 4 no a
reads), the median ms a call. bfp8 a / b, bf16 out, HiFi2, fp32 dest, `rot` as served (rot_pays). 256 on 110 cores
is 3 channels on 36 cores and 2 on the rest, so C 256 vs 220 / 330 says whether the call is paced by its busiest core.

usage: TT_VISIBLE_DEVICES=<chip> timeout 1500 python perf/rfd4/bmm512/split.py --out x.jsonl [--n 512] [--C 220,256,330]
"""
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))
ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--n", default="512")
ap.add_argument("--C", default="220,256,330")
ap.add_argument("--abl", default="0,1,2,4,3,6,5,7")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--inner", type=int, default=20)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--rstag", default="0", help="bench: bmm_res.RSTAG arms, e.g. 0,1 (interleaved, output sha per arm)")
ap.add_argument("--adepth", type=int, default=0, help="bench: bmm_res.A_DEPTH (0 keeps the module's)")
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, bmm_res as B

if A.adepth:
    B.A_DEPTH = A.adepth
dev = T.get_device()
ckc = ttnn.types.BlackholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                              fp32_dest_acc_en=True, packer_l1_acc=True)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev,  # noqa: E731
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)


def timed(fn):
    for _ in range(2):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(A.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(A.inner):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts.append(1e3 * (time.perf_counter() - t0) / A.inner)
    return statistics.median(ts), min(ts), max(ts)


def emit(rec):
    print(json.dumps(rec), flush=True)
    if A.out:
        with open(A.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


ops = {}
for n in map(int, A.n.split(",")):
    for C in map(int, A.C.split(",")):
        g = torch.Generator().manual_seed(C + n)
        ops[n, C] = (up(torch.randn(1, C, n, n, generator=g)), up(torch.randn(1, C, n, n, generator=g)))
for r in range(A.rounds):
    for (n, C), (a, b) in ops.items():
        rot = B.rot_pays(dev, n)
        for ta in (False, True):
            for abl in map(int, A.abl.split(",")):
                for rs in map(int, A.rstag.split(",")):
                    B.RSTAG = bool(rs)
                    fn = lambda: B.matmul(a, b, ckc, transpose_a=ta, abl=abl, rot=rot)  # noqa: E731
                    ms, lo, hi = timed(fn)
                    sha = None
                    if abl == 0 and r == 0:
                        o = fn(); sha = hashlib.sha256(ttnn.to_torch(o).view(torch.int16).numpy().tobytes()).hexdigest()[:16]
                        ttnn.deallocate(o)
                    emit({"round": r, "n": n, "C": C, "ta": ta, "abl": abl, "rot": rot, "rstag": rs, "adepth": B.A_DEPTH,
                          "ms": ms, "min": lo, "max": hi, "sha": sha, "t": time.time()})
