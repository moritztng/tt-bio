"""pair_swiglu_ws weight multicast on Wormhole at OpenDDE's pair transition: per-core weight reads (WEIGHTS_MCAST
False) against one read multicast to the grid (True). The call is the shipped one (transition_b8 + ws_b8_plain):
c_z 384, hidden 1536, bfp8 weights in one chunk, HiFi2, the residual added in place. Outputs must be torch.equal;
timing interleaved off, on over --reps rounds of --calls back-to-back calls, AICLK sampled per row.

usage: TT_VISIBLE_DEVICES=<chip> python perf/spd_m_opendde/wmc_ab.py --out x.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[640, 320])
ap.add_argument("--calls", type=int, default=20)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, pair_swiglu_ws as W

dev = T.get_device()
DR = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
FID = ttnn.MathFidelity.HiFi2
C, H = 384, 1536
out = open(A.out, "a")


def emit(rec):
    rec["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().split()[0])
    rec["load"] = os.getloadavg()[0]
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def up(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)


emit({"arch": str(dev.arch()), "sha": os.popen("git rev-parse HEAD").read().strip()})
g = torch.Generator().manual_seed(0)
bf = lambda t: t.to(torch.bfloat16).float()  # noqa: E731
w1, w2 = bf(torch.randn(C, H, generator=g) / C ** 0.5), bf(torch.randn(C, H, generator=g) / C ** 0.5)
w3 = bf(torch.randn(H, C, generator=g) / H ** 0.5)
parts = W.weights(w1, w2, w3, dev, ttnn.bfloat8_b)
assert len(parts) == 1, len(parts)
w12, w3t = parts[0]
for n in A.n:
    xn = up(torch.randn(1, n, n, C, generator=g))
    z0 = torch.randn(1, n, n, C, generator=g) * 40

    def call(on):
        W.WEIGHTS_MCAST = on
        z = up(z0)
        W.pair_swiglu(xn, w12, w3t, FID, resid=z)
        return z

    ref, got = ttnn.to_torch(call(False)), ttnn.to_torch(call(True))
    eq = bool(torch.equal(ref, got))
    z = up(z0)
    times = {False: [], True: []}
    for _ in range(A.reps):
        for on in (False, True):
            W.WEIGHTS_MCAST = on
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(A.calls):
                W.pair_swiglu(xn, w12, w3t, FID, resid=z)
            ttnn.synchronize_device(dev)
            times[on].append((time.perf_counter() - t0) / A.calls * 1e3)
    for on in times:
        times[on].sort()
    emit({"n": n, "equal": eq, "ms_off_med": times[False][A.reps // 2], "ms_on_med": times[True][A.reps // 2],
          "off_all": times[False], "on_all": times[True]})
    for t in (xn, z):
        t.deallocate()
emit({"done": True})
