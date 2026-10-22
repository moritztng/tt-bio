"""pair_swiglu_ws weight multicast A/B: per-core weight reads (WEIGHTS_MCAST False) against one read multicast to the
grid, on AF2's pair_relu forward and backward at BC2's 224 shapes (c 128 / hidden 512, template c 64 / hidden 128) and on
the SwiGLU form with the LayerNorm prologue (c 128 / hidden 512, as RFD3 / Protenix call it). Outputs must be
torch.equal; timing interleaved A, B, A, B over --reps rounds of --calls back-to-back calls, AICLK sampled per row.

usage: TT_VISIBLE_DEVICES=<chip> python perf/bc4_swiglu/mcast_ab.py --out x.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--calls", type=int, default=50)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", required=True)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T, pair_swiglu_ws as W
from tt_bio.af2 import compute_kernel_config

dev = T.get_device()
DR = ttnn.DRAM_MEMORY_CONFIG
CHIP = os.environ.get("TT_VISIBLE_DEVICES", "0")
FID = compute_kernel_config().math_fidelity
out = open(A.out, "a")


def emit(rec):
    rec["aiclk"] = int(open(f"/sys/class/tenstorrent/tenstorrent!{CHIP}/tt_aiclk").read().split()[0])
    rec["load"] = os.getloadavg()[0]
    print(json.dumps(rec), flush=True)
    out.write(json.dumps(rec) + "\n")
    out.flush()


def batch(fn):
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    keep = [fn() for _ in range(A.calls)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / A.calls * 1e3
    for o in keep:
        o.deallocate()
    return dt


def up(t):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DR)


def with_mcast(on, fn):
    W.WEIGHTS_MCAST = on
    return fn()


emit({"fidelity": str(FID), "sha": os.popen("git rev-parse HEAD").read().strip(),
      "dirty": os.popen("git status --porcelain --untracked-files=no").read().strip()})
n = A.n
cases = []
for C, H in ((128, 512), (64, 128)):
    g = torch.Generator().manual_seed(C)
    bf = lambda t: t.to(torch.bfloat16)  # noqa: E731
    x, dy = up(bf(torch.randn(1, n, n, C, generator=g) * 2 + 0.3)), up(bf(torch.randn(1, n, n, C, generator=g)))
    w1, w2 = bf(torch.randn(C, H, generator=g) / C ** 0.5).float(), bf(torch.randn(H, C, generator=g) / H ** 0.5).float()
    b1, b2 = bf(torch.randn(H, generator=g) * 0.1).float(), bf(torch.randn(C, generator=g) * 0.1).float()
    gam, bet = (ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                for v in (bf(1 + 0.1 * torch.randn(C, generator=g)), bf(0.1 * torch.randn(C, generator=g))))
    LN = (gam, bet, 1e-5)
    rw = W.relu_weights(w1, b1, w2, b2, dev)
    bw12, bw3 = W.relu_bw_weights(w1, w2, dev)
    cases.append((f"relu_fwd_c{C}", lambda x=x, rw=rw, LN=LN: W._run(x, rw[0], rw[2], FID, (), W.SILU_POLY, LN, None,
                                                                    (rw[1], rw[3]))))
    cases.append((f"relu_bw_c{C}", lambda x=x, dy=dy, rw=rw, b=(bw12, bw3), LN=LN: W._run(
        x, b[0], b[1], FID, (), W.SILU_POLY, LN, None, (rw[1], dy), bw=True)))
    if C == 128:
        w3 = bf(torch.randn(H, C, generator=g) / H ** 0.5).float()
        wg2 = bf(torch.randn(C, H, generator=g) / C ** 0.5).float()
        (s12, s3), = W.weights(w1, wg2, w3, dev)
        cases.append(("swiglu_ln_c128", lambda x=x, s12=s12, s3=s3, LN=LN: W.pair_swiglu(x, s12, s3, FID, (), 2,
                                                                                         ln=LN)))
for name, fn in cases:
    ref = ttnn.to_torch(with_mcast(False, fn))
    got = ttnn.to_torch(with_mcast(True, fn))
    eq = bool(torch.equal(ref, got))
    times = {False: [], True: []}
    for _ in range(A.reps):
        for on in (False, True):
            times[on].append(with_mcast(on, lambda: batch(fn)))
    for on in (False, True):
        times[on].sort()
    emit({"case": name, "equal": eq, "ms_off": times[False][0], "ms_on": times[True][0],
          "ms_off_med": times[False][A.reps // 2], "ms_on_med": times[True][A.reps // 2],
          "off_all": times[False], "on_all": times[True]})
W.WEIGHTS_MCAST = True
emit({"done": True})
