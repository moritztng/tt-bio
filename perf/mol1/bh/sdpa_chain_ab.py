"""mol1-bh-diffstep: dit_mask_chain op A/B. `_sdpa32_fused`, 5 samples, fp32 mask at 256 / 512 / 1024 tokens:
mask chain off vs on (with and without dit_qbox), bit-exact output check and trace-replay ms a call."""
import json, math, pathlib, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as TT
dev = TT.get_device()
BASE = frozenset({"dit_sdpa32", "dit_qkvf"})
H, DH, DR = 16, 64, 48


def aiclk():
    return [int(open(f).read()) for f in sorted(__import__("glob").glob("/sys/class/tenstorrent/*/tt_aiclk"))]


def replay(fn):
    o = fn(); ttnn.synchronize_device(dev); ref = ttnn.to_torch(o); ttnn.deallocate(o)
    tid = ttnn.begin_trace_capture(dev, cq_id=0); o = fn(); ttnn.end_trace_capture(dev, tid, cq_id=0)
    n = 40; ttnn.synchronize_device(dev); t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    clk = aiclk()
    ttnn.synchronize_device(dev); dt = (time.perf_counter() - t0) * 1000 / n
    ttnn.release_trace(dev, tid); ttnn.deallocate(o)
    return round(dt, 4), ref, clk


g = torch.Generator().manual_seed(0); res = {}
for n in [int(a) for a in sys.argv[2].split(",")]:
    S = 5
    qkv = torch.zeros(S, 1, n, 3 * H * DH)
    qkv.view(S, 1, n, 3 * H, DH)[..., :DR] = torch.randn(S, 1, n, 3 * H, DR, generator=g)
    tq = ttnn.from_torch(qkv, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = ttnn.from_torch(torch.randn(1, H, n, n, generator=g) * 2, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    outs = {}
    for arm in ("base", "dit_mask_chain", "dit_qbox", "dit_qbox+dit_mask_chain") * 2:
        TT._LEVERS = BASE | (frozenset(arm.split("+")) - {"base"})
        ms, o, clk = replay(lambda: TT._sdpa32_fused(tq, tm, 1 / math.sqrt(DR), H))
        same = bool(torch.equal(o, outs.setdefault("ref", o)))
        res.setdefault(f"{n} {arm}", []).append({"ms": ms, "bit_exact_vs_base": same, "aiclk": clk})
        print(n, arm, ms, "bit-exact" if same else "DIFFERS", clk, flush=True)
    ttnn.deallocate(tm); ttnn.deallocate(tq)
json.dump(res, open(sys.argv[1], "w"), indent=1)
