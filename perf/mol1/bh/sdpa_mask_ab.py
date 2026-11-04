"""mol1-bh-diffstep: is the DiT's fp32 SDPA bound by its mask stream? `_sdpa32_fused` at 1024 / 512 tokens, 5 samples:
fp32 mask (main), the same mask bf16 and bfp8, and one sample per call (the mask read once)."""
import json, math, os, pathlib, sys, time
import torch
ROOT = pathlib.Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
from tt_bio import tenstorrent as TT
dev = TT.get_device(); TT._LEVERS = frozenset({"dit_sdpa32", "dit_qkvf"})
H, DH, DR = 16, 64, 48


def replay(fn):
    o = fn(); ttnn.deallocate(o); ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0); o = fn(); ttnn.end_trace_capture(dev, tid, cq_id=0)
    n = 40; ttnn.synchronize_device(dev); t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev); dt = (time.perf_counter() - t0) * 1000 / n
    ttnn.release_trace(dev, tid); ttnn.deallocate(o)
    return round(dt, 4)


g = torch.Generator().manual_seed(0); res = {}
for n in (512, 1024):
    for S in (5, 1):
        qkv = torch.zeros(S, 1, n, 3 * H * DH)
        qkv.view(S, 1, n, 3 * H, DH)[..., :DR] = torch.randn(S, 1, n, 3 * H, DR, generator=g)
        tq = ttnn.from_torch(qkv, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        bias = torch.randn(1, H, n, n, generator=g) * 2
        for mdt in (ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b):
            tm = ttnn.from_torch(bias, dtype=mdt, layout=ttnn.TILE_LAYOUT, device=dev)
            res[f"{n} S{S} {mdt}"] = replay(lambda: TT._sdpa32_fused(tq, tm, 1 / math.sqrt(DR), H))
            print(f"{n} S{S} {mdt}", res[f"{n} S{S} {mdt}"], flush=True)
            ttnn.deallocate(tm)
        ttnn.deallocate(tq)
res["aiclk_after"] = [int(open(f).read()) for f in sorted(__import__("glob").glob("/sys/class/tenstorrent/*/tt_aiclk"))]
json.dump(res, open(sys.argv[1], "w"), indent=1)
