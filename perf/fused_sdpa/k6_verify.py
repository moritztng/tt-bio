#!/usr/bin/env python3
"""K6 off must be today byte for byte; K6 on must give one k chunk and the measured row sums."""
import sys, os, json
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/sdpa-rowsum-normalisation-kernel-fix")
sys.path.insert(0, str(ROOT))
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.triatt_sdpa as TS
assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__
ON = os.environ.get("TT_BIO_SDPA_ONE_K_CHUNK") == "1"
print("K6", "ON" if ON else "OFF", "chunks:",
      {S: T._sdpa_chunks_shipped(S, S) for S in (288, 320, 352, 384, 512, 704, 768, 864, 1024)},
      flush=True)

dev = T.get_device()
res = {}
for S in (320, 512, 768, 1024):
    B, H, D = 8, 4, 32
    torch.manual_seed(0)
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q, k = mk(torch.randn(B, H, S, D).to(torch.bfloat16)), mk(torch.randn(B, H, S, D).to(torch.bfloat16))
    v = mk(torch.ones(B, H, S, D, dtype=torch.bfloat16))
    bias = mk(torch.zeros(1, H, S, S, dtype=torch.bfloat16))
    for nm, fn in (("default", lambda: T._tri_att_sdpa(q, k, v, bias, D ** -0.5)),
                   ("hifi", lambda: T._tri_att_sdpa_hifi(q, k, v, bias, D ** -0.5))):
        pm = list(TS.STATS)
        try:
            o = fn()
        except Exception as exc:                                       # noqa: BLE001
            res[f"S{S}_{nm}"] = {"error": str(exc)[:200]}
            print(f"S{S} {nm:8s} RAISED {str(exc)[:100]}", flush=True)
            continue
        t = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
        d = t.mean(-1).flatten() - 1.0
        res[f"S{S}_{nm}"] = {"mean": d.mean().item(), "chunks": list(T._sdpa_chunks_shipped(S, S)),
                             "spread": (t.max(-1).values - t.min(-1).values).max().item(),
                             "pm_served": TS.STATS[0] - pm[0], "pm_declined": TS.STATS[1] - pm[1]}
        r = res[f"S{S}_{nm}"]
        print(f"S{S} {nm:8s} chunks={tuple(r['chunks'])} mean={r['mean']:+.6f} "
              f"pm_served={r['pm_served']} pm_declined={r['pm_declined']}", flush=True)
    for t_ in (q, k, v, bias):
        ttnn.deallocate(t_)
Path(f"/tmp/k6_{'on' if ON else 'off'}.json").write_text(json.dumps(res, indent=1) + "\n")
T.cleanup()
