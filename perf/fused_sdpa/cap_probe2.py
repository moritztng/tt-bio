#!/usr/bin/env python3
"""Does widening SDPA_CHUNK_MAX reach the path the other five models take? (corrected)

The first version of this cleared `_capped_sdpa_chunk_size` and `_sdpa_chunks_shipped` but NOT
`_dividing_sdpa_chunk_size`, which is the function that actually decides k. So it reported
"cap=4096 leaves k at 256", which was a stale lru_cache in the probe, not a property of the code.
Every chunk helper is cleared here, and the cap is taken from argv so each arm is its own process
and no program-cache churn crosses arms.
"""
import sys, json
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/sdpa-rowsum-normalisation-kernel-fix")
sys.path.insert(0, str(ROOT))
CAP = int(sys.argv[1])
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.triatt_sdpa as TS
assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__

T.SDPA_CHUNK_MAX = CAP
for f in (T._capped_sdpa_chunk_size, T._dividing_sdpa_chunk_size,
          T._sdpa_program_config_for_lengths, T._sdpa_chunks_shipped, T._tri_att_q_chunks):
    try:
        f.cache_clear()
    except AttributeError:
        pass
print("cap", CAP, "chunks:", {S: T._sdpa_chunks_shipped(S, S) for S in (320, 512, 768, 1024)},
      flush=True)

dev = T.get_device()
res = {}
for S in (320, 512, 768, 1024):
    B, HEADS, HEAD_DIM = 8, 4, 32
    torch.manual_seed(0)
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = mk(torch.randn(B, HEADS, S, HEAD_DIM).to(torch.bfloat16))
    k = mk(torch.randn(B, HEADS, S, HEAD_DIM).to(torch.bfloat16))
    v = mk(torch.ones(B, HEADS, S, HEAD_DIM, dtype=torch.bfloat16))
    bias = mk(torch.zeros(1, HEADS, S, S, dtype=torch.bfloat16))
    scale_inv = HEAD_DIM ** -0.5
    chosen = T._sdpa_chunks_shipped(S, S)
    for nm, fn in (("default", lambda: T._tri_att_sdpa(q, k, v, bias, scale_inv)),
                   ("hifi", lambda: T._tri_att_sdpa_hifi(q, k, v, bias, scale_inv))):
        pm = list(TS.STATS)
        try:
            o = fn()
        except Exception as exc:                                       # noqa: BLE001
            res[f"S{S}_{nm}"] = {"error": str(exc)[:200]}
            print(f"S{S} {nm:8s} RAISED {str(exc)[:100]}", flush=True)
            continue
        t = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
        spread = (t.max(-1).values - t.min(-1).values).max().item()
        d = t.mean(-1).flatten() - 1.0
        res[f"S{S}_{nm}"] = {"mean": d.mean().item(), "chosen_qk": list(chosen),
                             "head_dim_spread": spread, "cap": CAP,
                             "pm_served": TS.STATS[0] - pm[0], "pm_declined": TS.STATS[1] - pm[1],
                             "frac_below": (d < 0).double().mean().item()}
        r = res[f"S{S}_{nm}"]
        print(f"S{S} cap={CAP:5d} {nm:8s} chunks={tuple(chosen)} mean={r['mean']:+.6f} "
              f"pm_served={r['pm_served']} pm_declined={r['pm_declined']} spread={spread:.1e}",
              flush=True)
    for t_ in (q, k, v, bias):
        ttnn.deallocate(t_)
Path(f"/tmp/cap_probe2_{CAP}.json").write_text(json.dumps(res, indent=1) + "\n")
print("wrote", f"/tmp/cap_probe2_{CAP}.json", flush=True)
T.cleanup()
