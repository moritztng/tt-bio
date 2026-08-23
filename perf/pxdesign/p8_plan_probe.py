"""Which `fill_preconditions` term declines AF2-IGs triangle attention at 848 tokens.

`p8_triatt_screen.py` measured 208 of 208 calls SERVED at 208 tokens and 312 of 312 DECLINED at
848, all on `fill_preconditions`. That check is a conjunction of six terms, so the screen says the
gate closed and not which term closed it. This allocates the exact operands at both sizes and
prints `sdpa_generic.plan` in full -- no model, no weights, seconds rather than minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn

from tt_bio import sdpa_generic as SG
from tt_bio import tenstorrent as TT
from tt_bio import triatt_sdpa as TS


def probe(dev, tokens: int, heads: int = 4, head_dim: int = 32) -> dict:
    mk = lambda shape: ttnn.from_torch(
        __import__("torch").zeros(*shape), layout=ttnn.TILE_LAYOUT, device=dev,
        dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = mk((tokens, heads, tokens, head_dim))
    k, v = mk((tokens, heads, tokens, head_dim)), mk((tokens, heads, tokens, head_dim))
    bias = mk((1, heads, tokens, tokens))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([tokens, heads, tokens, head_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
        ttnn.DRAM_MEMORY_CONFIG)
    grid = tuple(TT.COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    shipped = TT._sdpa_chunks_shipped(tokens, tokens)[1]
    padded = TT._padded_sdpa_len(tokens)
    k_cands = [shipped] + [c for c in TT._tri_att_k_chunks(tokens, tokens) if c != shipped]
    rows = []
    for k_chunk in k_cands:
      for q_chunk in TT._tri_att_q_chunks(tokens, tokens):
          qnc = -(-tokens // q_chunk)
          q_pf = qnc if (TS._Q_SPLIT and tokens <= TS._Q_SPLIT_MAX_S and qnc > 1
                         and cores // (heads * qnc) >= 1) else 1
          split = (cores // (heads * q_pf), heads, q_pf)
          try:
              p = SG.plan(q, k, v, bias, out, q_chunk, k_chunk, grid,
                          (ttnn.MathFidelity.HiFi4, False, True, False), 0.25, split)
              terms = {"nh_per_core": p["nh_per_core"], "q_per_core": p["q_per_core"],
                       "bcast_batch": p["bcast_batch"], "use_padded_mask": p["use_padded_mask"],
                       "NKH": p["NKH"], "NVH": p["NVH"], "Sq_chunk_t": p.get("Sq_chunk_t"),
                       "Sk_chunk_t": p.get("Sk_chunk_t"), "k_num_chunks": p.get("k_num_chunks"),
                       "b_per_core": p.get("b_per_core")}
              ok = (p["nh_per_core"] == 1 and p["q_per_core"] == 1 and p["bcast_batch"]
                    and not p["use_padded_mask"] and p["NKH"] == heads and p["NVH"] == heads)
          except Exception as error:
              terms, ok = {"error": f"{type(error).__name__}: {error}"[:300]}, False
          rows.append({"q_chunk": q_chunk, "k_chunk": k_chunk, "k_divides": padded % k_chunk == 0,
                       "qnc": qnc, "q_pf": q_pf, "split": list(split), "serves": ok, **terms})
    for t in (q, k, v, bias, out):
        ttnn.deallocate(t)
    return {"tokens": tokens, "grid": list(grid), "cores": cores,
            "q_split_max_s": TS._Q_SPLIT_MAX_S, "rows": rows}


def main() -> int:
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    out = {"mode": "af2ig_fill_preconditions_probe",
           "sizes": [probe(dev, t) for t in (208, 848)]}
    print(json.dumps(out, indent=1))
    Path("perf/pxdesign/tt_pxd_p8_plan_probe.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
