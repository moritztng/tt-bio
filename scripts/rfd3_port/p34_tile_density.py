"""L1's screen: how many 32x32 tiles of the dense [1,4,3359,3360] atom-attention buffer
actually contain a neighbour?

L1 proposes to stop computing the tiles that hold no neighbour at all. That is bit-exact by
construction -- a dropped tile contributes exp(-inf)=0.0 to the softmax denominator for all
1024 of its entries, and adding exact zeros to a float sum in any grouping is exact, so the
surviving tiles keep both their internal accumulation order and their tile boundaries. This
is what p6's element-granular COMPACTION destroyed (trajectory PCC 0.890) and why only the
tile-granular form is live.

The lever is worth exactly the tile density, so this measures it before any device code gets
written. Two variants, because they degrade differently:

  * tile list  -- compute only the column tiles the row block's neighbours touch.
  * span       -- compute [min_col, max_col] rounded out to tile boundaries. Needs no new
                  kernel (it is a slice into the ops that already ship) and degrades
                  gracefully to the full width, where a tile list does not.

Pure host, no device. Reads the real trajectory dumped by p34_dump_traj.py, so the density
is measured at real coordinates at real diffusion steps rather than on a synthetic cloud.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tt_bio.rfd3.model import _create_attention_indices  # noqa: E402

TILE = 32


def densities(idx_row: torch.Tensor, L: int) -> tuple[float, float, float]:
    """(tile-list density, span density, mean union size) over the 32-row blocks.

    idx_row: [L, k] neighbour column indices per query row.
    Column axis is L padded up to a tile multiple, matching the device buffer.
    """
    n_col_tiles = -(-L // TILE)
    n_row_blocks = -(-L // TILE)
    touched_total = 0
    span_total = 0
    union_total = 0
    for b in range(n_row_blocks):
        block = idx_row[b * TILE:(b + 1) * TILE]
        cols = torch.unique(block.reshape(-1))
        union_total += cols.numel()
        ct = torch.unique(torch.div(cols, TILE, rounding_mode="floor"))
        touched_total += ct.numel()
        lo = int(cols.min()) // TILE
        hi = int(cols.max()) // TILE
        span_total += hi - lo + 1
    return (touched_total / (n_row_blocks * n_col_tiles),
            span_total / (n_row_blocks * n_col_tiles),
            union_total / n_row_blocks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="step indices to measure; default is a spread over the trajectory")
    ap.add_argument("--all", action="store_true", help="measure every step (slow, exact mean)")
    ap.add_argument("--n_keys", type=int, default=128)
    ap.add_argument("--n_seq", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = torch.load(a.traj, map_location="cpu", weights_only=False)
    X = d["X_noisy"]                        # [steps, L, 3]
    f = dict(d["f"])
    L = int(d["L"])
    n_steps = X.shape[0]
    tok_idx = f["atom_to_token_map"]
    n_chains = len(torch.unique(f["asym_id"])) if "asym_id" in f else 1
    print(f"[density] L={L} steps={n_steps} chains={n_chains} "
          f"n_keys={a.n_keys} n_seq={a.n_seq}", flush=True)

    if a.all:
        steps = list(range(n_steps))
    elif a.steps:
        steps = [s for s in a.steps if s < n_steps]
    else:
        steps = sorted({0, 1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180,
                        n_steps - 20, n_steps - 5, n_steps - 1})
        steps = [s for s in steps if 0 <= s < n_steps]

    rows = []
    for s in steps:
        idx = _create_attention_indices(f, X[s], tok_idx, a.n_keys, a.n_seq)
        idx_row = idx[0] if idx.ndim == 3 else idx
        dt, ds, us = densities(idx_row, L)
        rows.append({"step": s, "t_hat": float(d["t_hat"][s]), "tile_density": dt,
                     "span_density": ds, "union_size": us, "k_total": int(idx_row.shape[-1])})
        print(f"  step {s:3d}  sigma={float(d['t_hat'][s]):9.3f}  "
              f"tile_d={dt:.4f}  span_d={ds:.4f}  union={us:7.1f} atoms", flush=True)

    # The total win is set by the MEAN density over steps, since the indices are rebuilt
    # every step anyway and a per-step tile list/span costs nothing extra on host.
    mean_tile = sum(r["tile_density"] for r in rows) / len(rows)
    mean_span = sum(r["span_density"] for r in rows) / len(rows)
    worst_tile = max(r["tile_density"] for r in rows)
    worst_span = max(r["span_density"] for r in rows)
    elem_density = rows[-1]["k_total"] / (-(-L // TILE) * TILE)
    out = {"L": L, "n_steps": n_steps, "chains": n_chains, "sampled_steps": steps,
           "element_density": elem_density,
           "mean_tile_density": mean_tile, "mean_span_density": mean_span,
           "worst_tile_density": worst_tile, "worst_span_density": worst_span,
           "rows": rows, "traj": a.traj, "all_steps": bool(a.all)}
    print(f"\n[density] element density (the 3.8% figure)  = {elem_density:.4f}")
    print(f"[density] tile-list density  mean={mean_tile:.4f}  worst={worst_tile:.4f}")
    print(f"[density] span       density  mean={mean_span:.4f}  worst={worst_span:.4f}")
    # The kill gate, as written in the plan before the number existed.
    gate = ("STOP" if worst_tile > 0.30 else
            "SPAN-ONLY" if worst_tile > 0.10 else "BOTH-LIVE")
    print(f"[density] plan kill gate on the WORST step -> {gate}")
    out["gate_worst_step"] = gate
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"[density] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
