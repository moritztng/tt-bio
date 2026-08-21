#!/usr/bin/env python3
"""Why _acc_concat(host=True) and _acc_concat(host=False) do not agree at the fold.

p2 measured the host-concat budget as NOT bit-exact at 768 aa (state doc §31.3). The concat
itself is claimed bit-identical, and the logical values are; this asks what ELSE differs
between the two branches: the result's memory_config, and the tile padding the host round trip
drops and re-zeroes.

Runs at a small tile-padded shape first (H=100 pads to 128) so the mechanism is cheap to read,
then at the real OpenDDE refiner channel-join shape.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T


def padded_view(t):
    """A logical view of t including its tile padding, or None if ttnn will not give one."""
    for f in (lambda: ttnn.reshape(t, tuple(int(d) for d in t.padded_shape)),
              lambda: ttnn.reshape(t, tuple(int(d) for d in t.shape.with_tile_padding()))):
        try:
            return f()
        except Exception:
            continue
    return None


def one(H, width, n_chunks, dim=-1):
    dev = T.get_device()
    torch.manual_seed(0)
    # Blocks produced by a matmul, as the channel loop produces them: whatever a ttnn matmul
    # leaves in the tile padding is what the device branch carries and the host branch drops.
    blocks_dev, blocks_host = [], []
    for i in range(n_chunks):
        a = ttnn.from_torch(torch.randn(1, H, H, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(1, 1, 64, width, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16)
        b = ttnn.matmul(a, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(a); ttnn.deallocate(w)
        blocks_dev.append(b)
        blocks_host.append(ttnn.to_torch(b))
    out_h = T._acc_concat(list(blocks_host), dim, True)
    out_d = T._acc_concat(list(blocks_dev), dim, False)

    th, td = ttnn.to_torch(out_h), ttnn.to_torch(out_d)
    r = {"H": H, "width": width, "n_chunks": n_chunks, "dim": dim,
         "logical_shape": [int(d) for d in out_d.shape],
         "padded_shape_host": [int(d) for d in out_h.padded_shape],
         "padded_shape_dev": [int(d) for d in out_d.padded_shape],
         "logical_equal": bool(torch.equal(th, td)),
         "logical_max_abs_diff": float((th.float() - td.float()).abs().max()),
         "memcfg_host": str(out_h.memory_config()), "memcfg_dev": str(out_d.memory_config()),
         "memcfg_equal": str(out_h.memory_config()) == str(out_d.memory_config()),
         "dtype_host": str(out_h.dtype), "dtype_dev": str(out_d.dtype)}

    ph, pd = padded_view(out_h), padded_view(out_d)
    if ph is not None and pd is not None:
        a, b = ttnn.to_torch(ph).float(), ttnn.to_torch(pd).float()
        r["padded_view_shape"] = list(a.shape)
        r["padded_equal"] = bool(torch.equal(a, b))
        r["padded_max_abs_diff"] = float((a - b).abs().max())
        # the pad rows only, along the tiled row axis
        if a.shape[-2] > r["logical_shape"][-2]:
            L = r["logical_shape"][-2]
            r["pad_rows_host_absmax"] = float(a[..., L:, :].abs().max())
            r["pad_rows_dev_absmax"] = float(b[..., L:, :].abs().max())
            r["pad_rows_host_nonzero"] = int((a[..., L:, :] != 0).sum())
            r["pad_rows_dev_nonzero"] = int((b[..., L:, :] != 0).sum())
    else:
        r["padded_view"] = "unavailable"
    # a reduction ALONG the padded axis: reads the pad lanes if ttnn does not mask them
    try:
        sh = ttnn.to_torch(ttnn.sum(out_h, dim=-2)).float()
        sd = ttnn.to_torch(ttnn.sum(out_d, dim=-2)).float()
        r["sum_over_padded_axis_equal"] = bool(torch.equal(sh, sd))
        r["sum_over_padded_axis_max_abs_diff"] = float((sh - sd).abs().max())
    except Exception as e:
        r["sum_over_padded_axis"] = f"threw: {type(e).__name__}: {e}"[:200]
    ttnn.deallocate(out_h); ttnn.deallocate(out_d)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shapes", default="small")
    a = ap.parse_args()
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}"
    cases = {"small": [(100, 64, 6)], "refiner": [(1494, 64, 6)],
             "both": [(100, 64, 6), (1494, 64, 6)]}[a.shapes]
    out = {"ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "?", "rows": []}
    for H, w, n in cases:
        try:
            out["rows"].append(one(H, w, n))
        except Exception as e:
            out["rows"].append({"H": H, "error": f"{type(e).__name__}: {e}"[:400]})
        print(json.dumps(out["rows"][-1], indent=1), flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    T.close_device() if hasattr(T, "close_device") else None


main()
