#!/usr/bin/env python3
"""Where a last-dim ttnn.concat of sub-tile pieces in TILE layout puts its data.

move_ab.py found `_join_slices` wrong (not bit-equal) for a leading-gap join and for two width-16 parts,
both plain ttnn.concat paths that predate this branch. Per piece order, this prints which output columns
match which input, and the same concat in row-major layout (tile ttnn.pad refuses front padding).
    python perf/bc4_move/concat_diag.py out.json
"""
import json, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T
    dev = T.get_device()
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    g = torch.Generator().manual_seed(0)
    a = torch.randn(2, 4, 64, 16, generator=g).to(torch.bfloat16)
    b = torch.randn(2, 4, 64, 16, generator=g).to(torch.bfloat16)
    z = torch.zeros_like(a)
    out = {}

    def where(o, pieces):  # per 16-wide column block of o: which named piece it equals, else max |diff| to ref
        res = []
        for k in range(o.shape[-1] // 16):
            blk = o[..., 16 * k:16 * (k + 1)]
            hit = [n for n, p in pieces.items() if torch.equal(blk, p)]
            res.append(hit or f"none (max {blk.abs().max().item():.3g})")
        return res

    named = {"a": a, "b": b, "zeros": z}
    for label, order in [("a|zeros", "a z"), ("zeros|a", "z a"), ("a|b", "a b"), ("a|b|zeros", "a b z"),
                         ("zeros|a|b", "z a b")]:
        ts = [{"a": a, "b": b, "z": z}[c] for c in order.split()]
        ref = torch.cat(ts, -1)
        o = ttnn.to_torch(ttnn.concat([up(t) for t in ts], dim=3)).reshape(ref.shape)
        out[label] = {"bit_equal": bool(torch.equal(o, ref)), "blocks": where(o, named)}
        o = ttnn.to_torch(ttnn.concat([ttnn.to_layout(up(t), ttnn.ROW_MAJOR_LAYOUT) for t in ts], dim=3))
        out[label + " row-major"] = {"bit_equal": bool(torch.equal(o.reshape(ref.shape), ref))}
    print(json.dumps(out, indent=1))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
