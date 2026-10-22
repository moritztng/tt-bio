#!/usr/bin/env python3
"""Where a last-dim ttnn.concat of sub-tile pieces in TILE layout puts its data.

move_ab.py found `_join_slices` wrong (not bit-equal) for a leading-gap join and for two width-16 parts,
both plain ttnn.concat paths that predate this branch. Per piece order, this prints which output columns
match which input, and the same concat in row-major layout (tile ttnn.pad refuses front padding), at a small
shape and at the probe's [224, 4, 224, 16]; then the two `_join_slices` joins themselves.
    python perf/bc4_move/concat_diag.py out.json
"""
import json, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def where(o, pieces):  # per 16-wide column block of o: which named piece it equals, else its max |value|
    res = []
    for k in range(o.shape[-1] // 16):
        blk = o[..., 16 * k:16 * (k + 1)]
        hit = [n for n, p in pieces.items() if torch.equal(blk, p)]
        res.append(hit or f"none (max {blk.abs().max().item():.3g})")
    return res


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    dev = T.get_device()
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    out = {}
    for shape in [(2, 4, 64, 16), (224, 4, 224, 16)]:
        g = torch.Generator().manual_seed(0)
        a = torch.randn(shape, generator=g).to(torch.bfloat16)
        b = torch.randn(shape, generator=g).to(torch.bfloat16)
        z = torch.zeros_like(a)
        named, lead = {"a": a, "b": b, "zeros": z}, shape[:3]
        for label, order in [("a|zeros", "a z"), ("zeros|a", "z a"), ("a|b", "a b"), ("a|b|zeros", "a b z"),
                             ("zeros|a|b", "z a b")]:
            ts = [{"a": a, "b": b, "z": z}[c] for c in order.split()]
            ref = torch.cat(ts, -1)
            o = ttnn.to_torch(ttnn.concat([up(t) for t in ts], dim=3)).reshape(ref.shape)
            out[f"{shape} {label}"] = {"bit_equal": bool(torch.equal(o, ref)), "blocks": where(o, named)}
            o = ttnn.to_torch(ttnn.concat([ttnn.to_layout(up(t), ttnn.ROW_MAJOR_LAYOUT) for t in ts], dim=3))
            out[f"{shape} {label} row-major"] = {"bit_equal": bool(torch.equal(o.reshape(ref.shape), ref))}
        for label, parts, width, ref in [
                ("join lead gap", [([0, 0, 0, 16], [*lead, 32], a)], 32, [z, a]),
                ("join two parts tail", [([0, 0, 0, 0], [*lead, 16], a), ([0, 0, 0, 16], [*lead, 32], b)], 64,
                 [a, b, z, z])]:
            o = ag._join_slices([(s, e, up(t)) for s, e, t in parts], [*lead, width])[0]
            o = ttnn.to_torch(o).reshape(*lead, width)
            out[f"{shape} {label}"] = {"bit_equal": bool(torch.equal(o, torch.cat(ref, -1))), "blocks": where(o, named)}
    print(json.dumps(out, indent=1))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
