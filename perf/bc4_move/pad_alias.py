#!/usr/bin/env python3
"""Does a back pad that fits inside the last tile alias its input, and does it zero the tile padding?

move_ab.py lost bit-equality on the joins it checked after timing `_join_slices` on a single 16-wide part,
whose return it deallocated. If ttnn.pad 16 -> 32 only relabels the shape, that freed the probe input; and
if it does not write zeros, a producer that leaves non-zero tile padding (exp: exp(0) = 1) leaks into the pad.
    python perf/bc4_move/pad_alias.py
"""
import json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    dev = T.get_device()
    out = {}
    for shape in [(2, 4, 64, 16), (224, 4, 224, 16)]:
        t = torch.randn(shape, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        e = ttnn.exp(x)
        ref = torch.cat([ttnn.to_torch(e), torch.zeros_like(t)], -1)
        full = list(shape[:3]) + [32]
        r = {}
        for name, f in [("pad", lambda: ttnn.pad(e, [(0, 0)] * 3 + [(0, 16)], 0.0)),
                        ("join", lambda: ag._join_slices([([0] * 4, list(shape), e)], full)[0]),
                        ("concat", lambda: ttnn.concat([e, ag._zeros_like_along(e, 3, 16)], dim=3))]:
            o = f()
            got = ttnn.to_torch(o).reshape(ref.shape)
            r[name] = {"same_buffer": o.buffer_address() == e.buffer_address(),
                       "bit_equal": bool(torch.equal(got, ref)),
                       "tail_max": float(got[..., 16:].abs().max())}
        out[str(shape)] = r
        print(shape, r, flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
