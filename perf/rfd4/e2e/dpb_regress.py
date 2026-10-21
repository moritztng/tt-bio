"""dit_pair_bias at the shapes its other callers run, dumped for a bit-compare between two trees.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/dpb_regress.py --out <file.pt>

RFD3's DiT masks (c_z 128, 288 heads) and a one-head-tile c_z 128 case keep the held-rows path; RFD4's
c_z 384 group streams. A tree that only changed the streamed path must dump the first two unchanged.
"""
import argparse

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tt_bio import dit_pair_bias as D
    from tt_bio import tenstorrent as T

    dev = T.get_device()
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    g = torch.Generator().manual_seed(0)
    out = {}
    for name, c, h, n, scale in (("rfd3_dit", 128, 288, 128, 0.25), ("c128_h32", 128, 32, 128, 0.125),
                                 ("c384_h32", 384, 32, 128, 1.0)):
        z, wt = up(torch.randn(1, n, n, c, generator=g)), up(torch.randn(1, 1, h, c, generator=g) * 0.05)
        m = torch.zeros(1, 1, n, n)
        m[..., -8:] = -1e4
        m = up(m)
        if not D.eligible(z, wt, m):
            out[name] = None
            continue
        out[name] = ttnn.to_torch(D.dit_pair_bias(z, wt, m, scale))
        out[name + "_ref"] = ((ttnn.to_torch(z).double() @ ttnn.to_torch(wt).double()[0, 0].t()).permute(0, 3, 1, 2)
                              * scale + ttnn.to_torch(m).double())
    torch.save(out, a.out)
    for k, v in out.items():
        if v is not None and not k.endswith("_ref"):
            ref = out[k + "_ref"]
            print(k, list(v.shape), "rel_rms", float((v.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()))
        elif v is None:
            print(k, "not eligible")


if __name__ == "__main__":
    main()
