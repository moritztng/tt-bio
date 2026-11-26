"""PairUpdate sub-ops forward vs float64, one dtype: which part breaks (fp32 PairUpdate read 0.2-0.5 off, bf16 6e-3)."""
import argparse
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "tests")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--real", type=int, default=48)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    import rfd4_reference as REF
    from tt_bio import tenstorrent as tt
    from tt_bio.rfd4 import trunk as TR
    dev = tt.get_device()
    dt = ttnn.float32 if a.dtype == "fp32" else ttnn.bfloat16
    rd = (lambda t: t) if a.dtype == "fp32" else (lambda t: t.to(torch.bfloat16).to(torch.float64))  # noqa: E731
    x, z, _c, mask = (rd(t) for t in REF.block_inputs(1, a.n, 768, 384, 256, lengths=[a.real], seed=1))
    pm = mask[:, :, None] * mask[:, None, :]
    x, z = x * mask[..., None], z * pm[..., None]
    w = {k: rd(v.double()) for k, v in REF.pair_update_weights(768, 384, 256, seed=5, prefix="pu").items()}

    def up(t):
        return ttnn.from_torch(t.float().contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    def rel(dev_t, want, m):
        have = ttnn.to_torch(dev_t).double().reshape(want.shape)
        return float(((have - want) * m).norm() / (want * m).norm())

    with TR.lever_scope(set()):
        m = TR.PairUpdate(tt.WeightScope({k: v.float() for k, v in w.items()}).child("pu"), TR.compute_kernel_config(), dtype=dt)
        M = pm[..., None]
        zd, pmd, pmcd = up(z), up(pm), up(M)
        print(a.dtype, "tri_out", rel(m.tri_out(zd, pmd), REF.tri_mult(w, "pu.tri_mult_out", z, pm, True) * M, M), flush=True)
        print(a.dtype, "tri_in", rel(m.tri_in(zd, pmd), REF.tri_mult(w, "pu.tri_mult_in", z, pm, False) * M, M), flush=True)
        print(a.dtype, "transition", rel(m._transition(zd, pmcd), REF.pair_transition(w, "pu.transition_out", z, pm), M), flush=True)
        h = REF.layer_norm(x, w["pu.layer_norm_in.weight"], w["pu.layer_norm_in.bias"])
        pi, pj = (h @ w["pu.linear_x.weight"].T).chunk(2, dim=-1)
        u, _ = m._write(up(x), up(mask[..., None]))
        print(a.dtype, "write", rel(u, pi[:, None] * pj[:, :, None], M), flush=True)
        print(a.dtype, "whole", rel(m(up(x), zd, up(mask[..., None]), pmd, pmcd), REF.pair_update(w, "pu", x, z, mask), M), flush=True)


if __name__ == "__main__":
    main()
