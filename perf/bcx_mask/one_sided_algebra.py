"""Is masking the `a` half alone the same as AF2's both halves? float64, the shipped classes.

`tt_bio/af2.py`'s module docstring used to say a masked AF2 fold needs the both-halves form
written before a mask can be honoured, and that assertion held the padded regime shut. It is
wrong whenever `mask_2d` is an outer product, which is the only kind AF2 builds
(`mask_2d = seq_mask[:, None] * seq_mask[None, :]`).

`af2_reference.TriangleMultiplication` is AF2's own masking, verbatim. The one-sided arm is the
same module with `tt_bio/tenstorrent.py:7326`'s masking -- `a_chunk` alone -- and nothing else
changed. Both directions, random float64 weights, a random mask, and the distance taken over the
real block that the mask names and then over the whole tensor, because the claim is that they
agree on the first and disagree on the second.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import torch                                                           # noqa: E402

from tt_bio.af2_reference import TriangleMultiplication                 # noqa: E402

N, C, HID = 48, 16, 24
MASKED = [7, 8, 9, 30, 31, 32, 33, 47]


def one_sided(self, z, mask):
    """`tenstorrent.py:7326`: the mask multiplies `a_chunk`, and `b` is never masked."""
    x = self.norm_in(z)
    proj = self.p_in(x)
    a, b = proj.split(self.hidden, dim=-1)
    proj = torch.cat([mask.unsqueeze(-1).to(x.dtype) * a, b], dim=-1)
    proj = proj * torch.sigmoid(self.g_in(x))
    a, b = proj.split(self.hidden, dim=-1)
    equation = "kic,kjc->ijc" if self.ending else "ikc,jkc->ijc"
    act = self.p_out(self.norm_out(torch.einsum(equation, a, b)))
    return act * torch.sigmoid(self.g_out(x))


def main():
    torch.manual_seed(0)
    seq = torch.ones(N, dtype=torch.float64)
    seq[MASKED] = 0.0
    mask = seq[:, None] * seq[None, :]
    ones = torch.ones_like(mask)
    real = seq.bool()
    z = torch.randn(N, N, C, dtype=torch.float64)

    both = TriangleMultiplication.forward
    out = {"n": N, "n_masked": len(MASKED), "dtype": "float64",
           "mask_is_outer_product": True, "directions": {}}
    for ending in (False, True):
        mod = TriangleMultiplication(C, HID, ending=ending).double()
        for p in mod.parameters():
            with torch.no_grad():
                p.normal_()
        with torch.no_grad():
            af2 = mod(z, mask)
            TriangleMultiplication.forward = one_sided
            try:
                sided = mod(z, mask)
                unmasked = mod(z, ones)
            finally:
                TriangleMultiplication.forward = both
        sub = lambda t: t[real][:, real]
        out["directions"]["ending" if ending else "outgoing"] = {
            "one_sided_vs_af2_real_block_max_abs": float((sub(sided) - sub(af2)).abs().max()),
            "one_sided_vs_af2_whole_tensor_max_abs": float((sided - af2).abs().max()),
            "unmasked_vs_af2_real_block_max_abs": float((sub(unmasked) - sub(af2)).abs().max()),
            "af2_real_block_max_abs": float(sub(af2).abs().max()),
        }
    (HERE / "one_sided_algebra.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
