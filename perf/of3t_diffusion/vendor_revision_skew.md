# Our vendored openfold3 is not the revision this campaign compares against

`tt_bio/_vendor/openfold3` against `openfold3 0.5.0` (the tree `of3t-reference` built
BUNDLE-MIN with), 108 vendored .py files:

| | |
|---|---|
| identical | 41 |
| differ at all | 67 |
| **differ beyond the vendoring import rewrite** | **50 files, 1 989 lines** |

Most of the raw 2 495-line delta is `openfold3.core...` rewritten to
`tt_bio._vendor.openfold3.core...`, which is what vendoring does and means nothing.
Discounting those still leaves 50 files and 1 989 lines of real change, so the two trees are
different OF3 revisions rather than the same code in two places.

A worked example, because a count is not evidence on its own.
`core/model/structure/augmentation.py`, mean over atoms:

    ours     ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True)
    0.5.0    ) / torch.sum(atom_mask[..., None], dim=-2, keepdim=True).clamp(min=1)  # no 0-div

That is an upstream fix our vendored copy predates. `core/utils/relpos.py`, by contrast,
differs by exactly one import line and is substantively identical.

## Why this matters to instrument A

The diffusion transformer is NOT vendored, it is hand-written in `tt_bio`, so this does not
exhibit the specific change behind the DiT's measured 2.07e-02 per-block gap. What it does
establish is that our port was written against an earlier OF3 than the one the reference
bundle was generated with, which makes revision skew the leading explanation for that gap and
shifts the burden onto anyone claiming a defect in our port.

Settling it needs the revision our port targeted, then a diff of its
`core/model/layers/diffusion_transformer.py` against 0.5.0's. That is not done here.

Every per-parameter number in this campaign inherits this: instrument A currently compares our
port against a reference revision the port was never written against.
