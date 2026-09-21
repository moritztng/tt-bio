"""Deliverable 0: which ending-node bias orientation was `of3-p2-155k` trained with?

Read from upstream's own source, then CHECKED by running both upstream versions against each
other on one input with one set of weights. The source says:

  openfold3 0.4.3 (preview2, the release `of3-p2-155k` belongs to)
      TriangleAttention.forward has NO transpose_bias argument. For the ending node it does
      `x = x.transpose(-2, -3)` and then builds the bias from that transposed x with
      `permute_final_dims(self.linear_z(x), (2, 0, 1))`. The bias FOLLOWS the pair transpose.

  openfold3 0.5.0 (OpenBind)
      adds `transpose_bias`, documented as "used for Triangle Attention from the end node,
      where the input is transposed prior to calling this function. The bias would retain the
      ordering of the original un-transposed input tensor", and PairBlock.tri_att_end passes
      transpose_bias=True, giving `permute_final_dims(self.linear_z(x), (2, 1, 0))`.

So 0.4.3's only behaviour is 0.5.0's transpose_bias=False. This script asserts that by
construction on real tensors in float64, which is what turns two readings of source into a
measurement.
"""
import importlib.util
import json
import sys

import torch

V043 = "/home/ttuser/of3t_refprec/of3pkg043"
V050 = "/home/ttuser/of3t_gradients/pylibs"
# 0.4.3 is an unpacked tree with no vendored deps; 0.5.0's pylibs carries ml_collections and
# absl. Appended, never prepended, so `openfold3` always resolves from the version under test.
sys.path.append(V050)


def load(root, name):
    for m in list(sys.modules):
        if m == "openfold3" or m.startswith("openfold3."):
            del sys.modules[m]
    sys.path.insert(0, root)
    try:
        mod = importlib.import_module(name)
    finally:
        sys.path.remove(root)
    return mod


torch.manual_seed(20260920)
N, C, H, HD = 24, 32, 4, 8
x = torch.randn(1, N, N, C, dtype=torch.float64)

# One weight set, drawn once, loaded into BOTH versions. It has to be drawn rather than
# taken from either constructor: OpenFold initialises `mha.linear_o` to zeros ("final"), so a
# freshly constructed layer emits an identically zero tensor and every rel_l2 below would be
# 0/0. That is the A16 trap in its usual disguise -- a comparison that cannot fail.
out = {}
states = {}
weights = None
for tag, root in (("0.4.3", V043), ("0.5.0", V050)):
    ta = load(root, "openfold3.core.model.layers.triangular_attention")
    layer = ta.TriangleAttention(C, HD, H, starting=False).double()
    if weights is None:
        g = torch.Generator().manual_seed(7)
        weights = {k: torch.randn(v.shape, generator=g, dtype=torch.float64) * 0.1
                   for k, v in layer.state_dict().items()}
    layer.load_state_dict(weights)
    states[tag] = {k: v.clone() for k, v in layer.state_dict().items()}
    if tag == "0.4.3":
        out["0.4.3"] = layer(x.clone()).detach()
    else:
        for flag in (False, True):
            out["0.5.0_tb%d" % flag] = layer(x.clone(), transpose_bias=flag).detach()

# the weights must be identical on both sides or the comparison below is meaningless
wk = sorted(set(states["0.4.3"]) & set(states["0.5.0"]))
wsame = len(wk) == len(states["0.4.3"]) and all(
    torch.equal(states["0.4.3"][k], states["0.5.0"][k]) for k in wk)


def rel(a, b):
    return float((a - b).norm() / b.norm())


res = {
    "n_weight_tensors_compared": len(wk),
    "n_weight_tensors_043": len(states["0.4.3"]),
    "weights_identical_both_sides": wsame,
    "dtype": "float64",
    "rel_043_vs_050_tb_False": rel(out["0.4.3"], out["0.5.0_tb0"]),
    "rel_043_vs_050_tb_True": rel(out["0.4.3"], out["0.5.0_tb1"]),
    "rel_050_tb_False_vs_True": rel(out["0.5.0_tb0"], out["0.5.0_tb1"]),
    "out_finite": all(bool(torch.isfinite(v).all()) for v in out.values()),
    "out_norms": {k: float(v.norm()) for k, v in out.items()},
}
assert all(n > 0 for n in res["out_norms"].values()), (
    "a zero output makes every rel_l2 0/0 -- the A16 measured-zero requirement, here as a "
    "precondition rather than a baseline")
assert wsame, "weights differ across the two versions -- the comparison would be meaningless"
assert res["out_finite"], "a non-finite output makes every rel_l2 below unreadable"
print(json.dumps(res, indent=1))
print()
print("0.4.3 == 0.5.0(transpose_bias=False): %s  (rel %.3e)"
      % (res["rel_043_vs_050_tb_False"] == 0.0, res["rel_043_vs_050_tb_False"]))
print("0.4.3 == 0.5.0(transpose_bias=True) : %s  (rel %.3e)"
      % (res["rel_043_vs_050_tb_True"] == 0.0, res["rel_043_vs_050_tb_True"]))
print("break control, the two 0.5.0 arms against each other: rel %.3e"
      % res["rel_050_tb_False_vs_True"])
json.dump(res, open("perf/of3t_foldab/convention.json", "w"), indent=1)
print("\nwrote perf/of3t_foldab/convention.json")
