"""Does the PAD EXTENT change the REAL block in our torch reference?

`of3t-modelboundary` measured the trunk gradient norm at 12.3912543630 with 8 pad rows (c64) and
43.2103398400 with 328 (n384), 3.487164x, on real inputs verified bit-identical between the two
boundaries. If the model is correctly masked, the real block cannot see the pad, and that ratio
would have to come from the device rather than from the model's semantics. This asks torch.

Weights are random but deliberately NON-DEGENERATE: `of3t-ditmodel` found its own control vacuous
twice because a freshly built Transition has a zero LayerNorm bias and a zeroed fc3, so it returns
exactly 0 on a zero pad row whatever the masking does. Every LayerNorm bias and every final linear
here is forced non-zero, and the run asserts the pad output is non-zero before reporting anything.
"""
import sys, json, torch
sys.path.insert(0, "/home/moritz/.coworker/wt/of3t-orchestrator")
from tt_bio import reference as R

torch.manual_seed(1234)
TS, TZ, NREAL = 128, 64, 56
WIDTHS = [64, 384]
DEPTHS = [1, 2, 4, 8]


def build(nblocks):
    torch.manual_seed(99)
    m = R.PairformerModule(TS, TZ, nblocks, num_heads=8, dropout=0.0,
                           pairwise_head_width=16, pairwise_num_heads=4).eval()
    nz = 0
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.abs().max() == 0:                      # the vacuity trap, closed explicitly
                p.normal_(0.0, 0.02); nz += 1
            if name.endswith("bias") and "norm" in name.lower():
                p.normal_(0.0, 0.5); nz += 1
    return m, nz


# one fixed set of real tokens, reused at every width
s_real = torch.randn(1, NREAL, TS)
z_real = torch.randn(1, NREAL, NREAL, TZ)
pad_fill = 0.7                                          # pad carries GARBAGE, as the real pipeline does


def run(model, W):
    s = torch.full((1, W, TS), pad_fill); s[:, :NREAL] = s_real
    z = torch.full((1, W, W, TZ), pad_fill); z[:, :NREAL, :NREAL] = z_real
    mask = torch.zeros(1, W); mask[:, :NREAL] = 1.0
    pair_mask = mask[:, :, None] * mask[:, None, :]
    with torch.no_grad():
        so, zo = model(s, z, mask, pair_mask)
    return so, zo


out = {"what": "real-block forward at two pad widths, same real tokens, our torch reference",
       "real_tokens": NREAL, "widths": WIDTHS, "pad_fill": pad_fill, "rows": []}
for d in DEPTHS:
    m, nz = build(d)
    res = {w: run(m, w) for w in WIDTHS}
    a_s, a_z = res[WIDTHS[0]]
    b_s, b_z = res[WIDTHS[1]]
    ar_s, br_s = a_s[:, :NREAL].double(), b_s[:, :NREAL].double()
    ar_z = a_z[:, :NREAL, :NREAL].double(); br_z = b_z[:, :NREAL, :NREAL].double()
    pad_norm = float(b_s[:, NREAL:].double().norm())
    row = {"blocks": d, "forced_nonzero_params": nz,
           "pad_out_norm_n384": pad_norm,
           "s_real_absdiff": float((ar_s - br_s).norm()),
           "s_real_rel": float((ar_s - br_s).norm() / br_s.norm()),
           "s_bit_identical": bool(torch.equal(ar_s, br_s)),
           "z_real_absdiff": float((ar_z - br_z).norm()),
           "z_real_rel": float((ar_z - br_z).norm() / br_z.norm()),
           "z_bit_identical": bool(torch.equal(ar_z, br_z))}
    out["rows"].append(row)
    print("blocks %2d  pad_out %.4e  s_real rel %.6e (bitid %s)  z_real rel %.6e (bitid %s)"
          % (d, pad_norm, row["s_real_rel"], row["s_bit_identical"],
             row["z_real_rel"], row["z_bit_identical"]), flush=True)
    assert pad_norm > 0, "pad output is zero -- the control is vacuous, as of3t-ditmodel warned"

json.dump(out, open("/tmp/of3t/of3t-orchestrator/PADLEAK.json", "w"), indent=2)
print("\nnon-vacuity: pad output non-zero at every depth")
