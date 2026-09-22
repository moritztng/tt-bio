"""Two controls the width sweep cannot give on its own.

(1) depth 16 and 48, to show the width difference stays at the fp32 rounding level rather than
    compounding into something that could explain a 3.487164x device ratio;
(2) the STRONGER control: same width, different PAD VALUES. The width comparison changes the
    reduction extent, so a tiny difference there is confounded with summation order. Changing only
    the pad CONTENT at a fixed width is not: if the real block is bit-identical, the pad provably
    does not reach it.
"""
import sys, json, torch
sys.path.insert(0, "/home/moritz/.coworker/wt/of3t-orchestrator")
from tt_bio import reference as R

TS, TZ, NREAL = 128, 64, 56


def build(nblocks):
    torch.manual_seed(99)
    m = R.PairformerModule(TS, TZ, nblocks, num_heads=8, dropout=0.0,
                           pairwise_head_width=16, pairwise_num_heads=4).eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.abs().max() == 0:
                p.normal_(0.0, 0.02)
            if name.endswith("bias") and "norm" in name.lower():
                p.normal_(0.0, 0.5)
    return m


torch.manual_seed(1234)
s_real = torch.randn(1, NREAL, TS)
z_real = torch.randn(1, NREAL, NREAL, TZ)


def run(model, W, fill):
    s = torch.full((1, W, TS), fill); s[:, :NREAL] = s_real
    z = torch.full((1, W, W, TZ), fill); z[:, :NREAL, :NREAL] = z_real
    mask = torch.zeros(1, W); mask[:, :NREAL] = 1.0
    with torch.no_grad():
        return model(s, z, mask, mask[:, :, None] * mask[:, None, :])


out = {"width_sweep": [], "pad_value_control": []}
for d in (16, 48):
    m = build(d)
    a_s, a_z = run(m, 64, 0.7); b_s, b_z = run(m, 384, 0.7)
    ar, br = a_s[:, :NREAL].double(), b_s[:, :NREAL].double()
    az = a_z[:, :NREAL, :NREAL].double(); bz = b_z[:, :NREAL, :NREAL].double()
    r = {"blocks": d, "pad_out_norm": float(b_s[:, NREAL:].double().norm()),
         "s_real_rel": float((ar - br).norm() / br.norm()),
         "z_real_rel": float((az - bz).norm() / bz.norm())}
    out["width_sweep"].append(r)
    print("WIDTH  blocks %2d  pad_out %.4e  s_real rel %.6e  z_real rel %.6e"
          % (d, r["pad_out_norm"], r["s_real_rel"], r["z_real_rel"]), flush=True)

for d in (8, 48):
    m = build(d)
    a_s, a_z = run(m, 384, 0.7)
    b_s, b_z = run(m, 384, -3.1)          # same width, wildly different pad content
    ar, br = a_s[:, :NREAL].double(), b_s[:, :NREAL].double()
    az = a_z[:, :NREAL, :NREAL].double(); bz = b_z[:, :NREAL, :NREAL].double()
    pad_moved = float((a_s[:, NREAL:].double() - b_s[:, NREAL:].double()).norm())
    r = {"blocks": d, "pad_output_moved_between_arms": pad_moved,
         "s_real_absdiff": float((ar - br).norm()), "s_bit_identical": bool(torch.equal(ar, br)),
         "z_real_absdiff": float((az - bz).norm()), "z_bit_identical": bool(torch.equal(az, bz))}
    out["pad_value_control"].append(r)
    print("PADVAL blocks %2d  pad moved %.4e  s_real absdiff %.3e bitid %s  z_real absdiff %.3e bitid %s"
          % (d, pad_moved, r["s_real_absdiff"], r["s_bit_identical"],
             r["z_real_absdiff"], r["z_bit_identical"]), flush=True)
    assert pad_moved > 0, "pad did not move between arms -- vacuous control"

json.dump(out, open("/tmp/of3t/of3t-orchestrator/PADLEAK2.json", "w"), indent=2)
