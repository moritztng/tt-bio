"""The padded trunk's residual write as the card runs it under pair_bfp8z: the masked update typecast to bfloat8_b,
then the sum packed again (`trunk._residual`), against the model's single rounding of the sum. Random-weight stack of
test_trunk_stack, card-free.   python perf/rfd4/z8/cpu_pad.py --out <json> [--cases 4x2,14x6]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "perf" / "rfd4" / "z8")]
import torch  # noqa: E402
import rfd4_reference as REF  # noqa: E402
from zprec_cpu import bfp8, mode  # noqa: E402

DIM, DIM_PAIR, DIM_COND, HEADS = 768, 384, 256, 12
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="4x2,14x6")
a = ap.parse_args()
orig = REF.pair_update


def twice(w, p, x, z, mask, rnd=REF.ident, pair_write="hadamard"):
    """pair_update with each update rounded into bfp8 blocks before the add, as `_residual` does on the card."""
    if not getattr(rnd, "twice", False):
        return orig(w, p, x, z, mask, rnd, pair_write)
    pm = mask[:, None, :] * mask[:, :, None]
    res, u8 = rnd.z, lambda t: bfp8(t * pm[..., None])
    x = x * mask[..., None]
    h = rnd(REF.layer_norm(x, w[f"{p}.layer_norm_in.weight"], w[f"{p}.layer_norm_in.bias"]))
    pi, pj = rnd(h @ w[f"{p}.linear_x.weight"].T).chunk(2, dim=-1)
    z = res(z + u8(rnd(pi[:, None, :, :] * pj[:, :, None, :])))
    z = res(z + u8(REF.tri_mult(w, f"{p}.tri_mult_out", z, pm, True, rnd))) * pm[..., None]
    z = res(z + u8(REF.tri_mult(w, f"{p}.tri_mult_in", z, pm, False, rnd))) * pm[..., None]
    z = res(z + u8(REF.pair_transition(w, f"{p}.transition_out", z, pm, rnd))) * pm[..., None]
    return z


REF.pair_update = twice
rel = lambda g, r, m: ((g[m] - r[m]).pow(2).mean().sqrt() / r[m].pow(2).mean().sqrt()).item()  # noqa: E731
res, t0 = {}, time.time()
for case in a.cases.split(","):
    nl, cy = (int(v) for v in case.split("x"))
    w = REF.trunk_weights(nl, DIM, DIM_PAIR, DIM_COND, HEADS, seed=11)
    wb = {k: REF.bf16(v) for k, v in w.items()}
    s0, z0, _, mask = REF.block_inputs(1, 128, DIM, DIM_PAIR, DIM_COND, lengths=[96], seed=12)
    g = torch.Generator().manual_seed(13)
    conds = [torch.randn(1, 128, DIM_COND, generator=g, dtype=torch.float64) for _ in range(cy)]
    rs, rz = REF.trunk(w, s0, z0, conds, mask, nl, n_heads=HEADS)
    pm = (mask[:, :, None] * mask[:, None, :]).bool()
    tw = mode(False, True)
    tw.twice = True
    res[case] = {}
    for name, rnd in (("z8", mode(False, True)), ("z8_twice", tw)):
        _, z_ = REF.trunk(wb, REF.bf16(s0), REF.bf16(z0), [REF.bf16(c) for c in conds], mask, nl, n_heads=HEADS, rnd=rnd)
        res[case][name] = rel(z_, rz, pm)
        print(case, name, res[case][name], f"{time.time() - t0:.0f} s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
