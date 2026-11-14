"""tests/test_rfd4_trunk_parity.py::test_trunk_stack (random weights, 96 of 128 tokens real), card-free: float64 vs the
chain with the pair stream stored bf16 (main) and bfp8 (pair_bfp8z, zprec_cpu.bfp8 round to nearest).

    python perf/rfd4/z8/cpu_stack.py --out <json> [--cases 4x2,14x6]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "perf" / "rfd4" / "z8")]
import torch  # noqa: E402
import rfd4_reference as REF  # noqa: E402
from zprec_cpu import mode  # noqa: E402

DIM, DIM_PAIR, DIM_COND, HEADS = 768, 384, 256, 12
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="4x2,14x6")
a = ap.parse_args()
rel = lambda g, r, m: ((g[m] - r[m]).pow(2).mean().sqrt() / r[m].pow(2).mean().sqrt()).item()  # noqa: E731
pcc = lambda g, r, m: torch.corrcoef(torch.stack([g[m].flatten(), r[m].flatten()]))[0, 1].item()  # noqa: E731
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
    res[case] = {}
    for name, rnd in (("bf16_mixed", REF.bf16_mixed), ("main", mode(False, False)), ("z8", mode(False, True))):
        s_, z_ = REF.trunk(wb, REF.bf16(s0), REF.bf16(z0), [REF.bf16(c) for c in conds], mask, nl, n_heads=HEADS, rnd=rnd)
        res[case][name] = {"s": rel(s_, rs, mask.bool()), "z": rel(z_, rz, pm), "z_pcc": pcc(z_, rz, pm)}
        print(case, name, res[case][name], f"{time.time() - t0:.0f} s", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
