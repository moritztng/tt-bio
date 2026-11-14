"""The released-trunk 14x6 golden test (tests/test_rfd4_trunk_golden.py::test_released_trunk_14x6_recycles), card-free:
float64 vs the same chain with the pair stream stored as main stores it (bf16) and as pair_bfp8z stores it (bfp8, round
to nearest, zprec_cpu.bfp8), s fp32 and bf16 at every op boundary in both. What the device's 14x6 reading should be
compared to.

    python perf/rfd4/z8/cpu14x6.py --out <json> [--cycles 6]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "perf" / "rfd4" / "z8")]
import torch  # noqa: E402
import rfd4_reference as REF  # noqa: E402
from zprec_cpu import GOLDENS, HEADS, mode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cycles", type=int, default=6)
a = ap.parse_args()
w = {k: v.double() for k, v in torch.load(GOLDENS / "trunk-weights" / "flow_trunk.pt", map_location="cpu",
                                          weights_only=False)["weights"].items()}
d = torch.load(GOLDENS / "unconditional_128" / "MultiheadAttnAndTransition__nn_transformer_layers_1__call0.pt",
               map_location="cpu", weights_only=False)
x, z, cond, mask = [t.double() for t in d["args"]][:4]
conds = [cond] * a.cycles
rel = lambda g, r, m: ((g[m] - r[m]).pow(2).mean().sqrt() / r[m].pow(2).mean().sqrt()).item()  # noqa: E731
m, pm = mask.bool(), (mask[:, :, None] * mask[:, None, :]).bool()
t0 = time.time()
rs, rz = REF.trunk(w, x, z, conds, mask, 14, n_heads=HEADS)
wb = {k: REF.bf16(v) for k, v in w.items()}
res = {"cycles": a.cycles, "tokens": int(x.shape[1])}
for name, rnd in (("bf16_mixed", REF.bf16_mixed), ("main", mode(False, False)), ("z8", mode(False, True))):
    s_, z_ = REF.trunk(wb, x, z, conds, mask, 14, n_heads=HEADS, rnd=rnd)
    res[name] = {"s": rel(s_, rs, m), "z": rel(z_, rz, pm)}
    print(name, res[name], f"{time.time() - t0:.0f} s", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
