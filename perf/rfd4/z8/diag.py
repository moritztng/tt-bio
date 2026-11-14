"""Where the bfp8 pair stream's device excess comes from: test_trunk_stack's random-weight 14-layer trunk, padded (96 of
128 tokens) or full (128 of 128), 1 or 6 cycles, z stored bf16 (main) or bfloat8_b (pair_bfp8z). Each device z is graded
against float64 and beside the float64 model of its own storage (zprec_cpu.mode), so device / model isolates what the
card adds.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/z8/diag.py --out <jsonl> [--cases pad1,pad6,full1,full6] [--nl 14]
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "perf" / "rfd4" / "z8")]
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch, ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio.rfd4 import trunk as TR  # noqa: E402
import rfd4_reference as REF  # noqa: E402
from zprec_cpu import mode  # noqa: E402

DIM, DIM_PAIR, DIM_COND, HEADS = 768, 384, 256, 12
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="pad1,pad6,full1,full6")
ap.add_argument("--nl", type=int, default=14)
a = ap.parse_args()
rel = lambda g, r, m: ((g[m] - r[m]).pow(2).mean().sqrt() / r[m].pow(2).mean().sqrt()).item()  # noqa: E731
pcc = lambda g, r, m: torch.corrcoef(torch.stack([g[m].flatten(), r[m].flatten()]))[0, 1].item()  # noqa: E731
up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.float().contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                                 device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
dev = T.get_device()
w = REF.trunk_weights(a.nl, DIM, DIM_PAIR, DIM_COND, HEADS, seed=11)
wb = {k: REF.bf16(v) for k, v in w.items()}
scope = T.WeightScope({k: v.float() for k, v in w.items()})
t0 = time.time()
for case in a.cases.split(","):
    pad, cy = case.startswith("pad"), int(case.lstrip("padful"))
    s0, z0, _, mask = REF.block_inputs(1, 128, DIM, DIM_PAIR, DIM_COND, lengths=[96 if pad else 128], seed=12)
    g = torch.Generator().manual_seed(13)
    conds = [torch.randn(1, 128, DIM_COND, generator=g, dtype=torch.float64) for _ in range(cy)]
    rs, rz = REF.trunk(w, s0, z0, conds, mask, a.nl, n_heads=HEADS)
    pm = (mask[:, :, None] * mask[:, None, :]).bool()
    for arm, zd, rnd in (("main", ttnn.bfloat16, mode(False, False)), ("z8", ttnn.bfloat8_b, mode(False, True))):
        _, mz = REF.trunk(wb, REF.bf16(s0), REF.bf16(z0), [REF.bf16(c) for c in conds], mask, a.nl, n_heads=HEADS, rnd=rnd)
        tr = TR.Trunk(scope, TR.compute_kernel_config(), a.nl, HEADS, residual_dtype=ttnn.float32, pair_residual_dtype=zd)
        masks = TR.trunk_masks(mask, dev, residual_dtype=ttnn.float32)
        s, z = tr(up(s0, ttnn.float32), up(z0, ttnn.float32), [up(c) for c in conds], masks)
        gz = ttnn.to_torch(z).double()
        row = {"case": case, "arm": arm, "z_dtype": str(z.dtype), "dev": rel(gz, rz, pm), "model": rel(mz, rz, pm),
               "dev_vs_model": rel(gz, mz, pm), "dev_pcc": pcc(gz, rz, pm), "s": time.time() - t0}
        row["ratio"] = row["dev"] / row["model"]
        print(json.dumps(row), flush=True)
        open(a.out, "a").write(json.dumps(row) + "\n")
        for t in (s, z):
            ttnn.deallocate(t)
