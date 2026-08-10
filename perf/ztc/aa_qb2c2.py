#!/usr/bin/env python3
"""A/A bit-exactness floor for the c=256 Transition, and a repeat of the h=20/h=24 verdict.

`cwidth_qb2c2.py` found the pair-width (c=256) Transition is NOT bit-exact across chunk heights
(h=20 and h=24 vs production h=16, max_abs 0.015625). That claim is only worth anything if the
same height twice IS bit-exact, so this runs h=16 three times and diffs run-to-run before
re-taking the cross-height diff on the same weights and the same input.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
c, n = 256, 1024
torch.manual_seed(0)
sd = {"norm.weight": torch.ones(c), "norm.bias": torch.zeros(c),
      "fc1.weight": torch.randn(n, c) * 0.05, "fc2.weight": torch.randn(n, c) * 0.05,
      "fc3.weight": torch.randn(c, n) * 0.05}
inst = T.Transition(sd, ckc)
xt = torch.randn(1, 512, 512, c) * 0.5
prod = T.TRANSITION_H_CHUNK_SIZE
out = {"c": c, "shape": [1, 512, 512, c], "prod_h": prod, "ttnn": ttnn.__version__ if hasattr(ttnn,"__version__") else "0.68.0"}

def run(h):
    xz = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
    try:
        T.TRANSITION_H_CHUNK_SIZE = h
        y = inst(xz); r = ttnn.to_torch(y).clone(); ttnn.deallocate(y); return r
    finally:
        T.TRANSITION_H_CHUNK_SIZE = prod
        ttnn.deallocate(xz)

aa = [run(16) for _ in range(3)]
out["AA_h16_run1_vs_run2_torch_equal"] = bool(torch.equal(aa[0], aa[1]))
out["AA_h16_run1_vs_run3_torch_equal"] = bool(torch.equal(aa[0], aa[2]))
out["AA_h16_max_abs"] = float((aa[0].float() - aa[1].float()).abs().max())
print("  AA " + json.dumps({k: v for k, v in out.items() if k.startswith("AA")}), flush=True)

for h in (20, 24):
    y = run(h)
    d = y.float() - aa[0].float()
    rec = {"torch_equal_vs_h16": bool(torch.equal(y, aa[0])),
           "max_abs": float(d.abs().max()),
           "rel_rmsd": float(d.pow(2).mean().sqrt() / aa[0].float().pow(2).mean().sqrt()),
           "frac_elems_differing": float((d != 0).float().mean())}
    out[f"h{h}"] = rec
    print(f"  h={h} " + json.dumps(rec), flush=True)

Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print("wrote " + sys.argv[1], flush=True)
