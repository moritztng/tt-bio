"""Upstream Mol-1 on CPU fp32: save what the trunk hands the confidence head (s_inputs, s_trunk, z_trunk, coords), then exit.

    python -I trunk_dump.py OUT.pt -- <mol/runner/inference.py arguments>
Diffusion may be cut short with --sample_diffusion.N_step: the trunk outputs do not depend on it.
"""
import os, sys, torch
out = sys.argv[1]
rest = [a for a in sys.argv[2:] if a != "--"]
torch.set_num_threads(10)
import mol.runner.inference as inf
from mol.model.mol import Mol
orig = Mol.run_confidence_head
def run_confidence_head(self, *a, **kw):
    torch.save({k: kw[k].detach().cpu() for k in ("s_inputs", "s_trunk", "z_trunk", "x_pred_coords")}, out)
    print("TRUNK-DUMPED", out, flush=True)
    os._exit(0)
Mol.run_confidence_head = run_confidence_head
sys.argv = ["inference.py"] + rest
inf.run()
