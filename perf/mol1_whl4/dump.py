"""Upstream Mol-1 on CPU fp32: the trunk single track after every PairformerBlock (OUT_layers.pt) and what the trunk
hands the confidence head (OUT_trunk.pt: s_inputs, s_trunk, z_trunk, x_pred_coords), then exit.

    python -I dump.py OUT -- <mol/runner/inference.py arguments>
"""
import os, sys, torch
out = sys.argv[1]
rest = [a for a in sys.argv[2:] if a != "--"]
torch.set_num_threads(int(os.environ.get("THREADS", "6")))
import mol.runner.inference as inf
from mol.model.mol import Mol
from mol.model.modules.pairformer import PairformerBlock
layers = []
fwd = PairformerBlock.forward
def forward(self, s, *a, **kw):
    so, zo = fwd(self, s, *a, **kw)
    if so is not None:
        layers.append(so.detach().float().reshape(-1, so.shape[-1]).clone())
    return so, zo
PairformerBlock.forward = forward
def run_confidence_head(self, *a, **kw):
    torch.save(layers, out + "_layers.pt")
    torch.save({k: kw[k].detach().cpu() for k in ("s_inputs", "s_trunk", "z_trunk", "x_pred_coords")}, out + "_trunk.pt")
    print("DUMPED", out, len(layers), flush=True)
    os._exit(0)
Mol.run_confidence_head = run_confidence_head
sys.argv = ["inference.py"] + rest
inf.run()
