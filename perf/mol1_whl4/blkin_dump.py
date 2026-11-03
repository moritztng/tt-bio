"""Upstream Mol-1 on CPU fp32: input s, output z (what the s ops read), output s of every PairformerBlock of the LAST trunk cycle, plus the
block objects' qualified names, saved at the confidence head (OUT_blkin.pt), then exit.

    python -I blkin_dump.py OUT -- <mol/runner/inference.py arguments>
"""
import os, sys, torch
from collections import deque
out = sys.argv[1]
rest = [a for a in sys.argv[2:] if a != "--"]
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
import mol.runner.inference as inf
from mol.model.mol import Mol
from mol.model.modules.pairformer import PairformerBlock
buf = deque(maxlen=int(os.environ.get("KEEP", "48")))
fwd = PairformerBlock.forward
def forward(self, s, z, *a, **kw):
    si = None if s is None else s.detach().clone()
    so, zo = fwd(self, s, z, *a, **kw)
    if so is not None:
        buf.append((id(self), si, zo.detach().clone(), so.detach().clone()))
    return so, zo
PairformerBlock.forward = forward
def run_confidence_head(self, *a, **kw):
    names = {id(m): n for n, m in self.named_modules()}
    torch.save([(names.get(i), si, zi, so) for i, si, zi, so in buf], out + "_blkin.pt")
    print("DUMPED", out, len(buf), flush=True)
    os._exit(0)
Mol.run_confidence_head = run_confidence_head
sys.argv = ["inference.py"] + rest
inf.run()
