"""Upstream fp32 trunk block 43 (cycle 1) rerun on perturbed inputs: is WH's error input sensitivity or WH's block math?

    python -I blk_probe.py OUT.pt TT_LAYERS.pt -- <mol/runner/inference.py arguments>
Call 75 of PairformerBlock (upstream snapshot index 75 = cycle 1, stack block 43). Saves s_in/z_in/out per variant.
"""
import os, sys, torch
out, ttp = sys.argv[1], sys.argv[2]
rest = [a for a in sys.argv[3:] if a != "--"]
torch.set_num_threads(10)
import mol.runner.inference as inf
from mol.model.modules.pairformer import PairformerBlock
TARGET = int(os.environ.get("WHL3_CALL", "75"))
tt = torch.load(ttp, weights_only=False)
wh_in = tt[int(os.environ.get("WHL3_TT_IN", "90"))]
n = [0]
fwd = PairformerBlock.forward
bf = lambda x: x.to(torch.bfloat16).to(x.dtype)
def forward(self, s, z, *a, **kw):
    if self.c_s > 0:
        if n[0] == TARGET:
            res = {"s_in": s.detach().clone(), "z_in": z.detach().clone(), "wh_in": wh_in.clone()}
            wh = wh_in.to(s.dtype).reshape(s.shape)
            for name, si, zi in (("ref", s, z), ("s_wh", wh, z), ("s_bf", bf(s), z), ("z_bf", s, bf(z)),
                                 ("sz_bf", bf(s), bf(z)), ("s_wh_z_bf", wh, bf(z))):
                so, _ = fwd(self, si.clone(), zi.clone(), *a, **kw)
                res[name] = so.detach().clone()
                print("VARIANT", name, flush=True)
            torch.save(res, out)
            print("PROBE-DUMPED", out, flush=True)
            os._exit(0)
        n[0] += 1
    return fwd(self, s, z, *a, **kw)
PairformerBlock.forward = forward
sys.argv = ["inference.py"] + rest
inf.run()
