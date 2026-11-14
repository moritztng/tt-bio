"""Upstream fp32 trunk block 43 (cycle 1): inputs/outputs of its single-track submodules (attention_pair_bias,
single_transition and the transition's LN / a / b / out linears). python -I blk_split.py OUT.pt -- <inference args>"""
import os, sys, torch
out = sys.argv[1]
rest = [a for a in sys.argv[2:] if a != "--"]
torch.set_num_threads(10)
import mol.runner.inference as inf
from mol.model.modules.pairformer import PairformerBlock
TARGET = int(os.environ.get("WHL3_CALL", "75"))
n = [0]
fwd = PairformerBlock.forward
def forward(self, s, z, *a, **kw):
    if self.c_s > 0:
        if n[0] == TARGET:
            res = {}
            def cap(name):
                def h(m, i, o):
                    res[name + ".in"] = [x.detach().clone() for x in i if torch.is_tensor(x)]
                    res[name + ".out"] = o.detach().clone()
                return h
            st = self.single_transition
            hs = [self.attention_pair_bias.register_forward_hook(cap("apb")), st.register_forward_hook(cap("tr")),
                  st.layernorm1.register_forward_hook(cap("tr.ln")), st.linear_no_bias_a.register_forward_hook(cap("tr.a")),
                  st.linear_no_bias_b.register_forward_hook(cap("tr.b")), st.linear_no_bias.register_forward_hook(cap("tr.out"))]
            apb_kw = {}
            orig_apb = self.attention_pair_bias.forward
            def apb_fwd(*aa, **kk):
                res["apb.kw_a"] = kk.get("a").detach().clone(); res["apb.kw_z"] = kk.get("z").detach().clone()
                return orig_apb(*aa, **kk)
            self.attention_pair_bias.forward = apb_fwd
            so, _ = fwd(self, s, z, *a, **kw)
            res["s_in"] = s.detach().clone(); res["s_out"] = so.detach().clone()
            torch.save(res, out); print("SPLIT-DUMPED", out, flush=True); os._exit(0)
        n[0] += 1
    return fwd(self, s, z, *a, **kw)
PairformerBlock.forward = forward
sys.argv = ["inference.py"] + rest
inf.run()
