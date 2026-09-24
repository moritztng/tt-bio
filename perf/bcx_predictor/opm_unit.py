"""Why did the repeat-free contraction disagree? One op, one comparison, no 48-block stack.

_small_depth gives b the I batch with ttnn.repeat, which has no tape verb. The transposed
form -- [I, c_z, D] x [D, J] so a 2D in1 broadcasts over in0's batch -- is the same algebra
and read 0.606 against 0.151 through the full stack. This scores both against a torch
float64 contraction of the same numbers, so whichever is wrong is named rather than guessed.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)
import numpy as np, torch, ttnn
import afgrad as A, stack as S
from tt_bio.af2 import AF2MaskedOuterProductMean

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
opm = dm.device_evoformer[0].opm
assert isinstance(opm, AF2MaskedOuterProductMean), type(opm)

I = J = 64
rng = np.random.default_rng(0)
Sn, C = 2, int(opm.a_weight.shape[-1])
a_np = rng.standard_normal((Sn, I, C)).astype(np.float32) * 0.3
b_np = rng.standard_normal((Sn, J, C)).astype(np.float32) * 0.3

def up(x):
    return ttnn.from_torch(torch.from_numpy(x), layout=ttnn.TILE_LAYOUT,
                           device=dev.device, dtype=ttnn.bfloat16)
def down(t, shape):
    x = torch.Tensor(ttnn.to_torch(t)).float()
    while x.dim() > len(shape) and x.shape[0] == 1:
        x = x.squeeze(0)
    return x.reshape(shape).double()

# float64 truth: proj_o(sum_s a_s (x) b_s) + o_bias
w = torch.Tensor(ttnn.to_torch(opm._proj_o_folded(C, C))).double()   # [C, C*c_z]
o_b = torch.Tensor(ttnn.to_torch(opm.o_bias)).double().reshape(-1)
c_z = w.shape[1] // C
at, bt = torch.from_numpy(a_np).double(), torch.from_numpy(b_np).double()
Aw = torch.einsum("sic,cdk->sidk", at, w.reshape(C, C, c_z))
truth = torch.einsum("sidk,sjd->ijk", Aw, bt) + o_b

out = {"I": I, "J": J, "S": Sn, "C": C, "c_z": int(c_z)}
res = {}
sd = opm._small_depth(up(a_np), up(b_np), 1)
res["small_depth_repeat"] = down(sd, (I, J, c_z))
if hasattr(opm, "_sum_rows"):
    res["sum_rows_transposed"] = down(opm._sum_rows(up(a_np), up(b_np)), (I, J, c_z))

for name, got in res.items():
    d = (got - truth).flatten(); t = truth.flatten()
    out[name] = {"rel_l2": float(d.norm()/t.norm()),
                 "cos": float(got.flatten()@t/(got.flatten().norm()*t.norm())),
                 "norm_got": float(got.norm()), "norm_truth": float(t.norm())}
(HERE / "opm_unit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
