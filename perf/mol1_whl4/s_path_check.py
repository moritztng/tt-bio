"""ConfidenceHead._s_path (tt_bio/protenix.py, extracted, no ttnn) on a headblk.pt dump vs the float64 upstream stack.

    python -I s_path_check.py CKPT DUMP_DIR TT_BIO_PROTENIX_PY
Biases from the TT stack's own per-block z output, LN_z + linear_nobias_z in fp32, scaled by sqrt(head_dim) and
rounded to bf16 as the device hands them over. Prints pLDDT of the host s path and its s rel err vs float64.
"""
import ast
import sys
import textwrap
from pathlib import Path

import torch
import torch.nn.functional as F

ckpt, d, src = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.argv = [sys.argv[0], ckpt, str(d), "none"]
tree = ast.parse(open(src).read())
cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "ConfidenceHead")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_s_path")
ns = {}
exec(textwrap.dedent(ast.get_source_segment(open(src).read(), fn)), ns)
here = Path(__file__).resolve().parent
hc = {}
exec(compile(open(here / "head_chain.py").read().split("\nref = chain")[0], "head_chain", "exec"), hc)
sd, pre, recs, N = hc["sd"], hc["pre"], hc["recs"], hc["N"]


class H:
    _g = staticmethod(lambda k: sd[pre + k].float())
    _bias_scale = [24 ** 0.5] * len(recs)


biases = []
for i, r in enumerate(recs):
    a = f"pairformer_stack.blocks.{i}.attention_pair_bias."
    z = r[3].reshape(-1, *r[3].shape[-3:])[:, :N, :N].float()
    zn = F.layer_norm(z, (z.shape[-1],), H._g(a + "layernorm_z.weight"), H._g(a + "layernorm_z.bias"))
    biases.append((F.linear(zn, H._g(a + "linear_nobias_z.weight")) * 24 ** 0.5).permute(0, 3, 1, 2).to(torch.bfloat16))
s0 = recs[0][0].reshape(-1, recs[0][0].shape[-1])[:N].float()
with torch.no_grad():
    s = ns["_s_path"](H, s0, biases).double()
    ref = hc["chain"]("none")
print(f"host s path plddt {hc['plddt'](s):.4f}  float64 {hc['plddt'](ref):.4f}  s rel err {float((s - ref).norm() / ref.norm()):.3g}")
