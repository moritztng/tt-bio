"""Which step of the WH single Transition loses trunk block 43's per-token part, against float64.

    python -I tr_emul.py CKPT UP_blkin.pt
Block input from blkin_dump.py (last cycle), APB in float64, then the Transition with ONE (or a named set of)
device effects switched on: W = bf16 weights, Sin = Transition input rounded to bf16 (`_s_compute`), Sout = the
residual stored bf16, C = centred bf16 matmul inputs (`_centred_linear`), P = plain bf16 matmul inputs,
M = LayerNorm mean/var inputs truncated to tf32 (WH FPU source width). Metric: rel err of the 8 small
channels' token part vs float64.
"""
import sys, torch
from mol.model.modules.pairformer import PairformerBlock
torch.set_num_threads(6)
ckpt, blk = sys.argv[1:3]
sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
recs = torch.load(blk, weights_only=False)
bf = lambda x: x.to(torch.bfloat16).double()
def tf32(x):
    i = x.float().view(torch.int32)
    return (i & ~0x1FFF).view(torch.float32).double()
def mm_in(x, fx):
    if "C" in fx:
        c = x[..., :1, :]; return bf(c) + bf(x - c)
    if "P" in fx:
        return bf(x)
    return x
def tok(x): return x - x.mean(0)
small = recs[-1][3].reshape(-1, recs[-1][3].shape[-1]).abs().amax(0) < 512
VARIANTS = ["", "W", "Sin", "Sout", "C", "P", "M", "W C", "Sin C", "W Sin C", "W Sin C Sout", "W C M", "W Sin C M Sout"]
for i, (name, si, zo, so) in enumerate(recs):
    if name.split(".")[-1] not in ("40", "43"):
        continue
    pre = next(p for p in ("module." + name + ".", name + ".") if any(k.startswith(p) for k in sd))
    w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    c_z = w["attention_pair_bias.layernorm_z.weight"].shape[0]
    b = PairformerBlock(n_heads=16, c_z=c_z, c_s=384, hidden_scale_up=True)
    b.load_state_dict(w, strict=True); b.eval().double()
    t = b.single_transition
    s, z = si.double(), zo.double()
    with torch.no_grad():
        x0 = s + b.attention_pair_bias(a=s, s=None, z=z)
    def run(fx):
        fx = fx.split()
        W = (lambda p: bf(p)) if "W" in fx else (lambda p: p.double())
        with torch.no_grad():
            x = bf(x0) if "Sin" in fx else x0
            xm = tf32(x) if "M" in fx else x
            mu = xm.mean(-1, keepdim=True); xc = x - mu
            var = ((tf32(xc) if "M" in fx else xc) ** 2).mean(-1, keepdim=True)
            y = xc / torch.sqrt(var + 1e-5) * W(t.layernorm1.weight) + W(t.layernorm1.bias)
            yi = mm_in(y, fx)
            a = torch.nn.functional.silu(yi @ W(t.linear_no_bias_a.weight).T)
            h = a * (yi @ W(t.linear_no_bias_b.weight).T)
            out = x0 + mm_in(h, fx) @ W(t.linear_no_bias.weight).T
            if "Sout" in fx:
                out = bf(out)
        return out.reshape(-1, out.shape[-1])
    ref = run("")
    def e(x): return float((tok(x[:, small]) - tok(ref[:, small])).norm() / tok(ref[:, small]).norm())
    up = so.reshape(-1, so.shape[-1]).double()
    print(f"block {name.split('.')[-1]}: fp32-vs-64 {e(up):.3g}; ref small-ch |mean| {ref[:, small].mean(0).abs().tolist()} "
          f"token std {tok(ref[:, small]).std(0).tolist()}", flush=True)
    for v in VARIANTS[1:]:
        print(f"  {v:>16s}  {e(run(v)):.4g}", flush=True)
