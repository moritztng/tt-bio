"""CPU emulation of trunk block 43's single Transition, small-channel token rel err vs upstream fp32 output:
plain bf16 storage vs 'centred' (each matmul input split into its across-token mean, fp32, and the centred rest, bf16;
elementwise in fp32), vs tf32 inputs. python -I centre_emul.py CKPT SPLIT.pt UP_LAYERS.pt"""
import sys, torch
import torch.nn.functional as F
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); sd = sd.get("model", sd)
pre = "module.pairformer_stack.blocks.43.single_transition."
lw, lb, wa, wb, wo = (sd[pre + k].double() for k in ("layernorm1.weight", "layernorm1.bias", "linear_no_bias_a.weight", "linear_no_bias_b.weight", "linear_no_bias.weight"))
r = torch.load(sys.argv[2], weights_only=False); up = torch.load(sys.argv[3], weights_only=False)
small = up[-1].abs().amax(0) < 512
x = r["tr.in"][0].double().reshape(-1, 384); ref = r["tr.out"].double().reshape(-1, 384)
bf = lambda t: t.to(torch.bfloat16).double(); f32 = lambda t: t.float().double()
tf32 = lambda t: (t.float().view(torch.int32) & ~0x1FFF).view(torch.float32).double()
tok = lambda t: t - t.mean(0); rel = lambda a, b: float((tok(a[:, small]) - tok(b[:, small])).norm() / tok(b[:, small]).norm())
W = {k: bf(v) for k, v in dict(lw=lw, lb=lb, wa=wa, wb=wb, wo=wo).items()}   # tt-bio holds bf16 weights
def ln(x): return F.layer_norm(f32(x), (384,), W["lw"], W["lb"], 1e-5)
def mm(v, w, mode):
    if mode == "bf16": return bf(v) @ w.T
    if mode == "fp32": return f32(v) @ w.T
    if mode == "tf32": return tf32(v) @ w.T
    m = f32(v.mean(0, keepdim=True)); return f32(bf(v - m) @ w.T) + f32(bf(m) @ w.T) if mode == "centred_bfrow" else f32(bf(v - m) @ w.T) + f32(m @ w.T)
def run(mode, store):
    xn = store(ln(bf(x)))
    a, b = store(mm(xn, W["wa"], mode)), store(mm(xn, W["wb"], mode))
    h = store(F.silu(a) * b)
    return mm(h, W["wo"], mode)
print("bf16 weights, fp32 everything", rel(run("fp32", f32), ref))
print("plain bf16 stores (today)     ", rel(run("bf16", bf), ref))
print("tf32 matmul inputs, fp32 store", rel(run("tf32", f32), ref))
print("centred, fp32 row matmul      ", rel(run("centred", f32), ref))
print("centred, bf16 row matmul      ", rel(run("centred_bfrow", f32), ref))
print("centred, bf16 a/b/h stores    ", rel(run("centred", bf), ref))
resid = x + run("centred", f32)
print("centred, s_out = x + t stored bf16", rel(bf(resid) - x, ref), " stored fp32", rel(f32(resid) - x, ref))
print("small-channel s magnitude max", float((x + ref)[:, small].abs().max()), "tok std of t", float(tok(ref[:, small]).std()))
