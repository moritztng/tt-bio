"""CPU float64 check of an in-matmul LayerNorm on real ESMC activations: LN path (bf16 x_hat), shipped WH LN_FOLD, and raw bf16 x @ centred
gamma-folded W with rstd after. Usage: python -I perf/spd_m_esmc/lnacc.py esmc-300m scripts/gpu_vs_tt/fixtures/prot512.seq"""
import glob, math, sys, torch
torch.set_grad_enabled(False)
name = sys.argv[1]; seq = open(sys.argv[2]).read().split()[0][:512]
path = glob.glob(f"/home/moritz/.cache/huggingface/hub/models--biohub--{name}-2024-12/snapshots/*/data/weights/*.pth")[0]
sd = {k: v.double() for k, v in torch.load(path, map_location="cpu").items()}
V = ["<cls>","<pad>","<eos>","<unk>","L","A","G","V","S","E","R","T","I","D","P","K","Q","N","F","Y","M","H","W","C","X","B","U","Z","O",".","-","|","<mask>"]
ids = torch.tensor([0] + [V.index(c) if c in V else 3 for c in seq] + [2])
nl = max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.blocks.")) + 1
d = sd["embed.weight"].shape[1]; H = d // 64; L = len(ids); s = math.sqrt(nl / 36)
def ln(x, w=None, b=None):
    m = x.mean(-1, keepdim=True); v = ((x - m) ** 2).mean(-1, keepdim=True); y = (x - m) / torch.sqrt(v + 1e-5)
    return y * (w if w is not None else 1) + (b if b is not None else 0)
def bf(t): return t.to(torch.bfloat16).double()
def bfp8(w):  # [K, N]: groups of 16 along N share an exponent, 7-bit magnitude
    K, N = w.shape; g = w.reshape(K, N // 16, 16); e = torch.floor(torch.log2(g.abs().amax(-1, keepdim=True).clamp_min(1e-30)))
    q = 2.0 ** (e - 6); return (torch.round(g / q).clamp(-127, 127) * q).reshape(K, N)
def rope(x):
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64)); f = torch.outer(torch.arange(L, dtype=torch.float64), inv)
    c, sn = torch.cat([f.cos()] * 2, -1), torch.cat([f.sin()] * 2, -1); x1, x2 = x[..., :32], x[..., 32:]
    return x * c + torch.cat([-x2, x1], -1) * sn
def grade(tag, x, g, b, W):
    """x [L, d] float64 residual; W [d, N] torch [in, out]."""
    ref = ln(x, g, b) @ W
    xb = bf(x); xf = xb.float()
    m = xf.mean(-1, keepdim=True); r = torch.rsqrt(((xf - m) ** 2).mean(-1, keepdim=True) + 1e-5)
    out = {}
    for mode, q in (("n", bf), ("f", bfp8)):
        xh = bf(((xf - m) * r).double() * g + b)                       # LN path: affine in the norm, bf16 x_hat
        out["ln_" + mode] = (xh.float() @ q(W).float()).double()
        C = torch.eye(d, dtype=torch.float64) - 1.0 / d
        Wc = q(C @ (g[:, None] * W)); bw = (b @ W)                     # in-matmul: centred gamma-folded W
        out["mm_" + mode] = (r.double() * (xb.float() @ Wc.float()).double()) + bw
        # ps: producer stats from the residual matmul's epilogue, x in fp32 dest: packs bf16(x) and bf16(x^2) and row-sums
        # them per 32-column tile (the finest partial) (fp32 partials); the consumer sums the partials and takes E x^2 - (E x)^2
        x32 = x.float(); p1 = bf(x32).float().reshape(L, -1, 32).sum(-1); p2 = bf(x32 * x32).float().reshape(L, -1, 32).sum(-1)
        m1 = p1.sum(-1, keepdim=True) / d; rp = torch.rsqrt(p2.sum(-1, keepdim=True) / d - m1 * m1 + 1e-5)
        out["ps_" + mode] = (rp.double() * (xb.float() @ Wc.float()).double()) + bw
        # pb: the same partials taken from the bf16 x the consumer reads (x packed bf16 first), x^2 exact in fp32
        b1 = xf.reshape(L, -1, 32).sum(-1); b2 = (xf * xf).reshape(L, -1, 32).sum(-1)
        m3 = b1.sum(-1, keepdim=True) / d; rb = torch.rsqrt(b2.sum(-1, keepdim=True) / d - m3 * m3 + 1e-5)
        out["pb_" + mode] = (rb.double() * (xb.float() @ Wc.float()).double()) + bw
        Wg = q(W); gw = g @ Wg                                          # gamma on in0, mean as a correction
        acc = (bf(xb * g).float() @ Wg.float()).double()
        out["mg_" + mode] = r.double() * (acc - m.double() * gw) + bw
        out["lf_" + mode] = (bf(((xf - m) * r).double()).float() @ q(g[:, None] * W).float()).double() + b @ W  # shipped WH LN_FOLD
        Wf = q(g[:, None] * W); sf = Wf.sum(0)                          # gamma-folded W, mean from its exact column sums
        accf = (xb.float() @ Wf.float()).double()
        out["mf_" + mode] = r.double() * (accf - m.double() * sf) + bw
    e = {k: ((v - ref).norm() / ref.norm()).item() for k, v in out.items()}
    mu = (x.mean(-1).abs() / x.std(-1)).max().item()
    return tag, mu, e
rows = []
x = sd["embed.weight"][ids]
for i in range(nl):
    p = f"transformer.blocks.{i}."
    g, b, Wqkv = sd[p + "attn.layernorm_qkv.0.weight"], sd[p + "attn.layernorm_qkv.0.bias"], sd[p + "attn.layernorm_qkv.1.weight"].T
    rows.append(grade(f"L{i:02d} attn", x, g, b, Wqkv))
    qkv = ln(x, g, b) @ Wqkv; q, k, v = qkv.split(d, -1)
    q = ln(q, sd[p + "attn.q_ln.weight"]); k = ln(k, sd[p + "attn.k_ln.weight"])
    sh = lambda t: t.reshape(L, H, 64).transpose(0, 1)
    a = torch.softmax(rope(sh(q)) @ rope(sh(k)).transpose(-1, -2) / 8.0, -1) @ sh(v)
    x = x + (a.transpose(0, 1).reshape(L, d) @ sd[p + "attn.out_proj.weight"].T) / s
    g, b, W1 = sd[p + "ffn.0.weight"], sd[p + "ffn.0.bias"], sd[p + "ffn.1.weight"].T
    rows.append(grade(f"L{i:02d} ffn ", x, g, b, W1))
    h = ln(x, g, b) @ W1; h1, h2 = h.chunk(2, -1)
    x = x + (torch.nn.functional.silu(h1) * h2 @ sd[p + "ffn.3.weight"].T) / s
keys = list(rows[0][2])
print(f"{name} L={L}  rel_l2 vs float64 of LN(x) W.  ln_ = LN path, mm_ = centred gamma-folded W + rstd, mg_ = gamma on x + mean correction; _n bf16 W, _f bfp8 W")
print("site       max|mu|/sd " + " ".join(f"{k:>9}" for k in keys))
for t, mu, e in rows:
    print(f"{t}  {mu:9.2f} " + " ".join(f"{e[k]:9.2e}" for k in keys))
for k in keys:
    print(f"mean {k}: {sum(r[2][k] for r in rows) / len(rows):.3e}  worst {max(r[2][k] for r in rows):.3e}")
