"""CPU emulation: does a fused fp32 single Transition lose accuracy if each 32-token tile centres its matmul inputs on its
own first row instead of token 0 (fp32_single_transition today)? Block 43's captured input (8fg6 tap.pt), st_bench's
bf16 weights, small-channel token rel err vs float64. Centring is exact algebra, so only the bf16 rounding of v - c moves.
python -I tile_centre_emul.py TAP.pt LNW.pt [N]"""
import sys, torch
import torch.nn.functional as F
torch.manual_seed(0)
tap = torch.load(sys.argv[1], weights_only=False); lnw = torch.load(sys.argv[2], weights_only=False)
x0 = tap["in"][0].double().reshape(-1, 384)
N = int(sys.argv[3]) if len(sys.argv) > 3 else x0.shape[0]
x = x0[torch.arange(N) % x0.shape[0]] + (x0 - x0.mean(0)).std(0) * 0.3 * torch.randn(N, 384, dtype=torch.float64)
bf = lambda t: t.to(torch.bfloat16).double(); f32 = lambda t: t.float().double()
lw, lb = bf(lnw["w"].double()), bf(lnw["b"].double())
w1 = bf(torch.randn(384, 1536, dtype=torch.float64) / 384 ** .5 * 8); w2 = bf(torch.randn(384, 1536, dtype=torch.float64) / 384 ** .5 * 8)
w3 = bf(torch.randn(1536, 384, dtype=torch.float64) / 1536 ** .5)
xb = bf(x)
xn64 = F.layer_norm(xb, (384,), lw, lb, 1e-5)
ref = (F.silu(xn64 @ w1) * (xn64 @ w2)) @ w3
small = ref.abs().amax(0) < ref.abs().amax(0).median()
tok = lambda t: t - t.mean(0)
err = lambda o: float((tok(o[:, small]) - tok(ref[:, small])).norm() / tok(ref[:, small]).norm())

def ref_rows(v, how):
    if how == "token0": return v[:1].expand_as(v)
    if how == "tile": return v[(torch.arange(v.shape[0]) // 32) * 32]
    if how == "tile_last": return v[torch.clamp((torch.arange(v.shape[0]) // 32) * 32 + 31, max=v.shape[0] - 1)]
    raise ValueError(how)

def clin(v, w, how, row="bf16"):
    c = ref_rows(v, how)
    if row == "bf16": cw = bf(c) @ w              # tt-bio today: the c row goes through the matmul as bf16
    elif row == "hilo": cw = bf(c) @ w + bf(c - bf(c)) @ w   # two bf16 matmuls, c ~ hi + lo
    else: cw = f32(c) @ w
    return f32(f32(bf(v - c) @ w) + f32(cw))      # fp32 dest, fp32 add

def run(how, row="bf16"):
    xn = f32(F.layer_norm(f32(xb), (384,), lw, lb, 1e-5))
    a, b = clin(xn, w1, how, row), clin(xn, w2, how, row)
    h = f32(F.silu(a) * b)
    return clin(h, w3, how, row)

print(f"N={N}")
print("plain bf16 stores           ", err(bf((F.silu(bf(bf(xn64) @ w1)) * bf(bf(xn64) @ w2)) @ w3)))
for how in ("token0", "tile"):
    for row in ("bf16", "hilo", "fp32"):
        print(f"centred on {how:6s}, c row {row:4s}", err(run(how, row)))
