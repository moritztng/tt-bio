"""Can the multiply+add and the bf16->fp32 typecast be absorbed into ONE ttnn.add?

`scale_mask_softmax` is out (its mask must have all intermediate dims == 1, and a pair bias is
[1, n_heads, S, S]). The remaining bit-exact route is the binary op's own activation + output dtype:

    ttnn.add(sc_bf16, bias_bf16, dtype=float32, input_tensor_a_activations=[MUL_UNARY_SFPU(scale)])

replacing typecast -> multiply -> add. Small shapes: this is an API-surface question, not a timing one.
Every arm is checked against the three-op chain with torch.equal.
"""
import sys, torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/openfold3-sizes-perf")
from tt_bio.tenstorrent import get_device

dev = get_device()
S, H = 64, 4
SC = 32 ** -0.5
torch.manual_seed(0)
sc_t = (torch.randn(S, H, S, S) * 3).bfloat16()
b_t = (torch.randn(1, H, S, S) * 3).bfloat16()
mk = lambda t, d=ttnn.bfloat16: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=d,
                                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
sc_bf, b_bf = mk(sc_t), mk(b_t)
b_f = ttnn.typecast(b_bf, ttnn.float32)

ref = ttnn.to_torch(ttnn.add(ttnn.multiply(ttnn.typecast(sc_bf, ttnn.float32), SC), b_f))
print("ref dtype", ref.dtype)

def t(name, fn):
    try:
        o = fn()
        h = ttnn.to_torch(o)
        print(f"{name:34s} OK  dtype={h.dtype} torch_equal={torch.equal(ref, h)} "
              f"max_abs={float((ref.float()-h.float()).abs().max()):.3e}")
    except Exception as e:
        print(f"{name:34s} REFUSED {type(e).__name__}: {str(e).splitlines()[0][:110]}")

t("multiply(bf16, s, dtype=fp32)", lambda: ttnn.add(ttnn.multiply(sc_bf, SC, dtype=ttnn.float32), b_f))
print("--- UnaryWithParam present:", hasattr(ttnn, "UnaryWithParam"),
      "| MUL_UNARY_SFPU:", hasattr(ttnn.UnaryOpType, "MUL_UNARY_SFPU"))
if hasattr(ttnn, "UnaryWithParam"):
    act = lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SC)]
    t("add(fp32,fp32, a_act=mul)", lambda: ttnn.add(
        ttnn.typecast(sc_bf, ttnn.float32), b_f, input_tensor_a_activations=act()))
    t("add(bf16,fp32, a_act=mul, dt=fp32)", lambda: ttnn.add(
        sc_bf, b_f, dtype=ttnn.float32, input_tensor_a_activations=act()))
    t("add(bf16,bf16, a_act=mul, dt=fp32)", lambda: ttnn.add(
        sc_bf, b_bf, dtype=ttnn.float32, input_tensor_a_activations=act()))
# `softmax_in_place` is scored in probe_fuse2.py, against `ttnn.softmax` -- comparing it
# here would compare a post-softmax tensor against a pre-softmax reference.
