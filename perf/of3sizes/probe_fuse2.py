"""Two more bit-exactness questions, each against the exact op it would replace.

  1. `softmax_in_place` vs `softmax`  -- removes one fp32 allocation of the score tensor (16 GiB at
     1024 aa), which is half the capacity story. Compared against `ttnn.softmax`, not against the
     pre-softmax tensor.
  2. `add_` with the scale folded in as a MUL_UNARY_SFPU activation -- in-place AND fused, so it
     removes both the multiply's whole pass and its allocation.
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
act = lambda: [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SC)]
chain = lambda: ttnn.add(ttnn.multiply(ttnn.typecast(sc_bf, ttnn.float32), SC), b_f)

pre = ttnn.to_torch(chain())
post = ttnn.to_torch(ttnn.softmax(chain(), dim=-1))

def t(name, fn, ref):
    try:
        h = ttnn.to_torch(fn())
        print(f"{name:38s} OK  torch_equal={torch.equal(ref, h)} "
              f"max_abs={float((ref.float()-h.float()).abs().max()):.3e}")
    except Exception as e:
        print(f"{name:38s} REFUSED {type(e).__name__}: {str(e).splitlines()[0][:110]}")

t("softmax_in_place vs softmax", lambda: ttnn.softmax_in_place(chain()), post)
t("add_(fp32,fp32, a_act=mul)", lambda: ttnn.add_(
    ttnn.typecast(sc_bf, ttnn.float32), b_f, input_tensor_a_activations=act()), pre)
t("add(a_act=mul) + softmax_in_place", lambda: ttnn.softmax_in_place(ttnn.add(
    ttnn.typecast(sc_bf, ttnn.float32), b_f, input_tensor_a_activations=act())), post)
