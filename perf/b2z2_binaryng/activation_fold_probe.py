import sys
from pathlib import Path
ROOT = Path("/home/tt-admin/wt/b2z2-step-binaryng-fusion"); sys.path.insert(0, str(ROOT))
import torch; torch.set_grad_enabled(False)
import ttnn
import tt_bio.tenstorrent as T
dev = T.get_device(trace_region_size=512 << 20)
S = ttnn.UnaryOpType.SIGMOID
CG = T.CORE_GRID_MAIN
def mk(sh, s=0.5): return ttnn.from_torch(torch.randn(*sh)*s, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
for shape, wsh in (([1,512,768],(768,768)), ([1,140,32,128],(128,128))):
    s_in, y = mk(shape), mk(shape)
    W, Bi = mk(wsh, 0.05), mk((wsh[1],), 0.05)
    for cg_name, cg in (("core_grid=MAIN", CG), ("core_grid=None", None)):
      for bias_name, bias in (("with bias", Bi), ("no bias", None)):
        kw = {} if cg is None else {"core_grid": cg}
        lin_plain = ttnn.linear(s_in, W, bias=bias, **kw)
        lin_act   = ttnn.linear(s_in, W, bias=bias, activation="sigmoid", **kw)
        # reference: sigmoid applied AFTER the bias, as a standalone unary
        ref = ttnn.sigmoid(lin_plain)
        a_side = ttnn.multiply(lin_plain, y, input_tensor_a_activations=[S])
        folded = ttnn.multiply(lin_act, y)
        r_ref  = ttnn.to_torch(ref).float()
        r_act  = ttnn.to_torch(lin_act).float()
        r_a    = ttnn.to_torch(a_side).float()
        r_f    = ttnn.to_torch(folded).float()
        print(f"{str(shape):18s} {cg_name:15s} {bias_name:10s} "
              f"linear_act_vs_unary_sigmoid maxabs={(r_act-r_ref).abs().max().item():.6f}  "
              f"a_side_vs_folded maxabs={(r_a-r_f).abs().max().item():.6f} "
              f"exact={torch.equal(r_a,r_f)}", flush=True)
        for t in (lin_plain, lin_act, ref, a_side, folded): ttnn.deallocate(t)
print("DONE")
