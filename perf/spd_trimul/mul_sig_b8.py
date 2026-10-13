"""Is `multiply_(p bf16, g bfp8, input_tensor_b_activations=[SIGMOID])` right? The trimul tail runs it
when trimul_b8in leaves g_out in bfp8. Arms vs float64 p * sigmoid(g_bfp8-as-stored): g bf16, g bfp8,
g bfp8 typecast back to bf16 first, and bfp8 g with an explicit ttnn.sigmoid. One JSON line.

usage: TT_VISIBLE_DEVICES=<chip> timeout 300 python perf/spd_trimul/mul_sig_b8.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

dev = T.get_device()
gen = torch.Generator().manual_seed(0)
n, c = 256, 384
p = torch.randn(1, n, n, c, generator=gen).to(torch.bfloat16)
g = torch.randn(1, n, n, c, generator=gen).to(torch.bfloat16)
up = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
gb8 = ttnn.typecast(up(g), ttnn.bfloat8_b)
g8_host = ttnn.to_torch(ttnn.typecast(gb8, ttnn.bfloat16)).reshape(g.shape).double()
ref = {"bf16": p.double() * torch.sigmoid(g.double()), "b8": p.double() * torch.sigmoid(g8_host)}


def rel(out, r):
    out = ttnn.to_torch(out).reshape(r.shape).double()
    return round(float((out - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()), 5)


SIG = [ttnn.UnaryOpType.SIGMOID]
rec = {"n": n, "c": c, "ttnn": getattr(ttnn, "__version__", "?")}
rec["g_bf16"] = rel(ttnn.multiply_(up(p), up(g), input_tensor_b_activations=SIG), ref["bf16"])
rec["g_b8"] = rel(ttnn.multiply_(up(p), gb8, input_tensor_b_activations=SIG), ref["b8"])
rec["g_b8_vs_bf16ref"] = rel(ttnn.multiply_(up(p), gb8, input_tensor_b_activations=SIG), ref["bf16"])
rec["g_b8_cast"] = rel(ttnn.multiply_(up(p), ttnn.typecast(gb8, ttnn.bfloat16), input_tensor_b_activations=SIG), ref["b8"])
rec["g_b8_sigmoid_op"] = rel(ttnn.multiply_(up(p), ttnn.sigmoid(gb8)), ref["b8"])
rec["g_b8_nosig"] = rel(ttnn.multiply_(up(p), gb8), p.double() * g8_host)
print(json.dumps(rec), flush=True)
