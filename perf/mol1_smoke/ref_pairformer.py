"""Upstream Mol-1 PairformerStack in float64 on CPU: block 0 (PairMixer), block 8 (after the
448->320 adapter) and the full 48-block stack, on seeded inputs. Run in the upstream venv with -I."""
import sys
import torch
from mol.model.modules.pairformer import PairformerStack

ckpt, out, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.set_num_threads(8)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
st = PairformerStack(n_blocks=48, n_heads=16, c_z=448, c_s=384, dropout=0.25, hidden_scale_up=True,
                     n_pairmixer_blocks=8, c_z_pairmixer=448, c_z_pairformer=320)
p = "pairformer_stack."
missing = st.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)}, strict=True)
st = st.double().eval()
g = torch.Generator().manual_seed(0)
s = torch.randn(N, 384, generator=g, dtype=torch.float64)
z = torch.randn(N, N, 448, generator=g, dtype=torch.float64)
mask = torch.ones(N, N, dtype=torch.float64)
cap = {}
st.blocks[0].register_forward_hook(lambda m, a, o: cap.setdefault("b0", (o[0].clone(), o[1].clone())))
st.blocks[8].register_forward_pre_hook(lambda m, a, kw=None: cap.setdefault("b8_in", None))
def pre8(m, args, kwargs):
    cap["b8_in"] = (args[0].clone(), args[1].clone())
st.blocks[8].register_forward_pre_hook(pre8, with_kwargs=True)
st.blocks[8].register_forward_hook(lambda m, a, o: cap.setdefault("b8", (o[0].clone(), o[1].clone())))
with torch.no_grad():
    so, zo = st(s, z, pair_mask=mask)
torch.save({"s": s, "z": z, "b0": cap["b0"], "b8_in": cap["b8_in"], "b8": cap["b8"], "out": (so, zo)}, out)
print("REFDONE", so.shape, zo.shape, float(zo.abs().max()))
