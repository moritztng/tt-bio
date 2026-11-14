"""Self-consistent references from mol1-ref goldens whose args were captured post-forward: keep each golden's
(aliased) args as the input and replace out with upstream's float64 recompute on exactly those args.
Writes <out_dir>/<same file name> as {"args": (s, z), "out": (s, z)} float32. Run in the upstream venv with -I."""
import os
import sys
import torch
from mol.model.modules.pairformer import PairformerStack

ckpt, gdir, stack_pt, odir = sys.argv[1:5]
torch.set_num_threads(12)
os.makedirs(odir, exist_ok=True)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
st = PairformerStack(n_blocks=48, n_heads=16, c_z=448, c_s=384, dropout=0.25, hidden_scale_up=True,
                     n_pairmixer_blocks=8, c_z_pairmixer=448, c_z_pairformer=320)
p = "pairformer_stack."
st.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)}, strict=True)
st = st.double().eval()
r = torch.load(stack_pt, weights_only=False)
torch.save({"args": (r["s"].float(), r["z"].float()), "out": tuple(t.float() for t in r["out"])},
           f"{odir}/pairformer_stack.call0.pt")
for i in (0, 1, 8, 23, 47):
    f = f"pairformer_stack.blocks.{i}.call0.pt"
    g = torch.load(f"{gdir}/{f}", weights_only=False)
    s, z = (t.double() for t in g["args"][:2])
    N = s.shape[0]
    with torch.no_grad():
        so, zo = st.blocks[i](s.clone(), z.clone(), pair_mask=torch.ones(N, N, dtype=torch.float64))
    torch.save({"args": (s.float(), z.float()), "out": (so.float(), zo.float())}, f"{odir}/{f}")
    print(i, "done", flush=True)
print("REFSDONE")
