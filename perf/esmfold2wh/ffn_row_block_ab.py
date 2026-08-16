import glob, os, sys, torch, ttnn
sys.path.insert(0, "/home/ttuser/.coworker/wt/japanfold-esmfold2-wh-msa-cap-p2")
from safetensors.torch import load_file
from tt_bio import tenstorrent
import tt_bio.esmfold2 as E
from tt_bio.esmc import SwiGLUFFN
from tt_bio.tenstorrent import WeightScope

snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--biohub--ESMFold2/snapshots/*"))[0]
sd = {}
for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
    for k, v in load_file(f).items():
        if k.startswith("msa_encoder.blocks.0."):
            sd[k[len("msa_encoder."):]] = v
dev = tenstorrent.get_device()
ws = E._remap_transition_named(sd, "blocks.0.msa_transition")
print({k: tuple(v.shape) for k, v in ws.as_dict().items()}, flush=True)
ffn = SwiGLUFFN(ws, None)
ffn.device = dev if hasattr(ffn, "device") else None

B, L, M, C = 1, 128, 256, ws.as_dict()["0.weight"].shape[0]
g = torch.Generator().manual_seed(0)
xt = torch.randn(B, L, M, C, generator=g)
x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
full = torch.Tensor(ttnn.to_torch(ffn(x)))
blocks = [(s, s + 32) for s in range(0, L, 32)]
blk = torch.Tensor(ttnn.to_torch(ttnn.concat([ffn(x[:, s:e, :, :]) for s, e in blocks], dim=1)))
d = (full - blk).abs()
print(f"C={C} equal={torch.equal(full, blk)} maxdiff={d.max().item()} ndiff={int((d>0).sum())}/{d.numel()} |full|max={full.abs().max().item()}", flush=True)
# also: same call twice, no blocking -- determinism control
a2 = torch.Tensor(ttnn.to_torch(ffn(x)))
print("determinism control equal:", torch.equal(full, a2), flush=True)
