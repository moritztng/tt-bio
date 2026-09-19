"""What do nlp_create_qkv_heads and nlp_concat_heads actually do to the layout?

Derived from the device with an index-valued tensor, not from a docstring: a head-split
backward written from a guess about the packing is exactly how a wrong gradient ships.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import torch, ttnn
from tt_bio import tenstorrent as tt

dev = tt.get_device()
B, L, H, dh = 1, 64, 4, 32
# qkv packed [B, 1, L, 3*H*dh], each element its own flat index so any permutation shows.
n = B * 1 * L * 3 * H * dh
x = torch.arange(n, dtype=torch.float32).reshape(B, 1, L, 3 * H * dh)
xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    xt, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG)
for nm, t in (("q", q), ("k", k), ("v", v)):
    print(nm, "shape", tuple(int(d) for d in t.shape))
qh = ttnn.to_torch(q).to(torch.float32)
# candidate: [B,1,L,3,H,dh] -> q = x[...,0,:,:] transposed to [B,H,L,dh]
cand = x.reshape(B, 1, L, 3, H, dh)[:, 0, :, 0].permute(0, 2, 1, 3)
print("q == reshape(L,3,H,dh)[...,0,:,:].permute(0,2,1,3):",
      torch.equal(qh.to(torch.bfloat16), cand.to(torch.bfloat16)))
cand2 = x.reshape(B, 1, L, 3, H, dh)[:, 0, :, 1].permute(0, 2, 1, 3)
kh = ttnn.to_torch(k).to(torch.float32)
print("k matches slot 1:", torch.equal(kh.to(torch.bfloat16), cand2.to(torch.bfloat16)))

# nlp_concat_heads: [B,H,L,dh] -> [B,1,L,H*dh]
o = ttnn.experimental.nlp_concat_heads(q, memory_config=ttnn.DRAM_MEMORY_CONFIG)
oh = ttnn.to_torch(o).to(torch.float32)
print("concat_heads shape", tuple(int(d) for d in o.shape))
cc = qh.permute(0, 2, 1, 3).reshape(B, 1, L, H * dh)
print("concat == permute(0,2,1,3).reshape:", torch.equal(oh.to(torch.bfloat16),
                                                         cc.to(torch.bfloat16)))
