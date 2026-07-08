import os, sys
os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
sys.path.insert(0, '/home/ttuser/tt-bio-dev')
import torch, ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
B, H, NT, hd = 3, 4, 64, 32
torch.manual_seed(0)
# identical qkv across batch
one = torch.randn(1, NT, 3 * H * hd)
s = one.repeat(B, 1, 1).contiguous()
st = ttnn.from_torch(s, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
qkv = ttnn.unsqueeze(st, 1)   # (B,1,NT,3Hhd)
q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
print("q shape", tuple(q.shape))
qh = ttnn.to_torch(q).float()   # expect (B,H,NT,hd) identical across B
print("q s0-vs-s1 maxdiff", float((qh[0] - qh[1]).abs().max()))
# SDPA with a (1,H,NT,NT) mask, identical inputs -> outputs must be identical across B
mask = ttnn.from_torch(torch.randn(1, H, NT, NT), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
o = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, scale=hd ** -0.5)
oh = ttnn.to_torch(o).float()
print("o shape", tuple(oh.shape))
print("o s0-vs-s1 maxdiff", float((oh[0] - oh[1]).abs().max()))
print("MICRO_DONE")
