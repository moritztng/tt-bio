"""Does a ttnn reduction read the PHYSICAL tile tail? The RFD3 p23 method: two tensors with
IDENTICAL logical values but different upstream histories, so their tile padding differs.
Any difference in the result is proof the op read the padding. No way to poison a tail
directly (ttnn.reshape refuses to shrink logical volume), so provoke it instead.
"""
import torch, ttnn, json, sys
sys.path.insert(0, ".")
from tt_bio.tenstorrent import get_device
d = get_device()
W, H = 33, 32
V = 0.25
out = {}

def arms(shape, axis_is_last):
    """Two device tensors, same logical values, different tile-padding history."""
    t = torch.full(shape, V)
    a = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)  # tail 0 (tilize)
    # add writes whole tiles, so the tail becomes 0 + K if it writes padding at all
    b = ttnn.add(ttnn.from_torch(t - 7.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d), 7.0)
    # multiply by 1 after a subtract: a third history
    c = ttnn.multiply(ttnn.from_torch(t * 4.0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d), 0.25)
    return a, b, c

# ---- matmul reducing over a ragged K on the LHS last axis
A = arms((1, 1, H, W), True)
Bm = ttnn.from_torch(torch.ones((1, 1, W, H)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
mm = [float(ttnn.to_torch(ttnn.matmul(x, Bm))[0, 0, 0, 0]) for x in A]
out["matmul_ragged_k_arms"] = mm
out["matmul_logical_expect"] = V * W
print("matmul over ragged K=33, three tail histories:", mm, "logical expect", V * W)

# ---- matmul reducing over a ragged K on the RHS second-to-last axis
Br = arms((1, 1, W, H), False)
Am = ttnn.from_torch(torch.ones((1, 1, H, W)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
mm2 = [float(ttnn.to_torch(ttnn.matmul(Am, x))[0, 0, 0, 0]) for x in Br]
out["matmul_ragged_k_rhs_arms"] = mm2
print("matmul over ragged K (rhs dim -2):", mm2, "logical expect", V * W)

# ---- softmax over the ragged last dim, same three histories
sm = [float(ttnn.to_torch(ttnn.softmax(x, dim=-1))[0, 0, 0, 0]) for x in arms((1, 1, H, W), True)]
out["softmax_arms"] = sm
out["softmax_logical_expect"] = 1.0 / W
print("softmax over ragged W=33:", sm, "logical expect", 1.0 / W)

# ---- ttnn.sum over the ragged last dim
try:
    sums = [float(ttnn.to_torch(ttnn.sum(x, dim=-1)).flatten()[0]) for x in arms((1, 1, H, W), True)]
    out["sum_arms"] = sums
    print("sum over ragged W=33:", sums, "logical expect", V * W)
except Exception as e:
    out["sum_error"] = f"{type(e).__name__}: {e}"
    print("sum probe failed:", type(e).__name__)

# ---- ttnn.max over the ragged last dim (a max is the most tail-sensitive reduce)
try:
    mx = [float(ttnn.to_torch(ttnn.max(x, dim=-1)).flatten()[0]) for x in arms((1, 1, H, W), True)]
    out["max_arms"] = mx
    print("max over ragged W=33:", mx, "logical expect", V)
except Exception as e:
    out["max_error"] = f"{type(e).__name__}: {e}"
    print("max probe failed:", type(e).__name__)

print("RESULT " + json.dumps(out))
