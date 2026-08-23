"""Two load-bearing facts for the bucketing audit, measured not assumed.

1. Does ttnn.softmax(dim=-1) mask its ragged tile tail?  (the "IMMUNE by safe op" claim)
2. Does ttnn.matmul reduce over the PHYSICAL K, i.e. read a poisoned ragged tail?
"""
import torch, ttnn, json
print("ttnn", ttnn.__version__ if hasattr(ttnn, "__version__") else "?")
import sys; sys.path.insert(0, ".")
from tt_bio.tenstorrent import get_device
d = get_device()
out = {}

# ---- 1. softmax at a ragged last dim. All entries -5, so an unmasked tail of 31 zeros
# would add 31*exp(0) to a denominator of 33*exp(-5): masked -> 1/33 = 0.0303, unmasked -> 2.1e-4.
W = 33
x = torch.full((1, 1, 32, W), -5.0)
xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
print("softmax in: logical", tuple(xt.shape), "padded", tuple(xt.padded_shape))
s = ttnn.to_torch(ttnn.softmax(xt, dim=-1))[0, 0, 0, :W]
out["softmax_ragged_first"] = float(s[0])
out["softmax_ragged_sum"] = float(s.sum())
out["softmax_masked_expect"] = 1.0 / W
out["softmax_unmasked_expect"] = float(torch.exp(torch.tensor(-5.0)) /
                                       (W * torch.exp(torch.tensor(-5.0)) + (32 - W % 32) * 1.0))
print("softmax:", json.dumps({k: out[k] for k in out if k.startswith("softmax")}, indent=1))

# ---- 2. can we even build a logical-ragged tensor over a POISONED padded buffer?
# ttnn.pad aliases in TILE layout, so try the reverse relabel and check the poison survives.
big = torch.zeros((1, 1, 32, 64)); big[..., :W] = 1.0; big[..., W:] = 1.0e4
bt = ttnn.from_torch(big, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
poisoned = None
for name, fn in (("reshape", lambda t: ttnn.reshape(t, (1, 1, 32, W))),):
    try:
        poisoned = fn(bt)
        print(f"{name}: logical {tuple(poisoned.shape)} padded {tuple(poisoned.padded_shape)}")
        out["relabel_via"] = name
        break
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {e}")
        out[f"relabel_{name}_error"] = f"{type(e).__name__}: {e}"

if poisoned is not None and int(poisoned.shape[-1]) == W:
    # softmax on the poisoned tail: if the op masks, answer is 1/33 regardless of the 1e4.
    sp = ttnn.to_torch(ttnn.softmax(poisoned, dim=-1))[0, 0, 0, :W]
    out["softmax_poisoned_first"] = float(sp[0])
    print("softmax over POISONED tail, first elem:", out["softmax_poisoned_first"],
          "(masked -> 0.0303, unmasked -> ~0)")

# ---- 2b. matmul reducing over a ragged K with a poisoned tail.
# A[1,1,32,K=33] all ones (tail poisoned), B[1,1,33,32] all ones. Logical answer 33.
bigA = torch.zeros((1, 1, 32, 64)); bigA[..., :W] = 1.0; bigA[..., W:] = 1.0e4
at = ttnn.from_torch(bigA, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
try:
    aR = ttnn.reshape(at, (1, 1, 32, W))
    bR = ttnn.from_torch(torch.ones((1, 1, W, 32)), dtype=ttnn.float32,
                         layout=ttnn.TILE_LAYOUT, device=d)
    m = ttnn.to_torch(ttnn.matmul(aR, bR))
    out["matmul_ragged_k_poisoned_A"] = float(m[0, 0, 0, 0])
    out["matmul_expect_logical"] = float(W)
    print("matmul over ragged K, A tail poisoned 1e4:", out["matmul_ragged_k_poisoned_A"],
          "(logical answer 33; >33 means it read the physical tail)")
except Exception as e:
    print("matmul probe failed:", type(e).__name__, e)
    out["matmul_error"] = f"{type(e).__name__}: {e}"

# ---- 2c. and with B's K axis (dim -2) poisoned instead.
bigB = torch.zeros((1, 1, 64, 32)); bigB[:, :, :W, :] = 1.0; bigB[:, :, W:, :] = 1.0e4
btt = ttnn.from_torch(bigB, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=d)
try:
    bR2 = ttnn.reshape(btt, (1, 1, W, 32))
    aR2 = ttnn.from_torch(torch.ones((1, 1, 32, W)), dtype=ttnn.float32,
                          layout=ttnn.TILE_LAYOUT, device=d)
    m2 = ttnn.to_torch(ttnn.matmul(aR2, bR2))
    out["matmul_ragged_k_poisoned_B"] = float(m2[0, 0, 0, 0])
    print("matmul over ragged K, B tail poisoned 1e4:", out["matmul_ragged_k_poisoned_B"])
except Exception as e:
    print("matmul B probe failed:", type(e).__name__, e)
    out["matmul_B_error"] = f"{type(e).__name__}: {e}"

pass
print("RESULT " + json.dumps(out))
