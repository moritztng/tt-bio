"""C1 verification: the new _tri_att_sdpa is torch.equal to the shipped program config at every
production size, and the L1-overflow fallback recovers in-process without leaving the device dirty.

PREDICTIONS:
  V1  _tri_att_q_chunks picks: 320->(320,64), 512->(512,), 576->(576,), 768->(768,384), 1024->(1024,512)
      -- widest divisor first, shipped pick last.
  V2  torch.equal against the shipped config at 320, 512, 576.
  V3  at seq 768 the first candidate (768) overflows L1, is recorded in _SDPA_Q_CHUNK_OVER_L1, and
      the SECOND call at the same shape does not throw again (the set is consulted, not retried).
      The output after the caught throw is still torch.equal to the shipped config -- if a failed
      program creation left the device dirty this is where it shows.
"""
import json, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

OUT = sys.argv[1]
res = {"chunks": {}, "exact": {}, "fallback": {}}

print("--- V1: candidate lists (no device) ---", flush=True)
for s in (256, 320, 352, 384, 448, 512, 576, 640, 768, 1024):
    c = T._tri_att_q_chunks(s, s)
    res["chunks"][s] = list(c)
    print("  seq %4d -> %s   (shipped %s)" % (s, c, T._sdpa_chunks_shipped(s, s)), flush=True)

dev = ttnn.open_device(device_id=0)
T.COMPUTE_GRID_MAIN = dev.compute_with_storage_grid_size()
print("grid %dx%d" % (T.COMPUTE_GRID_MAIN.x, T.COMPUTE_GRID_MAIN.y), flush=True)


def tensors(s):
    q, k, v = (ttnn.from_torch(torch.randn(s, 8, s, 32) * 0.1, layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16) for _ in range(3))
    bias = ttnn.from_torch(torch.randn(1, 8, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
    return q, k, v, bias


def shipped(q, k, v, bias, s, scale):
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=scale,
        program_config=T._tri_att_sdpa_program_config(s, s))


print("--- V2/V3: on device ---", flush=True)
for s in (320, 512, 576, 768):
    q, k, v, bias = tensors(s)
    scale = 32 ** -0.5
    ref = ttnn.to_torch(shipped(q, k, v, bias, s, scale))
    t0 = time.perf_counter()
    got = ttnn.to_torch(T._tri_att_sdpa(q, k, v, bias, scale))
    first_ms = (time.perf_counter() - t0) * 1e3
    eq = torch.equal(got, ref)
    # second call at the same shape: must not pay the throw again
    t0 = time.perf_counter()
    got2 = ttnn.to_torch(T._tri_att_sdpa(q, k, v, bias, scale))
    second_ms = (time.perf_counter() - t0) * 1e3
    eq2 = torch.equal(got2, ref)
    over = sorted(x for x in T._SDPA_Q_CHUNK_OVER_L1 if x[0] == s)
    res["exact"][s] = {"torch_equal": bool(eq), "torch_equal_2nd": bool(eq2),
                       "first_call_ms": round(first_ms, 2), "second_call_ms": round(second_ms, 2),
                       "over_l1": [list(x) for x in over]}
    print("  seq %4d  torch.equal %s / %s   1st %8.2f ms  2nd %8.2f ms  over-L1 %s" %
          (s, eq, eq2, first_ms, second_ms, over), flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)

json.dump(res, open(OUT, "w"), indent=1)
ttnn.close_device(dev)
print("wrote", OUT, flush=True)
