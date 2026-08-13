"""S1: screen the WHOLE fp32-softmax tail, height-sharded in L1, against what main runs today.

The tail begins at the bf16 score tensor the first batched_matmul leaves in DRAM and ends at the
bf16 attention weights the second batched_matmul reads back, so every arm pays its own transitions.
probe_l1_chain2.py only covered the last two steps; this covers all of:

    typecast(bf16 -> fp32) -> add_(bias, a_act=MUL(scale_inv)) -> softmax_in_place -> typecast(-> bf16)

Arms:
  dram          all four steps DRAM-interleaved. What main runs today.
  shard_bcast   shard the bf16 scores, then all four in L1, bias left interleaved and broadcasting.
                This is the open question (L2b): does add_ take a sharded destination with a
                [1,H,S,S] operand broadcasting over the leading dim.
  shard_rep     same, but the bias is repeated to the block's leading dim ONCE and sharded, so the
                add_ sees two identically-sharded operands. The repeat is hoisted out of the timed
                region because a real call reuses one repeated bias across all of its blocks; its
                own cost is timed separately and reported as repeat_ms.
  shard_softmax the fallback of kill gate 3: add_ stays interleaved, shard only from the softmax on.

Every arm returns a bf16 DRAM tensor and is torch.equal-compared against dram.
"""
import json, sys, time
import torch, ttnn

sys.path.insert(0, ".")
import tt_bio.tenstorrent as T

S, H = 512, 4
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
GX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SCALE_INV = 0.125

res = {"S": S, "H": H, "rows": ROWS, "cores": GY * GX,
       "fp32_MB": round(ROWS * H * S * S * 4 / 2**20, 2),
       "fp32_B_per_core": int(ROWS * H * S * S * 4 / (GY * GX))}

dev = T.get_device()
DRAM = ttnn.DRAM_MEMORY_CONFIG


def sharded(dtype_rows):
    return ttnn.create_sharded_memory_config(
        shape=(dtype_rows, S), core_grid=ttnn.CoreGrid(y=GY, x=GX),
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)


try:
    torch.manual_seed(0)
    sc_t = (torch.randn(ROWS, H, S, S) * 3.0).to(torch.bfloat16)
    bias_t = (torch.randn(1, H, S, S) * 0.5).to(torch.bfloat16)
    sc0 = ttnn.from_torch(sc_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=dev, memory_config=DRAM)
    bias = ttnn.from_torch(bias_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=DRAM)
    bias_f = ttnn.typecast(bias, ttnn.float32, memory_config=DRAM)

    shard = sharded(ROWS * H * S)
    act = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, SCALE_INV)]

    def dram_arm():
        sc = ttnn.typecast(sc0, ttnn.float32, memory_config=DRAM)
        attn = ttnn.add_(sc, bias_f, input_tensor_a_activations=act)
        attn = ttnn.softmax_in_place(attn)
        out = ttnn.typecast(attn, ttnn.bfloat16, memory_config=DRAM)
        ttnn.deallocate(attn)
        return out

    def _sharded_softmax(x):
        try:
            return ttnn.softmax_in_place(x)
        except Exception:
            y = ttnn.softmax(x, dim=-1, memory_config=shard)
            ttnn.deallocate(x)
            return y

    def shard_bcast_arm():
        scl = ttnn.to_memory_config(sc0, shard)
        sc = ttnn.typecast(scl, ttnn.float32, memory_config=shard)
        ttnn.deallocate(scl)
        attn = ttnn.add_(sc, bias_f, input_tensor_a_activations=act)
        attn = _sharded_softmax(attn)
        ob = ttnn.typecast(attn, ttnn.bfloat16, memory_config=shard)
        ttnn.deallocate(attn)
        out = ttnn.to_memory_config(ob, DRAM)
        ttnn.deallocate(ob)
        return out

    # bias repeated to the block's leading dim, sharded once and reused across every block of a call
    bias_rep_l1 = None

    def shard_rep_arm():
        scl = ttnn.to_memory_config(sc0, shard)
        sc = ttnn.typecast(scl, ttnn.float32, memory_config=shard)
        ttnn.deallocate(scl)
        attn = ttnn.add_(sc, bias_rep_l1, input_tensor_a_activations=act)
        attn = _sharded_softmax(attn)
        ob = ttnn.typecast(attn, ttnn.bfloat16, memory_config=shard)
        ttnn.deallocate(attn)
        out = ttnn.to_memory_config(ob, DRAM)
        ttnn.deallocate(ob)
        return out

    def shard_softmax_arm():
        sc = ttnn.typecast(sc0, ttnn.float32, memory_config=DRAM)
        attn = ttnn.add_(sc, bias_f, input_tensor_a_activations=act)
        al = ttnn.to_memory_config(attn, shard)
        ttnn.deallocate(attn)
        al = _sharded_softmax(al)
        ob = ttnn.typecast(al, ttnn.bfloat16, memory_config=shard)
        ttnn.deallocate(al)
        out = ttnn.to_memory_config(ob, DRAM)
        ttnn.deallocate(ob)
        return out

    def timed(fn, n=7):
        fn()
        ts = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(o)
        return round(sorted(ts)[len(ts) // 2] * 1e3, 4)

    ref = dram_arm()
    ref_out = ttnn.to_torch(ref)
    ttnn.deallocate(ref)

    ARMS = [("dram", dram_arm), ("shard_bcast", shard_bcast_arm),
            ("shard_rep", shard_rep_arm), ("shard_softmax", shard_softmax_arm)]
    for name, fn in ARMS:
        try:
            if name == "shard_rep":
                br = ttnn.repeat(bias_f, ttnn.Shape([ROWS, 1, 1, 1]))
                t0 = time.perf_counter()
                ttnn.synchronize_device(dev)
                br2 = ttnn.repeat(bias_f, ttnn.Shape([ROWS, 1, 1, 1]))
                ttnn.synchronize_device(dev)
                res["repeat_ms"] = round((time.perf_counter() - t0) * 1e3, 4)
                ttnn.deallocate(br2)
                bias_rep_l1 = ttnn.to_memory_config(br, shard)
                ttnn.deallocate(br)
            o = fn()
            got = ttnn.to_torch(o)
            ttnn.deallocate(o)
            res[name + "_equal"] = bool(torch.equal(ref_out, got))
            res[name + "_maxabs"] = float((ref_out.float() - got.float()).abs().max())
            res[name + "_ms"] = timed(fn)
            if name == "shard_rep" and bias_rep_l1 is not None:
                ttnn.deallocate(bias_rep_l1)
                bias_rep_l1 = None
        except Exception as e:
            res[name + "_error"] = f"{type(e).__name__}: {e}"[:300]
            if name == "shard_rep" and bias_rep_l1 is not None:
                try:
                    ttnn.deallocate(bias_rep_l1)
                except Exception:
                    pass
                bias_rep_l1 = None

    if res.get("dram_ms"):
        # DRAM traffic of the dram arm, in bytes: typecast r bf16 + w fp32; add_ r fp32 (x2) w fp32;
        # softmax r+w fp32; typecast r fp32 + w bf16
        el = ROWS * H * S * S
        by = el * (2 + 4) + el * (4 + 4 + 4) + el * (4 + 4) + el * (4 + 2)
        res["dram_GBs"] = round(by / 1e9 / (res["dram_ms"] / 1e3), 1)
        for k in ("shard_bcast", "shard_rep", "shard_softmax"):
            if res.get(k + "_ms"):
                res[k + "_speedup"] = round(res["dram_ms"] / res[k + "_ms"], 4)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:400]

print("RESULT " + json.dumps(res))
