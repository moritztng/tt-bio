"""Does the tail of _fp32_softmax_attention run height-sharded in L1, bit-exactly?

TriangleAttention is 26.98 s of the 51.04 s OpenFold3 fold on qb2 and sits at ~88 % of the measured
DRAM copy roof, so the only lever class that can pay is deleting DRAM passes. The chain's last two
passes (fp32 softmax, then the typecast back to bf16) read and write a 2 GiB fp32 score tensor
through DRAM. If a row block of it fits L1 and both ops accept a height-sharded tensor, those passes
move to L1.

Bit-exactness is by construction -- same ops, same dtypes, same reduction axis, only the memory
config moves -- but "by construction" is not evidence, so this probes torch.equal at the real shape.

usage: probe_l1_chain.py [rows] [grid_y] [grid_x]
Constraint found the hard way: this p150a has 110 L1 banks in COMPUTE cores, not the 130 of the
compute grid, so a height shard may use at most 110 shards and tile_rows must divide the core count.
"""
import json, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

S, H = 512, 4
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
GX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
res = {"S": S, "H": H, "rows": ROWS, "core_grid": [GY, GX], "cores": GY * GX}

dev = T.get_device()
try:
    torch.manual_seed(0)
    sc_t = torch.randn(ROWS, H, S, S, dtype=torch.float32)
    dram = ttnn.DRAM_MEMORY_CONFIG

    def mk():
        return ttnn.from_torch(sc_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=dram)

    def timed(fn, n=3):
        fn()
        ts = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(out)
        return sorted(ts)[len(ts) // 2]

    def ref_arm():
        sc = mk()
        a = ttnn.softmax(sc, dim=-1)
        ttnn.deallocate(sc)
        return ttnn.typecast(a, ttnn.bfloat16)

    r = ref_arm()
    ref_out = ttnn.to_torch(r)
    ttnn.deallocate(r)
    res["dram_ms"] = round(timed(ref_arm) * 1e3, 3)

    height = ROWS * H * S
    res["height_rows"] = height
    res["tile_rows"] = height // 32
    res["tile_rows_per_core"] = (height // 32) / (GY * GX)
    res["fp32_MB_per_core"] = round(ROWS * H * S * S * 4 / (GY * GX) / 2**20, 3)
    shard = ttnn.create_sharded_memory_config(
        shape=(height, S), core_grid=ttnn.CoreGrid(y=GY, x=GX),
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)

    for step in ("to_sharded", "softmax_sharded", "typecast_sharded"):
        res[step] = "not reached"

    def l1_arm():
        sc = mk()
        sl = ttnn.to_memory_config(sc, shard)
        ttnn.deallocate(sc)
        res["to_sharded"] = "ok"
        a = ttnn.softmax(sl, dim=-1, memory_config=shard)
        res["softmax_sharded"] = "ok"
        ttnn.deallocate(sl)
        b = ttnn.typecast(a, ttnn.bfloat16, memory_config=shard)
        res["typecast_sharded"] = "ok"
        ttnn.deallocate(a)
        return b

    l = l1_arm()
    l1_out = ttnn.to_torch(l)
    ttnn.deallocate(l)
    res["equal"] = bool(torch.equal(ref_out, l1_out))
    res["max_abs_diff"] = float((ref_out.float() - l1_out.float()).abs().max())
    res["l1_ms"] = round(timed(l1_arm) * 1e3, 3)
    res["speedup"] = round(res["dram_ms"] / res["l1_ms"], 4)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:400]
finally:
    pass

print("RESULT " + json.dumps(res))
