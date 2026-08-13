"""Corrected: time ONLY the device chain, not the host upload of the operand.

probe_l1_chain.py put from_torch() inside the timed region, so both arms paid a 33 MB host->device
tilize that swamped the device work (117 MB of implied traffic in 11.5 ms = 10 GB/s, 2.7 % of the
383.9 GB/s measured copy roof -- the giveaway that the number was not measuring the chain). The
bit-exactness verdict from that probe stands; its ratios do not.

Three arms over the same device-resident fp32 score block:
  dram     softmax + typecast, DRAM-interleaved. What main runs today.
  to_l1    shard it first, then softmax + typecast in L1. Pays the interleaved->sharded pass.
  in_l1    input already sharded, softmax + typecast in L1. The state a sharded producer would give.
"""
import json, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

S, H = 512, 4
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
GY = int(sys.argv[2]) if len(sys.argv) > 2 else 8
GX = int(sys.argv[3]) if len(sys.argv) > 3 else 8
res = {"S": S, "H": H, "rows": ROWS, "cores": GY * GX,
       "fp32_MB": round(ROWS * H * S * S * 4 / 2**20, 2),
       "fp32_MB_per_core": round(ROWS * H * S * S * 4 / (GY * GX) / 2**20, 3)}

dev = T.get_device()
try:
    torch.manual_seed(0)
    sc_t = torch.randn(ROWS, H, S, S, dtype=torch.float32)
    sc = ttnn.from_torch(sc_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                         device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    height = ROWS * H * S
    shard = ttnn.create_sharded_memory_config(
        shape=(height, S), core_grid=ttnn.CoreGrid(y=GY, x=GX),
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)

    def timed(fn, n=5):
        fn()
        ts = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            ttnn.deallocate(out)
        return round(sorted(ts)[len(ts) // 2] * 1e3, 4)

    def dram_arm():
        a = ttnn.softmax(sc, dim=-1)
        b = ttnn.typecast(a, ttnn.bfloat16)
        ttnn.deallocate(a)
        return b

    def to_l1_arm():
        sl = ttnn.to_memory_config(sc, shard)
        a = ttnn.softmax(sl, dim=-1, memory_config=shard)
        ttnn.deallocate(sl)
        b = ttnn.typecast(a, ttnn.bfloat16, memory_config=shard)
        ttnn.deallocate(a)
        return b

    sc_l1 = ttnn.to_memory_config(sc, shard)

    def in_l1_arm():
        a = ttnn.softmax(sc_l1, dim=-1, memory_config=shard)
        b = ttnn.typecast(a, ttnn.bfloat16, memory_config=shard)
        ttnn.deallocate(a)
        return b

    ref = dram_arm()
    ref_out = ttnn.to_torch(ref)
    ttnn.deallocate(ref)
    for name, fn in (("dram", dram_arm), ("to_l1", to_l1_arm), ("in_l1", in_l1_arm)):
        try:
            o = fn()
            got = ttnn.to_torch(o)
            ttnn.deallocate(o)
            res[name + "_equal"] = bool(torch.equal(ref_out, got))
            res[name + "_ms"] = timed(fn)
        except Exception as e:
            res[name + "_error"] = f"{type(e).__name__}: {e}"[:220]
    if res.get("dram_ms"):
        # DRAM traffic of the dram arm: softmax r+w fp32, typecast r fp32 + w bf16
        mb = ROWS * H * S * S * (4 + 4 + 4 + 2) / 2**20
        res["dram_GBs"] = round(mb / 1024 / (res["dram_ms"] / 1e3), 1)
        for k in ("to_l1", "in_l1"):
            if res.get(k + "_ms"):
                res[k + "_speedup"] = round(res["dram_ms"] / res[k + "_ms"], 4)
except Exception as e:
    res["error"] = f"{type(e).__name__}: {e}"[:400]

print("RESULT " + json.dumps(res))
