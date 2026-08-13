import sys, json
import torch, ttnn
sys.path.insert(0, ".")
import tt_bio.tenstorrent as T
S, H, R = 512, 4, 12
dev = T.get_device(); res = {}
sh = ttnn.create_sharded_memory_config(shape=(R*H*S, S), core_grid=ttnn.CoreGrid(y=8, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
t = ttnn.from_torch(torch.randn(R,H,S,S), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                    device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
for name in ("in_place", "out_of_place"):
    x = ttnn.to_memory_config(t, sh)
    try:
        y = ttnn.softmax_in_place(x) if name == "in_place" else ttnn.softmax(x, dim=-1, memory_config=sh)
        res[name] = str(y.memory_config().memory_layout)
        ttnn.deallocate(y)
    except Exception as e:
        res[name] = f"ERR {type(e).__name__}: {e}"[:180]
        ttnn.deallocate(x)
print("RESULT " + json.dumps(res))
