"""Fill one chip's DRAM in big blocks and run untilize -> reshape -> tilize over each, the op chain
chip 2 hung in (rm_reshape_interleaved waiting on a DRAM write ack). A hang raises after
TT_METAL_OPERATION_TIMEOUT_SECONDS; the block index and its DRAM address say where it hung.
    TT_VISIBLE_DEVICES=N dramstress.py ROUNDS [BLOCK_ROWS]"""
import os, sys, time
os.environ.setdefault("TT_METAL_OPERATION_TIMEOUT_SECONDS", "10")
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn, torch

rounds = int(sys.argv[1]); rows = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
dev = ttnn.open_device(device_id=0)
print("dram channels", dev.dram_grid_size() if hasattr(dev, "dram_grid_size") else "?", flush=True)
blocks = []
t0 = time.time()
while True:
    try:
        b = ttnn.full([rows, rows], fill_value=float(len(blocks) % 7 + 1), dtype=ttnn.bfloat16,
                      layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    except Exception as e:
        print(f"full at {len(blocks)} blocks: {str(e)[:120]}", flush=True)
        break
    blocks.append(b)
mb = rows * rows * 2 / 2**20
print(f"{len(blocks)} blocks of {mb:.0f} MiB ({len(blocks)*mb/1024:.1f} GiB) in {time.time()-t0:.0f}s", flush=True)
# leave room for one block of scratch for the op outputs
for b in blocks[-4:]:
    ttnn.deallocate(b)
del blocks[-4:]
bad = 0
for r in range(rounds):
    for i, b in enumerate(blocks):
        t = time.time()
        x = ttnn.to_layout(b, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.reshape(x, [rows * 2, rows // 2])
        z = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        m = float(ttnn.to_torch(ttnn.max(z)).float().max())
        want = float(i % 7 + 1)
        ok = m == want
        bad += not ok
        print(f"round {r} block {i} addr {b.buffer_address():#x} max {m} {ok if ok else BAD} {time.time()-t:.2f}s", flush=True)
        ttnn.deallocate(x); ttnn.deallocate(y); ttnn.deallocate(z)
print(f"DONE rounds={rounds} blocks={len(blocks)} bad={bad} {time.time()-t0:.0f}s", flush=True)
ttnn.close_device(dev)
