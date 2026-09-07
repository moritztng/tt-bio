"""Print the Transition W-chunk constants as this part actually resolves them."""
import sys
import tt_bio.tenstorrent as T

dev = T.get_device()
print("grid COMPUTE_GRID_MAIN =", T.COMPUTE_GRID_MAIN)
print("_IS_SMALL_GRID         =", T._IS_SMALL_GRID)
import ttnn
print("l1_unreserved/core     =", ttnn.get_max_worker_l1_unreserved_size())
print("SEQ_LEN_MORE_CHUNKING          =", T.SEQ_LEN_MORE_CHUNKING)
print("TRANSITION_W_CHUNKING_THRESHOLD=", T.TRANSITION_W_CHUNKING_THRESHOLD)
print("TRANSITION_W_CHUNK_SIZE        =", T.TRANSITION_W_CHUNK_SIZE)
print("TRANSITION_L1_CHUNK_BYTES_PER_CORE =", T.TRANSITION_L1_CHUNK_BYTES_PER_CORE)
for W in (512, 608, 640, 672, 768, 896, 1024, 1088):
    thr = T.TRANSITION_W_CHUNKING_THRESHOLD
    cs = T.TRANSITION_W_CHUNK_SIZE
    if W > thr:
        parts = [min(w + cs, W) - w for w in range(0, W, cs)]
        print(f"  W={W:5d} CHUNKED  parts={parts}  all_tile_aligned={all(p % 32 == 0 for p in parts)}")
    else:
        print(f"  W={W:5d} whole")
