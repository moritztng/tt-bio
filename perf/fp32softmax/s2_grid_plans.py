import sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/openbind-perf-p5")
import tt_bio.tenstorrent as T
L1 = 1532416
T._batched_matmul_config = lambda b, m, k, n, e, rung=0: T._batched_matmul_search(
    b, m, k, n, e, tuple(T.COMPUTE_GRID_MAIN), L1, rung)
T._FP32_SOFTMAX_L1_FLOAT_CORES = True

def bmm(tok, heads, hd=32):
    t = -(-tok // 32)
    return heads, t, -(-hd // 32), t, -(-hd // 32)

for grid in ((13, 10), (11, 10), (8, 8)):
    T.COMPUTE_GRID_MAIN = grid
    T._fp32_softmax_l1_plan.cache_clear()
    print("=== grid %s (%d cores)" % (grid, grid[0]*grid[1]))
    for heads in (2, 4, 8):
        line = []
        for S in (256, 512, 576, 640, 768, 1024):
            hpr = heads * S
            per_row = hpr * S * 4
            tuned = T._fp32_softmax_l1_rows(per_row, hpr)
            r, c = T._fp32_softmax_l1_plan(per_row, hpr, S, None, bmm(S, heads))
            nb_a = -(-S // tuned) if tuned else 1
            nb_c = -(-S // r) if r else 1
            line.append("%d:%d/%d->%d/%d(%d->%dblk)" % (S, tuned, 64, r, c, nb_a, nb_c))
        print("  heads=%d  %s" % (heads, "  ".join(line)))
