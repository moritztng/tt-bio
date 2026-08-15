import sys, json
import ttnn
import tt_bio.tenstorrent as T

WH_L1 = 1466080          # MEASURED on UF-EV-A13-GWH02, ttnn.get_max_worker_l1_unreserved_size()
BH_L1 = 1499136          # p150 reference, filled in below if a BH probe is available

NAMES = ["SEQ_LEN_MORE_CHUNKING", "TRANSITION_BATCH_CHUNKING_THRESHOLD",
         "TRANSITION_W_CHUNKING_THRESHOLD", "TRIANGLE_ATT_CHUNK_SIZE_FAST",
         "TRANSITION_W_CHUNK_SIZE", "TRIANGLE_MULT_L1_MAX_SEQ_FAST",
         "TRIANGLE_MULT_L1_MAX_SEQ", "SMALL_GRID_SEQ_TILE",
         "SMALL_GRID_PAIR_TILE_AREA", "_IS_SMALL_GRID",
         "TRIANGLE_MULT_CHUNK_SIZE", "TRIANGLE_ATT_CHUNK_SIZE",
         "SDPA_CHUNK_MAX", "PAIRFORMER_PAD_MULTIPLE"]

def snapshot():
    return {n: getattr(T, n, "<absent>") for n in NAMES}

base = snapshot()

# --- Wormhole arm: grid (8,9) measured, per-core unreserved L1 measured.
ttnn.get_max_worker_l1_unreserved_size = lambda: WH_L1
T.CORE_GRID_MAIN = ttnn.CoreGrid(y=9, x=8)
T.COMPUTE_GRID_MAIN = (8, 9)
T._apply_grid_thresholds((8, 9))
wh = snapshot()

SIZES = [128, 256, 298, 320, 384, 512, 640, 768, 1024]
def derived(grid, gx13):
    T.COMPUTE_GRID_MAIN = grid
    d = {}
    for fast in (False, True):
        T._FAST_MODE = fast
        d["trimul_l1_max_seq_fast" if fast else "trimul_l1_max_seq"] = T._trimul_l1_max_seq()
        d["trimul_chunk_%s" % ("fast" if fast else "norm")] = {
            s: T._trimul_chunk_size(s, 128, 1) for s in SIZES}
        d["trimul_in_l1_%s" % ("fast" if fast else "norm")] = {
            s: str(T._triangle_mul_memory_config(s).buffer_type).split(".")[-1] for s in SIZES}
    T._FAST_MODE = False
    d["sdpa_chunks"] = {s: T._sdpa_chunks_shipped(T._padded_sdpa_len(s), T._padded_sdpa_len(s))
                        for s in SIZES}
    d["pair_row_tile"] = {s: T.pair_row_tile(s) for s in SIZES}
    return d

wh_d = derived((8, 9), False)

print("WH_JSON " + json.dumps({"constants": {k: wh[k] for k in NAMES},
                               "blackhole_baseline": {k: base[k] for k in NAMES},
                               "derived": wh_d}, default=str, indent=1))
