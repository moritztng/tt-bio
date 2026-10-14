# K1: TRISC2 waits for its own Tensix pack thread to drain after every tiles_received publish (tensix_sync).
# Stuck packer -> TRISC2 now hangs in llk_push_to_brisc; lost store -> no change in where it hangs.
REL = "tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h"
OLD = "    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr_tensix[0]);\n}"
NEW = "    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr_tensix[0]);\n    tensix_sync();\n}"
def _p(src):
    assert src.count(OLD) == 1, "k1 anchor"
    return src.replace(OLD, NEW)
PATCHES = {REL: _p}
