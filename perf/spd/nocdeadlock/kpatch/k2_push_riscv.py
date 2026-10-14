# K2: publish tiles_received from the RISC-V after the pack thread has drained, instead of a Tensix STOREREG.
REL = "tt_metal/hw/ckernels/blackhole/metal/llk_io/llk_io_pack.h"
OLD = """    TT_SETDMAREG(0, tiles_received_new, 0, LO_16(p_gpr_pack::NUM_MSGS_RECEIVED));
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);  // wait for pack to finish
    TT_STOREREG(p_gpr_pack::NUM_MSGS_RECEIVED, (uint32_t)&tiles_received_ptr_tensix[0]);
"""
NEW = """    tensix_sync();
    *get_cb_tiles_received_ptr(operand) = tiles_received_new;
"""
def _p(src):
    assert src.count(OLD) == 1, "k2 anchor"
    return src.replace(OLD, NEW)
PATCHES = {REL: _p}
