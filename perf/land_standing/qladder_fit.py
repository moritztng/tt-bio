#!/usr/bin/env python3
"""Does a NARROW q make the full-width k fit, at the lengths where the wide q does not?

The dividing-k lever lands on a chunked k at 832 only because every (q, k=832) rung it is allowed
to try is refused on l1_budget. The rungs it is allowed to try are the q values that divide the
padded length AND survive TT_BIO_TRIATT_NARROW_Q_FALLBACK's `q >= prod/2` bound. At 832 that bound
removes 64 and 32, and 64 is the one the neighbours' equivalent already runs at 1088.

This asks the same question the engine asks, with the engine's own model -- `plan_for_shape` and
`cb_fits_l1`, the pair `_k_chunks_up` uses -- over EVERY 32-aligned dividing q rather than only
the widest. No device, no fold: it is a host-side budget calculation.

704 and 864 are the controls. Both already serve a single-chunk k in production, so the model has
to reproduce that or it is not measuring what the device measures.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing")
import ttnn                                                        # noqa: E402
from tt_bio import tenstorrent as T                                # noqa: E402
from tt_bio import sdpa_generic as SG                              # noqa: E402
from tt_bio import triatt_sdpa as TS                               # noqa: E402

HEADS, HEAD_DIM = 4, 32              # OpenFold3 trunk triangle attention
DTYPE = ttnn.bfloat16
CORES = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
TILE = T.SDPA_CHUNK_TILE

print("grid %s -> %d cores, heads=%d head_dim=%d dtype=%s"
      % (T.COMPUTE_GRID_MAIN, CORES, HEADS, HEAD_DIM, DTYPE))


def divisors(padded):
    return sorted({padded // n for n in range(1, padded // TILE + 1)
                   if padded % n == 0 and (padded // n) % TILE == 0}, reverse=True)


def fits(padded, qc, kc):
    """True when the engine's own budget model accepts this (q_chunk, k_chunk)."""
    q_pf = TS.q_parallel_factor(padded, HEADS, qc, CORES, cap=0)
    p = SG.plan_for_shape(padded, HEADS, HEAD_DIM, qc, kc, grid=(CORES, 1),
                          split=(max(CORES // (HEADS * q_pf), 1), HEADS, q_pf), dtype=DTYPE)
    if p["q_per_core"] != 1 or p["nh_per_core"] != 1 or p["use_padded_mask"]:
        return None, p
    pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
    ok = SG.cb_fits_l1(p, mask_cb_tiles=pers,
                       **{"%s_dtype" % o: DTYPE for o in ("q", "k", "v", "mask", "out")})
    return bool(ok), p


for n in (704, 832, 864, 1088):
    padded = T._padded_sdpa_len(n)
    prod = T._sdpa_chunks_shipped(n, n)[0]
    offered = T._tri_att_q_chunks(n, n)
    divs = divisors(padded)
    print("\n=== n=%d padded=%d  shipped q=%d  ladder offers %s ===" % (n, padded, prod, offered))
    print("    32-aligned dividing q: %s" % divs)
    print("    full-width k = %d, asking every dividing q:" % padded)
    for qc in divs:
        ok, p = fits(padded, qc, padded)
        tag = "OFFERED" if qc in offered else "not offered"
        why = "" if ok is not None else "  (rejected by plan: q_per_core=%s nh_per_core=%s padded_mask=%s)" % (
            p["q_per_core"], p["nh_per_core"], p["use_padded_mask"])
        print("      q=%-5d k=%-5d fits_l1=%-5s  [%s]%s" % (qc, padded, ok, tag, why))
