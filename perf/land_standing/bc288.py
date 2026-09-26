#!/usr/bin/env python3
"""Does the flip change BC2's taped path at 288, as the flipped default's comment claims?

My sweep modelled `one_k_chunk=True`, which is what openfold3.trunk gets. BindCraft 2 reaches
`_tri_att_sdpa_hifi_inner` WITHOUT it, and the source comment says 288 is the length where the old
ladder declines every rung and the lever serves (288, 288). That claim is load-bearing in the
justification I wrote into the default, so it should be checked on this tree rather than quoted
from a comment.
"""
import sys

sys.path.insert(0, "/home/ttuser/.coworker/wt/land-standing/perf/land_standing")
import capreach as C                                               # noqa: E402

print("\n\nBC2's path (one_k_chunk=False), the length the comment names:")
for n in (256, 288, 320, 384):
    off = C.serves(n, dividing_k=False, one_k_chunk=False)
    on = C.serves(n, dividing_k=True, one_k_chunk=False)
    verdict = "OPENED by the lever" if (off is None and on) else (
        "changed" if off and on and (off[1:] != on[1:]) else "inert")
    print("  n=%-5d off=%-24s on=%-24s %s" % (n, off, on, verdict))

print("\nand openfold3.trunk's path (one_k_chunk=True) at the same lengths, for contrast:")
for n in (256, 288, 320, 384):
    off = C.serves(n, dividing_k=False)
    on = C.serves(n, dividing_k=True)
    verdict = "OPENED" if (off is None and on) else (
        "changed" if off and on and (off[1:] != on[1:]) else "inert")
    print("  n=%-5d off=%-24s on=%-24s %s" % (n, off, on, verdict))
