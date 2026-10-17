"""Write the size-ladder exemption reasons T1 (the fused trimul in-projection) left as TODO.

Every release gate since 50bcbd42c fails the same lines on these, and each needed a hand waiver.

protenix-v2, TRIMUL_TAIL_F1 at 512 and 640 (p300c, p150a, tt-galaxy-wh-l). F1's one call site is
guarded by `g_out_fused is None`. At 512 and 640, trimul_t1_c256 takes the in-projection on the
DRAM route with one channel chunk and returns g_out with it, so F1 is never offered: 0 served,
0 declined. The other 160 tails (the narrow 2x4x2 key, which T1 declines) also arrive with a
fused g_out, from the incumbent in-projection. At 256 the channel loop runs on L1 and at 768+
it splits into several chunks, T1 is not on either route, and F1 serves its 1048 calls. The
TRIMUL_IN_PROJ_DUAL_NOC counts mark the same boundary: 160 served at 512/640 against 2256 at
768.

nesso1, REBLOCK_PERMUTE_GATED and TRIMUL_MASK_AFTER_MOVE at 256..896 (all three cards).
trimul_t1 + trimul_t1_dram serve every trimul in-projection there, and T1 replaces the gated
channel move and the mask-after-move, so neither is called. From 1024 the channel loop splits,
T1 declines and both serve 384 calls. Gate rel-803db0fc1 on WH reads the same 0/0 cells as the
baseline, so the BH reason holds there.

opendde at 256. TRIMUL_IN_PROJ_DUAL_NOC declines every call on `memory` (all cards): at 256 the
trimul channel loop runs on L1, and the dual-NOC matmul only takes a DRAM-interleaved activation
and output (mm_dualnoc.eligible). REBLOCK_PERMUTE_GATED and TRIMUL_MASK_AFTER_MOVE read 0/0 on
the Blackhole cards only: the gated move's L1 leg (reblock_permute.eligible_gated) opens for
288 <= N <= 352, or below 288 when Nt*Nt groups reach 0.68 per core. N=256 is 64 groups; a
Wormhole chip's 72 cores need 49 and both serve 25792 there, the 110-core p300c needs 75 and the
130-core p150a 88, so the gated move declines (its rejects land in REBLOCK_PERMUTE's counter as
gated_window_BufferType.L1 x6448) and the mask-after-move that rides it is never reached.

Each entry must still be dark with the counts this was written against, or the script refuses.
"""
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
import release_gate as rg

D = pathlib.Path("docs/size_ladder_baseline.d")
CARDS = ("p300c", "p150a", "tt-galaxy-wh-l")

PX_WHY = (
    "F1's one call site (`fused_tail`) is guarded by `g_out_fused is None`, and at 512/640 "
    "trimul_t1_c256 takes the in-projection (DRAM route, one channel chunk) and returns g_out with "
    "it; the 160 narrow-key tails T1 declines (t1:key=2x4x2) get a fused g_out from the incumbent "
    "in-projection. So F1 is never offered. At 256 (L1 channel loop) and 768+ (several channel "
    "chunks) T1 is off the route and F1 serves 1048; TRIMUL_IN_PROJ_DUAL_NOC marks the same "
    "boundary, 160 served here vs 2256 at 768. Falsifier: F1 serving at 512/640 with T1 on means "
    "g_out stopped riding T1")
NS_WHY = (
    "trimul_t1 + trimul_t1_dram serve every trimul in-projection at 256-896, and T1 replaces the "
    "gated channel move and the mask-after-move, so neither is called. From 1024 the channel loop "
    "splits, T1 declines and both serve 384. Same on BH and WH (gate rel-803db0fc1 reads these "
    "cells 0/0 on WH). Falsifier: a call here means T1 stopped taking the in-projection")

OD_DUAL_WHY = (
    "at 256 the trimul channel loop runs on L1, and the dual-NOC matmul only takes a DRAM-"
    "interleaved activation and output (mm_dualnoc.eligible rejects on `memory`). From 512 the "
    "loop is on DRAM and it serves 184. Falsifier: a `memory` decline at 512+ means the DRAM "
    "route moved")
OD_GATED_WHY = (
    "the gated move's L1 leg (reblock_permute.eligible_gated) opens for 288 <= N <= 352, or below "
    "288 when the Nt*Nt groups reach L1_GROUPS_PER_CORE=0.68 per core. N=256 is 64 groups: the "
    "110-core p300c needs 75 and the 130-core p150a 88, so the gated move declines (counted under "
    "REBLOCK_PERMUTE as gated_window_BufferType.L1 x6448) and the mask-after-move that rides it is "
    "never reached. A Wormhole chip's 72 cores need 49, and both serve 25792 there. Falsifier: a "
    "call here on Blackhole means the L1 floor or the grid changed")
BH = ("p300c", "p150a")

JOBS = [("protenix-v2", CARDS, ("512", "640"), ("TRIMUL_TAIL_F1",), PX_WHY),
        ("nesso1", CARDS, ("256", "512", "640", "768", "896"),
         ("REBLOCK_PERMUTE_GATED", "TRIMUL_MASK_AFTER_MOVE"), NS_WHY),
        ("opendde", CARDS, ("256",), ("TRIMUL_IN_PROJ_DUAL_NOC",), OD_DUAL_WHY),
        ("opendde", BH, ("256",), ("REBLOCK_PERMUTE_GATED", "TRIMUL_MASK_AFTER_MOVE"), OD_GATED_WHY)]

for model, cards, rungs, flags, why in JOBS:
    path = D / f"{model}.json"
    data = json.loads(path.read_text())
    n = 0
    for card in cards:
        levers = data["cards"][card]["models"][model]["levers"]
        for rung in rungs:
            for flag in flags:
                e = levers[rung][flag]
                assert rg._size_ladder_dark(e), (model, card, rung, flag, "not dark")
                assert not e.get("served"), (model, card, rung, flag, e)
                old = str(e.get("reason") or "")
                if old.endswith(why):
                    continue                    # written by an earlier run
                assert not old or old.startswith("TODO"), (model, card, rung, flag, old)
                e["reason"] = rg._size_ladder_reason_evidence(e) + ": " + why
                n += 1
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{model}: {n} reasons written")
