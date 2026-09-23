# What these three walks should do, written before their rungs landed

Pre-registered 2026-09-23T01:37Z, while rfd3 512 and pxdesign 512 were still running and no rung
above any recorded top had returned. The campaign's standing rule is to pre-register a falsifier
before an arm; this is that, and it is committed so it cannot be fitted afterwards to whatever
came out. Every number below is read from `tt_bio/size_limits.py` or from the fixture, not
guessed.

## pxdesign — the sharp one

The row's own evidence already names a measured failure above its top, on a different axis:
"no measured failure until **1664 TOKENS** (ws:ceiling-pxdesign, one 1.42 GB pair-transition
buffer), with 1088 and 1408 tokens both passing twice."

This walk counts DESIGN_TARGET and adds an 80-residue binder, so tokens = target + 80:

| rung (target) | 512 | 768 | 896 | 1024 | 1280 | 1536 |
|---|---|---|---|---|---|---|
| tokens | 592 | 848 | 976 | 1104 | 1360 | **1616** |

**Prediction: all six rungs PASS.** 1616 tokens is 48 short of the 1664 where the only measured
pxdesign failure sits, and 1408 tokens is already known to pass twice. The top rung is therefore
predicted to clear by a margin thinner than one rung step, which is what makes this falsifiable
rather than safe: if 1536 fails, it should fail on **one oversized pair-transition buffer of
about 1.4 GB**, not on fragmentation and not on cumulative residency. A failure of any other
class at 1536, or a failure at 1280 (1360 tokens) or below, refutes the reading that the 1664
wall is a token wall and means the binder or the multi-chain crop changed the shape of the
problem.

## boltzgen — arithmetic, then a confirmation

`big_1831.cif` carries 8.08 heavy atoms per residue, so a 1536-residue target is **12405 atoms**
(counted by `check_structure.py` on the crop itself, not multiplied out). The row's Wormhole
ceiling is 14786 TARGET_ATOMS, walked on THIS box, so 1536 residues sits at 84 % of a size that
already designed here in 2242.9 s.

**Prediction: PASS**, in roughly 1500-1900 s, and it tells us nothing new if it does. It is worth
the chip only because it converts an inference ("12405 < 14786") into a measured cell. A FAILURE
would be the interesting outcome and would mean the atom count is not the axis that governs, i.e.
that the 14786 row is denominated wrongly.

## rfd3 — genuinely open

This is the one with no prior. Its 1024 is a LADDER_TOP set by the platform's own `max_residues`
fence, so nothing above it has ever been tried on any part, and the row records host RSS rising
to 21.4 GB at 1024 against a 12 GiB chip. The pair track is O(N^2) in memory, so 1536 asks for
2.25x the pair bytes of 1024.

**Prediction: rfd3 fails before 1536**, most likely at 1280 or 1536, and the failure class is
DRAM rather than L1. Stated this way so it can be wrong in a useful direction: if rfd3 reaches
1536 cleanly, the recorded 1024 was never a capability statement at all and the platform fence is
the only thing that was ever stopping it.

The L1 criticals its 512 rung logs (`33554432 B L1 buffer across 72 banks`, circular buffers on
[(0,0)-(7,1)] growing to 1602848 B) are predicted to be **non-fatal** — the documented
`_swiglu_resident` path drops the residency and not the fold. If a rung dies on one of those
instead, the L1 budget is the wall and this row hands the finding to whoever owns the budget.

## What would void all three

Any rung whose log carries `device contention, nothing ran`, or whose passing rungs did not share
one chip. Those are measurements of the fleet, not of the model, and `report.py` refuses them.
