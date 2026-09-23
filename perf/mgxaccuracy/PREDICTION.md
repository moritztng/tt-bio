# Pre-registered: what the requeued pxdesign jobs must read, and what refutes the mechanism

Written **before** any of these jobs held a chip. The point of writing it down is that the
mechanism below was arrived at after seeing the 95.183 A, so on its own it is a retrodiction;
it becomes evidence only if it predicts a measurement not yet taken.

## The mechanism, stated so it can be wrong

pxdesign sees the target only as a 64-bin distogram over **2-22 A**
(`tt_bio/pxdesign/featurize.py:40`, `bins = sum(cdist > linspace(2, 22, 63))`). Beyond 22 A two
conditioned tokens are indistinguishable to it.

**The discriminator is not how much of the distogram saturates.** Measured
(`perf/mgxaccuracy/contact.py conditioning_graph`):

| target crop | saturated | sub-22 A components | inter-chain edges | fit_rmsd |
|---|---|---|---|---|
| big_1831 512 | 73.5 % | **1** (512) | 0 (one chain) | **0.0755 A** |
| gpb dimer 512 | 75.9 % | **1** (512) | 0 (one chain) | not yet run |
| gpb dimer 1024 | 85.9 % | **1** (1024) | 4715 | not yet run |
| gpb dimer 1536 | 90.0 % | **1** (1536) | 8316 | not yet run |
| big_1831 1536 | 90.9 % | **2** (1008 + 528) | **0** | **95.183 A** |

Saturation separates nothing: a target at 73.5 % gives 0.0755 A and the valid 1536 crop at
90.0 % is *less* saturated than the invalid one at 90.9 %. **Connectivity separates them.** A
conditioning graph with two components carries nothing about where one component sits relative
to the other, so the rigid fit that recovers the output frame
(`tt_bio/pxdesign/write.py:47`) has no determined answer and lands at the scale of the
separation. 95.183 A against a 246.9 A chain separation is that scale.

## One half of it is a proof, not a prediction

`perf/mgxaccuracy/rigidity.py` moves chain B rigidly and counts how many inter-chain pairs
leave the bin they were in. Bins are (22-2)/63 = **0.317 A** wide, so a resolvable pair
constrains its distance to about +/-0.16 A.

| motion of chain B | gpb dimer 1536 crop | big_1831 1536 crop |
|---|---|---|
| resolvable inter-chain pairs | **8316** (1.42 %) | **0** (0.00 %) |
| translate 0.1 A | 1111 change bin (13.4 %) | **0** |
| translate 0.5 A | 5124 (61.6 %) | **0** |
| translate 2.0 A | 7487 (90.0 %) | **0** |
| translate 10 A | 8144 (97.9 %) | **0** |
| rotate 1 deg | 3462 (41.6 %) | **0** |
| rotate 30 deg | 8083 (97.2 %) | **0** |

**On `big_1831.cif`'s 1536 crop the conditioning is invariant under every one of those
motions.** Chain B can be moved 10 A or turned 30 degrees and not one bit of the input
changes. The relative placement is therefore not a function of the input, so no model and no
algorithm recovers it, and a fit that tries lands at the scale of the separation. 95.183 A
against a 246.9 A separation is that. **This needs no device run and is not falsifiable by
one** — it is a property of the fixture.

What is still a prediction is SUFFICIENCY on the valid target: 8316 pairs at +/-0.16 A
overdetermine 6 degrees of freedom by three orders of magnitude, and a 0.1 A translation
already moves 13.4 % of them, so the placement is pinned at about the 0.1 A scale. Whether the
model exploits that is what P1 tests.

## The predictions

**P1 — the mechanism.** `pxdesign` on `gpb_dimer_1646.cif` at a 1536-residue crop (1616 tokens,
1 component, 8316 inter-chain edges) reads **fit_rmsd well under the 15 A
`PXDESIGN_MAX_FIT_RMSD` gate, and in the same order as the 512 rungs** — call it under 2 A.

**REFUTED IF** it comes back in the tens of angstrom. That would mean token count, not graph
connectivity, is the carrier, and the next step is the shared-noise step comparison against the
fp32 reference on the same 1616-token input.

**P2 — the live-reachable region, which nobody has measured.**
`aiand-bio/japanfold/catalog.py:21` caps a user at `max_residues = 1024` and pxdesign has no
lower override, so **1024 target residues + a 200-residue binder = 1224 tokens is reachable
today**. The highest token count any pxdesign `fit_rmsd` has ever been measured at is **1024**
(824 + 200 and 960 + 64, both sub-angstrom). 1024 -> 1224 is unmeasured. Predicted **sub-2 A**,
because that crop is 1 component with 4715 inter-chain edges.

**If P2 fails this is a live customer bug**, not a campaign blocker, and it changes the
severity of the whole finding. That is why it runs first.

**P3 — the ordering.** If P1 and P2 both hold, then no pxdesign conditioning defect exists on
any valid target at any reachable size, and the 95.183 A belongs entirely to
`perf/bhdesign/make_big_target.py`'s fixture. If P1 holds and P2 fails, the carrier is between
1024 and 1224 tokens and is not connectivity.

## What would make this whole line wrong

The 95.183 A could be the FIT rather than the design — `write.py` computes the residual and
places the binder with the same transform, so a misindexed fit produces both symptoms. The
observation that argues against it: the 512 rungs use the identical code path and read
0.0755 A, and `conditioned_tokens` is recorded as 1536 on the failing runs, so the mask is the
right size. It is not fully excluded, and P1 discriminates it: a misindexing that depends only
on multi-chain input would fail on the gpb dimer too.
