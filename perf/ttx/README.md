# TTX phase 0 — is the newer stack usable on Blackhole, and whose bug is the 18 A fold?

`b2z-ttnn-upgrade` found that ttnn 0.78.0 folds the 298 aa control 18.3 A away from the 0.68.0 arm
at half the plDDT, on Wormhole, through a binary-patched UMD. Nobody had said which side was wrong,
and "the new build disagrees with the old build" names no culprit. This directory answers both on
Blackhole, the part the published number runs on.

## Running the newer stack on a p300c

`run_arm.sh new <card> 298 <outdir>` folds on ttnn 0.78.0; `old` folds on the pin. Two things 0.78.0
needs on qb2 that 0.68.0 did not:

* `TT_MESH_GRAPH_DESC_PATH=perf/ttx/mgd/bh_1x1.textproto`. Without it `ttnn.open_device` dies at
  `tt_metal/llrt/tt_cluster.cpp:275`, "Custom fabric mesh graph descriptor path must be specified for
  CUSTOM cluster type". The box lands in CUSTOM because upstream's `mobo_to_bus_ids` table has no
  entry for its motherboard (`Unknown motherboard 'B850M-C'`), and upstream's own p300 descriptors
  describe 2 and 4 chips, which cannot map onto one pinned card. The committed descriptor is 1x1.
* SFPI 7.72.0, which 0.78.0 pins in `tt_metal/sfpi-version`. The system toolchain in
  `/opt/tenstorrent/sfpi` is 7.35.3 and rejects `-ftt-nttp`, `-ftt-constinit`, `-ftt-consteval`,
  `-ftt-no-dyninit`. Unpack it into the venv's own `runtime/sfpi`, which tt-metal resolves first, so
  the shared system toolchain stays where every other task on the box expects it.

No UMD patch and no source build: the released cp312 wheel opens a Blackhole chip in 2.6 s.

## The answer, scored against something that is neither stack

Both arms fold `cdk2x2_298` on qb2 card 2 with `TT_BIO_SHARED_DRAW_SEED=0`, so the diffusion noise
is the same stream upstream's own reference consumed (first draw `(1, 2400, 3)`,
`84c748a64d2a8963`, checked in both run records). Every hand-written fused kernel is off on both
arms, since they do not compile against 0.78's LLK surface. Scored with
`perf/b2z2_fusebias/score.py` unmodified against the committed `gpurefshared-s0` CIF, which is
upstream `boltz==2.2.1` at fp32.

| arm | all-atom RMSD vs upstream fp32 | CA-lDDT vs upstream | CA RMSD vs 1HCL | plDDT |
|---|---|---|---|---|
| ttnn 0.68.0, the pin | **0.4241 A** | 0.99636 | 0.7457 A | 0.9132 |
| ttnn 0.78.0 | **19.5562 A** | 0.31453 | 19.1921 A | 0.3597 |
| upstream fp32, 4 seeds | (reference) | | 0.5974 - 0.7398 A | 0.9092 - 0.9115 |

The third column is the one that closes the attribution. 1HCL is the experimental crystal structure
and no sampler argument reaches it. The pin lands at 0.7457 A, inside the reference's own four-seed
spread. 0.78.0 lands at 19.1921 A with CA-lDDT 0.31. It is not a drift and it is not fixture
sensitivity: on 0.78.0 the model does not fold the protein.

So the 18.3 A is **not ours in the sense of a fixture or a harness artifact**, and it is not the UMD
patch either, which this Blackhole run does not use. What remains open, and is the next step, is
whether 0.78.0 mis-serves a *configured* call tt-bio makes or is broken in an op outright: 20 ops at
their default configurations agree between the stacks, 16 of them bit-identically
(`perf/b2z_ttnn`), so the failure lives somewhere those defaults do not reach.
