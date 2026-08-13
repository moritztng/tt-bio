# S1 prediction, written BEFORE the run (2026-08-13, exec pass 1)

Byte model for the whole tail, in units of the bf16 score tensor (el = rows*H*S*S, 1 unit = 2*el B):

    dram arm      typecast 3 + add_ 4.25 + softmax 4 + typecast 3   = 14.25 units
    shard_rep     DRAM->L1 read 1 + (all four steps in L1) + L1->DRAM write 1 = 2.0 units

Pure byte model says 7.1x. It will not land there, because the L1 arm is no longer DRAM-bound and
`probe_l1_chain2.py` already showed the L1 pair costs 0.1656 ms for work the byte model prices at
zero DRAM traffic. So:

  * plan's band, carried forward: `shard_rep` 1.7x-2.1x, `shard_softmax` 1.2x-1.4x.
  * this pass's own band from the byte model plus the measured L1 residual: `shard_rep` 2.0x-3.5x.

  * `shard_bcast` (bias left interleaved, broadcasting over the leading dim onto a height-sharded
    destination) is predicted to REFUSE. That is kill gate 3's trigger and why `shard_rep` exists.
