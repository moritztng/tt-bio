# bfp8 on a real design round: the prediction, written before the measurement

Base: `origin/wk/bcx-tracewire` f8b9070b2, merged into `wk/bcx-bfp8` as 3b82b505a.

The 1.58x (7.71 -> 4.89 s) this row predicted on 2026-09-24 is on the n=256 gradient step and
assumes both arms sit at their own DRAM roofs. Neither holds for the unit the campaign now runs,
so it does not carry. Converted here from bcx-tracewire's traced arm
(`perf/bcx_tracewire/round_interleave3_xmsa_final_seed100.json`: pdl1, n=211 padded to 224,
extra-MSA swap on, AICLK median 1350):

* Traced design round 15.93 s, of which trunk step 6.93 s, of which device wait 6.48 s
  (`TraceWire.seg`, 51.84 s over 8 rounds) and host 0.21 s.
* bcx-realcensus graded 100 % of the real block's device time against byte roofs, so bfp8's
  device ceiling is the tile byte ratio, 2048/1088 = 1.882x, applied to device time only.
* Ceiling: device 6.48 -> 3.44 s, 3.04 s removed. Trunk step 6.93 -> 3.89 s (1.78x).
  **Design round 15.93 -> 12.89 s, 1.24x.** On the campaign's 16.870 s round (bcx-extramsa's ON
  arm) the same 3.04 s is 1.22x.
* That is an upper bound. It halves every byte, including float32 residual temporaries and
  layout traffic a dtype flag may not reach, and charges nothing for typecasts.
* Point expectation for the conversion that exists today, `tenstorrent.set_fast_mode(True)`:
  well under the ceiling. Its own comment (`tt_bio/tenstorrent.py:1642`) records 0.95x on an
  inference fold, because it bundles chunk-size and weight-dtype changes with the activation
  dtype. I predict 1.00x to 1.10x on the round.
* Worth gating only at >= 1.05x on the round, interleaved, AICLK >= 1300 sampled during.
