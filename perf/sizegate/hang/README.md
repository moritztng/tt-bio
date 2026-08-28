# A nondeterministic hang in the pair transpose, caught live

`opendde-640-pyspy.txt` is a py-spy dump of a real hang, taken while it was still up, during the
full size-ladder check of 2026-08-28 on qb2 card 0 (p300c).

The 640 aa opendde warmup fold printed `trunk 7/10` at 00:58:33 and then nothing. Twelve minutes
later the worker was still there, one thread runnable at 100% CPU and every other thread in
`futex_do_wait`, which is what a tt-metal host-side completion spin looks like when the device
never finishes. Two dumps two minutes apart show the identical frame:

    ttnn/decorators.py:473
    _pair_transpose_impl (tt_bio/tenstorrent.py:2506)   ttnn.to_layout(p, TILE_LAYOUT, DRAM)
    _pair_transpose (tt_bio/tenstorrent.py:2490)
    tt_bio/tenstorrent.py:4906                          pair-line-attention ending transpose
    _msa (tt_bio/protenix.py:2682)
    fold (tt_bio/opendde.py:538)

So it is the ROW_MAJOR route second `to_layout`, the one that tiles the permuted pair tensor back.
Not a matmul, so not the forced-`core_grid` multicast deadlock that the protenix-v1 512 aa hang
turned out to be.

Rare, not systematic. The same rung ran clean in the 91-minute check the day before, and the fold
that hung had already done the same transpose hundreds of times over seven trunk recycles before it
stopped. One hang in roughly two full checks of ~64 folds each is the rate the two runs support:
enough to threaten a green arm, not enough to characterise.

`_pair_transpose` lives in `tenstorrent.py`, not in a model file, so every folding model runs it.
opendde at 640 aa is only where it was seen.

`opendde-640-warmup.log` is the fold own log, ending at `trunk 7/10`.
