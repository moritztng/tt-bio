# narrow-q: attribution of the size-ladder reds (2026-09-23)

All three card-2 slices of the size-ladder arm ended red at `4cbada0e8` / `243f30bb2`. The arm has
been red on main since 2026-09-17, so the only question is which red cells belong to the lever.

## By construction

`_tri_att_q_chunks` returns the flag-off tuple whenever `padded % prod == 0`, which holds at every
multiple of 256. So nothing at 256, 512, 768 or 1024 can be the lever's. That covers openbind's
TRIATT_PERSISTENT_MASK at those rungs, nesso1/1024 SDPA_Q_CHUNK_FITS, and every SDPA_WIDE_K,
TRIATT_SDPA_HIFI, QKV_MM_CONFIG, PAIR_PROJ_MINIMAL_MATMUL and REBLOCK_PERMUTE line (main commits
named in `narrowq_size_ladder_rf3.md`).

## By a flag-only census A/B at the cells where the lever can fire

`narrowq_cell_ab.py` calls the arm's own `_run_census_fold` (same fixture, 6 steps, 1 sample,
seed 0) on the candidate tree with only `TT_BIO_TRIATT_NARROW_Q_FALLBACK` flipped. qb2 cards 0/2,
p300c. Host loadavg ~12 from a CPU-only f64 job, so no runtime here is a measurement.

    cell              lever-census diff off -> on                  CIF digest
    protenix-v2/896   none (PAIR_TRANSPOSE 0.868 in BOTH arms)     identical
    openbind/640      none                                         identical
    openbind/896      TRIATT_PERSISTENT_MASK declined 5 -> 4       DIFFERENT
    openfold3/768     none                                         identical

- protenix-v2/896 PAIR_TRANSPOSE_VIA_ROW_MAJOR 0.998 -> 0.868 is main's.
- openfold3/768 warm-up timeout (07:39Z, stuck at trunk 1/4 for 1800 s) did not reproduce: the
  cell folded in 38.4 s off and 40.1 s on. Transient, and at a length where the lever is a no-op.
- **openbind/896 is the lever's, and it is not bit-exact.** Three legs per arm, interleaved: every
  off leg is one structure, every on leg is another. The narrower q chunk frees enough L1 for one
  more triangle-attention call to take the persistent-mask kernel, and that kernel rounds
  differently. The branch's "q_chunk splits output rows, so no reduction order changes" argument
  is right for the SDPA itself and misses this second-order effect. rf3 896 took the same
  PERSISTENT_MASK shift (0 -> 0.333) and stayed bit-exact over 12 legs; openbind does not.
  At the 6-step census config the move is 2.218 A CA, on a structure with chain_break 1.0 and
  pLDDT 0.32, so that figure scores nothing.

## Consequence

narrow-q is a user-visible change on openbind at firing lengths and has to be scored there at the
production config, against the bar with a seed floor, before its default can ship. Result, uninformative on this fixture:
`narrowq_openbind_acc.md`.
