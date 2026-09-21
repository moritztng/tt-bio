# Resuming this row

The candidate under audit is `70b2c4796` and its parent `ce7467175` on `origin/wk/allm-gates`.
It is NOT committed on this branch. Apply it into the working tree with

    git checkout 70b2c4796 -- tt_bio/tenstorrent.py tt_bio/swiglu_fused.py

and restore with `git checkout HEAD -- tt_bio/tenstorrent.py tt_bio/swiglu_fused.py` when done.
Only `tt_bio/esmfold2.py` carries a change of this row's own, the uniform `gated_move`.

Instruments, all committed:

  deriv_audit.py         replays the pass-1 census keys through main's dict and the candidate's
                         dict-plus-rule and prints every key whose answer moves. Host-only.
  selfpair_check.py      proves `widths[i+1:]` keeps all six literals and opendde's win and drops
                         only the two self-pair models. Host-only.
  fused_key_ab_fixed.py  the A/B whose control arm is actually main. NOTE: interleaving is not a
                         valid instrument for this lever (it writes the process-lifetime
                         `_L1_OUT_REFUSED`). Next run is one process per arm.
  arm_rmsd.py            Kabsch between the CIFs each leg keeps, all-atom and CA-only.

Cards on qb1 at the end of pass 2: card 0 free (used here), card 1 held by worker:tmk-writersplit,
cards 2 and 3 held by worker:cov-ladder-p150a-p3.
