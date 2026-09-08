# ceilrfd3 results

Two eras of measurement live here and they are NOT comparable. Read the `steps` field before
reading anything else.

`base.jsonl`, `base2.jsonl`, `boundary.jsonl`, `fix.jsonl`, `attrib.jsonl`, `fix_ladder.txt`,
`digest_*.jsonl`, `init_digest.jsonl` were all walked at **2 diffusion steps**, which was
`rfd3_cap.py`'s default until 2026-09-07. A 2-step run never enters the self-conditioning recycle
path (`model.py:3646` passes `D_II_self` only on a recycle), and the pair transition's SwiGLU asks
1.94x as much L1 once it does. So the 992 ceiling those files support is not a ceiling at the
settings anybody runs, and it is withdrawn. They are kept because the row-block A/B digests in
them are still valid: that comparison is between two arms at the same step count.

`s100.jsonl`, `s100rep.jsonl`, `steps100.jsonl` are walked at **100 diffusion steps**, what the
platform sends. These are the ceiling of record.

| total residues | runs at 100 steps | result | wall |
| --- | --- | --- | --- |
| 640 | 1 | PASS | 162.2 s |
| **704** | **3** | **PASS, all three** | 169.6 / 161.5 / 148.0 s |
| 768 | 3 | **FAIL 1, PASS 2** | 70.2 s to the throw; 176.2 / 175.4 s to fold |
| 992 | 1 | FAIL | 104.3 s to the throw |

704 is the cap: the largest size below the FIRST failure. 768 usually folds and cannot be
published, because a cap has to be a size below which everything works.

`steps100.jsonl` also records two chips that `sudo lsof` reported free and that are wedged --
rows `card=0` (node 16) and `card=1` (node 17), each `failed to initialize FW` after a 10 s
timeout on physical cores. Neither was reset: every chip on this host reports one board number.

## whverify_*.jsonl — the Wormhole ladder, 2026-09-08

The rows that moved the `rfd3/wormhole_b0` ceiling from 704 to 1024, measured on GWH02 through
the shipped `tt-bio design` CLI rather than through `rfd3_cap.py`, so every PASS is a CIF on disk.

* `whverify_up.jsonl` — UMD 3, ascending 640 → 1024.
* `whverify_down.jsonl` — UMD 0, descending 1024 → 640. Same seven rungs, opposite order, so no
  rung inherits the one before it. Every CIF is byte-identical to its `up` twin.
* `whverify_ab768.jsonl` — 768 with `TT_BIO_L1_RESIDENT_BUDGET_BYTES=0`, i.e. every residency
  granted, which is the path the engine shipped with. Folds 3/3 and returns the same CIF md5 as
  the arm that declines. This is the bit-exactness proof ON a Wormhole part.
* `whverify_postfix.jsonl` — 1024 re-run after the block loop learned to read the refusal memo.
  Same md5, and the census reads 2 granted / 794 declined instead of -13 / 809. The two 10.5 s
  rows are a second chain the device lease correctly refused; nothing ran in them.

`../whverify/wh_cap704.cif` and `wh_cap1024.cif` are the scored structures for the old cap and
the new one.
* `whverify_seed7.jsonl` — 768/832/896 at seed 7 instead of 42. The backbone break inside the
  designed binder at 768 and 832 reproduces under both seeds and 896 is clean under both, so it
  is a property of those sizes on this target and not of the noise draw. It is not this ceiling
  and not the L1 lever: the 768 CIF is byte-identical with the residency granted.
