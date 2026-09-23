# narrow-q on openbind at 896 aa: the fixture cannot score it (2026-09-23)

The size-ladder attribution (`narrowq_size_ladder_attribution.md`) found narrow-q moves openbind's
output at 896 aa, deterministically: three legs per arm, every off leg one digest, every on leg
another. So the default flip is user-visible on openbind and needs an accuracy score.

`narrowq_openbind_acc.sh` / `_msa.sh`: `tt_bio.main predict` at CLI defaults (200 steps), lever off
vs on, seeds 0 and 1, qb2 cards 0 and 2 (p300c). `narrowq_openbind_acc.py` scores CA RMSD against
1HCL per CDK2 copy (three full copies in `cdk2x2_896`, residue identity asserted) and arm-to-arm.
Float64 Kabsch throughout.

    config           mean CA vs 1HCL (off/on)       arm move s0 / s1     seed floor (off s0|s1)
    single-seq       s0 19.289/19.342  s1 19.318/19.277   0.958 / 0.401 A    24.728 A
    ColabFold MSA    s0 11.711/16.078  s1 16.097/13.397  10.249 / 24.623 A   13.386 A

Neither row is a verdict. Single-sequence openbind does not fold CDK2 (19 A off the deposited
structure in every leg). With an MSA it is chaotic: the lever's move is the same order as
re-seeding, and the distance to 1HCL goes worse at seed 0 and better at seed 1. This is the M18
failure mode (an unconfident fixture makes any effect/floor ratio meaningless), so it is recorded
as uninformative, not as a pass.

Owed before the default can ship: an openbind target that is confident, at a padded length where
the lever fires and moves TRIATT_PERSISTENT_MASK, with a three-seed floor.
