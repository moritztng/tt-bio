# b2p-ship: what the pad-up does to the other models on the shared triangle-attention path

`_tri_att_hifi_pad_up` sits inside `_tri_att_sdpa_hifi_inner`, which is triangle attention for
every model in the repo, and it was graded against float64 at BindCraft 2's shape. These are the
counters read on a real fold of two other models, on qb1 UMD card 1 (a p150a), commit `a5985d3e4`.

`inert_drv.py` is the driver. It exists because `tt-bio predict` spawns its worker with
`mp.get_context("spawn")` and stops it with `terminate()`, so the child never runs `atexit`: a
first attempt read the PARENT's counters, which are all zeros because the parent never folds. The
driver registers its dump at import, on SIGTERM and on SIGINT, and writes one file per pid.

| leg | worker pid | `TT_BIO_TRIATT_HIFI_PAD_UP` | served | declined | padded |
|---|---|---|---|---|---|
| OpenFold3, 544 residues | 4131200 | 2 | 384 | 0 | 544 -> 576 |
| OpenFold3, 608 residues | 4136132 | 2 | 384 | 0 | 608 -> 640 |
| OpenFold3, 544, pad-up off | 4141223 | 0 | 0 | 384 | none |
| Boltz-2, 544 residues | 4138430 | 2 | 0 | 0 | none |
| Boltz-2, 608 residues | 4143745 | 2 | 0 | 0 | none |

**Boltz-2 is inert, measured rather than reasoned:** served 0, declined 0 and `too_short` 0 means
the fused arm is never offered a call, because Boltz-2 keeps the materialised block.

**OpenFold3 is not inert.** Every one of its 384 trunk calls takes the pad-up, and with the pad-up
off all 384 decline, so the change moves OpenFold3 from the fallback onto the padded fused route
at both lengths.

The op-level accuracy question is answered by `perf/b2p_ceiling/grade.json`, which grades this
kernel against torch float64 forward and VJP at heads 4, head_dim 32 -- OpenFold3's trunk
triangle-attention shape as well as BindCraft 2's. A padded rung's relative L2 equals its
natively-served neighbour's to the fourth decimal (544 padded to 576 reads 0.02162 against 576
native's 0.02162), so the pad costs nothing measurable at the op.

**What is still owed, and it blocks the pad-up merging as a default:** a structural reading on a
real OpenFold3 target. The A/B here (`inert_ab.sh`, `kabsch.py`) folded a random 544-residue
sequence with `msa: empty` on both arms and got CA-RMSD 7.336 A unsuperposed, 7.210 A after
Kabsch -- against **mean pLDDT 27.93 and 28.33** and a 28.0 A radius of gyration. Both arms folded
to noise, so that number carries no accuracy signal and is not a parity result. It needs a
deposited target with an MSA and matched seeds, the way the OpenFold3 HiFi-route entry in the
CHANGELOG was measured.

`design_wheel.py` and `wheel312.sh` are the release proof: tt-bio from the built wheel in a clean
3.12 venv, BindCraft 2 from its own checkout because `pip install` of BindCraft 2 does not carry
its `settings/` or `examples/` trees.

`wheel_design/` is that proof's result on qb1 UMD card 2 (a p150a), take 3 at `471e4fc59`:
`DESIGN_CARD_OPEN` at 21:54:49Z, AICLK 1350 MHz on the card during the trajectory, and
`DESIGN_RESULT` after one PD-L1 trajectory at a 60-residue binder (192 tokens) in 456.9 s, rc=0.
The trajectory ran to the screen stage (i_pTM 0.77, pLDDT 0.53, rejected on pLDDT), which is the
expected end for a trajectory-only run with no ProteinMPNN redesign.
