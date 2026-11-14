# Issue #21, round 4: the validation stage alone, on a card

qb1 card 2 (logical 2 = `/dev/tenstorrent/3`), 01:03:44Z-02:10:35Z 2026-10-08, three arms rc=0.
AICLK median 1350 MHz on every arm, sampled every second DURING the fold from
`/sys/class/tenstorrent/tenstorrent!3/tt_aiclk` (n=1305 / 1353 / 1188).

Complex 4ZQK: chain A human PD-L1 (115 aa) target, chain B PD-1 (118 aa) the backbone
ProteinMPNN redesigns. Ten candidates, the same ten sequences in every arm, scored through
BindCraft 2's own 32 filters resolved by `build_design_settings(settings).filters`.

Each arm folds the MULTIMER design model first so `campaign_predictor` routes it to the card and
selects a checkpoint into the trunk pool, then builds the MONOMER validation model, which is the
fold the reporter's numbers come from. That ordering is the precondition for the bug.

| | prefix (6ba0f975d) | fixed (575ac9fa3) | control (6ba0f975d, `--no-extra-msa`) |
|---|---|---|---|
| `Target_pLDDT` | **0.325 - 0.520** | 0.939 - 0.961 | 0.939 - 0.961 |
| `Interface_Residues` | **118.0, all 10 of 10** | 0 - 5 | 0 - 5 |
| `Binder_RMSD` | 2.61 - 6.13 | 1.97 - 25.76 | 1.97 - 25.76 |
| `i_pTM` | 0.611 - 0.696 | 0.059 - 0.141 | 0.059 - 0.141 |
| `Backbone_Clashes` (cand 0) | **3231.0** | 0.0 | 0.0 |
| device extra-MSA stacks built in the validation fold | **2 on candidate 0**, 0 after | 0 | 0 |
| accepted | 0 of 10 | 0 of 10 | 0 of 10 |
| wall | 1391.7 s | 1393.9 s | 1199.8 s |

## What the three arms establish

**The prefix arm reproduces the reporter's signature on a card, on all three of their metrics at
once.** `Interface_Residues` is exactly 118.0 for all ten candidates and the binder is 118 aa,
as theirs was exactly 90.0 for a 90-aa binder and 94.0 for a 94-aa binder in two independent
campaigns. `Target_pLDDT` 0.325-0.520 brackets their 0.28-0.30. `Binder_RMSD` 2.61-6.13 overlaps
their 4.9-7.7.

**The control arm is bit-identical to the fixed arm on every validation metric.** Turning the
extra-MSA splice off on the UNFIXED tree gives exactly the fixed tree's numbers, to the last
digit. That is the separation the CPU arms could not make: the whole of the reporter's gap is the
extra-MSA stack reaching a host validation fold. Not the template path, not the Evoformer, not
device numerics in the fold itself.

**The design trunk is untouched by the fix.** `design.metrics` is identical in the prefix and
fixed arms (`Target_pLDDT` 0.8577, `pTM` 0.5179, `i_pTM` 0.1375, `Interface_Residues` 17.0,
`Binder_RMSD` 15.360). This matches what the reporter saw, that the design trajectory agrees with
CUDA while validation does not, and it shows the fix does not move the design path.

**The mechanism is caught in the act.** The counter reads 2 device extra-MSA stacks built during
the prefix arm's first validation fold and 0 during every fold after the fix. `after_design`
records `trunk_selected: model_1_multimer_v3` in both spliced arms: the design loop's multimer
checkpoint is what a monomer validation fold was reading out of `pool.current`.

**`Backbone_Clashes` 3231 vs 0 is the structural proof of the collapse.** It confirms the reading
that `Interface_Residues == binder_length` is a consequence rather than a separate bug: the target
folds into a blob that interpenetrates the binder, so every binder residue is within 4 A of a
target atom. It also explains why `i_pTM` is HIGHER in the broken arm (0.61-0.70 vs 0.06-0.14).
A collapsed target that interpenetrates the binder scores as an excellent interface, which is why
the reporter could not have caught this from `i_pTM`.

## Accepted counts: 0 of 10 in all three arms, and why that is not a failure of the fix

The two cases are not the same failure, and the difference is the verdict.

Pre-fix, acceptance was impossible for ANY sequence. The ten candidates fail
`Backbone_Clashes` 10/10 (3231 clashes) and `Unbound_Binder_pLDDT` 10/10. Both are driven by the
collapsed target, not by the binder, so no amount of design quality could have passed them. This
is the reporter's 0 of 10.

Post-fix, the candidates are judged on their own merits and these particular ten are not good
binders: `pTM` 10/10, `i_pTM` 10/10 (0.059-0.141), `i_pAE` 10/10, `Interface_Residues` 10/10 for
being too FEW, `Binder_RMSD` 8/10. They are ProteinMPNN redesigns of the PD-1 backbone at its
`PRNGKey(0)` default with no design optimization behind them, so failing a real binder filter is
the correct outcome. Round 4 deliberately bypasses the design loop, which is the only reason it
could measure #21 at all after three campaigns spent their chip time on the design loop's dice.

So this row does not quote an accept rate against upstream's 3 of 10. Upstream's 3 of 10 came from
a full campaign whose design loop produced optimized binders; these ten did not come from a design
loop at all, and comparing them would be a fabricated comparison. The accept rate from a full
campaign belongs to `bci-accept`.
