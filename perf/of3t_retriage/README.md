# of3t-retriage — re-classing the six USER-FACING defects

Pass 528, card-free. Every defect re-read by content against `origin/main` cdd2c2f38; the filed
line numbers are stale (`autograd.py` moved 673 lines, D32's `triatt_qkv` sites moved by six).

| defect | before | after | why |
|---|---|---|---|
| D205 | USER-FACING | **FIXED**, out of the set | 576 runs and `docs/training.md:194` + `dryrun.py:108` say so; the 544/576 wall was contiguity and e558bfb06 removed it |
| D32 | USER-FACING | **CAMPAIGN-INTERNAL** | its own filing is "UNFIXED as a METHOD GAP", fix "in the brief rather than in code"; the user-reachable residue is 1.00508x of a training step |
| D55 | USER-FACING | USER-FACING | `softmax_bw_inner` (`autograd.py:158-171`) still withholds the config from `inner = sum(g*y)`; reached by `triangle_attention`'s backward at `:2069` |
| D184 | USER-FACING | USER-FACING | `TT_BIO_OF3_DEVICE_REFATOM` default OFF, so 17 parameters read rel_l2 exactly 1.0 on the shipped arm |
| D210 | USER-FACING | USER-FACING | only the attention output is sliced; the pads are inert only while exactly 0.0 |
| D250 | USER-FACING | USER-FACING | the fp32-z lever is training-only and off, which is why inference still carries 8.176e-03 |

Counts: 96 UNFIXED (5 / 6 / 85) -> 95 UNFIXED (5 / **4** / 86).

`UNFIXED_TRIAGE.before.json` and `.after.json` are the artifact either side of
`perf/of3t_orchestrator/defecttriage/triage.py`. The classification rule, the two disagreements
with the dispatching evidence and the generator defect are in `~/.coworker/state/of3t-retriage.md`.

## The generator defect

`triage.py` reads `DEFECTS.md` only to learn which defects are UNFIXED. Every class comes from a
hardcoded `TABLE` in its own source, so a reclassification argued in a `### D<N> UPDATE` body is
invisible to a regeneration. It also refused to run on arrival (`TRIAGE IS STALE`), and the
artifact the GO gate reads was being maintained by `stamp_row_counts.py:30-38`, which mutates the
previous JSON in place, defaults new UNFIXED defects to the least severe class and writes `reasons`
in a different schema. Both need fixing by whoever owns the orchestrator's tooling.
