# Censuses behind the affinity/ligand coverage arm

Measured 2026-08-21 on qb1 (tt-quietbox) card 0/1/2, p150a at a 13x10 grid, branch
`wk/sizegate-affinity-coverage`. Folds are the arm's cheap config: `--single_sequence
--sampling_steps 6 --diffusion_samples 1 --seed 0`, plus `--sampling_steps_affinity 6
--diffusion_samples_affinity 1` where the affinity module runs.

The three ladders, all the same protein at a given rung:

| file | fixture | what it is |
|---|---|---|
| `census_holo_boltz2_<rung>.json` | `perf/sizegate/inputs/holo/` | protein + ligand, structure only |
| `census_boltz2-affinity-<rung>-rep0.json` | `perf/nesso1/inputs/ladder/` | + `properties.affinity`, so the module runs |
| `census_b2aff_768_fixed.json` | same | the 768 rung, taken after the census fix |

The apo column of the table in `docs/size-generality.md` is not here: it comes from the
recorded baseline in `docs/size_ladder_baseline.json`.

No 640 aa rung. Boltz-2 does not fold a ligand at 640 aa on this part; it dies at trunk 0/4 on
an L1 circular-buffer clash. OpenDDE folds the same fixture at the same rung.

## The two A/B files

`census_ab_loaded_before_fix.json` and `census_ab_loaded_after_fix.json` are the same 256 aa
affinity fold under 32 busy loops, either side of the wrap-install fix in
`scripts/lever_census.py`. 7456 calls against 11446, the gap being exactly the seven
`wrap`-counted levers at `0/0`. 11446 is what an idle host measures, so the after arm shows the
fix removes the load sensitivity without moving the number.

Read them with `python3 /dev/stdin` style ad-hoc diffs, or `scripts/lever_census.py --report
<files>` for the table form.
