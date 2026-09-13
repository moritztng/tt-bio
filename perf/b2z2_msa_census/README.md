# The MSA track, measured

`msa_probe.py` grabs one settled `MSALayer.__call__` out of a real 512 aa fold and replays it.

    --mode ops [--mark]   ordered top-level ttnn op list from one graph capture
    --mode time           bare synced wall, profiler off
    --mode prof --mark    one fenced window of `--reps` calls, for a profiler-armed capture
    --mode ab   --arms    paired interleaved arms in one process, with their own A/A floor
    --mode parity --arms  same inputs, two settings, torch.equal plus a negative control
    --mode fold           a real fold at the full protocol with the MSA track bracketed by syncs

`msa_split.py` reads the armed capture: the CB stall split and the byte / tile / per-program fit,
whole and per sub-unit, cut at the four markers `--mark` puts in the program stream. The arithmetic
under it is `b2z2_sampler_stall/stall_split.py` and `b2z2_tile_census/census_tiles.py`, unchanged.

Run (whglx, one pinned card, profiler overlay for the armed capture only):

    P=/home/mthuening/work/tt-bio/env/bin/python3
    export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_TRACE_REGION_SIZE=536870912
    $P perf/b2z2_msa_census/msa_probe.py --mode time --out perf/b2z2_msa_census/time_wh_c1.json

    source /home/mthuening/work/b2z2-profiler/profenv.sh
    python3 -m tracy -v -r --enable-sum-profiling --op-support-count 12000 \
      -o perf/b2z2_msa_census/prof_layer perf/b2z2_msa_census/msa_probe.py \
      --mode prof --reps 3 --mark --out perf/b2z2_msa_census/prof_wh_c1.json
    $P perf/b2z2_msa_census/msa_split.py --csv src/ops_perf_msalayer_whglx_c1.csv.gz \
      --meta perf/b2z2_msa_census/prof_wh_c1.json --out .../split_layer_wh_c1.json --label X --arch WH

`--op-support-count` sizes a DRAM buffer. 200000 refuses the weight load outright; 12000 covers a
precursor plus nine replayed calls.

Results and what they mean: `FINDINGS.md`. Prediction, written first: `PREDICTED.md`.
