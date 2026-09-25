# abagstale — OpenDDE-abag's wh-galaxy size record, re-walked on the tree we ship

`PROVEN["opendde-abag"]` in the platform's `japanfold/size_evidence.py` recorded 1536 tokens at
16384 alignment rows, measured 2026-09-24 at tt-bio `428670d60` on j10glx02. 195 commits have
touched `SIZE_AFFECTING_PATHS` since, so the coverage sweeper re-scheduled it. This rig re-runs
the same two rungs on today's `origin/main`, on the GWH02 Galaxy.

The antibody checkpoint gets its own rungs on purpose. `opendde` and `opendde-abag` share every
tensor shape, and the file's convention is still that neither inherits a number from the other:
the two checkpoints have already been measured 4.6x apart on the same fixture and the same pair
of engine shas (`state/cov-below-bar-openddeabag-whgalaxy.md`).

    ./leg.sh <tag> <rung> <model> <umd_card> <dev_node> <timeout_s>
    python3 collect.py . <tag> ... > results/<file>.jsonl

`leg.sh` runs the shipped `tt-bio predict` CLI at exactly the flags
`japanfold/jobs.py::_build_cmd` sends a served predict job, with the size guard left ON, and it
exits `ENGINE_MISMATCH` before it opens a card if `tt_bio` does not resolve to the tree the leg
claims. AICLK, card power and host load are sampled every 10 s from the tenstorrent CLASS node
DURING the fold; `collect.py` trims those to the leg's own START..END window so the 500 MHz
device-open and device-close edges do not drag the median down.

Rungs are `perf/ceilings/make_rung.py` output: CDK2 (PDB 1HCL) tiled on its 298-aa period, apo,
one chain, so tokens equal residues. Structure is scored with
`perf/wh-correctness/check_structure.py`.
