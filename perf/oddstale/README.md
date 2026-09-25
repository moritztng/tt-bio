# oddstale — OpenDDE's wh-galaxy size record, re-walked

`japanfold/size_evidence.py` records OpenDDE at 1536 tokens / 16384 alignment rows on a
Wormhole Galaxy, measured at tt-bio `428670d60` on 2026-09-24. 195 commits have touched
`SIZE_AFFECTING_PATHS` since, so the coverage sweeper re-scheduled it: the number described a
tree we no longer ship.

`leg.sh` runs one fold through the shipped `tt-bio predict` CLI at the flags
`japanfold/jobs.py::_build_cmd` assembles for a served predict job, with the size guard left
on (`size_limits` caps opendde/wormhole_b0 at 1536). It asserts the engine on `PYTHONPATH`
before the card is spent, and samples `tt_aiclk`, hwmon power and host load every 10 s during
the fold, so the runtime carries the clock it was measured at.

    ./leg.sh <tag> <rung> <model> <umd-card> <kernel-node> <timeout-s>

The card argument is a **UMD id**, the node argument its `/dev/tenstorrent/N`. They are not
the same number on this box: UMD 0-31 map to nodes 16..31, 8..15, 0..7 in PCI BDF order.

Rungs are built by `perf/ceilings/make_rung.py <dir> --depth=16384 1024 1536`, the same CDK2
period every other OpenDDE rung on this board uses. Results in `results/`.
