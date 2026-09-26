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

`collect.py <dir> <tag>...` folds each leg's run marker, clock samples and `results.json` into
one JSONL row. `clashsep.py <struct.cif>` answers the one question the clash count on a tiled
fixture cannot: the cell is CDK2 repeated on a 298-aa period, so it reports how many of the
sub-2.0 A heavy-atom pairs sit at a sequence separation that is a multiple of 298. Copies
packing onto each other cluster there; a tensor-layout tear is local and has no way to know
about 298. It reuses `check_structure.py`'s clash definition verbatim so the two counts agree.

## Result, 2026-09-26, tt-bio a1c35618e

| rung | tokens | MSA rows | runtime | AICLK during fold | clashes | backbone breaks |
|---|---:|---:|---:|---:|---:|---:|
| 1536 | 1536 | 16384 | 3983.0 s | 1000 MHz (400 samples) | 712 / 12345 | 0 |
| 1024 | 1024 | 16384 | 1456.9 s | 1000 MHz (147 samples) | 107 / 8240 | 0 |
| 1024, second card | 1024 | 16384 | 1443.7 s | 1000 MHz (146 samples) | 107 / 8240 | 0 |

Both 1024 legs return a byte-identical CIF. The clash rule fails on all three and the
separation histogram says it is the fixture: at 1536, 61 % of the pairs are within +-5 of a
multiple of 298 and 0.7 % are local.
