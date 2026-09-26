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

Two questions the runtime number cannot answer on its own get their own instruments. `phases.py`
splits the fold into trunk cadence, the trunk-to-diffusion transition and the diffusion loop, so a
long quiet window can be told apart from a stall. `perf/oddstale/clashsep.py` bins clashing atom
pairs by sequence separation against the 298-aa tiling period, which is what separates "the model
packed identical copies" from "the layout tore" with no reference structure to compare against.

## Results, 2026-09-26, GWH02 (`japanfold-ssh`), engine tree verified against `8ed707b93`

| | 1536 x 16384 | 1024 x 16384 | 1024 again, other chip |
|---|---:|---:|---:|
| leg / UMD card / node | `a1536` / 1 / 17 | `a1024` / 3 / 19 | `a1024b` / 0 / 16 |
| engine runtime, leg wall | 3885.9 s, 3899 s | 1452.4 s, 1465 s | 1442.7 s, 1455 s |
| AICLK during the fold | 1000 MHz over 390 | 1000 MHz over 147 | 1000 MHz over 146 |
| tokens / MSA depth at the model | 1536 / 16384 | 1024 / 16384 | 1024 / 16384 |
| pLDDT, pTM | 0.7370, 0.5001 | 0.7728, 0.5881 | 0.7728, 0.5881 |
| backbone breaks, CA-CA in band | 0, 98.96 % | 0, 98.73 % | 0, 98.73 % |
| clashes under 2.0 A | 818 (6.63 %) | 20 (0.24 %) | 20 (0.24 %) |
| CIF sha256 | `06bfc47d246c6c76` | `01c13e1670ac2235` | `01c13e1670ac2235` |
| `max_runtime_s` a served job gets | 5062 s | 1500 s | 1500 s |

The bar rung is byte-identical on two different chips of this Galaxy, so nothing in the reading
depends on which card ran it.

The clash rule fails at 1536 and the fixture is why: 75.6 % of clashing pairs sit within +-5 of a
multiple of 298 and 414 of them at exactly 5x298, the first tiled copy against the last, while
1.7 % are local. A torn layout is local and shows up as 19-22 A backbone steps, of which there are
none. Tiling CDK2 five times and asking the model to pack the copies is what produces this.

1536 is comfortably inside its watchdog and 1024 is not. Adding the 301 s of served overhead
`cov-below-bar-opendde-whgalaxy` measured, 1536 lands near 4200 s against 5062 s, while 1024 lands
near 1766 s against 1500 s. `max_runtime_s` is cubic off a 1024-residue reference, so it grows
faster than OpenDDE's runtime does and the SMALL size is the tight one. That is a runtime fact
about 1024-token OpenDDE jobs, not a capacity gap.
