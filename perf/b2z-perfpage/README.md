# b2z-perfpage-recell — the published Boltz-2 Blackhole cell, re-taken on the merged tree

`wk/b2z-levers-default-on` merged at `0f3f9f67` and turned three levers on by default:
`TT_BIO_FUSE_BIAS_STACKS` (not bit-exact), `TT_BIO_SDPA_ADD_GRANULARITY` and
`TT_BIO_GATE_GRANULARITY=2` (both bit-exact). The perf page was not re-measured, and its parity
sentence claimed the shipped default still writes `dd1c2a12f97772fb` through the CLI. It does not.

| through the CLI, own process, no lever in the environment | CIF sha256 | complex plDDT | confidence |
|---|---|---|---|
| shipped default | `a91aa44441f0d9c5` | 0.842856 | 0.800144 |
| all three forced off (the pre-merge path) | `dd1c2a12f97772fb` | 0.845488 | 0.801141 |

`dd1c2a12f97772fb` at 0.845488 is exactly what the cell published, so the sentence described the
arm a user no longer gets.

Fold time, one session, benchlock at loadavg 1.92, qb2 card 0, 11x10 grid, 3 recycles / 200 steps /
1 sample / seed 0, one warmup discarded per arm:

| arm | median of n=6 | min / max | spread |
|---|---|---|---|
| shipped default | **20.113 s** | 19.932 / 20.185 | 1.27 % |
| all three off | 20.190 s | 20.037 / 20.341 | 1.52 % |

The default was the faster fold in all six pairs, mean 0.077 s (0.38 %). That is inside either
arm's own spread, so the run fixes the direction and not the size; the merge row's paired run reads
1.00814x. The cell moved 20.079 -> 20.113 s, which is noise on a cell whose cross-session spread is
4.60 % — the digest is what made this re-record necessary.

Structure, shipped default against all-three-off, same tree, same session:

* `cdk2x2_298`, the monomeric control the 0.35 A bar is written against: **0.2181 A all-atom,
  0.1373 A CA, PASS**, A/A structural floor exactly 0 over six reps of each arm.
* `cdk2x2_512`, per pseudo-domain, which is the only reading this chimeric fixture supports:
  0.3526 and 0.3057 A with a 3.22 deg hinge, 0.3338 A pooled. Whole molecule 0.5051 A all-atom /
  0.3954 A CA, above 0.35 but not a bar the fixture can carry — and `domain_split.py` reports this
  is **not** the saturation signature, so the hinge is not hiding anything either.
* plDDT flat: 0.845488 -> 0.842856 at 512 aa, 0.917806 -> 0.917809 at 298 aa.

The preceding default generation moved the same 512 aa fixture 0.52 A pooled with a 1.9 deg hinge
and shipped; this one moves it 0.33 A. It is the smaller of the two steps the cell has taken.

## Files

* `recell.sh`, `recell_512_qb2c0.json` — the 28-fold interleaved run through
  `perf/b2z_levers/fold_ab.py --reps 6 --warmup`, both sizes, every fold's time, digest and plDDT.
* `cli_default.py`, `cli_default.json`, `cli_base.json` — the shipped default and the pre-merge
  path, each through `python3 -m tt_bio.main predict` in its own process. Both reproduce their
  arm in the interleaved run byte for byte.
* `control298.json`, `split512.json`, `cif298/`, `cif512/` — the scorers
  (`perf/b2x-flag-levers/score298.py`, `domain_split.py`) and one CIF per arm per size. The other
  five reps of each arm were byte-identical to the one kept; their digests are in the run JSON.
* `loadsamples.txt` — loadavg every 15 s from 18:49:22 to 18:55:07, which covers the whole 512 aa
  phase, because benchlock only checks the box once, at acquisition. It peaks at 2.75 in the first
  sample while the model loads and sits at 1.23-1.71 through the timed folds. The one-minute figure
  counts this run's own process, so the box had nothing else of size on it.
