# b2x-wh-neutral — the two default-on Boltz-2 levers, checked on Wormhole

`BOLTZ2_TOKEN_DIT_SDPA` and `TT_BIO_ATOM_AXIS_BUCKET` ship on since `93f75b10`, and both were only
ever measured on a Blackhole p300c. This is the same harness, fixtures, protocol and accuracy bar
re-run on the Wormhole Galaxy `j10glx02` (chips 0 and 1, 8x9 = 72 cores, so the small-grid path
Blackhole never takes), so the two platforms' numbers are directly comparable.

Correctness, not perf, is the deliverable. **Both arms fold, nothing OOMs, and the 4649 accuracy
bar passes with more margin than on Blackhole.**

| arm | atom axis | fold median | sampler | 512 CIF | 298 CA vs base | verdict |
|---|---|---|---|---|---|---|
| base (both off) | 7168 | 45.5045 s | 14.4131 s | `36f9a30ce5105c2c` | — | runs |
| `BOLTZ2_TOKEN_DIT_SDPA` | 7168 | 43.207 s | 11.8665 s | `47f81c6293dbcb51` | 0.128221 A | PASS |
| `TT_BIO_ATOM_AXIS_BUCKET` | 4480 | 43.680 s | 12.4337 s | `6dce58c09b274b3c` | 0.000000 A | PASS, byte-exact |
| **both, the shipped default** | 4480 | **41.336 s** | 10.0868 s | `2e3b25a5520c8ff8` | **0.128221 A** | **PASS** |

Blackhole scored the same arms at 0.177490 A CA, so Wormhole moves the structure 28 % less. The
298 aa A/A structural floor is exactly 0.000000 A, the trunk is flat to 0.75 % across all four
arms, and every arm is bit-identical across its own reps.

The fold ratio is **not** a Wormhole perf claim: a sibling leg loading its model on chip 1 put the
first base fold at loadavg 40.5 and the session's A/A floor at 8.2 %. The uncontaminated rep-1
base pair is 0.64 % apart. The sampler ratios, which are the comparable denominator, land close to
Blackhole's: A 1.2146x against 1.170x, B 1.1592x against 1.157x, AB 1.4289x against 1.401x.

## Files

* `run_wh.sh` — both legs. `ab` is the in-process A/B, the 298 aa control and the
  `TT_BIO_TOKEN_BUCKET=0` leg; `cli [CHIP] [FIXTURE]` is the same two arms through the real CLI on
  any chip.
* `ab512_wh.json` — every fold's time, shape, plDDT and CIF digest, plus the 298 aa control and the
  `TT_BIO_TOKEN_BUCKET=0` result. With the atom bucket off, 298 tokens ask for 4172 atoms and
  Wormhole dies in `reshape_common.cpp:52`; with it on, the fold runs and writes the same digest
  the shipped token bucket does.
* `control298_wh.json`, `cif298/` — the monomeric fixture the 4649 thresholds are written against.
* `split512_wh.json`, `cif512/` — each pseudo-domain of the chimeric 512 aa fixture superposed
  alone. Needed because its free inter-domain hinge saturates whole-molecule RMSD for any
  non-bit-exact change. Every domain is under the 0.60 A per-domain bar. One CIF per arm is kept;
  the second rep of each arm was bit-identical to it and its digest is in `ab512_wh.json`.
* `cli_{default,off}_{512,298}_chip{0,1}.json` — the shipped default and the documented `=0`
  escape hatch through `tt_bio.main predict` and its worker spawn, on two chips. Coordinates are
  identical to the matching in-process arm in all six runs, so the CIFs themselves are not kept;
  at 298 aa the files match byte for byte too. At 512 aa the files differ in the per-atom plDDT
  column alone (63.933 against 63.864) with every coordinate identical, which is the one size
  where Wormhole's L1 refusal ladder fires and memoises.
* `leak/shas.txt` — OpenFold3 and Protenix-v2 each folded `examples/615.yaml` with both flags
  forced off, on and off again. One digest per model across all three arms, `ce6ffe152d8d0bf3` and
  `95815f56ae090343`, so neither lever moves another model here. Both are Boltz-2-exclusive by
  construction (`TT_BIO_ATOM_AXIS_BUCKET` is read only in `tenstorrent.DiffusionModule`, and the
  SDPA branch needs `token_dit`, which only `DiffusionTransformerLayer` sets) and nothing in either
  guard reads the grid, so this is a backstop for that argument rather than the argument.

Full evidence and the verdict: `~/.coworker/state/b2x-integrate-wh-neutral.md`.
