# b2x-integrate — the two Boltz-2 diffusion levers, re-verified at 512 aa and landed on by default

`BOLTZ2_TOKEN_DIT_SDPA` and `TT_BIO_ATOM_AXIS_BUCKET` were measured default-off by
`b2x-flag-levers` on a co-tenanted card. This pass re-took both on an idle card under benchlock,
with N=3 per arm and the domain-split diagnostic rather than a whole-molecule RMSD, and flipped
both defaults.

| arm | fold median | speedup | sampler | atom axis | CIF sha256 |
|---|---|---|---|---|---|
| base (previous default) | 22.195 s (n=6) | 1.000x | 7.396 s | 7168 | `4f3995a69be5d610` |
| `BOLTZ2_TOKEN_DIT_SDPA` | 21.095 s | 1.0522x | 6.324 s | 7168 | `d597d0ba59ec2e26` |
| `TT_BIO_ATOM_AXIS_BUCKET` | 21.198 s | 1.0470x | 6.390 s | 4480 | `0918b3092da1b527` |
| **both, the new default** | **20.079 s** | **1.1054x** | **5.280 s** | 4480 | `dd1c2a12f97772fb` |

A/A floor 0.19 % on the fold, trunk flat at 12.20-12.24 s in all four arms, 2.097 s summed against
2.116 s measured together, so the two are independent.

## Files

* `run512.sh`, `ab512.json` — the 19-fold interleaved A/B and the 298 aa control, through
  `perf/b2x-flag-levers/ab_flag_levers.py --keep-512`. Every fold's own time, shape, plDDT and
  CIF digest is in the JSON; the run log is not committed, `*.log` is repo-ignored.
* `control298.json`, `cif298/` — the monomeric control the 4649 thresholds are written against:
  CA 0.177490 A, all-atom 0.383745 A, A/A structural floor exactly 0.
* `split512.json`, `cif512/` — each pseudo-domain superposed alone, plus the hinge and the pooled
  hinge-free RMSD. One CIF per arm is kept; the other two reps of each arm were bit-identical to it
  and their digests are in `ab512.json`.
* `leakcheck.sh`, `leak/shas.txt` — the 4649 engine-leak requirement: OpenFold3 and Protenix-v2
  each fold byte-identically with both flags forced off, forced on, and off again. Three digests
  per model, one distinct value each: `d4da7b0ee098cb05` and `6744de383e0c3610`.
* `verify_default.py` — the shipped default through the real CLI and its worker spawn, not through
  a module global the harness assigned. Run 2026-09-11 on card 0: rc=0, 29.248 s for the whole
  command, 4116 atoms, file digest `dd1c2a12f97772fb`, atom identity and coordinates both
  identical to `cif512/512_AB_0`. Re-run after merging `origin/main` (which had since gained
  `wk/b2x-host-residual`): same digest, coordinates still identical. The two CLI walls, 29.248
  and 29.227 s, are n=1 each and cannot resolve that merge's 0.395 s claim, so nothing is
  claimed about it here -- the A/B timings above were taken at c7dab1dc, before it.
