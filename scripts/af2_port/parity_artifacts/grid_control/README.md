# af2ig-trunk-device grid control

Ran 2026-09-09 on qb2 (p300c, tt-kmd 2.11.0, card firmware 19.15.0.0), card 0 held exclusively,
verified free by an fd-scan of every `/proc/*/fd` before each arm. Full writeup in the release
state doc §71.

Each arm is the exact argv `scripts/full_parity_gate.py::run_inprocess` builds for the `af2ig-trunk-device`
leg. The only difference between arms is `TT_BIO_FORCE_GRID` (`tt_bio/tenstorrent.py`
`_configure_active_compute_grid`) and, for arm C, the checkout.

| arm | checkout | grid | taps_failed | scalars_failed | pcc_min |
|---|---|---|---|---|---|
| A | `fb2bc462` | forced 11,10 (= the engine's own pick) | 13 | 2 | 0.9960112623229 |
| B | `fb2bc462` | forced 10,10 | 13 | 2 | 0.9960112623229 |
| C | `afad85a0`, the floor's own commit | unpinned, 11,10 | 9 | 3 | 0.9964227723349416 |
| — | committed floor, qb1 p150a, fw 19.8.1.0, 13x10 | — | 9 | 3 | 0.9964227723349419 |

Arm A is bit-for-bit identical to the leg's report inside the L4 record, all 29 fields including
94 tap rows and 6 scalars, which is what makes arm B readable. Arm B is bit-for-bit identical to
arm A. Arm C reproduces the committed floor's failing set exactly, with all 94 rows agreeing to
about 1e-15 relative (float64 summation order in the scorer).

Conclusion: the grid, the board type, the driver and the firmware are all refuted as the cause of
this leg's drift, and the tap set is grid-invariant over 13x10, 11x10 and 10x10. What remains is
`afad85a0..fb2bc462` in the port's own kernels.

`grid_kernel_path.txt` holds the positive control that arm B was not a dead knob: arm A fires
tt-metal's L1 circular-buffer clash on core range [(0,0)-(10,9)] and tt-bio's retry-narrower
notice, arm B fires neither.

Scripts: `readgrid.py` reads the active grid through `tt_bio.get_device` (a bare
`ttnn.open_device` dies on qb2 without an MGD). `armrun.py` runs one arm out of this checkout,
`armrun2.py` out of any checkout, `cmp.py` diffs two reports field by field.
