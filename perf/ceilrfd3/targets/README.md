# The one target the RFD3 ceiling ladder runs on

`laczc_1008.cif` — 1DP0 chain A (E. coli beta-galactosidase), 1008 residues, 8095 atoms,
renumbered 1..1008 so `A1-N` selects the first N of them for any N on the ladder.
sha256 `a6f9bdf4ea77bb800ba7e8dc3d4202bcb37746f9dcfd3e4fbacf3d4b3a234bd5`.

Same source and same cut as the PXDesign size ladder in `perf/pxdesign/targets`
(`manifest.json` there records the recipe and the 1DP0 sha256); this is the 1008 rung,
which that manifest does not carry because 1DP0 chain A is 1011 residues and PXDesign's
own ladder stops at 768.

One target for the whole ladder, on purpose. RFD3's previous Wormhole cap was bisected on
9ma0 for exactly this reason: two rungs cut from different PDBs differ in size and in
everything else, which is fine for a perf ladder and not fine for a ceiling.
