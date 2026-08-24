# rf3-1024aa, first run through full_parity_gate.py

`scripts/full_parity_gate.py --leg rf3-1024aa --workers pc:0`, 2026-08-24, pc card 0 (p150a),
one seed, 326 s. Device 1.9576 Å from the deposited 7EIP coordinates over 966 modelled
residues, the H200 reference 2.0092 Å, X 0.467 Å. Floor is 4.0 Å.

`GATE_CODE.json` pins the `tt_bio/` + `scripts/` tree the numbers were measured on.

The release anchor stays `release_gate.RF3_1024AA_XTAL_MEASURED` (1.9687 Å, measured on qb2):
pc's card is the one with the location-keyed matmul fault, so a number measured here is
evidence the leg runs and reads the right quantity, not a replacement for that anchor.
