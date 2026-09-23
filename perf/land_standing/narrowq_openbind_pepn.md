# narrow-q on openbind: accuracy on a confident target (2026-09-23)

narrow-q moves openbind's output at 896 aa (`narrowq_size_ladder_attribution.md`), and the cdk2x2
fixture could not score the move (`narrowq_openbind_acc.md`). This scores it on E. coli
aminopeptidase N, PDB 3B34 (1.30 A, one chain, 891 aa with its His tag, so it pads to 896).

`narrowq_openbind_pepn.sh`: `tt_bio.main predict` at CLI defaults (200 steps), one ColabFold MSA
fetched once and shared by every leg, lever off vs on at seeds 0, 1, 2, qb2 cards 0 and 2
(p300c). `narrowq_openbind_pepn.py`: float64 Kabsch CA RMSD over the 866 residues 3B34 resolves,
residue identity asserted per pair.

    seed   CA vs 3B34 off   CA vs 3B34 on   delta      off -> on move
    0      0.713806 A       0.709204 A      -0.0046    0.060841 A
    1      0.601173 A       0.600358 A      -0.0008    0.070799 A
    2      0.782470 A       0.788276 A      +0.0058    0.081084 A

Seed floor, off arm to off arm: 0.819 / 3.288 / 3.961 A. Every digest differs between arms, as the
6-step census predicted.

Verdict: accurate enough to ship. The lever moves the structure 0.071 A on average, 8.5x under
the 0.60 A bar and 11.5x under the smallest seed-to-seed distance. Its distance to the experimental
structure moves by -0.0001 A on average, with the sign split across seeds. This is a
precision change (a different L1-feasible kernel for one call), not a wrong transform.
