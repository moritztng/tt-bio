# AbAg-XM antigen de-duplication audit (addendum A2)

CoFold Arena panel rules applied as an AUDIT of the 164-target panel: one entry per antibody, at most one antibody per antigen UniProt accession, one physical copy per mmCIF. Antigen entities were resolved by SEQUENCE (the fold yaml's A chain matched by containment against every RCSB entity; ARK's interface rows do not put the antigen on a consistent side) and mapped to UniProt via the RCSB GraphQL API (cached). PRIMARY reporting stays the full 164-target panel: panel identity with ARK-164 anchors the 67.1%/66.4% harness validation. The deduplicated view is the sensitivity analysis.

## Accession mapping

- 138/164 antigens map to >=1 UniProt accession; 26 are null-mapping (engineered/construct antigens without a SIFTS UniProt mapping). Null-mapping antigens are a reported class, NOT auto-duplicates of each other. Note: the manifest has_peptide_antigen flag is target-level (ANY interface of the entry); the fold antigen of a flagged target can still be a full protein (e.g. 9d73, 274 aa), and short true-peptide fold antigens (7-24 aa) DO carry UniProt accessions via their parent protein.
- Multi-accession antigen entities: 5 (first accession = primary key).

## Accession multiplicity (primary accession per target)

- multiplicity 1: 86 targets
- multiplicity 2: 28 targets
- multiplicity 3: 15 targets
- multiplicity 4: 4 targets
- multiplicity 5: 10 targets
- multiplicity 9: 9 targets
- multiplicity 12: 12 targets

24 accessions are shared by >1 target (78 targets). Duplicate groups:

| group | accession | members (kept first = earliest release) |
|---|---|---|
| G00 | A0A0B5KY46 | 9uoc (2026-04-29T00:00:00Z) KEEP, 9uoi (2026-04-29T00:00:00Z) |
| G01 | A0A1S6XXK1 | 9loe (2026-01-28T00:00:00Z) KEEP, 9lof (2026-01-28T00:00:00Z), 9log (2026-01-28T00:00:00Z) |
| G02 | A0A2L1GIK5 | 9rig (2026-04-29T00:00:00Z) KEEP, 9rih (2026-04-29T00:00:00Z) |
| G03 | A0A590UJY2 | 21du (2026-02-11T00:00:00Z) KEEP, 9lwc (2026-02-18T00:00:00Z) |
| G04 | D2CJY3 | 9mz6 (2026-01-21T00:00:00Z) KEEP, 9mz8 (2026-01-21T00:00:00Z) |
| G05 | O95760 | 9wwh (2026-03-18T00:00:00Z) KEEP, 9x05 (2026-03-18T00:00:00Z), 9x0j (2026-03-18T00:00:00Z) |
| G06 | P02768 | 9q6h (2026-03-25T00:00:00Z) KEEP, 9q6n (2026-03-25T00:00:00Z), 9q6y (2026-03-25T00:00:00Z), 9q6z (2026-03-25T00:00:00Z) |
| G07 | P03303 | 9u5p (2026-03-25T00:00:00Z) KEEP, 9u5q (2026-03-25T00:00:00Z), 9u5r (2026-03-25T00:00:00Z), 9wb3 (2026-03-25T00:00:00Z), 9wb4 (2026-03-25T00:00:00Z) |
| G08 | P04626 | 9t3r (2026-03-11T00:00:00Z) KEEP, 9t3s (2026-03-11T00:00:00Z) |
| G09 | P0DTC2 | 9k6j (2026-01-07T00:00:00Z) KEEP, 9lh2 (2026-01-14T00:00:00Z), 9rn6 (2026-01-21T00:00:00Z), 9loz (2026-01-28T00:00:00Z), 9lp1 (2026-01-28T00:00:00Z), 9pso (2026-02-25T00:00:00Z), 9sat (2026-03-18T00:00:00Z), 9sbb (2026-03-18T00:00:00Z), 9w14 (2026-03-18T00:00:00Z), 9ynx (2026-03-18T00:00:00Z), 9dsg (2026-04-08T00:00:00Z), 9zdu (2026-04-22T00:00:00Z) |
| G10 | P13747 | 9nw7 (2026-03-25T00:00:00Z) KEEP, 9nw8 (2026-03-25T00:00:00Z), 9nw9 (2026-03-25T00:00:00Z) |
| G11 | P15494 | 9y0a (2026-02-11T00:00:00Z) KEEP, 9y0e (2026-02-11T00:00:00Z) |
| G12 | P17870 | 9lz1 (2026-03-11T00:00:00Z) KEEP, 9lz0 (2026-03-18T00:00:00Z) |
| G13 | P30518 | 9ly2 (2026-03-04T00:00:00Z) KEEP, 9ly3 (2026-03-04T00:00:00Z), 9lz2 (2026-03-11T00:00:00Z) |
| G14 | P35414 | 9lqw (2026-02-04T00:00:00Z) KEEP, 9lr1 (2026-02-04T00:00:00Z) |
| G15 | P38405 | 9ldx (2026-01-28T00:00:00Z) KEEP, 9le0 (2026-01-28T00:00:00Z) |
| G16 | P51679 | 9ull (2026-04-22T00:00:00Z) KEEP, 9ulm (2026-04-22T00:00:00Z) |
| G17 | P63092 | 9m40 (2026-03-11T00:00:00Z) KEEP, 9mxu (2026-04-22T00:00:00Z), 9mze (2026-04-22T00:00:00Z), 9mzf (2026-04-22T00:00:00Z), 9n05 (2026-04-22T00:00:00Z), 9n0e (2026-04-22T00:00:00Z), 9n1p (2026-04-22T00:00:00Z), 9n1q (2026-04-22T00:00:00Z), 9n2i (2026-04-22T00:00:00Z) |
| G18 | P63096 | 9ppw (2026-03-04T00:00:00Z) KEEP, 9ppy (2026-03-04T00:00:00Z) |
| G19 | P68874 | 9n8i (2026-02-04T00:00:00Z) KEEP, 9n8n (2026-02-04T00:00:00Z) |
| G20 | Q5ZPR3 | 9lme (2026-01-21T00:00:00Z) KEEP, 9ly5 (2026-03-04T00:00:00Z), 9ly6 (2026-03-04T00:00:00Z) |
| G21 | Q72J04 | 9rye (2026-04-08T00:00:00Z) KEEP, 9ryf (2026-04-08T00:00:00Z) |
| G22 | Q7K740 | 9nkz (2026-03-04T00:00:00Z) KEEP, 9nl0 (2026-03-04T00:00:00Z), 9nl1 (2026-03-04T00:00:00Z), 9ncy (2026-03-11T00:00:00Z), 9nzf (2026-04-08T00:00:00Z) |
| G23 | Q860B7 | 9d73 (2026-02-04T00:00:00Z) KEEP, 9d74 (2026-02-11T00:00:00Z) |

Sensitivity (ANY shared accession, not just primary): 26 accessions shared.

## Same-antibody-same-antigen pairs: 74

- 9d73 x 9d74 (shared cdrh3_cluster + antigen accession)
- 9ldx x 9le0 (shared cdrh3_cluster + antigen accession)
- 9loz x 9lp1 (shared cdrh3_cluster + antigen accession)
- 9ly2 x 9ly3 (shared cdrh3_cluster + antigen accession)
- 9ly2 x 9lz2 (shared cdrh3_cluster + antigen accession)
- 9ly3 x 9lz2 (shared cdrh3_cluster + antigen accession)
- 9ly5 x 9ly6 (shared cdrh3_cluster + antigen accession)
- 9lz0 x 9lz1 (shared cdrh3_cluster + antigen accession)
- 9m40 x 9mxu (shared cdrh3_cluster + antigen accession)
- 9m40 x 9mze (shared cdrh3_cluster + antigen accession)
- 9m40 x 9mzf (shared cdrh3_cluster + antigen accession)
- 9m40 x 9n05 (shared cdrh3_cluster + antigen accession)
- 9m40 x 9n0e (shared cdrh3_cluster + antigen accession)
- 9m40 x 9n1p (shared cdrh3_cluster + antigen accession)
- 9m40 x 9n1q (shared cdrh3_cluster + antigen accession)
- 9m40 x 9n2i (shared cdrh3_cluster + antigen accession)
- 9m8k x 9ppw (shared cdrh3_cluster + antigen accession)
- 9m8k x 9ppy (shared cdrh3_cluster + antigen accession)
- 9mxu x 9mze (shared cdrh3_cluster + antigen accession)
- 9mxu x 9mzf (shared cdrh3_cluster + antigen accession)
- 9mxu x 9n05 (shared cdrh3_cluster + antigen accession)
- 9mxu x 9n0e (shared cdrh3_cluster + antigen accession)
- 9mxu x 9n1p (shared cdrh3_cluster + antigen accession)
- 9mxu x 9n1q (shared cdrh3_cluster + antigen accession)
- 9mxu x 9n2i (shared cdrh3_cluster + antigen accession)
- 9mz6 x 9mz8 (shared cdrh3_cluster + antigen accession)
- 9mze x 9mzf (shared cdrh3_cluster + antigen accession)
- 9mze x 9n05 (shared cdrh3_cluster + antigen accession)
- 9mze x 9n0e (shared cdrh3_cluster + antigen accession)
- 9mze x 9n1p (shared cdrh3_cluster + antigen accession)
- 9mze x 9n1q (shared cdrh3_cluster + antigen accession)
- 9mze x 9n2i (shared cdrh3_cluster + antigen accession)
- 9mzf x 9n05 (shared cdrh3_cluster + antigen accession)
- 9mzf x 9n0e (shared cdrh3_cluster + antigen accession)
- 9mzf x 9n1p (shared cdrh3_cluster + antigen accession)
- 9mzf x 9n1q (shared cdrh3_cluster + antigen accession)
- 9mzf x 9n2i (shared cdrh3_cluster + antigen accession)
- 9n05 x 9n0e (shared cdrh3_cluster + antigen accession)
- 9n05 x 9n1p (shared cdrh3_cluster + antigen accession)
- 9n05 x 9n1q (shared cdrh3_cluster + antigen accession)
- 9n05 x 9n2i (shared cdrh3_cluster + antigen accession)
- 9n0e x 9n1p (shared cdrh3_cluster + antigen accession)
- 9n0e x 9n1q (shared cdrh3_cluster + antigen accession)
- 9n0e x 9n2i (shared cdrh3_cluster + antigen accession)
- 9n1p x 9n1q (shared cdrh3_cluster + antigen accession)
- 9n1p x 9n2i (shared cdrh3_cluster + antigen accession)
- 9n1q x 9n2i (shared cdrh3_cluster + antigen accession)
- 9nkz x 9nl0 (shared cdrh3_cluster + antigen accession)
- 9nkz x 9nl1 (shared cdrh3_cluster + antigen accession)
- 9nkz x 9nzf (shared cdrh3_cluster + antigen accession)
- 9nl0 x 9nl1 (shared cdrh3_cluster + antigen accession)
- 9nl0 x 9nzf (shared cdrh3_cluster + antigen accession)
- 9nl1 x 9nzf (shared cdrh3_cluster + antigen accession)
- 9nw7 x 9nw8 (shared cdrh3_cluster + antigen accession)
- 9nw7 x 9nw9 (shared cdrh3_cluster + antigen accession)
- 9nw8 x 9nw9 (shared cdrh3_cluster + antigen accession)
- 9ppw x 9ppy (shared cdrh3_cluster + antigen accession)
- 9q6y x 9q6z (shared cdrh3_cluster + antigen accession)
- 9rye x 9ryf (shared cdrh3_cluster + antigen accession)
- 9t3r x 9t3s (shared cdrh3_cluster + antigen accession)
- 9u5p x 9u5q (shared cdrh3_cluster + antigen accession)
- 9u5p x 9u5r (shared cdrh3_cluster + antigen accession)
- 9u5p x 9wb3 (shared cdrh3_cluster + antigen accession)
- 9u5p x 9wb4 (shared cdrh3_cluster + antigen accession)
- 9u5q x 9u5r (shared cdrh3_cluster + antigen accession)
- 9u5q x 9wb3 (shared cdrh3_cluster + antigen accession)
- 9u5q x 9wb4 (shared cdrh3_cluster + antigen accession)
- 9u5r x 9wb3 (shared cdrh3_cluster + antigen accession)
- 9u5r x 9wb4 (shared cdrh3_cluster + antigen accession)
- 9wb3 x 9wb4 (shared cdrh3_cluster + antigen accession)
- 9wwh x 9x05 (shared cdrh3_cluster + antigen accession)
- 9wwh x 9x0j (shared cdrh3_cluster + antigen accession)
- 9x05 x 9x0j (shared cdrh3_cluster + antigen accession)
- 9y0a x 9y0e (shared cdrh3_cluster + antigen accession)

One physical copy per mmCIF: 2 targets carry a superseded_by pointer; no superseding entry is in the panel (asserted).

## Sequence-level fallback (MMseqs2 all-vs-all, >=90% identity)

235 unique cross-target pairs at >=90% identity over >=50 aa; 119 of them span targets the accession route does NOT group (different accessions, or null-mapping antigens). Clusters below merge accession-sharing and sequence hits into one graph (connected components).

| cluster | members (accession or null) | accession-consistent? |
|---|---|---|
| C00 (13) | 9dsg [P0DTC2], 9k6j [P0DTC2], 9lh2 [P0DTC2], 9loz [P0DTC2], 9lp1 [P0DTC2], 9pso [P0DTC2], 9rn6 [P0DTC2], 9sat [P0DTC2], 9sbb [P0DTC2], 9ssm [A0A8A5XRG7], 9w14 [P0DTC2], 9ynx [P0DTC2], 9zdu [P0DTC2] | **NOVEL span** |
| C01 (12) | 21du [A0A590UJY2], 9j87 [null], 9lwc [A0A590UJY2], 9m0x [null], 9m0z [null], 9m1p [null], 9m2o [null], 9m2s [null], 9m3q [null], 9m3s [null], 9n09 [null], 9uk2 [null] | **NOVEL span** |
| C02 (9) | 9m40 [P63092], 9mxu [P63092], 9mze [P63092], 9mzf [P63092], 9n05 [P63092], 9n0e [P63092], 9n1p [P63092], 9n1q [P63092], 9n2i [P63092] | yes |
| C03 (8) | 9m0j [null], 9v0x [null], 9v1h [null], 9vmn [null], 9vmo [null], 9vo2 [null], 9x3z [A0A5A9NRD5], 9xqc [null] | **NOVEL span** |
| C04 (5) | 9u5p [P03303], 9u5q [P03303], 9u5r [P03303], 9wb3 [P03303], 9wb4 [P03303] | yes |
| C05 (5) | 9ncy [Q7K740], 9nkz [Q7K740], 9nl0 [Q7K740], 9nl1 [Q7K740], 9nzf [Q7K740] | yes |
| C06 (5) | 21av [A0A7S9CEK0], 9loe [A0A1S6XXK1], 9lof [A0A1S6XXK1], 9log [A0A1S6XXK1], 9ve0 [W5VWE0] | **NOVEL span** |
| C07 (4) | 9q6h [P02768], 9q6n [P02768], 9q6y [P02768], 9q6z [P02768] | yes |
| C08 (3) | 9wwh [O95760], 9x05 [O95760], 9x0j [O95760] | yes |
| C09 (3) | 9uo0 [A0A0A7E6A4], 9uoc [A0A0B5KY46], 9uoi [A0A0B5KY46] | **NOVEL span** |
| C10 (3) | 9nw7 [P13747], 9nw8 [P13747], 9nw9 [P13747] | yes |
| C11 (3) | 9n8i [P68874], 9n8n [P68874], 9obn [Q8I6T1] | **NOVEL span** |
| C12 (3) | 9ly2 [P30518], 9ly3 [P30518], 9lz2 [P30518] | yes |
| C13 (3) | 9lxp [P49407], 9lz0 [P17870], 9lz1 [P17870] | **NOVEL span** |
| C14 (3) | 9lme [Q5ZPR3], 9ly5 [Q5ZPR3], 9ly6 [Q5ZPR3] | yes |
| C15 (3) | 9l9o [Q13585], 9tmp [P0ABE7], 9xqb [P29274] | **NOVEL span** |
| C16 (3) | 9jno [A0A2Z5DTY1], 9xsx [null], 9xth [null] | **NOVEL span** |
| C17 (2) | 9y0a [P15494], 9y0e [P15494] | yes |
| C18 (2) | 9ull [P51679], 9ulm [P51679] | yes |
| C19 (2) | 9t3r [P04626], 9t3s [P04626] | yes |
| C20 (2) | 9rye [Q72J04], 9ryf [Q72J04] | yes |
| C21 (2) | 9rig [A0A2L1GIK5], 9rih [A0A2L1GIK5] | yes |
| C22 (2) | 9qqe [U5YJM1], 9qqf [A0A5H2UF23] | **NOVEL span** |
| C23 (2) | 9ppw [P63096], 9ppy [P63096] | yes |
| C24 (2) | 9mz6 [D2CJY3], 9mz8 [D2CJY3] | yes |
| C25 (2) | 9mnt [null], 9mnu [null] | yes |
| C26 (2) | 9lqw [P35414], 9lr1 [P35414] | yes |
| C27 (2) | 9ldx [P38405], 9le0 [P38405] | yes |
| C28 (2) | 9d73 [Q860B7], 9d74 [Q860B7] | yes |

Peptide-fragment hits (<50 aa alignment, not merged into clusters): 10

- 9jkr x 9ssm: 100.0% over 15 aa (P04141 vs A0A8A5XRG7)
- 9k6j x 9jkr: 100.0% over 15 aa (P0DTC2 vs P04141)
- 9l9o x 9sat: 93.0% over 43 aa (Q13585 vs P0DTC2)
- 9l9o x 9vnp: 93.3% over 30 aa (Q13585 vs P01579)
- 9mmj x 9mz6: 93.0% over 43 aa (P20871 vs D2CJY3)
- 9mmj x 9mz7: 93.0% over 43 aa (P20871 vs A8DM03)
- 9mmj x 9mz8: 93.0% over 43 aa (P20871 vs D2CJY3)
- 9mz6 x 9mz8: 97.6% over 43 aa (D2CJY3 vs D2CJY3)
- 9ulm x 9ull: 100.0% over 11 aa (P51679 vs P51679)
- 9vnp x 9sat: 100.0% over 31 aa (P01579 vs P0DTC2)


## Headline metrics, full panel vs antigen-deduplicated

Full panel: 161 scorable targets. Accession-deduplicated: 109 scorable (110 total). Sequence-deduplicated (accession + >=90%-identity clusters merged): 80 scorable (81 total). Point estimates over the same seeded budget-N fold constants as the bootstrap doc; CIs for the full panel are in abag-xm-ranker-cis.md.

| generator | N | thr | metric | full | acc-dedup | seq-dedup |
|---|---|---|---|---|---|---|
| opendde-abag | 5 | 0.23 | oracle | 0.695 | 0.689 (-0.7) | 0.672 (-2.3) |
| opendde-abag | 5 | 0.23 | random | 0.663 | 0.648 (-1.5) | 0.621 (-4.2) |
| opendde-abag | 5 | 0.23 | ranked (ranking_score) | 0.671 | 0.656 (-1.5) | 0.631 (-4.0) |
| opendde-abag | 5 | 0.23 | ranked (deeprank_ab) | 0.668 | 0.654 (-1.5) | 0.628 (-4.1) |
| opendde-abag | 5 | 0.23 | deeprank_ab gap-recovered | 0.169 | 0.143 (-2.6) | 0.121 (-4.8) |
| opendde-abag | 5 | 0.8 | oracle | 0.328 | 0.253 (-7.5) | 0.280 (-4.7) |
| opendde-abag | 5 | 0.8 | random | 0.271 | 0.205 (-6.6) | 0.248 (-2.2) |
| opendde-abag | 5 | 0.8 | ranked (ranking_score) | 0.277 | 0.208 (-6.9) | 0.242 (-3.4) |
| opendde-abag | 5 | 0.8 | ranked (deeprank_ab) | 0.269 | 0.200 (-6.9) | 0.245 (-2.4) |
| opendde-abag | 5 | 0.8 | deeprank_ab gap-recovered | -0.031 | -0.101 (-7.0) | -0.119 (-8.8) |
| opendde-abag | 50 | 0.23 | oracle | 0.745 | 0.734 (-1.1) | 0.713 (-3.3) |
| opendde-abag | 50 | 0.23 | random | 0.664 | 0.648 (-1.6) | 0.621 (-4.2) |
| opendde-abag | 50 | 0.23 | ranked (ranking_score) | 0.671 | 0.651 (-1.9) | 0.625 (-4.6) |
| opendde-abag | 50 | 0.23 | ranked (deeprank_ab) | 0.665 | 0.642 (-2.2) | 0.613 (-5.2) |
| opendde-abag | 50 | 0.23 | deeprank_ab gap-recovered | 0.011 | -0.066 (-7.7) | -0.098 (-10.9) |
| opendde-abag | 50 | 0.8 | oracle | 0.435 | 0.367 (-6.8) | 0.338 (-9.7) |
| opendde-abag | 50 | 0.8 | random | 0.270 | 0.204 (-6.6) | 0.248 (-2.2) |
| opendde-abag | 50 | 0.8 | ranked (ranking_score) | 0.273 | 0.202 (-7.1) | 0.225 (-4.8) |
| opendde-abag | 50 | 0.8 | ranked (deeprank_ab) | 0.273 | 0.202 (-7.1) | 0.237 (-3.6) |
| opendde-abag | 50 | 0.8 | deeprank_ab gap-recovered | 0.021 | -0.014 (-3.5) | -0.122 (-14.3) |
| protenix-v2 | 5 | 0.23 | oracle | 0.523 | 0.481 (-4.2) | 0.502 (-2.2) |
| protenix-v2 | 5 | 0.23 | random | 0.413 | 0.365 (-4.9) | 0.404 (-0.9) |
| protenix-v2 | 5 | 0.23 | ranked (ranking_score) | 0.452 | 0.410 (-4.2) | 0.438 (-1.4) |
| protenix-v2 | 5 | 0.23 | ranked (deeprank_ab) | 0.456 | 0.415 (-4.1) | 0.438 (-1.8) |
| protenix-v2 | 5 | 0.23 | deeprank_ab gap-recovered | 0.386 | 0.428 (+4.1) | 0.349 (-3.7) |
| protenix-v2 | 5 | 0.8 | oracle | 0.186 | 0.132 (-5.5) | 0.166 (-2.0) |
| protenix-v2 | 5 | 0.8 | random | 0.138 | 0.090 (-4.8) | 0.119 (-1.9) |
| protenix-v2 | 5 | 0.8 | ranked (ranking_score) | 0.134 | 0.087 (-4.7) | 0.117 (-1.8) |
| protenix-v2 | 5 | 0.8 | ranked (deeprank_ab) | 0.140 | 0.094 (-4.6) | 0.125 (-1.5) |
| protenix-v2 | 5 | 0.8 | deeprank_ab gap-recovered | 0.036 | 0.099 (+6.3) | 0.128 (+9.2) |
| protenix-v2 | 50 | 0.23 | oracle | 0.652 | 0.606 (-4.7) | 0.625 (-2.7) |
| protenix-v2 | 50 | 0.23 | random | 0.411 | 0.362 (-4.9) | 0.406 (-0.5) |
| protenix-v2 | 50 | 0.23 | ranked (ranking_score) | 0.447 | 0.413 (-3.4) | 0.438 (-1.0) |
| protenix-v2 | 50 | 0.23 | ranked (deeprank_ab) | 0.484 | 0.459 (-2.6) | 0.463 (-2.2) |
| protenix-v2 | 50 | 0.23 | deeprank_ab gap-recovered | 0.304 | 0.396 (+9.2) | 0.259 (-4.5) |
| protenix-v2 | 50 | 0.8 | oracle | 0.255 | 0.202 (-5.3) | 0.212 (-4.2) |
| protenix-v2 | 50 | 0.8 | random | 0.137 | 0.089 (-4.8) | 0.119 (-1.8) |
| protenix-v2 | 50 | 0.8 | ranked (ranking_score) | 0.130 | 0.092 (-3.9) | 0.125 (-0.5) |
| protenix-v2 | 50 | 0.8 | ranked (deeprank_ab) | 0.149 | 0.092 (-5.7) | 0.125 (-2.4) |
| protenix-v2 | 50 | 0.8 | deeprank_ab gap-recovered | 0.105 | 0.025 (-8.0) | 0.066 (-3.9) |
| boltz2 | 5 | 0.23 | oracle | 0.376 | 0.360 (-1.7) | 0.364 (-1.2) |
| boltz2 | 5 | 0.23 | random | 0.287 | 0.253 (-3.4) | 0.283 (-0.5) |
| boltz2 | 5 | 0.23 | ranked (ranking_score) | 0.300 | 0.267 (-3.3) | 0.295 (-0.5) |
| boltz2 | 5 | 0.23 | ranked (deeprank_ab) | 0.311 | 0.279 (-3.2) | 0.313 (+0.3) |
| boltz2 | 5 | 0.23 | deeprank_ab gap-recovered | 0.263 | 0.240 (-2.2) | 0.378 (+11.5) |
| boltz2 | 5 | 0.8 | oracle | 0.134 | 0.094 (-4.1) | 0.092 (-4.2) |
| boltz2 | 5 | 0.8 | random | 0.109 | 0.076 (-3.4) | 0.075 (-3.5) |
| boltz2 | 5 | 0.8 | ranked (ranking_score) | 0.118 | 0.078 (-4.0) | 0.080 (-3.8) |
| boltz2 | 5 | 0.8 | ranked (deeprank_ab) | 0.122 | 0.084 (-3.8) | 0.082 (-4.0) |
| boltz2 | 5 | 0.8 | deeprank_ab gap-recovered | 0.507 | 0.483 (-2.3) | 0.406 (-10.1) |
| boltz2 | 50 | 0.23 | oracle | 0.497 | 0.486 (-1.1) | 0.475 (-2.2) |
| boltz2 | 50 | 0.23 | random | 0.286 | 0.253 (-3.3) | 0.283 (-0.3) |
| boltz2 | 50 | 0.23 | ranked (ranking_score) | 0.292 | 0.257 (-3.5) | 0.275 (-1.7) |
| boltz2 | 50 | 0.23 | ranked (deeprank_ab) | 0.329 | 0.303 (-2.6) | 0.350 (+2.1) |
| boltz2 | 50 | 0.23 | deeprank_ab gap-recovered | 0.206 | 0.214 (+0.8) | 0.349 (+14.3) |
| boltz2 | 50 | 0.8 | oracle | 0.149 | 0.110 (-3.9) | 0.113 (-3.7) |
| boltz2 | 50 | 0.8 | random | 0.107 | 0.074 (-3.3) | 0.072 (-3.5) |
| boltz2 | 50 | 0.8 | ranked (ranking_score) | 0.099 | 0.073 (-2.6) | 0.087 (-1.2) |
| boltz2 | 50 | 0.8 | ranked (deeprank_ab) | 0.124 | 0.083 (-4.2) | 0.075 (-4.9) |
| boltz2 | 50 | 0.8 | deeprank_ab gap-recovered | 0.405 | 0.241 (-16.4) | 0.067 (-33.8) |

Per-target Spearman (mean across panel; the diagnostic whose independence assumption duplicates would break):

| generator | ranker | full | acc-dedup | seq-dedup |
|---|---|---|---|---|
| opendde-abag | ranking_score | 0.023 | 0.041 | 0.002 |
| opendde-abag | deeprank_ab | 0.035 | 0.034 | 0.043 |
| opendde-abag | abag_rank | 0.000 | -0.013 | 0.006 |
| protenix-v2 | ranking_score | 0.134 | 0.147 | 0.089 |
| protenix-v2 | deeprank_ab | 0.106 | 0.112 | 0.060 |
| protenix-v2 | abag_rank | 0.051 | 0.043 | 0.031 |
| boltz2 | ranking_score | 0.021 | -0.005 | 0.049 |
| boltz2 | deeprank_ab | 0.079 | 0.065 | 0.076 |
| boltz2 | abag_rank | 0.025 | 0.005 | 0.057 |

