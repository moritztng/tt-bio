# §6's last uncovered loss term: five targets that satisfy its predicate, and two famous ones that do not

`of3t-auxheads` closed the question of *why* `bond` has never fired and left the question of *what
would fire it* open, with the predicate named exactly. This file answers the second half, verified
against real structure annotations rather than named from reputation.

## The predicate

`core/loss/diffusion.py:208-210` builds

    bond_mask = token_bonds * (is_polymer[..., None, :] * is_ligand[..., None])

so the term is a **polymer–ligand** bond loss. A bond between two ligand tokens contributes nothing,
which is why counting `token_bonds` answers a different question — all 8 of 8 corpus targets carry
inter-token bonds (65–150 pairs each) and **0 of 8** carry a polymer–ligand one.

At mmCIF level the predicate is: a `_struct_conn` row with `conn_type_id == covale` whose two
partners sit in entities of **different kind**, one `polymer` and one `non-polymer` or `branched`.

## Result — 5 of 10 candidates satisfy it

| target | covale links | polymer–ligand | what the bond is |
|---|---|---|---|
| **6VXX** | 63 | **48** | ASN–NAG, SARS-CoV-2 spike, heavily N-glycosylated |
| **7KJ2** | 55 | **38** | ASN–NAG |
| **5T3X** | 114 | **19** | ASN–NAG |
| **4BYH** | 20 | **2** | ASN–NAG |
| **4G5J** | 1 | **1** | CYS–0WN, afatinib covalently bound to EGFR Cys797 |
| 6LU7 | 4 | 0 | — |
| 1HZH | 16 | 0 | — |
| 1OXR / 3PTE / 6O0K | 0 | 0 | — |

**4G5J is the cleanest single-bond case** (one covalent inhibitor, one link) and the glycoproteins
are the high-count cases. Either shape fires the term; the single-bond one is easier to reason about
when the gradient contribution is read.

## The two that fail are the interesting half, and they are why this was checked rather than named

- **6LU7** is *the* textbook covalent complex — the N3 inhibitor bonded to SARS-CoV-2 Mpro's
  Cys145. It fails, because the PDB models N3 as a **peptide-like polymer entity**, so its Cys145
  link is polymer–**polymer**. A structure can be covalent in the literature and polymer–polymer in
  the file.
- **1HZH** is a glycosylated antibody, and all 16 of its covale links are **glycan–glycan**
  (NAG–NAG, NAG–BMA, …). The ASN attachment is not annotated as a covale row at all, so the
  glycan hangs unlinked as far as this predicate is concerned.
- **1OXR, 3PTE, 6O0K** carry **no covale rows whatsoever**; where the chemistry is real it is
  carried as a modified residue or left unannotated.

So "pick a structure with a covalent ligand" would have picked 6LU7, and it would not have fired the
term.

## What this does NOT establish

That OF3's featuriser turns any of these into a `token_bonds` entry with `is_polymer × is_ligand`
set. This verifies the predicate **at the mmCIF annotation level**, one step short of the
featuriser. That step is the remaining work and it is what `of3t-bondcov` is dispatched for: run
one of these through the vendored pipeline, confirm the mask is non-zero, and read the gradient
contribution the way `of3t-auxheads` read the zero one — one forward, two losses differing only in
the bond weight, two backwards.

`check_predicate.py` re-derives the table and fetches what it needs; no structure is redistributed
here, per the campaign's provenance rule.
