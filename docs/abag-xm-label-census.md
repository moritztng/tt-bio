# AbAg-XM label census

164 targets; 161 scorable (the 3 anti-phosphoepitope targets 9ly2, 9ly3, 9lz2 have no scorable native interface -- their contacts are carried by phosphoserine residues that DockQ's loader discards). Every success-rate table uses denominator 161. This census lists every null-label fold x column with its verified cause.

## `dockq` -- 9 folds, 450 samples

| target | gen | null/50 | cause |
|---|---|---|---|
| 9ly2 | boltz2 | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9ly2 | opendde-abag | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9ly2 | protenix-v2 | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9ly3 | boltz2 | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9ly3 | opendde-abag | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9ly3 | protenix-v2 | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9lz2 | boltz2 | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9lz2 | opendde-abag | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |
| 9lz2 | protenix-v2 | 50/50 | anti-phosphoepitope: the native interface is carried by SEP (phosphoserine) residues that DockQ's loader discards -- no scorable interface atoms |

## `interface_lddt` -- 1 folds, 3 samples

| target | gen | null/50 | cause |
|---|---|---|---|
| 9mnu | boltz2 | 3/50 | model pose docked away from the antigen (n_interface_residues: 0) -- correct null |

## `cdr_h3_rmsd` -- 15 folds, 750 samples

| target | gen | null/50 | cause |
|---|---|---|---|
| 9l9y | boltz2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9l9y | opendde-abag | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9l9y | protenix-v2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9lwc | boltz2 | 50/50 | partial CDR numbering (H1 scored; H2,H3 absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9lwc | opendde-abag | 50/50 | partial CDR numbering (H1 scored; H2,H3 absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9lwc | protenix-v2 | 50/50 | partial CDR numbering (H1 scored; H2,H3 absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9mnu | boltz2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9mnu | opendde-abag | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9mnu | protenix-v2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9msc | boltz2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9msc | opendde-abag | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9msc | protenix-v2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; none absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9udq | boltz2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; H1 absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9udq | opendde-abag | 50/50 | partial CDR numbering (L1,L2,L3 scored; H1 absent) -- consistent with the native heavy chain unresolved at CDR-H3 |
| 9udq | protenix-v2 | 50/50 | partial CDR numbering (L1,L2,L3 scored; H1 absent) -- consistent with the native heavy chain unresolved at CDR-H3 |

