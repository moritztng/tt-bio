# What `TOTAL: 22.6335 s` is made of, and whether it is still true

Audited 2026-09-26 against `origin/main`. This row has carried that number unchanged for
nineteen passes, and the passes that built it were rotated out of the state doc by
`scripts/rotate_state_docs.sh` on 2026-09-25, so nothing in the live doc says what it consists of
any more. Re-derived from the archive and re-checked lever by lever.

## The four parts

| seconds | lever | where | still true on main? |
|---|---|---|---|
| **1.438** | the derived fused `_MM_BLOCK` key | OpenDDE, 512 aa | **yes** — `widths[i + 1:]` present, and the self-pair bug (`widths[i:]`) has **0** occurrences, so the derivation did not regress |
| **11.564** | the fused HiFi triangle-attention route with the faithful reduction order | OpenFold3, 512 aa | **yes** — `triatt_sdpa_hifi_site("openfold3.trunk", True)` at `openfold3_trunk.py:193` |
| **0.1315** | K4, `TT_BIO_SDPA_BAND_DIV_K` | 256 < seq <= 384 | **yes** — `env_flag(..., True)` at `tenstorrent.py:1478` |
| **9.5000** | `TT_BIO_TRIATT_NARROW_Q_FALLBACK` | rf3, 896 aa | **yes** — `env_flag(..., True)` at `tenstorrent.py:1727` |
| **22.6335** | | | |

The arithmetic also closes: 1.438 + 11.564 = 13.002, + 0.1315 = 13.1335, + 9.5 = 22.6335. The
apparent unexplained jump from 13.1335 to 22.6335 in the live doc is the rotation, not a missing
justification — the pass that landed narrow-q is in the archive.

## Two caveats a reader of the headline needs

**It sums across models and sizes.** 22.6335 s is not one fold getting 22.6 s faster. It is
OpenDDE at 512 aa plus OpenFold3 at 512 aa plus a band between 256 and 384 plus rf3 at 896 aa.
That is the row's stated convention and it is easy to misread.

**One of the four has a reach condition in its guard.** K4 reads
`if _SDPA_BAND_DIV_K and not _IS_SMALL_GRID`, so its 0.1315 s does not apply on a small-grid part
even though the flag is on. The other three are unconditional at the sizes they were measured at.
Given how often this session found a lever's reach to be narrower than its flag suggests, that is
worth carrying rather than discovering later.

## What this does not claim

Nothing here re-measures anything. It checks that every lever the total counts is still present
and still default-on, which is the failure mode a carried number actually has — a default flipped
off, or a derivation regressed, under a total that keeps being copied forward.
