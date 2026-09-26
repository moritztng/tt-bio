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

## Where the seconds land: one of the four is Blackhole-only

The audit above checked that each lever is present and default-on. That is necessary and not
sufficient — a lever can be on and still not reach the hardware users run on. Checked per lever:

`_IS_SMALL_GRID` is `grid_x * grid_y < 11 * 10` (`tenstorrent.py:5412`), so it is **False on a
p300c (11x10 = 110)** and **True on a Wormhole Galaxy (8x9 = 72**, the grid this repo records in
202 places**)**.

| seconds | lever | grid-gated? |
|---|---|---|
| 1.438 | derived fused `_MM_BLOCK` key | no |
| 11.564 | fused HiFi triangle attention | no |
| **0.1315** | **K4 `TT_BIO_SDPA_BAND_DIV_K`** | **YES — `if _SDPA_BAND_DIV_K and not _IS_SMALL_GRID`** (`tenstorrent.py:1403`), and the line above it says so in words: *"`not _IS_SMALL_GRID` is the Blackhole scope"* |
| 9.5000 | `TT_BIO_TRIATT_NARROW_Q_FALLBACK` | no — its docstring discusses core count as an L1 term, not as a gate |

**So K4's 0.1315 s is Blackhole-only and does not fire on a Galaxy.** JapanFold production runs
on Wormhole Galaxies, so on the hardware that serves users this row has delivered **22.5020 s**,
not 22.6335 s. On a QuietBox p300c the full 22.6335 s is real.

This is not a defect in K4 — it is scoped deliberately and the code says so. It is a defect in
how the total is stated: **a single cumulative number cannot be true on two hardware families at
once**, and this row's own charter is that a win users never receive did not happen. Quote the
figure with the part it was measured on.

## What this does not claim

Nothing here re-measures anything. It checks that every lever the total counts is still present
and still default-on, which is the failure mode a carried number actually has — a default flipped
off, or a derivation regressed, under a total that keeps being copied forward.

## Re-verified against current main, 2026-09-26 07:0xZ

`perf/land_standing/totalrecheck.sh` re-derives this table instead of trusting it. It checks the
SYMBOL on `origin/main`, never a remembered line number, because line numbers move and this row
has already been bitten by a flag whose stated reason had drifted away from it.

At `origin/main` = `4cf8db279`, many commits after the audit above was written:

    1.438 s   derived fused _MM_BLOCK key      OK  widths[i + 1:] present at tenstorrent.py:8479
                                                   regression check: widths[i:] occurrences = 0
    11.564 s  fused HiFi triatt, of3 trunk     OK  triatt_sdpa_hifi_site("openfold3.trunk", True)
                                                   at openfold3_trunk.py:193
    0.1315 s  K4 TT_BIO_SDPA_BAND_DIV_K        OK  env_flag(..., True) at tenstorrent.py:1478
                                                   and the grid gate still present at :1403
    9.5000 s  TT_BIO_TRIATT_NARROW_Q_FALLBACK  OK  env_flag(..., True) at tenstorrent.py:1727

So the total stands: **22.6335 s on a p300c, 22.5020 s on a Wormhole Galaxy** — the difference
being K4, which is gated on `not _IS_SMALL_GRID` and therefore does not fire on an 8x9 Galaxy.

Run it again rather than quoting this block; a lever that was default-on when audited is not
necessarily default-on now, which is `merged-lever-defaults-off-is-not-a-landed-win` read
forwards.
