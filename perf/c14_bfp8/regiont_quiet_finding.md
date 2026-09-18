# Region T on a quiet p300c, current main: +0.6640 s does not reproduce

`wk/c14-regiont-on-main` @ `f468f46f6` = `origin/main@a4319c696` + region T (131+/39- under
`tt_bio/`). qb2 node 3, board `...410D`, sibling node 2 with no fd for the whole session, under
`benchlock`, 4 blocks x 3 folds an arm interleaved, one warmup fold discarded per process,
AICLK asserted at 1350 and sampled during every fold: **1350-1350 MHz over 1478 samples**.
Result file `regiont_quiet_ab.json`, comparison `regiont_compare.py`.

## As run, the session verdict is NULL

    session                          base med    on med     ratio      A/A floor   verdict
    record, contended (4x3)          15.135 s    14.471 s   1.04588    1.02219     WIN, p=2/70
    today, quiet (4x3)               14.669 s    14.459 s   1.01459    1.02333     NULL, p=14/70

`1.01459 < 1.02333`, so by the harness's own rule the session is a null. That is the number of
record for this pass and it is not the interesting part.

## What moved is the BASE arm, not the lever

    arm     record folds                     today folds
    base    14.845 .. 15.516                 14.598 .. 14.969
    on      14.445 .. 14.586                 14.403 .. 15.364 (block 0 only, see below)

The on arm reproduces the record almost exactly: today's admissible on folds span
14.403-14.551 against the record's 14.445-14.586. The base arm came in **0.35-0.68 s faster than
the record's base**. So the 0.6640 s did not shrink because region T got worse; it shrank because
the arm it was measured against was slow, and on a quiet box it is not.

That is what the record's own A/A floor was already saying. Its base blocks spread **1.02219**
while its on blocks spread **1.00318** — a 7x asymmetry between two arms doing the same amount of
work. `state/bfp8/LEDGER.md` flagged the floor as "inflated by co-tenancy" and then argued the
sign survives because block 3 inverts the load ordering. The sign does survive. **The magnitude
does not**, and no sign test can rescue a magnitude.

## Block 0 was inadmissible and the harness recorded why

    arm  blk  folds (s)                   loadavg1 during     foreign device holders
    base  0   14.956 14.969 14.776        2.12 3.07 3.61      1 1 1
    on    0   15.236 15.106 15.364        3.78 4.18 4.83      0 1 1
    base  1   14.681 14.629 14.608        1.50 1.39 1.37      0 0 0
    on    1   14.499 14.445 14.551        1.20 1.23 1.70      0 0 0
    base  2   14.675 14.615 14.598        1.95 2.81 2.54      0 0 0
    on    2   14.472 14.434 14.422        1.75 2.14 1.89      0 0 0
    base  3   14.664 14.721 14.631        2.03 2.36 2.06      0 0 0
    on    3   14.406 14.435 14.403        1.17 1.13 1.10      0 0 0

Block 0 is the only block where a **foreign process held a tt device during the fold**, on 5 of its
6 folds, at loadavg up to 4.83. Blocks 1-3 have zero foreign holders. The per-arm guard passed
before block 0 because loadavg had dipped to 2.00 at that instant; the box went loud again during
the folds. **Part of that load was me**: I was running ssh status polls against the host through
block 0 and stopped afterwards. The agent driving a measurement is a co-tenant of it.

Excluding block 0 on a criterion the instrument records and that is symmetric between arms -- a
block in which any fold saw a foreign device holder is inadmissible, which is the same channel the
between-arm guard exists to exclude -- blocks 1-3 give:

    base median 14.631 s   on median 14.435 s   delta +0.196 s   ratio 1.01358
    A/A floor (base block medians 14.629 14.615 14.664)          1.00335
    effect is 4.05x its own floor; permutation over block medians p = 2/20
    COMPLETE SEPARATION: slowest on fold 14.551 < fastest base fold 14.598, gap +0.047 s

**This exclusion is post-hoc and is labelled as such.** It is not the verdict. It is the reason the
confirmatory session below is worth its 25 minutes rather than being a repeat for its own sake.

## Pre-registered, before it runs

The confirmatory session is `--blocks 5 --folds 3`, and **block 0 is discarded unconditionally as
a session warm-up**, declared here before the run rather than chosen after it. Rationale from this
session's own data, independent of the contamination: block 0's discarded warmup folds cost 18.55 s
(base) and 16.86 s (on) against ~15.0 s in every later block, so the first process of a session
pays a first-touch cost that its single warmup fold does not absorb.

Predictions, so the next reader can score this rather than re-argue it:

* base and on medians land inside 14.60-14.70 and 14.40-14.50 respectively;
* ratio lands in **1.010-1.018**, i.e. it confirms ~+0.2 s and **refutes +0.6640 s**;
* the A/A floor over 4 admissible base blocks lands under 1.006;
* both arms reproduce the digests below exactly. If either moves, the tree changed under us and
  the seconds are not comparable.

What would make this negative: an on median above the base median in two or more admissible
blocks, or an A/A floor above the effect again on a session with no foreign holder in any block.

## Accuracy is inherited by identity, not assumed

Every fold in all 8 blocks:

    base  cif 45781db716ebf020   plDDT 0.845919
    on    cif 791696c3e443676d   plDDT 0.845902

Bit-identical to `region_t_ab.json`'s 12-fold-an-arm record and to pass 7's census on today's main.
The on-arm structure is therefore exactly the one scored at **0.42886 A worst at 512 aa against the
0.60 A bar**, 0.30x the 1.42336 A same-session seed floor. Kernel path 560 served / 0 declined on
all four levers in both arms, every block, so neither arm fell back and it is the same programs on
narrower operands.

## Consequence for the ledger

Region T stays a live candidate and drops from **+0.6640 s** to **~+0.20 s**, pending the
confirmatory session. At ~0.2 s it is still the second largest unshipped item in C14's tail and
four times `TT_BIO_MM_SHORT_M_BW`'s +0.047 s, so under this row's no-lower-bound rule it remains
worth landing -- but it is no longer "larger than everything else in the tail combined", and the
headline in `state/bfp8/LEDGER.md` needs correcting to the quiet-box figure.

Both quiet repeats now agree with each other and disagree with the contended one:

    qb1 p150a, quiet   1.00829   inside its floor 1.00826-1.01162
    qb2 p300c, quiet   1.01358   4.05x its floor 1.00335, complete separation
    qb2 p300c, loud    1.04588   2.07x a floor its own arm asymmetry had already flagged

The board-class story the bfp8 campaign told -- 5.6x smaller on p150a because region T deletes DRAM
bytes and p150a is relatively more compute-bound -- is not needed to explain the two quiet numbers.
They are within 0.5 % of each other.
