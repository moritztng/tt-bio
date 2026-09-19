# Region T on shipping main: +0.2020 s (1.01422x)

Scored against `compose_prereg.md`, registered at `ca662c12e`/`d5c9cbcfc` before the first block
landed. Session `regiont_compose_ab.json`, tree = `origin/main@67f1ecaa8` + region T, 5 blocks x 3
folds an arm interleaved, qb2 node 3, benchlock, **1350-1350 MHz over 1827 during-fold samples**.

    admissible blocks (0 discarded per the registration)
    base [14.387 14.439 14.346 14.428]   on [14.209 14.207 14.209 14.162]
    base 14.410 s   on 14.208 s   delta +0.2020 s   ratio 1.01422
    A/A floor 1.00648   effect is 2.19x its floor   permutation p = 3/70
    all four admissible blocks favour the on arm
    whole session including block 0: +0.2190 s, 1.01541, same floor

## Four of six registered checks held, and both misses have one cause

    [PASS] base median in 14.35-14.50        14.410 s
    [PASS] on median in 14.18-14.35          14.208 s
    [PASS] ratio in 1.007-1.015              1.01422
    [FAIL] A/A floor under 1.006             1.00648
    [PASS] the two arms' digests differ      f2d1dce81ffa3f34 vs 1ee654ed6730ba5f
    [FAIL] no foreign device holder          trix-floor and trix-scaffold-attribute, blocks 1-4

Both failures are the same event: Moritz's TRIX campaign started folding on the OTHER board at
around 23:36Z, four minutes into the session. **My board pair stayed clean throughout** —
`pair_idle` read "sibling 2 idle" before all ten arms, and `trix-transaction`, the row that holds
my sibling card 2, never appears in the holder list. So the channel here is the host (same PSU,
host DRAM, PCIe), which is the channel that retracted `c14-matmul-ceiling`'s +0.2510 s, not the
board-power coupling.

The floor is what it cost: 1.00648 against 1.00335 and 1.00432 in the two quiet pre-hoist
sessions, inflated about 1.5-2x. The effect still clears it by 2.19x with every admissible block
favouring the on arm. There is no complete separation this time, and one fold is why — `on` block
1 fold 0 reads 15.068 s against 14.130 and 14.209 beside it, the signature of a transient rather
than of the lever.

**Do not read this as a clean number.** It is a real number on a clean pair and a loud host, and
it is stated that way. A fully quiet repeat would tighten the floor; it would be surprising if it
moved the estimate, because three sessions now agree.

## Region T across every session it has

    session                         base       on         delta      ratio     floor
    contended, pre-hoist            15.135 s   14.471 s   +0.6640 s  1.04588   1.02219
    quiet #1, pre-hoist             14.631 s   14.435 s   +0.1960 s  1.01358   1.00335
    quiet #2, pre-hoist             14.630 s   14.455 s   +0.1750 s  1.01211   1.00432
    loud host, clean pair, COMPOSED 14.410 s   14.208 s   +0.2020 s  1.01422   1.00648

The three sessions that were not taken against a contended base arm agree on +0.175 to +0.202 s.
Composition with cond-hoist did not eat region T: the absolute saving is the same size after the
hoist as before it, within session-to-session spread.

## A free confirmation of a shipped lever

This session's base arm **is** `origin/main` by construction — the branch's only production delta
is region T and the base arm has it off. Same harness, same box, same clock, base medians:

    pre-hoist main   14.630 s
    shipping main    14.410 s      difference 0.220 s

`TT_BIO_DIT_COND_HOIST` was landed at `15f1ab0ac` on a measured **+0.2052 s**. An independent
harness on a different night reads its shipped value at **0.220 s**. That is the first time this
campaign has confirmed one of its own shipped levers from the outside.

## Accuracy: cond-hoist moved the structure, region T's delta on top of it did not change character

    arm     cif digest          plDDT       identical across all 15 folds of the arm
    base    f2d1dce81ffa3f34    0.863008    yes
    on      1ee654ed6730ba5f    0.865392    yes

Both differ from the pre-hoist pair (`45781db716ebf020`/0.845919 and `791696c3e443676d`/0.845902),
which is cond-hoist, not region T: cond-hoist ships and is not bit-exact, and it was approved on
CA-lDDT rather than on a digest. Region T's own contribution is the base-to-on step, and on this
tree it moves plDDT by **+0.002384**, the same sign and size as the -0.000267 to +0.001972 steps
pass 1 measured for the other levers at this size.

**The structure has not been re-scored on this tree and must be before the flip.** Region T's
0.42886 A against the 0.60 A bar was measured on the pre-hoist structure, and the pre-hoist
structure is no longer what ships. Inheriting the accuracy verdict by digest identity worked while
the base was `45781db716ebf020`; it does not transfer across a base that moved. That is the one
thing this session does not close, and it is named here rather than left for a reader to notice.
