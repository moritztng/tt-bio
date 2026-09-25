# TT_BIO_TRIATT_DIVIDING_K: what it is worth, and what is still unknown

Written 2026-09-26 by `land-standing` so the next reader does not re-derive any of it. Every
number here was measured on qb2 card 3, p300c; the artifacts are under
`perf/land_standing/out/`.

## What the lever does

`_tri_att_sdpa_hifi_inner` builds its k ladder from `_dividing_k_chunks` instead of the shipped
pick alone. Where the shipped k does not divide the padded length, the fused route today offers
only illegal rungs and declines, falling through to `_fp32_softmax_attention`.

## Reach: one length, and it is a hole rather than a frontier

OpenFold3's trunk is the only site with `tri_att_sdpa_hifi` on by default (`boltz2.trunk`,
`rf3.tri_att` and OpenFold3's confidence/msa/template sites are all `False`). It pads the pair
axis to a multiple of 64, so of the 20 lengths the lever changes only five can ever be presented.
All five measured, by reading `TRIATT_FUSED_HIFI_STATS` and `PICKS` out of real folds:

| tokens | shipped arm | with the lever | verdict |
|--------|-------------|----------------|---------|
| 704  | 384 served, `(704,704) q352 k704`   | identical, CIF byte-identical | inert |
| 832  | **0 served, 384 declined**          | 384 served, `(832,832) q416 k416` | **live** |
| 1088 | 384 served, `(1088,1088) q64 k1088` | not needed | inert |
| 1216 | 384 served, `(1216,1216) q64 k1216` | not needed | inert |
| 1472 | 384 served, `(1472,1472) q64 k1472` | not needed | inert |

At the four inert lengths `_tri_att_fused_large_s` serves before the k ladder is reached. So 832
is a hole between 704 and 1088, which both serve. It is not a size ceiling: 1472 is fine without
the lever.

A user folding 769-832 tokens is the only one who loses the fused route.

## Accuracy: favourable where it can be measured, unanswerable where it cannot

At the kernel, against a float64 reference, the route the lever unlocks is closer to the
reference than the path it replaces at every length checked — 8.7 % to 12.6 %, and 11.4 % at 832
(`perf/land_standing/out/khole_fixed/`).

At the structure, the question is open and **this box cannot close it**. Five folds at 832, three
with the lever on and two without, give

    window        within-arm (seed)   across-arm (seed+lever)   permutation p
    full 1-832        19.574 A            16.143 A                 0.30
    copy 1-298        15.882 A            12.975 A                 0.30
    core 1-150         9.937 A             8.407 A                 0.90

Across-arm distances are if anything smaller than within-arm ones, so there is no separation to
see — but the test has no power to see one either. Five folds give ten labellings, so p cannot
fall below 0.1, and a fold that disagrees with itself by ~10 A across seeds inside a 150-residue
core cannot resolve an effect orders below that.

The cause is the fixture, not the lever. 832 tokens of tiled CDK2 folded single-sequence is about
the least determined input available, and narrowing the window does not rescue it: the floor
falls from 19.6 to 9.9 A and stops. What is needed is a confident target at 832 tokens folded
with an MSA. There is no local ColabFold DB on this box and `~/.boltz/msa` is empty, so that
needs a cached a3m via `RELEASE_GATE_MSA_DIR` or the online server.

## Speed: about 1.5x on that one length, unpriced

    lever on     86, 90, 94 s
    lever off   137, 302 s

The 302 s reading was taken under contention; the same arm read 137 s later. Every fold in this
record ran with loadavg 12-20 and with card 2, card 3's board-pair sibling, serving another row,
and no AICLK was sampled during any of them. **Do not quote a speedup from this.** A real number
needs interleaved arms in one process on one device open, the sibling verified idle, and AICLK
sampled during each leg from `/sys/class/tenstorrent/tenstorrent!N/tt_aiclk` (note the
`tenstorrent!` prefix, and that N is not the `TT_VISIBLE_DEVICES` index —
`perf/bcx_stack/stack.py:sysfs_node` sorts by PCI address).

## Recommendation

Keep it off by default until the structural question is answered on a fixture that could detect a
problem. The gain is one length, the speed term is unpriced, and nothing is being lost meanwhile
except consistency at 769-832 tokens.

If someone wants it on sooner, the honest minimum is one confident 832-token fold per arm with an
MSA, showing the structures agree to within that fixture's own seed floor.
