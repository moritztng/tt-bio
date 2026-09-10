Annotations for the second Blackhole design + embedding walk. A row states what the artifact
was; these state what a row cannot say about itself.

THE LADDER COULD ONLY EVER RUN ON CARD 1. `ladder.py` passed `--devices 1` to `tt-bio design`
with a comment asserting that `--devices` is a COUNT. It is not. `main.py` declares the design
option as `@click.option("--devices", "--device_ids", "devices", ...)`, a comma-separated list
of physical card ids. The first pass held card 1, so the literal matched its grant and the bug
was invisible; the first rung of this pass, on card 3, died in 3.2 s with
  ValueError: Requested Tenstorrent device id(s) [1] not available (present: 3).
The flag now carries the card the ladder was told to use. This is the same shape as the ambient
`TT_VISIBLE_DEVICES` pin: two independent card selectors, and only one of them was being set
from the grant.

THE TARGET AXIS RAN PAST A SINGLE CHAIN, AND BOTH FIXTURES WERE SINGLE-CHAIN. boltzgen's spec
named `include: - chain: id: A`, which silently drops every residue past the first chain, and
pxdesign cropped chain A only. Above 1008 residues (1DP0 chain A entire) both would have
reported a bigger rung while conditioning on the same 1008 residues, which is a ladder that
looks like it climbed. boltzgen now uses `include: all` over a crop that already holds exactly
the residues the rung wants, and pxdesign spills its crop across chains in file order.

THE TARGET ITSELF IS EIGHT REAL CHAINS, NOT A GENERATED BACKBONE.
`perf/bhdesign/targets/big_7324.cif` is 7324 residues / 59144 atoms, built by
  python3 perf/bhdesign/make_big_target.py \
    --parts perf/ceilrfd3/targets/laczc_1008.cif:A perf/ceilrfd3/targets/gpb_823.cif:A \
            (that pair repeated four times) \
    --out perf/bhdesign/targets/big_7324.cif
Only two real chains exist in this repo, so the pair is repeated as chains A-H, each translated
150 A clear of the last. Repeated copies, not invented coordinates: every atom keeps its
deposited geometry and the atoms-per-residue ratio stays at the ~8.1 a deposited structure
carries, where a synthetic backbone would carry ~4 and would report an atom ceiling about twice
the real one. The file is generated and is not committed; the command above rebuilds it.

A REAL DRAM WALL, AND IT IS THE ATTENTION MATRIX. saprot-35m passes 99999 residues in 160.6 s
and FAILS at 131072 in 50.1 s, mechanism `dram`. The allocator's own numbers off the throw:
  requested 34376517632 B, largest free block 4268474176 B
34376517632 = 131104^2 x 2, and 131104 is 131072 rounded UP to the next multiple of 32. So the
allocation is the full L x L attention score matrix in bf16, at the padded token length, as ONE
buffer. The card is 8 DRAM banks x 3.984 GiB = 31.875 GiB = 34.22 GB total, so a 34.38 GB
request does not fit an EMPTY card. That puts this squarely in the oversized-single-tensor
class, shape-determined, and NOT in the cumulative-residency / fragmentation class: the largest
free block reported is a whole bank, there is nothing to coalesce, and no packing would help.
The predicted ceiling is therefore L_max ~ sqrt((DRAM - weights) / 2), which is ~130.8k for a
35m model and lower for the bigger ones in proportion to their resident weights. Every embedding
model in this family should hit the same wall at its own L, and the ladder is what says where.

THE BUCKETING CONTROL AT SCALE. 99999 is 3124.97 x 32, deliberately off the bucket, and the npz
check reads both the row count and the nonzero fraction: it came back [99999, 480], finite,
nonzero_frac 1.0. A run that silently rounded the token axis up to 100000 would return the wrong
row count and a run that left a padded tail unmasked would return a block of zero rows. Both
fail that check. Every other rung in this walk is a power of two and could not have told the
difference. The 34376517632 B request above is the same rounding seen from the allocator side,
and it is doing the right thing: 131072 -> 131104 is a pad to 32, not a silent truncation.

CARD 3 WAS WEDGED WHEN THIS PASS REACHED IT. Its first boltzgen rung aborted (rc -6) at
teardown with
  Device 0: Timed out while waiting for active ethernet core (x=31,y=25) to become active
  again. Try resetting the board.
the same signature the first pass hit on card 1 on arrival. `~/.local/bin/tt-smi -r 3
--no_reinit` cleared it. Not an OOM and not a capacity wall: a sweep that reads this row as a
ceiling has read a wedge.

CARD FANOUT, AND WHAT THE GRANT ACTUALLY IS. This task's grant is card 2. Cards 1 and 3 were
idle (`fuser /dev/tenstorrent/*` showed only the `bh-1536-structure` sibling on node 1), so the
embedding walk ran on card 2, pxdesign on card 1 and boltzgen on card 3, three streams at once,
each pinned with its own `TT_VISIBLE_DEVICES` / `TT_BIO_LEASE_CARDS` and every rung under
`TT_BIO_LEASE_TIMEOUT` 1800-2400 s so a queue behind a co-tenant is not read as a capacity wall.
