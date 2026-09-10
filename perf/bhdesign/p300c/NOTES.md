Annotations for the p300c design + embedding sweep (qb2). A row states what the artifact was;
these state what a row cannot say about itself.

THE WALL IS SHARED, AND IT IS NOT THE MODEL. All six embedding models fail at 131072 residues
with the same DRAM throw, and the request is the same size in every one of them: 34,376,517,632 B
against a largest free block of 3,924,449,088 B (one bank, 3.655 GiB). 34,376,517,632 / 2 bytes
= 17,188,258,816 elements = 131072 x 131136, i.e. an L x L(+64 pad) bf16 matrix. A 35M-parameter
model and a 6B-parameter model asking for the identical allocation is the tell: what runs out of
DRAM is the quadratic per-sequence attention term, not the weights. That is the oversized-
single-allocation class, shape-determined, and it wants one shared fix, not six.

WHERE EACH CEILING SITS. esmc-300m and esmc-600m are proven at 98304 and refused at 131072, so
their ceiling is inside that interval. saprot-35m, saprot-650m, saprot-1.3b and esmc-6b are
proven at 65536 and refused at 131072; 98304 was not run on them, so their ceiling is somewhere
in 65536..131072 and the lower number is a ladder top, not a measured wall.

ESMC-6B IS NOT SPECIAL ON BLACKHOLE. The ~12.8 GB resident case embeds 65536 residues in 206.7 s
here. The Wormhole ceiling of 1968 residues (DRAM oom at 1984) does not transfer and must not be
copied into a Blackhole row.

BUCKETING NEGATIVE CONTROL. The npz check reads the row count and the nonzero fraction, so an
unmasked padded tail returns rows of zeros and a wrong bucket returns the wrong row count; both
fail. That check only proves something at a length that is NOT already a multiple of 32, because
1536 (48x32) cannot distinguish a correct implementation from one that silently rounds up. Every
embedding model is therefore also walked at 2000 (62.5x32), and every one returned exactly 2000
rows with nonzero_frac 1.0. saprot's 0.9999 rows are one exactly-zero value in a 1536x480 block,
not a padded tail.

THREE HARNESS DEFECTS, ALL FOUND HERE, ALL PRESENT IN THE qb1 WALK.
 1. A failed rung kept blob[-2500:]. On a ttnn throw that window is entirely C++ backtrace frames
    and nanobind "leaked instance" teardown lines, and the message that says why sits thousands of
    characters above it. Both remaining defects were invisible until the rung started keeping the
    diagnostic lines instead of the last ones.
 2. pxdesign conditioned on chain A alone. big_1831.cif is A=1008 + B=823, so a rung asking for
    1536 or 1831 conditioned on 1008 and still wrote a perfectly good 80-residue binder: the
    binder's size does not depend on the target, so the artifact check could not see it. Rows
    px1536 and px1831 first landed as PASS at 12.5 s and 11.2 s, which is the plateau that gave it
    away. The fixture now spills into later chains and the check reads conditioned_tokens out of
    designs.json; rescoring the old artifacts turns both rows FAIL, and the re-run with the fixed
    fixture passes at 44.8 s and 50.3 s. The qb1 pxdesign row is a 256 rung, below chain A's
    length, so it is unaffected in value but was checked by the blind check.
 3. boltzgen was invoked with --devices 1. That flag is an ID LIST, not a count, and it worked on
    qb1 only because the grant there happened to be card 1. On card 3 every boltzgen rung died in
    2 s inside detect_tenstorrent_devices, before the model loaded, which the old tail reported as
    "no output directory" with mechanism unknown. Unset, it uses every visible card, and
    TT_VISIBLE_DEVICES has already narrowed that to one.

BOLTZGEN VS THE qb1 ROWS. Here 256, 512 and 1008 target residues all PASS (59.7 s, 59.0 s,
180.4 s) with the designed 80-residue chain in the artifact next to the target chain. qb1 recorded
l1-classified failures at 512, 768 and 1008. Those rows predate the artifact-check fix on that
branch, so this is reported as a difference that needs qb1 re-run with the current ladder, not as a
p150a-vs-p300c hardware finding.

CO-TENANCY. Cards 0..3 were free at 00:05Z and no other tt-bio process appeared during the walk.
The design rungs were fanned onto card 3, the idle chip on this task's own board, with the lease
widened to 2,3 on those commands only. Nothing was killed and no card was reset.

# --- round 3 annotations ----------------------------------------------------------------------
#
# BOTH OOM CLASSES ARE NOW MEASURED. Round 2 reported that no rung had hit the cumulative-residency
# class. That was true of the rungs it ran and is no longer true; four more rungs found it twice.
#
#  Oversized single allocation, shape-determined. All six embedding models fail at 131072 with
#  byte-identical requests: 34,376,517,632 B DRAM across 8 banks, 4,297,066,496 B per bank, i.e.
#  131072 x 131136 bf16. A 35M model and a 6B model asking for exactly the same allocation is the
#  tell: what runs out is the quadratic per-sequence attention term, not the weights. The largest
#  free block ranges from 4,268,474,176 B (saprot-35m) to 2,681,800,512 B (esmc-6b), tracking how
#  much each model's weights already hold, and even the emptiest of the six is 28.6 MB per bank
#  short.
#
#  Cumulative residency. Two rungs fail asking for a fraction of what the rung above asked for:
#  esmc-6b at 98304 wants 1,510,440,960 B (188,805,120 B/bank) with 134,580,032 B free, and
#  pxdesign at 3662 target residues wants 14,346,289,152 B (1,793,286,144 B/bank) with
#  1,154,575,040 B free. The same card, at the same 98304 rung, carried saprot-1.3b to a PASS.
#  The binding constraint is what is already resident, not the shape of the request, and the two
#  classes want different fixes: chunk or shard the big tensor vs get weights off the card between
#  stages.
#
# ESMC-6B IS THE ONLY MODEL-DEPENDENT CEILING IN THE SWEEP. Five embedding models are proven at
# 98304 and refused at 131072. esmc-6b is proven at 65536 and refused at 98304, because the
# activation at 98304 is the same for all six and only esmc-6b carries ~12.8 GB of weights on top
# of it. tt_bio/size_limits.py already records this mechanism for esmc-6b on Wormhole ("the 6B
# weights nearly fill the chip", 1968 pass / 1984 fail). The mechanism transfers to Blackhole; the
# number does not, by a factor of 50.
#
# A FOURTH HARNESS DEFECT, AND IT IS THE SECOND SPELLING OF THE THIRD. The BoltzGen fixture wrote
# `include: - chain: id: A`, so a crop of 3662 residues whose chain A is 1008 conditioned on 1008
# and designed a perfectly good 80-residue binder. The 1831 and 3662 rungs first landed PASS at
# 151.0 s and 150.2 s -- two sizes apart, 0.8 s apart, the same plateau that gave the pxdesign
# fixture away one round earlier. The designcif check could not see it either: it asserted that
# some chain was the designed one, never that the rest of the complex was the size asked for.
# Fixture now includes every chain the crop produced, with the binder moved to chain Z so it cannot
# collide with a crop labelled A..H; the check sums the non-binder chains and requires the
# asked-for target. Rescoring the two old artifacts turns them FAIL and leaves 256, 512 and 1008
# PASS -- those three were inside chain A and were always honest. That is the negative control.
# Re-run, 1831 passes in 548.9 s with chains {A: 80, B: 1008, C: 823} and 15323 atoms, against
# 180.4 s at 1008. Three times the wall for 1.8x the atoms is what the real target costs.
#
# THE FIXTURE SHELF IS NO LONGER THE LIMIT. make_big_target.py stacks real deposited chains side by
# side, each keeping its geometry and translated clear along +x. big_3662.cif (3662 residues,
# 29572 atoms) and big_7324.cif (7324 residues, 59144 atoms) are built from laczc chain A and gpb
# chain A repeated, so both design models can now be walked past the largest single chain on hand.
# pxdesign found its wall on the first rung above it.
#
# THE PUSH THAT NEVER LANDED. Round 2 recorded the branch as unpushable "because this qb2 worktree
# has no github credentials". That was wrong. The credentials work over ssh with a forwarded agent;
# GitHub rejected the push with GH001 because the walk had committed perf/bhdesign/p300c/work/,
# 1.9 GB of per-rung output including seven .npz embeddings over the 100 MB file limit, the largest
# 301 MB. The work dirs are now gitignored the way perf/sizegate/work/ and perf/capacity/work*/
# already were, the two commits were rewritten without them, and the branch is on origin.
