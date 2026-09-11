"""Every model's MEASURED size ceiling, and the refusal that keeps a user off it.

A model that cannot fold N residues on the hardware in front of it should say so in a second, by
name, before anything opens a device. Until this module existed tt-bio said nothing: the ceilings
lived only in the serving platform (``japanfold/catalog.py``), as a record of where a worker had
been observed to die, so a plain CLI user who asked OpenDDE for 1024 residues got the crash itself
-- an L1 throw, an OOM, or a chip left wedged for the next job. The ceilings are a property of the
engine and the chip, not of the web front end, so they belong here.

WHAT IS IN THIS TABLE, AND WHAT IS DELIBERATELY NOT
---------------------------------------------------
Only a **hardware** ceiling: a size at which the engine does not produce an answer, or does not
produce one inside a runtime anybody would wait for. Three things are excluded on purpose.

*Demo policy is not a ceiling.* The platform's ``LIMITS`` also carry a free-demo fence -- concurrent
jobs, per-IP rates, structures per run, retention -- and ``catalog.DEMO_NOTE`` says out loud that
"the full platform has no such limits". Mirroring those here would import a business rule into an
inference engine. None of them appear below.

*An inherited number is not a measurement.* Most models carry no ``max_residues`` of their own in
the catalog and inherit the platform's advertised 1024. That 1024 is the demo fence, not a fold
anybody walked to a failure. Those models get an ``UNMEASURED`` row here and are **never refused**:
absence of a limit is not a limit, and inventing one would refuse work the engine can do.

*A number from the wrong chip is not a measurement either.* Every row is keyed by
``ttnn.get_arch_name()``, and a number measured on one part is never copied to another: a Blackhole
p150a has 2.7x the DRAM of a Galaxy chip and a 13x10 grid against 8x9, and OpenDDE -- capped at 544
on Wormhole -- folded every rung to 1024 aa on a p150a (``state/sizes-recheck-opendde.md``). Most
rows here are still Wormhole-only, from the GWH02 Galaxy pool, and on Blackhole those models are
refused nothing. The ``blackhole`` rows that do exist were each walked to a failure on a p150a with
their own ladder, and the guard test asserts a Blackhole row names the part it was measured on and
does not repeat the Wormhole numbers.

WHY EVERY ROW CARRIES ITS NEGATIVE CONTROL
------------------------------------------
A ceiling nobody has crossed is a guess. ``fail_at`` is the size ABOVE the cap that was measured to
fail, and it is what separates "we walked a ladder until it broke" from "we stopped testing here".
The guard test refuses a row that claims a memory- or runtime-bound ceiling without one, so the
distinction cannot decay into a convention. Three states are expressible and they mean different
things:

  * ``fail_at=<int>``   -- a failing size is on record. The cap is the largest size below the FIRST
                           failure, never merely the largest size that happens to work. Pass/fail is
                           not monotonic in residue count for the L1-clash class (OpenDDE folds 544,
                           throws at 576, folds 608, throws at 640), so publishing the largest
                           passing size would promise a size that throws.
  * ``fail_at=None``    -- nothing above the cap has ever failed, because nothing above it was ever
                           run. The row must declare ``binds=LADDER_TOP``: the cap is the top of a
                           ladder, and the real ceiling may be higher.
  * ``fail_at=UNRECORDED`` -- a failure above the cap was witnessed, but its size was not written
                           down. The cap is real; the control is an evidence gap, and naming it here
                           keeps it visible instead of laundering it into one of the other two.

A CEILING IS ONLY VALID IN THE CONFIGURATION IT WAS MEASURED IN
---------------------------------------------------------------
Every number below was walked with the flags the serving platform sends, and for the MSA-dependent
models that means **the MSA on**. Single-sequence folding is measurably roomier: OpenFold3 caps at
576 with a real alignment and folds **768 single-sequence in 301 s** (catalog.py). The same is
likely true of the other ``dram_msa`` rows.

It is true of the ``l1_clash`` rows too, which this file used to deny. The throw does come from a
static circular-buffer layout in the structural refiner, and the refiner runs on the structural
token axis, so its own shapes do not depend on alignment depth -- which is why "nothing to do with
alignment depth" looked safe. But a CB *clash* is not about one op's size, it is about what else is
resident: the message names an L1 buffer at 352256 that the static CB region, ending at 382240,
runs into. Deeper alignments leave more live, so the same op clashes at a smaller residue count.
Measured on OpenDDE 2026-09-07, one fixture walked at four depths: it folds past 1056 residues at
35 alignment rows, 832 at 512 rows, and fails at 576 with the platform's default 8192. A ladder is
therefore a (residues, depth) point and ``msa_rows`` is the field that says which.

Rather than guess a second set of numbers, the table keeps the measured (MSA-on) ceiling and
``TT_BIO_SIZE_LIMIT=0`` turns any refusal into a warning -- see ``enforced()``. A single-sequence
ladder for the MSA-dependent models would let these become two rows instead of one, and until
somebody walks it, guessing which configuration a user is in would be inventing a ceiling.

THE UNITS ARE RESIDUES, AND A LIGAND IS COUNTED ON TOP
-----------------------------------------------------
Every ceiling below was measured by walking residue counts, so residues is what this table can
honestly express -- the model-internal expansion is already folded into the measured number.
OpenDDE's structural axis is ``Ns = 2*n_res - n_gly``, roughly 1.9x the residue count; that needs
no restating here, because a ladder walked in residues already paid for it.

A ligand was the one input this could not see. Its heavy atoms are TOKENS the trunk pays for
exactly like residues and they are nowhere in the residue count, so a request at the residue cap
plus any ligand was admitted here and then died on the chip -- the failure this whole file exists
to replace with a sentence. Four rows now say where their wall really is. Three carry evidence
that names it as a token wall (openbind, esmfold2, esmfold2-fast) and one was walked on the token
axis outright (protenix-v2); each declares ``ladder_ligand_atoms``, the ligand ITS OWN rungs
carried, which is what turns its residue numbers into token numbers without inventing anything.
``check`` then refuses a ligand-bearing input whose PADDED token width is past that wall, padded
by ``token_axis``'s own bucket so the guard and the model cannot drift apart.

It is an EXTRA refusal and never a relaxation. An input with no ligand is compared on residues
exactly as it always was, which matters more than it looks: openbind's 960 was walked WITH a
35-atom ligand, so its token wall is the 1024 that 995 pads to, and letting that wall speak for a
ligand-free input would quietly raise a published ceiling nobody re-walked.

A row without ``ladder_ligand_atoms`` keeps the residue check alone. Asserting a token wall from a
residue ladder that never saw a ligand would be a units substitution, not a measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# --- What binds at the top of the range ------------------------------------------------------
# The brief's question "memory or runtime?" is a field, not a footnote: a ceiling that used to be a
# memory wall and is now a wall-clock wall is a different piece of work, and the two are fixed by
# different means.
MEMORY = "memory"          # the engine does not fold at all: an OOM, an L1 throw, a wedge
RUNTIME = "runtime"        # it folds, but past a wall-clock anybody would wait for
FREEZE = "freeze"          # it neither folds nor fails: progress stops and never resumes, so
#                            there is no answer AND no error. Distinct from MEMORY (which raises)
#                            and from RUNTIME (which finishes, late).
LADDER_TOP = "ladder_top"  # nothing failed; this is simply the largest size proven
UNMEASURED = "unmeasured"  # nobody walked a ladder. Never refuses.
BINDS = (MEMORY, RUNTIME, FREEZE, LADDER_TOP, UNMEASURED)

# The named failure mechanism, so a row says WHY and not just WHERE. Raising a ceiling means
# attacking one of these, and they do not yield to the same fix -- chunking is the answer for a DRAM
# wall and useless against an L1 clash, where the throw lands at a consumer's program creation with
# DRAM nowhere near full.
L1_CLASH = "l1_clash"              # L1 static circular buffers overlap the tensor. Not monotonic.
# L1 is simply too full for a residency the code asked for on purpose, and that is a THIRD thing,
# distinct from both neighbours above and below it. Not L1_CLASH: nothing static is in the way, the
# allocator just has fewer bytes per bank than the request. Not FRAGMENTATION either, and the throw
# proves it -- `free` and `largest free block` come back EQUAL, so there is no block to coalesce and
# a defragmenting fix would buy nothing. What sets the size is a chunk height chosen against an L1
# budget that was tuned at a smaller size, so the lever is the budget or the residency, not the
# layout (`one-size-tuning-is-a-standing-defect-class`).
L1_BUDGET = "l1_budget"            # an L1 residency sized by a budget too optimistic at this size
DRAM = "dram"                      # a single allocation the chip cannot serve
DRAM_MSA = "dram_msa"              # DRAM, in the MSA track, growing with tokens and depth
FRAGMENTATION = "fragmentation"    # enough free DRAM, no block big enough
TRUNK_FREEZE = "trunk_freeze"      # the trunk stops mid-recycle: CPU still burning, RSS flat to
#                                    the byte, no log line, no allocator refusal, forever
NO_FAILURE = "none"                # nothing broke
UNKNOWN = "unknown"                # not diagnosed
MECHANISMS = (L1_CLASH, L1_BUDGET, DRAM, DRAM_MSA, FRAGMENTATION, TRUNK_FREEZE,
              NO_FAILURE, UNKNOWN)

# WHAT THE NUMBER COUNTS. Not decoration: the two design models were measured in DIFFERENT
# denominators, and holding one against the other would be a silent unit substitution. RFD3's 704 is
# motif PLUS designed residues, while PXDesign's 768 is TARGET residues only, with its 80-residue
# binder on top and outside the number. A guard that compared a total against a target-only cap
# would refuse correct work on one model and pass oversized work on the other. Each row names its
# denominator, each model has a sizer that produces that denominator, and the guard test asserts the
# two agree -- so the slip is a test failure rather than a subtle wrong answer.
RESIDUES = "residues"              # polymer residues in the input, summed over chains and copies
DESIGN_TOTAL = "design_total"      # motif + designed, i.e. everything the model tokenises
DESIGN_TARGET = "design_target"    # the conditioned target only; the binder is extra
MAX_SEQUENCE = "max_sequence"      # residues in the LONGEST single sequence, not their sum
COUNTS = (RESIDUES, DESIGN_TOTAL, DESIGN_TARGET, MAX_SEQUENCE)

# How each denominator reads in a refusal. A message that just said "residues" for all three would
# leave a design user unable to tell which number of theirs is too big.
_COUNT_NAMES = {
    RESIDUES: "residues",
    DESIGN_TOTAL: "residues (motif + designed)",
    DESIGN_TARGET: "target residues",
    MAX_SEQUENCE: "residues in its longest sequence",
}


class _Unrecorded:
    """``fail_at`` when a failure above the cap is witnessed but its size was never written down."""

    def __repr__(self) -> str:
        return "UNRECORDED"

    def __bool__(self) -> bool:
        return True


UNRECORDED = _Unrecorded()


@dataclass(frozen=True)
class Ceiling:
    """One model's ceiling on one architecture. Every field is load-bearing; see the guard test."""

    residues: int | None       # the published cap. None only when binds is UNMEASURED.
    pass_at: int | None        # a size MEASURED to fold
    fail_at: int | _Unrecorded | None  # the negative control -- see the module docstring
    binds: str
    mechanism: str
    evidence: str              # who measured it, when, on what. Never empty.
    msa_rows: int | None = None  # alignment depth the ladder was walked at, where it was varied
    counts: str = RESIDUES     # what `residues` counts -- see COUNTS
    # The ligand EVERY rung of this ladder carried, in heavy atoms, and the flag that says the
    # wall is on tokens. 0 is a real value and means "walked apo, and the wall is still on
    # tokens"; None means nobody established that, so the row is checked on residues alone. Set it
    # only from the row's own evidence: it is what converts residue numbers into token numbers,
    # and a guessed value would refuse work nothing ever measured to fail.
    ladder_ligand_atoms: int | None = None
    #: The SAME ceiling measured with block-fp8 weights, i.e. under ``--fast``. None means nobody
    #: walked that arm and the row above speaks for both, which is the right default: for every
    #: model whose chip is filled by ACTIVATIONS the weight dtype barely moves the wall.
    #:
    #: It moves it enormously where the WEIGHTS are what fills the chip. ESMC-6B in bf16 embeds
    #: 1968 residues on a Galaxy chip and is refused at 1984 on a 30965760 B request with
    #: 2525280 B free -- the wall is 31 MB of headroom behind ~12 GB of resident weights, not the
    #: sequence. Under ``--fast`` the same chip does 8192. One number cannot be right for both,
    #: and the one that shipped was refusing work the engine does.
    fast: "Ceiling | None" = None

    @property
    def measured(self) -> bool:
        return self.binds != UNMEASURED

    @property
    def token_bound(self) -> bool:
        """The wall is on TOKENS, so a ligand's heavy atoms count against it."""
        return self.ladder_ligand_atoms is not None

    @property
    def tokens(self) -> int | None:
        """The cap in tokens: the residue cap plus the ligand the ladder was walked with."""
        if self.ladder_ligand_atoms is None or self.residues is None:
            return None
        return self.residues + self.ladder_ligand_atoms


def _unmeasured(evidence: str, counts: str = RESIDUES) -> Ceiling:
    return Ceiling(None, None, None, UNMEASURED, UNKNOWN, evidence, counts=counts)


# Shared by every model that carries no ceiling of its own. Spelled once because the reason is one
# reason, and a reader who sees it eight times should see the same sentence eight times.
_INHERITS_DEMO_FENCE = (
    "no measured engine ceiling. The catalog's 1024 for this model is LIMITS['max_residues'], the "
    "platform's free-demo fence, which no ladder walked to a failure -- so tt-bio does not refuse "
    "on it. Walking a ladder here is outstanding work, not a missing guard."
)

# model id (as it appears in a CLI --model choice) -> arch (ttnn.get_arch_name()) -> Ceiling
#
# The Wormhole rows are copied from the ladders recorded in japanfold/catalog.py, each attributed to
# the task that walked it. Nothing here was measured by the change that introduced this file, and
# the evidence strings say so; a row re-measured later should say that instead.
CEILINGS: dict[str, dict[str, Ceiling]] = {
    "opendde": {
        "blackhole": Ceiling(
            residues=1024, pass_at=1024, fail_at=1536, binds=FREEZE, mechanism=TRUNK_FREEZE,
            evidence="the first Blackhole row in this file, measured via perf/bh1536/run_rung.py, and it exists because the failure it "
                     "guards costs a card. 1536 residues freezes: the trunk walks nine of its ten "
                     "recycles at a steady 91-93 s each and stops at `trunk 9/10` forever. "
                     "Measured on BOTH boards -- qb1 card 0 (p150a) 2026-09-10 and qb2 card 1 "
                     "(p300c) 2026-09-10, tasks bh-1536-structure and p300c-1536-structure -- and "
                     "on BOTH checkpoints, which share every tensor shape. What it is not, from "
                     "samples on the frozen process: not an OOM (no DRAM or L1 refusal anywhere "
                     "in either log), not idle (110-160 % CPU, ticks advancing), not the 0 %-CPU "
                     "wedge this repo already records, and not slow (RSS pinned to the byte and "
                     "the log unchanged for 25 minutes). It also leaves the chip refusing every "
                     "device open at risc_firmware_initializer.cpp:1115, so an unguarded attempt "
                     "costs the next job on that card too. Both freezes were measured "
                     "--single_sequence, i.e. in the roomiest configuration this engine has, so a "
                     "cap set from them cannot be over-refusing on alignment depth. The cap is "
                     "1024 because that is the largest size on record folding on this arch (the "
                     "p150a ladder in state/sizes-recheck-opendde.md, cited in this module's own "
                     "docstring); 1025-1535 is unmeasured and the convention caps below the FIRST "
                     "failure rather than at the largest passing size",
        ),
        "wormhole_b0": Ceiling(
            residues=1024, pass_at=1024, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            msa_rows=8192,
            evidence="its own ladder, re-measured 2026-09-08 on GWH02 at the platform's default "
                     "8192 alignment rows after the trimul in-projection re-probe fix: 128, 256, "
                     "512, 576, 640, 768, 896 and 1024 all fold, 0 backbone breaks on every rung. "
                     "1024 folds 3/3 in separate processes, byte-identical (CIF md5 3b3099f9, "
                     "~1553 s, clash_frac 0.01226, plDDT 0.774); 896 folds 2/2 byte-identical "
                     "(md5 5ee2bacb, ~1040 s). The previous row capped this at 544 because 576 "
                     "threw an L1 static-CB clash: the cause was a fused in-projection width DRAM "
                     "had already refused being re-probed once per pairformer block, which "
                     "ratcheted a fold to the narrowest channel chunk. Parity is bit-exact against "
                     "the merge base at 128/256/512 and the row cap that fixes the clash is "
                     "bit-exact at 512 with it forced off. See state/opendde-l1-clash-to-1024.md",
        ),
    },
    "opendde-abag": {
        "blackhole": Ceiling(
            residues=1024, pass_at=1024, fail_at=1536, binds=FREEZE, mechanism=TRUNK_FREEZE,
            evidence="its OWN 1536 rung (perf/bh1536/run_rung.py), not inherited from opendde by architecture "
                     "argument. 1536 residues freezes: the trunk walks nine of its ten "
                     "recycles at a steady 91-93 s each and stops at `trunk 9/10` forever. "
                     "Measured on BOTH boards -- qb1 card 0 (p150a) 2026-09-10 and qb2 card 1 "
                     "(p300c) 2026-09-10, tasks bh-1536-structure and p300c-1536-structure -- and "
                     "on BOTH checkpoints, which share every tensor shape. What it is not, from "
                     "samples on the frozen process: not an OOM (no DRAM or L1 refusal anywhere "
                     "in either log), not idle (110-160 % CPU, ticks advancing), not the 0 %-CPU "
                     "wedge this repo already records, and not slow (RSS pinned to the byte and "
                     "the log unchanged for 25 minutes). It also leaves the chip refusing every "
                     "device open at risc_firmware_initializer.cpp:1115, so an unguarded attempt "
                     "costs the next job on that card too. Both freezes were measured "
                     "--single_sequence, i.e. in the roomiest configuration this engine has, so a "
                     "cap set from them cannot be over-refusing on alignment depth. The cap is "
                     "1024 because that is the largest size on record folding on this arch (the "
                     "p150a ladder in state/sizes-recheck-opendde.md, cited in this module's own "
                     "docstring); 1025-1535 is unmeasured and the convention caps below the FIRST "
                     "failure rather than at the largest passing size",
        ),
        "wormhole_b0": Ceiling(
            residues=1024, pass_at=1024, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            msa_rows=8192,
            evidence="its OWN rungs at 8192 alignment rows, 2026-09-08, not inherited from "
                     "opendde by architecture argument: 1024 folds 3/3 in separate processes, "
                     "byte-identical (CIF md5 a27f0d67, 1545-1549 s, clash_frac 0.00971, plDDT "
                     "0.717) and 512 folds (md5 7cc87160, 296 s, clash_frac 0.00534). The two "
                     "checkpoints share the trunk, the refiner and every tensor shape and differ "
                     "only in weight values, which is why the same fix serves both -- but these "
                     "numbers are measured on this checkpoint. Same history as the opendde row: "
                     "the old 544 cap was the re-probed in-projection width, not a capacity wall",
        ),
    },
    "openfold3": {
        "wormhole_b0": Ceiling(
            residues=1024, pass_at=1024, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            msa_rows=14190,
            evidence="its own ladder, measured 2026-09-07 on GWH02 at 14190 alignment rows on "
                     "every rung -- the deepest real ColabFold alignment this pipeline has "
                     "produced. Monotone with NO failure found: 640/672/704/736/768/800/832/896/"
                     "960/1024 all fold, in 219/229/287/319/324/384/435/666/453/642 s. 1024 is "
                     "the platform's own max_residues fence, so there is nothing above it to walk "
                     "to. Every rung was scored for structure and not just for returning: 0 "
                     "backbone breaks everywhere, Ca-Ca median 3.849-3.863 A at 99.48-100 % in "
                     "band, Rg ratio 1.155-1.213, pLDDT 0.733-0.773 declining smoothly with size, "
                     "and clashes 0 except 2/6171 at 768, 2/7711 at 960 and 8/8231 at 1024 -- all "
                     "marginal contacts inside the budget real crystal structures show. The 576 "
                     "this replaces was measured 2026-08-17 against an engine whose "
                     "OuterProductMean materialised its whole z matmul and whose MSA track held "
                     "four redundant full-width copies of the representation; 614 dying at 2.01 GB "
                     "was that engine, and 614 buckets to 640, which folds. The wall clocks above "
                     "are the ws:ceiling-openfold3-1024 arm. The shipped engine takes the "
                     "refusal-narrowed route instead, and the top two rungs were re-walked on it "
                     "(ws:ceiling-1024-integration-and-gate): 960 folds in 768 s and 1024 in "
                     "963 s, each absorbing OuterProductMean's single-shot z refusal (1887436800 "
                     "and 2147483648 B) through the row block. Slower than the arm above and "
                     "structurally at least as good: 1024 scores PASS with ZERO clashes and 960 "
                     "WARNs at 2/7711 marginal contacts, both with the backbone intact (Ca-Ca "
                     "median 3.842/3.863 A, 99.61/99.48 % in band). Every size that folds on the "
                     "previous engine is bit-exact against it. A refusal narrows the row block, "
                     "which partitions independent rows and is bit-exact too; only "
                     "OuterProductMean's un-joined form reassociates a bf16 depth sum, and it ran "
                     "on NEITHER rung (join_split=0 at 960 and 1024), so nothing on this ladder "
                     "moved a bit. Single-sequence is roomier still and is not the default",
        ),
    },
    "openbind": {
        "wormhole_b0": Ceiling(
            residues=960, pass_at=960, fail_at=1024, binds=MEMORY, mechanism=DRAM_MSA,
            msa_rows=14190, ladder_ligand_atoms=35,
            evidence="its own ladder, and walked WITH A LIGAND BOUND rather than inherited from "
                     "openfold3 -- measured 2026-09-07 on GWH02 at 14190 alignment rows, "
                     "ws:ceiling-1024-integration-and-gate on tt-bio e9cb5b70. Every rung carries "
                     "CCD STU, 35 heavy atoms: 768 folds in 525 s, 896 in 747 s, 960 in 1080 s, "
                     "and 1024 FAILS. Scored for structure and not just for returning: 768 and "
                     "896 PASS with zero clashes, 960 WARNs at 3/7746 marginal contacts inside "
                     "the 0.1 % budget, and no rung breaks a backbone (Ca-Ca median 3.854-3.861 "
                     "A, 99.66-99.74 % in band). The wall is the TOKEN count, not the residue "
                     "count, and the same input apo proves it: 1024 residues with no ligand fold "
                     "in 863 s. 1024 residues plus 35 ligand atoms is 1059 tokens, which buckets "
                     "to 1088, and 1088 is refused twice over -- OuterProductMean's z at "
                     "2424307712 = 1088 x 1088 x 1024 x 2, and the MSA Transition's "
                     "host-assembled result uploaded whole at 1976016896 = 14189 x 1088 x 64 x 2. "
                     "So the wall this row records is on TOKENS, and ladder_ligand_atoms=35 says "
                     "which ligand every rung carried: the 960 cap is 995 tokens, which pad to "
                     "the 1024 measured to fold. At 960 residues it therefore holds for a ligand "
                     "of 64 atoms or fewer and refuses 65. A larger ligand at a residue count "
                     "this row admits used to cross the wall unseen and die on the chip; it is "
                     "refused at submission now. The 576/614 this replaces was "
                     "measured 2026-08-17 against an engine whose OuterProductMean materialised "
                     "its whole z matmul; openbind dedups its main MSA "
                     "(af3_spec_main_msa_dedup is keyed on the checkpoint), so the depth reaching "
                     "its model at a given alignment is not openfold3's and this ladder is its "
                     "own. Single-sequence is roomier still and is not the default",
        ),
    },
    "rf3": {
        "wormhole_b0": Ceiling(
            residues=1095, pass_at=1095, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            msa_rows=27317,
            evidence="was 627 (fail_at 630) until the two allocations that set it were fixed; "
                     "state/ceiling-rf3-1024.md, measured 2026-09-07 on GWH02 card UMD 2 with real "
                     "ColabFold alignments. Both walls were shape choices, not memory the model "
                     "needs. The template embedder and the MSA module were the last two RF3 "
                     "triangle-attention sites on the materialised fp32-softmax route, which writes "
                     "[tokens, heads, S_pad, S_pad] over the RAW token axis: 656 aa asked "
                     "2369912832 B = 656 x 4 x 672 x 672 x 2 at trunk 0/10. The confidence head's "
                     "global layer norm flattened the pair tensor to one row, and TILE_LAYOUT pads "
                     "one row to 32, so it asked 32x: 640 aa died on 3355443200 B = 32 x (640 x 640 "
                     "x 128 x 2), reproduced on this card at 179 s. Ladder above the old cap, fix "
                     "arm, all folded with zero backbone breaks and clash fractions of 0.2-0.7%: "
                     "630 (22936 rows), 656 (23951), 716 (11615), 796 (21448), 891 (16253), 980 "
                     "(12267), 1095 (25815, pLDDT 80.5 in 584 s). 640 aa carries the deepest "
                     "alignment walked, 27317 rows. Nothing above 1095 has been run, hence "
                     "LADDER_TOP: the real ceiling may be higher. The fix is on by default; "
                     "TT_BIO_RF3_TEMPLATE_FUSED_SDPA=0, TT_BIO_RF3_MSA_FUSED_SDPA=0 or "
                     "TT_BIO_RF3_GLN_ROW_FOLD=0 restore the old route and the old 627 wall with it",
        ),
    },
    "protenix-v2": {
        "wormhole_b0": Ceiling(
            residues=1024, pass_at=1024, fail_at=1095, binds=MEMORY, mechanism=DRAM,
            msa_rows=8832, ladder_ligand_atoms=0,
            evidence="1024 measured 2026-09-08 on GWH02 (8x9, 12 GiB/card), "
                     "ws:wh-transition-wchunk-hang-fix-p2, AT PRODUCTION MSA DEPTH: "
                     "capacity_gate.py --tokens 1024 folds screen and full residency at 8832 "
                     "unique alignment rows, 730.5 s, DRAM peak 5.79 GiB = 48% of the card "
                     "(perf/capacity/wh_1024_protenix-v2_msa8832.json). Depth is the load-bearing "
                     "part: the failing tensor here scales with tokens x rows, so the "
                     "single-sequence 1024 fold measured the same day (476.8 s, pLDDT 0.745, "
                     "digest 2226d5a9adfa135810f3f2df9a4a3460, bit-identical to the same rung "
                     "pre-merge on the fix branch) would not have been evidence for this row on "
                     "its own. Both needed 938989d7: under the shipped W-chunk gate this band did "
                     "not OOM, it hung the chip, which is why the row called 1024 UNTESTED for a "
                     "month. 1095 still OOMs on DRAM (catalog.py, 2026-08-11, tree d0ff69b2). "
                     "The wall is on TOKENS and not on the polymer: the failing tensor scales "
                     "with tokens x rows, and the ladder was walked with capacity_gate.py "
                     "--tokens. ladder_ligand_atoms=0 records that those rungs were apo, so 1024 "
                     "residues is 1024 tokens and a ligand's heavy atoms are counted against the "
                     "same 1024 instead of being invisible to the residue count",
        ),
    },
    "rfd3": {
        "wormhole_b0": Ceiling(
            residues=1024, pass_at=1024, fail_at=None, binds=LADDER_TOP,
            mechanism=NO_FAILURE, counts=DESIGN_TOTAL,
            evidence="state/rfd3-swiglu-dram-resident-768-wh-verify.md, its own ladder measured 2026-09-08 on GWH02 "
                     "cards UMD 3 and UMD 0, ws:rfd3-swiglu-dram-resident-768-wh-verify. 640/704/768/832/896/960/1024 "
                     "total residues (target + a 100-residue binder) all design, TWICE each, once per card, the two "
                     "walks run in opposite directions so no rung inherits the one before it. ONE target (laczc_1008, "
                     "1DP0 chain A) cut to every rung, through the shipped `tt-bio design --model rfd3 --from_pdb` "
                     "CLI at the platform's 100 diffusion steps, so a rung passes only when a real CIF lands: "
                     "4798/5309/5814/6322/6869/7353/7855 heavy atoms in 140.5/150.4/182.1/201.3/221.3/253.0/285.6 s, "
                     "host RSS 9.7 GB at 640 rising to 21.4 GB at 1024. Every rung is BYTE-IDENTICAL between the two "
                     "cards, 1024 included. This replaces 704, which was the largest size below a 768 that failed 1 "
                     "run in 3. That failure was an L1 residency the pair transition's SwiGLU asked for against a "
                     "138000000 B constant -- 1.37x the 100470528 B a Wormhole part actually has, so it declined "
                     "nothing at any size and 768 asked for two 700416 B-per-bank buffers out of a 1395424 B bank, "
                     "5408 B short, which is why the boundary flipped run to run instead of failing cleanly. The "
                     "budget reads the part now, so the residency is declined by shape from 768 up: 398 declines "
                     "against 398 grants at every rung from 768 to 1024 (z_transition at hidden=512 declines, the "
                     "narrower Pairformer transitions at hidden=256 do not). BIT-EXACT ON THIS PART and not only in "
                     "simulation: 768 with TT_BIO_L1_RESIDENT_BUDGET_BYTES=0 -- every residency granted, the path the "
                     "engine shipped with -- folds 3/3 on UMD 0 and returns the same CIF md5 615c2d57 as the declined "
                     "arm. At 1024 the budget grants a hidden=256 residency the allocator then refuses; "
                     "`_swiglu_resident` drops the residency rather than the fold and the run returns md5 69c66e47, "
                     "identical to a run where no refusal happens at all. Structure was SCORED, not assumed: 0 "
                     "backbone breaks and worst Ca-Ca 3.92-3.95 A at 640, 704, 896, 960 and 1024, clash_frac "
                     "0.016-0.051 against 0.00098 for the deposited 9SAT through the same instrument. 768 and 832 "
                     "each carry a break inside the DESIGNED binder (28.99 A at 669->670, 37.63 A at 733->734). That "
                     "is not this ceiling and not this lever -- the same 768 CIF comes back byte for byte with the "
                     "residency granted -- and it is recorded here rather than left out. LADDER_TOP because 1024 is "
                     "the platform's own max_residues fence, so there is nothing above it to walk to and the real "
                     "ceiling may be higher. The cap still DEPENDS on the atom-pair row block: unblocked, the token "
                     "initializer dies at 640 total residues on a 2114887680 B DRAM request, and "
                     "TT_BIO_ATOM_PAIR_BUDGET_BYTES=0 restores that wall.",
        ),
    },
    "pxdesign": {
        "blackhole": Ceiling(
            residues=2500, pass_at=2500, fail_at=3000, binds=MEMORY, mechanism=DRAM,
            counts=DESIGN_TARGET,
            evidence=
                "its own Blackhole ladder, walked 2026-09-10 on qb1 p150a (task "
                "bh-1536-design-embed-p2, perf/bhdesign/ladder.py): 1536, 1831 and 2500 "
                "conditioned target residues each "
                "design an 80-residue, 321-atom binder in 97.8, 177.1 and 208.2 s, and 3000 "
                "throws on DRAM at 199.6 s. The target above 1008 residues is "
                "perf/bhdesign/targets/big_7324.cif, real deposited chains placed side by side, "
                "with the crop spilling across chains in file order. The throw is NOT the "
                "single-oversized-tensor class the embedding rows carry: Not enough space to "
                "allocate 9789767680 B DRAM buffer across 8 banks, where each bank needs to store "
                "1223720960 B, but bank size is 4278190016 B (allocated: 3152167424 B, free: "
                "1126022592 B, largest free block: 890649024 B). The request would fit an empty "
                "card three times over; what fails is that 3.15 GB per bank is already resident, "
                "leaving 1.13 GB against a 1.22 GB ask. Free and largest-free-block differ by 235 "
                "MB, so there is some fragmentation on top, but the first-order cause is "
                "cumulative residency and the lever is what stays live, not the layout. "
                "Wormhole's 768 is a LADDER TOP from a different chip and does not bound this",
        ),
        "wormhole_b0": Ceiling(
            residues=768, pass_at=768, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            counts=DESIGN_TARGET,
            evidence="catalog.py, measured 2026-08-29 on the serving pool (engine a189fdbb) with "
                     "the ladder in perf/pxdesign/targets -- 1DP0 chain A cut around one epitope, "
                     "80-residue binder, n_step 400, one design. EVERY rung ran: 128 aa in 62.0 s, "
                     "256 in 50.8 s, 512 in 70.7 s, 768 in 99.9 s. No crash-class failure anywhere, "
                     "so 768 is the largest size PROVEN and not the rung below a first failure. "
                     "Nobody has run 1024, and 1DP0 chain A is 1011 residues so this ladder's own "
                     "fixture source cannot reach it",
        ),
    },
    "esmc-6b": {
        "wormhole_b0": Ceiling(
            residues=1968, pass_at=1968, fail_at=1984, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence="catalog.py, measured 2026-08-11 (tree 7b6ab185, live pool): 1968 aa embeds in "
                     "32 s, 1984 aa OOMs on DRAM. The 6B weights nearly fill the chip, so past the "
                     "ceiling the activation allocation has nowhere to go. This is an embed model: "
                     "the binding constraint is DRAM and not wall-clock, which is why its number is "
                     "so much higher than any folding model's. RE-MEASURED 2026-09-11 on a QUIET "
                     "j10glx02 Galaxy chip 5 by ws:wh-seqlen-design-embed and it reproduces exactly: "
                     "512 aa 45.0 s, 1968 aa 45.0 s, 1984 aa throws in 30.9 s, 4096 aa throws in "
                     "26.0 s. That matters because the original was taken on the shared serving "
                     "pool, where a co-tenant's residency could have set the number; it did not. "
                     "The allocator says why in one line -- 1984 dies on a 30965760 B request with "
                     "'bank size is 1073741792 B (allocated: 1068605312 B, free: 5136480 B, largest "
                     "free block: 2525280 B)'. 31 MB of headroom behind ~12 GB of resident weights, "
                     "so the wall is the WEIGHTS and barely the sequence at all, which is exactly "
                     "why this row needed a `fast` sibling",
            fast=Ceiling(
                residues=8192, pass_at=8192, fail_at=None, binds=LADDER_TOP,
                mechanism=NO_FAILURE, counts=MAX_SEQUENCE,
                evidence="its own ladder in the OTHER weight dtype, walked 2026-09-11 on j10glx02 "
                         "Galaxy chip 6 by ws:wh-seqlen-design-embed through the shipped `tt-bio "
                         "embed --model esmc-6b --fast` CLI: 1968 aa in 110.1 s, 4096 in 130.1 s, "
                         "8192 in 180.2 s, every one to [L, 2560], finite, nonzero_frac 1.0. The "
                         "bf16 arm on chip 5 the same day cannot do 4096 at all. Block-fp8 roughly "
                         "halves the resident weights and this model is weight-bound, so the "
                         "ceiling moves by at least 4.16x -- which is the whole reason this field "
                         "exists: the 1968 above was being enforced against a configuration it "
                         "never measured, refusing a 4096 aa --fast job that runs in 130 s. "
                         "LADDER_TOP and not MEMORY because nothing above 8192 has failed: 16384 "
                         "and up are the open rungs, so the real fast ceiling may be higher and "
                         "this cap is the largest size PROVEN rather than the rung below a "
                         "failure. Wall-clock is not free here either -- fp8 is SLOWER per call at "
                         "equal length (110.1 s against bf16's 45.0 s at 1968 aa), so --fast on "
                         "this model buys capacity and costs latency, which is the opposite of "
                         "what its name suggests and is worth knowing before reaching for it",
            ),
        ),
    },
    # --- No measured ceiling. Never refused. ---------------------------------------------------
    "boltz2": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE)},
    "esmfold2": {
        "wormhole_b0": Ceiling(
            residues=1024, pass_at=1024, fail_at=1057, binds=MEMORY, mechanism=DRAM,
            msa_rows=0, ladder_ligand_atoms=0,
            evidence="its own ladder, walked 2026-09-09 on GWH02 card 1 by "
                     "ws:esmfold2-cocrystal-everywhere at the settings a user gets: "
                     "single-sequence, which is this model's default, and --fast, which "
                     "tt_bio/main.py forces on Wormhole because ESMC-6B needs ~12.8 GB against "
                     "a ~12 GB chip. 895 folds in 213 s, 960 in 273 s, 1024 in 287 s; 1057 does "
                     "not fold. The failure is one DRAM allocation the chip cannot serve, and it "
                     "is refused on BANK size rather than total memory: 588808192 B across 12 "
                     "banks wants 49068032 B per bank against a 43118592 B largest free block. "
                     "It is a TOKEN wall, not a polymer one -- a cocrystal at 1024 tokens (991 "
                     "residues plus a 33-atom ligand) folds in 278 s and one at 1057 tokens is "
                     "refused with the identical message, so a ligand costs nothing beyond the "
                     "tokens it adds. This model read UNMEASURED until now because its published "
                     "ladder (docs/size-generality.md, 2026-09-08) stopped at 1024, the "
                     "platform's demo fence, and so never recorded a failing size above the cap: "
                     "the ceiling was there, the negative control was not. The wall is recorded "
                     "in TOKENS on the strength of that cocrystal pair, and ladder_ligand_atoms=0 "
                     "says the rungs above were walked apo, so the cap is 1024 tokens: 991 "
                     "residues + 33 atoms is admitted because it is the 1024 that folded, and "
                     "1024 residues plus any ligand at all is refused. It used to be admitted "
                     "here and fail on the chip",
        ),
    },
    "esmfold2-fast": {
        "wormhole_b0": Ceiling(
            residues=1152, pass_at=1152, fail_at=1280, binds=MEMORY, mechanism=DRAM,
            msa_rows=0, ladder_ligand_atoms=0,
            evidence="its OWN ladder, walked 2026-09-09 on GWH02 card 1 by "
                     "ws:esmfold2-cocrystal-everywhere, not inherited from esmfold2 by "
                     "architecture argument -- and it had to be walked separately, because the "
                     "two checkpoints do NOT share a ceiling. 960 folds in 202 s, 1024 in 214 s, "
                     "1057 in 261 s, 1152 in 278 s; 1280 does not fold (838860800 B DRAM buffer "
                     "across 12 banks, 69906432 B wanted per bank). 1057 is the rung that stops "
                     "the full trunk and this checkpoint clears it, which is the expected "
                     "direction: same architecture at half the trunk depth (24 blocks against "
                     "48), so a smaller DRAM peak. Same settings as the esmfold2 row -- "
                     "single-sequence (this checkpoint has no MSA encoder at all) and --fast, "
                     "forced on Wormhole. The wall is on TOKENS for the same reason esmfold2's "
                     "is -- same trunk, same allocation, same per-atom ligand tokenisation -- and "
                     "it is closed the same way rather than left as a caveat: "
                     "ladder_ligand_atoms=0 records apo rungs, so the cap is 1152 tokens and a "
                     "ligand is counted against it. The NUMBERS are still this checkpoint's own; "
                     "only the denominator is shared",
        ),
    },
    "protenix-v1": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE)},
    "nesso1": {
        "wormhole_b0": _unmeasured(
            "no ceiling, and unusually this is a positive result rather than an untested gap: the "
            "range is measured on GWH02 from 128 aa (28.3 s) through 1024 aa (72.0 s), and 1152 "
            "still scores (catalog.py). Nothing in that ladder failed, so there is nothing to "
            "refuse on"),
    },
    "boltzgen": {
        "blackhole": _unmeasured(
            "no residue-denominated ceiling on Blackhole either, and this row is a RECORD rather "
            "than a gap: on qb1 p150a a 2100-residue / 16948-atom target designs an 80-residue "
            "binder in 805.3 s (task bh-boltzgen-sdpa-circbuf-p3, 2026-09-10), which is 3.6x the "
            "top of the Wormhole 3158-4651 atom band and 2.1x the 8095 atoms the first Blackhole "
            "pass could reach with the largest single chain on hand; 1831 residues / 14786 atoms "
            "is 692.0 s. Above 16948 the ladder has to be re-walked before anything is claimed: "
            "the rungs above it were read while the trimul in-projection wedged the card on a "
            "sub-tile slice above 2048 padded tokens (_TRIMUL_MIN_CHUNK), so a rung that 'ran out "
            "its 3000 s budget with no allocator throw' was a spinning card and bounds nothing. "
            "The axis is atom-denominated on top of that -- COUNTS carries no atom unit, so "
            "writing it as residues would be a unit substitution",),
        "wormhole_b0": _unmeasured(
            "its measured cap is NOT residue-denominated and so cannot be expressed as a row here: "
            "wh-design-models-l1-budget-and-size-caps puts it between 3158 and 4651 ATOMS, in the "
            "trunk Pairformer's triangle attention, at shipped settings -- the same 8x9 per-core L1 "
            "wall that killed that task's chunk-width lever. Atoms per residue vary with "
            "composition, so converting that to residues would be a guess. Refusing on it needs an "
            "atom-denominated dimension this table does not yet carry"),
    },
    "esmc-300m": {
        "wormhole_b0": Ceiling(
            residues=69632, pass_at=69632, fail_at=73728, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Wormhole ladder, walked 2026-09-11 on j10glx02 Galaxy chips 5, 7 and 8 by "
                "ws:wh-seqlen-design-embed (perf/bhdesign/ladder.py, one rung per subprocess "
                "through the shipped CLI, verdict off the .npz). 65537 (160.2 s) and 69632 "
                "(170.2 s) embed to [L, 960], finite, nonzero_frac 1.0; 73728 throws in 51.7 s on "
                "424857600 B with a 24740448 B largest free block. 49153 and 65537 are NOT "
                "multiples of 32 and their npz files carry exactly that many rows, so the token "
                "axis's pad tail is masked at lengths that exercise it. THE WALL IS NOT THE SINGLE "
                "padded_L^2 x 2 BUFFER the Blackhole row below blames: that buffer lands, and a "
                "later, smaller request dies with the chip full. On a p150a, with 2.65x the memory "
                "(8 banks x 4278190016 B against 12 x 1073741792 B), the L^2 buffer IS the wall "
                "and everything after it fits, which is why one Blackhole ladder could read one "
                "shape off five models and call it the cause. THIS NUMBER IS 1.13x WHAT THE ENGINE "
                "HELD THIS MORNING, and the difference is a fix rather than a re-measurement: "
                "before it, 61440 was the top rung and 65537 died in 25.0 s, because loading this "
                "model reserved 268435456 B PER BANK for a ttnn trace region -- 3 GiB of a 12.8 GB "
                "chip -- that a single long sequence can never replay, since the forward captures "
                "only on the SECOND sighting of a bucketed shape. `esmc.trace_pays` now asks that "
                "question before the device opens. Proven as a ONE-CHIP A/B rather than across "
                "chips, because the first failure and the first pass landed on different cards and "
                "that is a perfectly good alternative explanation: on card 7, same day, the "
                "pre-fix tree (a detached worktree at 389adde1) fails 65537 in 44.2 s on the same "
                "377671680 B request, and the post-fix tree passes it in 160.2 s",
        ),
        "blackhole": Ceiling(
            residues=114688, pass_at=114688, fail_at=126976, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Blackhole ladder, walked 2026-09-10 on qb1 p150a (task "
                "bh-1536-design-embed-p2, perf/bhdesign/ladder.py: one rung per subprocess "
                "through the shipped CLI, verdict read off the .npz and not off the exit "
                "code). 99999 (157.0 s) and 114688 (179.5 s) residues all embed to [L, 960], "
                "finite, nonzero_frac 1.0; 126976 throws in 52.4 s on the 32262064128 B buffer "
                "itself, with 3924465472 B free in the largest bank. The failing allocation is "
                "the same on all five models in this family and that is the finding: "
                "padded_L^2 x 2, the full L x L attention matrix in bf16 at the token length "
                "rounded up to a multiple of 32, as ONE DRAM buffer spread over 8 banks. At "
                "131072 it asks for 34376517632 B (131104^2 x 2), which needs 4297066496 B per "
                "bank against a 4278190016 B bank, so it does not fit an EMPTY card. At 126976 "
                "it asks for 32262064128 B (127008^2 x 2), 4032758016 B per bank, which an "
                "empty bank WOULD hold -- so that rung is decided by what else is resident, "
                "which is why the five caps differ although the shape does not. The first pass "
                "stopped this model at 65536 with nothing failing, and every sibling was then "
                "held to that as the bar. The Wormhole row above is UNMEASURED and this does "
                "not fill it in: nobody walked this ladder on a Galaxy chip.",
        ),
    },
    "esmc-600m": {
        "wormhole_b0": Ceiling(
            residues=69632, pass_at=69632, fail_at=73728, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Wormhole ladder, walked 2026-09-11 on j10glx02 Galaxy chip 9 by "
                "ws:wh-seqlen-design-embed (perf/bhdesign/ladder.py, one rung per subprocess "
                "through the shipped CLI, verdict off the .npz). 61440 (190.2 s), 65537 (230.4 s) "
                "and 69632 (220.3 s) embed to [L, 1152], finite, nonzero_frac 1.0; 73728 throws in "
                "47.3 s on 509829120 B with a 39219552 B largest free block. 65537 is not a "
                "multiple of 32 and its npz carries exactly 65537 rows. Same mechanism as "
                "esmc-300m and the same cause behind the same fix: the pre-fix engine capped this "
                "model at 57344 and failed 61440 in 25.0 s, because loading it reserved "
                "268435456 B PER BANK for a trace region a single long sequence can never replay "
                "(see esmc.trace_pays); this number is 1.21x that. THE CAP BEING EQUAL TO "
                "esmc-300m's IS A RESOLUTION ARTEFACT, NOT A CLAIM THAT THEY SHARE A WALL: this "
                "ladder steps in 4096s, and one step near the top adds "
                "(73728^2 - 69632^2) x 2 / 12 = 97 MB per bank, while the two models' resident "
                "weights differ by only ~75 MB per bank (1.24 GB against 2.14 GB over 12 banks). "
                "So the step is coarser than the difference, and a finer walk would be expected to "
                "separate them. Not walked finer because the rungs cost 3-4 minutes each and a "
                "1024-token refinement would not change what any user can ask for",
        ),
        "blackhole": Ceiling(
            residues=114688, pass_at=114688, fail_at=126976, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Blackhole ladder, walked 2026-09-10 on qb1 p150a (task "
                "bh-1536-design-embed-p2, perf/bhdesign/ladder.py: one rung per subprocess "
                "through the shipped CLI, verdict read off the .npz and not off the exit "
                "code). 65536 (110.8 s), 99999 (186.7 s) and 114688 (220.8 s) residues all "
                "embed to [L, 1152], finite, nonzero_frac 1.0; 126976 throws in 55.2 s on the "
                "32262064128 B buffer itself, with 3863293760 B free in the largest bank. The "
                "failing allocation is the same on all five models in this family and that is "
                "the finding: padded_L^2 x 2, the full L x L attention matrix in bf16 at the "
                "token length rounded up to a multiple of 32, as ONE DRAM buffer spread over 8 "
                "banks. At 131072 it asks for 34376517632 B (131104^2 x 2), which needs "
                "4297066496 B per bank against a 4278190016 B bank, so it does not fit an "
                "EMPTY card. At 126976 it asks for 32262064128 B (127008^2 x 2), 4032758016 B "
                "per bank, which an empty bank WOULD hold -- so that rung is decided by what "
                "else is resident, which is why the five caps differ although the shape does "
                "not.  The Wormhole row above is UNMEASURED and this does not fill it in: "
                "nobody walked this ladder on a Galaxy chip.",
        ),
    },
    "saprot-35m": {
        "wormhole_b0": Ceiling(
            residues=73728, pass_at=73728, fail_at=77824, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Wormhole ladder, walked 2026-09-11 on the j10glx02 Galaxy chip 7 by "
                "ws:wh-seqlen-design-embed with the same harness the Blackhole row below used "
                "(perf/bhdesign/ladder.py, one rung per subprocess through the shipped CLI, "
                "verdict read off the .npz). 65537 (90.1 s) and 73728 (100.1 s) residues embed to "
                "[L, 480], finite, nonzero_frac 1.0; 77824 throws in 35.0 s. 65537 is deliberately "
                "NOT a multiple of 32 and its npz carries exactly 65537 rows, so the pad tail the "
                "token axis adds is masked at a length that exercises it. THE WALL IS NOT THE "
                "BUFFER THE BLACKHOLE ROW BLAMES, and that is the finding: the padded_L^2 x 2 "
                "attention matrix lands, and what fails is a LATER 298967040 B request with the "
                "chip already full -- 'each bank needs to store 24913920 B, but bank size is "
                "1073741792 B (allocated: 1055840384 B, free: 17901408 B, largest free block: "
                "11671392 B)'. On a p150a, with 2.65x the memory, that same L^2 buffer IS the wall "
                "and everything after it fits; on a 12.76 GB Galaxy chip the ceiling is cumulative "
                "residency. So a chunked-attention fix would buy Blackhole a lot and Wormhole "
                "comparatively little. This model reserves no trace region, so its number is "
                "unaffected by the 3 GiB reservation that moved esmc-300m's",
        ),
        "blackhole": Ceiling(
            residues=126976, pass_at=126976, fail_at=131072, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Blackhole ladder, walked 2026-09-10 on qb1 p150a (task "
                "bh-1536-design-embed-p2, perf/bhdesign/ladder.py: one rung per subprocess "
                "through the shipped CLI, verdict read off the .npz and not off the exit "
                "code). 65536 (78.2 s), 99999 (160.6 s), 114688 (149.6 s) and 126976 (176.1 s) "
                "residues all embed to [L, 480], finite, nonzero_frac 1.0; 131072 throws in "
                "50.1 s on the 34376517632 B buffer, the one that does not fit an empty card. "
                "The failing allocation is the same on all five models in this family and that "
                "is the finding: padded_L^2 x 2, the full L x L attention matrix in bf16 at "
                "the token length rounded up to a multiple of 32, as ONE DRAM buffer spread "
                "over 8 banks. At 131072 it asks for 34376517632 B (131104^2 x 2), which needs "
                "4297066496 B per bank against a 4278190016 B bank, so it does not fit an "
                "EMPTY card. At 126976 it asks for 32262064128 B (127008^2 x 2), 4032758016 B "
                "per bank, which an empty bank WOULD hold -- so that rung is decided by what "
                "else is resident, which is why the five caps differ although the shape does "
                "not. This is the only model in the family that clears 126976, and it carries "
                "the smallest weights, which is the whole reason the caps differ. The Wormhole "
                "row above is UNMEASURED and this does not fill it in: nobody walked this "
                "ladder on a Galaxy chip.",
        ),
    },
    "saprot-650m": {
        "wormhole_b0": Ceiling(
            residues=65537, pass_at=65537, fail_at=73728, binds=MEMORY,
            mechanism=DRAM, counts=MAX_SEQUENCE,
            evidence="its own Wormhole ladder, walked 2026-09-11 on j10glx02 Galaxy chip 5 (rungs to 57344) and chip 7 (65537, 73728) by ws:wh-seqlen-design-embed, same harness as the Blackhole row below (perf/bhdesign/ladder.py, one rung per subprocess through the shipped CLI, verdict off the .npz). 32768 (65.1 s), 40960 (100.1 s), 49153 (145.2 s), 57344 (210.3 s) and 65537 (210.3 s) embed to [L, 1280], finite, nonzero_frac 0.9999; 73728 throws in 47.3 s. 49153 and 65537 are NOT multiples of 32 and their npz files carry exactly that many rows, so the token axis's pad tail is masked at lengths that exercise it. THE WALL IS NOT THE SINGLE padded_L^2 x 2 BUFFER THE BLACKHOLE ROW BLAMES: that buffer lands, and what fails is a LATER request with the chip already full. On a p150a, with 2.65x the memory (8 banks x 4278190016 B against 12 x 1073741792 B), the L^2 buffer IS the wall and everything after it fits; on a Galaxy chip the ceiling is cumulative residency. A chunked-attention fix therefore buys Blackhole a lot and Wormhole comparatively little. The failing line: 566476800 B wanted, 'each bank needs to store 47206400 B, but bank size is 1073741792 B (allocated: 1052934272 B, free: 20807520 B, largest free block: 20807520 B)' -- free and largest-free-block are EQUAL, so there is nothing to coalesce and this is residency rather than fragmentation. Reserves no trace region, so unaffected by the 3 GiB reservation that moved esmc-300m",
        ),
        "blackhole": Ceiling(
            residues=114688, pass_at=114688, fail_at=126976, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Blackhole ladder, walked 2026-09-10 on qb1 p150a (task "
                "bh-1536-design-embed-p2, perf/bhdesign/ladder.py: one rung per subprocess "
                "through the shipped CLI, verdict read off the .npz and not off the exit "
                "code). 65536 (122.3 s), 99999 (219.7 s) and 114688 (236.0 s) residues all "
                "embed to [L, 1280], finite, nonzero_frac 0.9999; 126976 throws in 50.9 s and "
                "does it one allocation LATER than its siblings: the 32262064128 B buffer "
                "lands, and a following 325140480 B request dies with 4247369856 B per bank "
                "already allocated and 30820160 B free. Same wall, caught by the next "
                "allocation instead of by that one. The failing allocation is the same on all "
                "five models in this family and that is the finding: padded_L^2 x 2, the full "
                "L x L attention matrix in bf16 at the token length rounded up to a multiple "
                "of 32, as ONE DRAM buffer spread over 8 banks. At 131072 it asks for "
                "34376517632 B (131104^2 x 2), which needs 4297066496 B per bank against a "
                "4278190016 B bank, so it does not fit an EMPTY card. At 126976 it asks for "
                "32262064128 B (127008^2 x 2), 4032758016 B per bank, which an empty bank "
                "WOULD hold -- so that rung is decided by what else is resident, which is why "
                "the five caps differ although the shape does not.  The Wormhole row above is "
                "UNMEASURED and this does not fill it in: nobody walked this ladder on a "
                "Galaxy chip.",
        ),
    },
    "saprot-1.3b": {
        "wormhole_b0": Ceiling(
            residues=57344, pass_at=57344, fail_at=65537, binds=MEMORY,
            mechanism=DRAM, counts=MAX_SEQUENCE,
            evidence="its own Wormhole ladder, walked 2026-09-11 on j10glx02 Galaxy chips 7 and 9 by ws:wh-seqlen-design-embed. 24576 (90.1 s), 32768 (115.1 s), 40961 (160.2 s), 49152 (210.2 s) and 57344 (230.3 s) embed to [L, 1280], finite, nonzero_frac 0.9999; 65537 throws in 52.5 s on 671416320 B, 'each bank needs to store 55951360 B, but bank size is 1073741792 B (allocated: 1026390144 B, free: 47351648 B, largest free block: 33363808 B)'. 40961 is not a multiple of 32 and comes back with exactly 40961 rows. THE WALL IS NOT THE SINGLE padded_L^2 x 2 BUFFER THE BLACKHOLE ROW BLAMES: that buffer lands, and what fails is a LATER request with the chip already full. On a p150a, with 2.65x the memory (8 banks x 4278190016 B against 12 x 1073741792 B), the L^2 buffer IS the wall and everything after it fits; on a Galaxy chip the ceiling is cumulative residency. A chunked-attention fix therefore buys Blackhole a lot and Wormhole comparatively little. The lowest Wormhole cap of the five, and the largest weights of the five, which is the same ordering the Blackhole ladder found and the reason the caps differ although the shape does not. Its Blackhole row notes it was one of the four models scripts/capacity_gate.EXEMPT skipped; on Wormhole it is measured now too",
        ),
        "blackhole": Ceiling(
            residues=114688, pass_at=114688, fail_at=126976, binds=MEMORY, mechanism=DRAM,
            counts=MAX_SEQUENCE,
            evidence=
                "its own Blackhole ladder, walked 2026-09-10 on qb1 p150a (task "
                "bh-1536-design-embed-p2, perf/bhdesign/ladder.py: one rung per subprocess "
                "through the shipped CLI, verdict read off the .npz and not off the exit "
                "code). 65536 (146.4 s), 99999 (278.6 s) and 114688 (326.7 s) residues all "
                "embed to [L, 1280], finite, nonzero_frac 0.9999; 126976 throws in 53.5 s on "
                "the 32262064128 B buffer itself, with 3943739200 B free in the largest bank. "
                "The failing allocation is the same on all five models in this family and that "
                "is the finding: padded_L^2 x 2, the full L x L attention matrix in bf16 at "
                "the token length rounded up to a multiple of 32, as ONE DRAM buffer spread "
                "over 8 banks. At 131072 it asks for 34376517632 B (131104^2 x 2), which needs "
                "4297066496 B per bank against a 4278190016 B bank, so it does not fit an "
                "EMPTY card. At 126976 it asks for 32262064128 B (127008^2 x 2), 4032758016 B "
                "per bank, which an empty bank WOULD hold -- so that rung is decided by what "
                "else is resident, which is why the five caps differ although the shape does "
                "not. One of the four models scripts/capacity_gate.EXEMPT skips, now measured "
                "rather than exempted. The Wormhole row above is UNMEASURED and this does not "
                "fill it in: nobody walked this ladder on a Galaxy chip.",
        ),
    },
}

_NO_ROW = _unmeasured("no row for this architecture: nothing has been measured on it")


class SizeTooLargeError(ValueError):
    """The input exceeds a MEASURED ceiling for the model it was sent to."""


# ---------------------------------------------------------------------------------------------
# The mechanism. One lookup, one check, one scanner.
# ---------------------------------------------------------------------------------------------


def enforced(default: bool = True) -> bool:
    """The one escape hatch, ``TT_BIO_SIZE_LIMIT=0``: refusals become warnings.

    It exists because every ceiling here was measured in ONE configuration, and a user can be in a
    roomier one. The clearest case is on the record: OpenFold3's 576 was measured with the MSA on,
    and the same model folds 768 single-sequence in 301 s. Refusing that would be a false refusal,
    which is the worst thing a guard can do -- it stops work the machine can actually finish, and
    unlike a crash the user cannot even retry past it.

    So the default protects and the hatch is named IN the refusal message, which is what makes it
    usable: a limit you can not get past is a bug report, a limit that tells you how to override it
    is a safety rail. An env var rather than a flag because the guard sits on five commands and this
    is an escape hatch, not an operating knob.
    """
    from .envflags import env_flag
    return env_flag("TT_BIO_SIZE_LIMIT", default)


def current_arch() -> str | None:
    """``ttnn.get_arch_name()``, or None if ttnn is not importable or names no card.

    Cheap and opens NO device, which is the whole point: the refusal has to land before anything
    takes a chip. A host with no Tenstorrent card gets None and every ceiling goes dormant -- a CPU
    run has different limits and none of them are these.
    """
    try:
        from .tenstorrent import arch_name
        return arch_name() or None
    except Exception:
        return None


def ceiling(model: str, arch: str | None = None, *, fast: bool = False) -> Ceiling:
    """The ceiling for `model` on `arch`. Unknown model or unknown arch -> UNMEASURED, never a cap.

    Defaulting an unknown pair to "no limit" rather than to some other model's number is deliberate:
    every way of guessing is a way of refusing a fold the engine can do.

    `fast` selects the block-fp8 arm where the row carries one. A row without a `fast` sibling
    answers the same for both, so passing the flag can only ever raise the cap and never lower it
    -- which is what makes threading it through safe for every model that has not been walked in
    both dtypes.
    """
    arch = arch if arch is not None else current_arch()
    if arch is None:
        return _NO_ROW
    c = CEILINGS.get(model, {}).get(arch, _NO_ROW)
    return c.fast if (fast and c.fast is not None) else c


def padded_tokens(model: str, tokens: int) -> int:
    """`tokens` as the width the chip actually allocates for `model`.

    Straight through ``token_axis``'s own bucket rather than a second ceil-to-32 written here: the
    guard has to pad by the SAME rule the model does, and a drift between them would be silent and
    in the permissive direction. It is also the width the measurements are written in -- openbind's
    row records its failing allocation as 1088 x 1088 x 1024 x 2, and 1088 is 1059 tokens bucketed.
    """
    from .token_axis import bucket_multiple, bucketed_width
    return bucketed_width(tokens, bucket_multiple(model))


def _verdict(model: str, c: Ceiling, residues: int, ligand_atoms: int):
    """``(refused, tokens, padded, wall)`` for one input against one ceiling. THE predicate.

    One copy of it, because ``check`` and ``models_accepting`` have to agree: a refusal that sent
    the user to a model this same rule would also refuse is worse than no hint at all.

    The token arm only ever ADDS a refusal. A ligand-free input never reaches it, so every
    residue-denominated ceiling behaves exactly as it did before the arm existed.
    """
    if not c.measured or c.residues is None:
        return False, None, None, None
    if c.token_bound and ligand_atoms > 0:
        tokens = residues + ligand_atoms
        padded, wall = padded_tokens(model, tokens), padded_tokens(model, c.tokens)
        if padded > wall:
            return True, tokens, padded, wall
    return residues > c.residues, None, None, None


def models_accepting(residues: int, arch: str | None = None, exclude: str | None = None,
                     ligand_atoms: int = 0, *, fast: bool = False) -> list[str]:
    """Models with a MEASURED ceiling that admits this input, so a refusal can point somewhere
    instead of only saying no.

    Only measured rows, and that keeps the list a promise we can keep: a model whose ladder nobody
    walked might well fold the input, but sending someone to it is advertising an untested size.
    A ligand-bearing input narrows it further, from ``capabilities.CAPABILITY`` rather than a
    second list here: OpenFold3 has plenty of room at these sizes and refuses a ligand by name, so
    naming it would send a cocrystal to a model that cannot take one.
    """
    arch = arch if arch is not None else current_arch()
    if ligand_atoms > 0:
        from .capabilities import CAPABILITY, HONOURED
        folds_ligand = {m for m, row in CAPABILITY.items() if row.get("ligand") == HONOURED}
    out = []
    for name in sorted(CEILINGS):
        if name == exclude or (ligand_atoms > 0 and name not in folds_ligand):
            continue
        c = ceiling(name, arch)
        if c.measured and c.residues is not None and not _verdict(name, c, residues, ligand_atoms)[0]:
            out.append(name)
    return out


def check(model: str, residues: int, *, ligand_atoms: int = 0, arch: str | None = None,
          fast: bool = False, where: str = "This input") -> None:
    """Refuse an input of `residues` plus `ligand_atoms` ligand heavy atoms on `model`, if a
    MEASURED ceiling says it will not fold. Otherwise silent.

    Raises ``SizeTooLargeError``. The message names the model, the dimension, the value, the limit
    and the architecture, because a refusal that omits the architecture is unactionable -- the same
    input at the same size folds on Blackhole for several of these models. Where the ligand is what
    crosses the wall it shows the token arithmetic as well, and how many ligand atoms this input
    did have room for: a user who sent 1000 residues and was told "1024 residues is the limit" has
    been told something true and useless.
    """
    arch = arch if arch is not None else current_arch()
    c = ceiling(model, arch, fast=fast)
    refused, tokens, padded, wall = _verdict(model, c, residues, ligand_atoms)
    if not refused:
        return
    if tokens is None:
        had = f"{residues} {_COUNT_NAMES[c.counts]}"
        limit = f"{c.residues} {_COUNT_NAMES[c.counts]}"
        ligand_note = ""
    else:
        had = (f"{residues} {_COUNT_NAMES[c.counts]} and a {ligand_atoms}-atom ligand, which is "
               f"{tokens} tokens padded to {padded}")
        limit = f"{wall} tokens"
        ligand_note = (
            f" A ligand's heavy atoms are tokens the trunk pays for exactly like residues, so they "
            f"count against the same wall: at {residues} {_COUNT_NAMES[c.counts]} there is room "
            f"for {max(0, wall - residues)} ligand atoms, not {ligand_atoms}.")
    if not enforced():
        import warnings
        warnings.warn(
            f"{where} has {had}, above {model}'s measured limit of "
            f"{limit} on {arch}. TT_BIO_SIZE_LIMIT=0 is set, so this runs anyway and may "
            f"fail on the device.", stacklevel=2)
        return
    alts = models_accepting(residues, arch, exclude=model, ligand_atoms=ligand_atoms, fast=fast)
    # "no model accepts this size" is only true where every model HAS a row. On an arch that is
    # mostly unmeasured -- blackhole, where two freeze rows exist and the other nine models were
    # measured folding 1536 by the 2026-09-10 ladder without earning a row -- the same sentence
    # would report missing rows as a hardware fact. Absence of a row means unmeasured, and this
    # message must not turn that into "cannot".
    unmeasured_here = [m for m in shipped_models() if not ceiling(m, arch).measured]
    fits = (f"above {residues}" if tokens is None else f"that admit this input")
    hint = (f" Models with a measured ceiling {fits} on this hardware: "
            f"{', '.join(alts)}." if alts else
            f" {len(unmeasured_here)} of this engine's models have no measured ceiling on "
            f"{arch} at all, so there is no answer here about what else fits; try a smaller "
            f"construct or domain." if unmeasured_here else
            " No model has a measured ceiling this high on this hardware; try a smaller "
            "construct or domain.")
    depth = (f" (measured with alignments up to {c.msa_rows} rows)" if c.msa_rows else "")
    top = ("the largest size proven on a ladder that never failed above it"
           if c.binds == LADDER_TOP else
           "the largest size below the first measured failure")
    raise SizeTooLargeError(
        f"{where} has {had}, and {model} is measured to handle at "
        f"most {limit} on {arch}{depth} -- {top}.{ligand_note}{hint}"
        f" If you have reason to think this input is roomier than the ladder that set the limit "
        f"(a single-sequence run of an MSA-dependent model is), set TT_BIO_SIZE_LIMIT=0 to run it "
        f"anyway."
    )


# ---------------------------------------------------------------------------------------------
# The size of an input, read off the file. No device, no CCD, no weights.
# ---------------------------------------------------------------------------------------------

_NON_LETTER = re.compile(r"[^A-Za-z]")
_POLYMER_KEYS = ("protein", "dna", "rna")


def _seq_residues(seq) -> int:
    """Residues in one sequence field. A 'low..high' binder range counts as its HIGH end.

    An upper bound is the only safe reading for a guard: a design spec that says 80..120 will
    allocate for 120, so scoring it at 80 would let the largest case through the check it exists for.
    """
    if not isinstance(seq, str):
        return 0
    s = seq.strip()
    if ".." in s:
        parts = [p.strip() for p in s.split("..", 1)]
        if all(p.isdigit() for p in parts):
            return max(int(p) for p in parts)
        return 0
    return len(_NON_LETTER.sub("", s))


def scan_residues(text: str) -> int:
    """Residues in one YAML or FASTA input. Best effort and an UPPER bound; never raises.

    Never raising is deliberate. This runs ahead of the real parser purely to decide a refusal, and
    a malformed input has to reach that parser to get its own proper error -- a guard that threw its
    own YAML exception first would replace a good message with a worse one.
    """
    t = text.lstrip()
    if t.startswith(">"):
        return sum(len(_NON_LETTER.sub("", ln.strip()))
                   for ln in text.splitlines() if ln.strip() and not ln.startswith(">"))
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    total = 0
    entries = data.get("sequences") or data.get("entities") or []
    for e in entries if isinstance(entries, list) else []:
        if not isinstance(e, dict):
            continue
        for key, body in e.items():
            if not isinstance(body, dict) or str(key).lower() not in _POLYMER_KEYS:
                continue
            idv = body.get("id")
            # `id: [A, B]` is N copies of one sequence, and each copy is folded.
            copies = len(idv) if isinstance(idv, list) and idv else 1
            total += _seq_residues(body.get("sequence")) * copies
    return total


def _heavy_atoms(spec: str) -> int:
    """Heavy atoms in one ligand spec: ``CCD_<code>[,<code>...]`` or a SMILES string.

    The two sources the FEATURIZER tokenises, not a third: the CCD component's own Mol out of the
    ``mols`` library, and RDKit's parse of a SMILES. Heavy atoms only, the same
    ``GetAtomicNum() > 1`` filter ``protenix_data.ligand_atom_features`` applies when it lays one
    token per atom. So this is the token count the model will build, not an estimate of it --
    checked against the two ligand counts already written into this file's own evidence: CCD STU
    comes back 35 (openbind's ladder) and BTN 16 (the census arm at 98 aa + BTN = 114 tokens).
    """
    try:
        if spec.upper().startswith("CCD_"):
            from .data.mol import load_molecules
            from .protenix_data import _default_mol_dir
            codes = [x.strip().upper() for x in spec[4:].split(",") if x.strip()]
            mols = load_molecules(_default_mol_dir(), codes)
            return sum(sum(1 for a in mols[c].GetAtoms() if a.GetAtomicNum() > 1) for c in codes)
        from rdkit import Chem
        mol = Chem.MolFromSmiles(spec)
        return mol.GetNumHeavyAtoms() if mol is not None else 0
    except Exception:
        return 0


def scan_ligand_atoms(path) -> int:
    """Ligand heavy atoms in one predict input, summed over every ligand chain. 0 if it has none.

    Takes a PATH rather than text because it reads the chains through ``main._read_bio_chains`` --
    the engine's own reader, the one the fold itself uses -- instead of a second copy of the input
    grammar. ``scan_rfd3_total`` goes through the engine's contig parser for the same reason: a
    size rule that reimplements the grammar it sizes drifts away from it, silently, and in the
    permissive direction.

    Never raises, and anything it cannot count scores 0 and refuses nothing -- a host with no CCD
    library, an unparseable SMILES, a file the real parser will reject on its own terms. That is
    ``scan_residues``'s rule and it holds here: a size guard is the wrong place to learn that an
    input is malformed.
    """
    try:
        from .main import _read_bio_chains
        chains = _read_bio_chains(Path(path))
    except Exception:
        return 0
    return sum(_heavy_atoms(c[1]) for c in chains if len(c) > 3 and c[3] == "ligand")


_BARE_SEQUENCE = re.compile(r"^[A-Za-z]+$")


def scan_rfd3_total(text: str) -> int:
    """Motif + designed residues in one RFD3 spec file (DESIGN_TOTAL). 0 if it cannot be sized.

    Sized from the CONTIG and not from the pasted structure, because the contig is what decides how
    big the run is: RFD3 tokenises the contig-selected motif plus the designed regions, so a
    nine-character contig like ``A1-2,4000`` asks for thousands of residues while looking tiny. The
    count comes from the engine's OWN parser (``rfd3.input.contig_residue_count``), which mirrors the
    featurizer's token plan term for term -- so the size rule cannot drift away from the grammar it
    sizes, which it would if this file reimplemented the arithmetic.
    """
    try:
        import json
        import yaml
        from .rfd3.input import contig_residue_count, parse_contig
    except Exception:
        return 0
    try:
        data = yaml.safe_load(text)
    except Exception:
        try:
            data = json.loads(text)
        except Exception:
            return 0
    if not isinstance(data, dict):
        return 0
    # Each top-level key is one independent design, and they run one after another rather than
    # together, so the ceiling applies to the LARGEST of them and not to their sum.
    largest = 0
    for spec in data.values():
        if not isinstance(spec, dict):
            continue
        contig = spec.get("contig")
        if not isinstance(contig, str) or not contig.strip():
            continue
        try:
            largest = max(largest, contig_residue_count(parse_contig(contig)))
        except Exception:
            continue
    return largest


def scan_pxdesign_target(text: str) -> int:
    """Conditioned TARGET residues in one PXDesign target YAML (DESIGN_TARGET). 0 if unsizable.

    Counted from the per-chain ``crop`` ranges, which is the only part of the spec that gives a
    residue count without opening the structure file the YAML points at. A spec with no crop
    conditions on the whole chain and returns 0, so it is NOT refused: sizing it needs the structure
    parsed, and a guard that guessed there would refuse real work on a number it did not have. The
    binder is deliberately excluded -- this model's ladder was walked in target residues with the
    binder held at 80, so counting it in would compare against the wrong denominator.
    """
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    chains = (data.get("target") or {}).get("chains") if isinstance(data.get("target"), dict) else None
    if not isinstance(chains, dict):
        return 0
    total = 0
    for body in chains.values():
        crop = (body or {}).get("crop") if isinstance(body, dict) else None
        if isinstance(crop, str):
            crop = [crop]
        if not isinstance(crop, (list, tuple)):
            return 0          # one uncropped chain and the whole spec is unsizable from text alone
        for rng in crop:
            m = re.match(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$", str(rng))
            if not m:
                return 0
            lo, hi = int(m.group(1)), int(m.group(2))
            total += max(0, hi - lo + 1)
    return total


def scan_longest_sequence(text: str) -> int:
    """Residues in the LONGEST single sequence of an embed/saprot input (MAX_SEQUENCE).

    The MAX and not the sum, which is the whole point. `tt-bio embed` and `tt-bio saprot` take
    INDEPENDENT sequences and run the trunk over each one separately, so what has to fit is the
    longest of them; the ceiling has nothing to say about how many there are. Summing instead
    refuses correct work -- a 50-record FASTA of 100 aa each scores 5000 against esmc-6b's 1968 and
    gets turned away, though every sequence in it embeds comfortably. That is worse than having no
    guard at all, and it is exactly what the first version of this file did.

    Predict is the opposite case and keeps ``scan_residues``: there one file is ONE complex whose
    chains are folded together, so the sum is what occupies the chip.

    Three input shapes, because ``embed`` documents all three: FASTA, a flat ``{id: sequence}``
    mapping, and the ``sequences:`` list the predict path uses.
    """
    t = text.lstrip()
    if t.startswith(">"):
        best = cur = 0
        for ln in text.splitlines():
            if ln.startswith(">"):
                best, cur = max(best, cur), 0
            elif ln.strip():
                cur += len(_NON_LETTER.sub("", ln.strip()))
        return max(best, cur)
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    entries = data.get("sequences") or data.get("entities")
    if isinstance(entries, list):
        best = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            for key, body in e.items():
                if isinstance(body, dict) and str(key).lower() in _POLYMER_KEYS:
                    best = max(best, _seq_residues(body.get("sequence")))
        return best
    # The flat `{id: sequence}` mapping. Every plain letter string counts, and that OVER-counts:
    # a `pool: mean` config line scores 4, because nothing in a flat mapping distinguishes a value
    # from a sequence, and `mean` happens to be four valid amino-acid letters, so no alphabet
    # filter separates them either. Harmless by construction rather than by luck -- this returns a
    # MAXIMUM, so a stray short word can only lose to a real sequence, and it could only cause a
    # wrong refusal if a config value ran to thousands of letters.
    return max((len(v.strip()) for v in data.values()
                if isinstance(v, str) and _BARE_SEQUENCE.match(v.strip())), default=0)


# What a design spec is allowed to be called. `design` takes JSON as well as the YAML the predict
# path accepts, so the shared suffix list would miss an RFD3 spec written as .json.
_DESIGN_SUFFIXES = (".json", ".yml", ".yaml")

# model -> (what its sizer counts, the sizer, the file suffixes it applies to).
#
# The suffixes are carried EXPLICITLY per model and not derived from the denominator. Deriving them
# was a real bug: the first version keyed them on `counts != RESIDUES` as a proxy for "is a design
# model", and adding a third denominator silently gave every embed model the design suffix list, so
# `.fasta` inputs were skipped and an oversized sequence was admitted. A proxy that happens to hold
# for two cases is not a rule, and it fails silently in the permissive direction.
_SIZERS: dict[str, tuple] = {
    "rfd3": (DESIGN_TOTAL, scan_rfd3_total, _DESIGN_SUFFIXES),
    "pxdesign": (DESIGN_TARGET, scan_pxdesign_target, _DESIGN_SUFFIXES),
    # Every embed / saprot model: independent sequences, so the longest one binds, not their sum.
    # FASTA is the common input here, so these keep the predict path's suffix list.
    **{m: (MAX_SEQUENCE, scan_longest_sequence, None) for m in
       ("esmc-300m", "esmc-600m", "esmc-6b", "saprot-35m", "saprot-650m", "saprot-1.3b")},
}
_DEFAULT_SIZER = (RESIDUES, scan_residues, None)


def sizer_for(model: str):
    """``(denominator, callable, suffixes)`` for one model. Unknown models size as plain residues.

    ``suffixes`` of None means the predict path's ``runtime.INPUT_SUFFIXES``.
    """
    return _SIZERS.get(model, _DEFAULT_SIZER)


def check_input(data, model: str, *, arch: str | None = None, fast: bool = False) -> None:
    """Refuse every oversized input in `data`: one file, a directory of them, or a bare sequence.

    THE call site for the CLI, and it runs before a device is opened -- which is the entire point.
    A refusal that lands after the device open has already done the damage this guard exists to
    prevent: on the L1-clash models the failure mode is a throw that can leave the chip wedged for
    the next job, so "fails fast with a clear message" and "fails after taking a chip" are different
    outcomes even though both end in an error.

    All three input shapes, because `tt-bio embed` and `tt-bio saprot` document a bare sequence
    string as a valid DATA argument, and a guard that only understood paths would skip the one input
    form a user is most likely to paste something huge into.

    Unreadable or unparseable files are skipped rather than reported, and a sizer that cannot size
    its input returns 0 and refuses nothing. Job discovery runs immediately after and fails on a bad
    file with its own message; a size guard is the wrong place to learn that a file is missing, and
    raising here would replace a good error with a worse one.
    """
    from .runtime import INPUT_SUFFIXES
    _, sizer, suffixes = sizer_for(model)
    suffixes = suffixes or INPUT_SUFFIXES
    arch = arch if arch is not None else current_arch()
    # The ligand is only counted where the ceiling's wall is measured on tokens. Anywhere else it
    # would load the CCD library and RDKit to produce a number nothing compares against.
    tokenwise = ceiling(model, arch, fast=fast).token_bound
    text = str(data).strip()
    p = Path(text).expanduser()
    try:
        exists = p.exists()
    except OSError:      # an over-long "path" that is really a pasted sequence
        exists = False
    if not exists:
        if _BARE_SEQUENCE.match(text):
            check(model, len(text), arch=arch, fast=fast, where="The input sequence")
        return
    files = sorted(q for q in (p.glob("*") if p.is_dir() else [p])
                   if q.suffix.lower() in suffixes)
    for q in files:
        try:
            n = sizer(q.read_text())
        except Exception:
            continue
        if n:
            check(model, n, ligand_atoms=scan_ligand_atoms(q) if tokenwise else 0,
                  arch=arch, fast=fast, where=f"'{q.name}'")


def shipped_models() -> set:
    """Every name reachable from a CLI ``--model`` choice, discovered from main.py's own tuples.

    Discovered and not named, for the reason token_axis.shipped_models() records: a hand-typed list
    is exactly how a model gets added to the CLI without getting a row here.
    """
    from tt_bio import main as _main
    tuples = {n: getattr(_main, n) for n in dir(_main) if n.endswith("_MODELS")}
    return set().union(*tuples.values())


# ---------------------------------------------------------------------------------------------
# The refusal that arrives too late for the table above.
# ---------------------------------------------------------------------------------------------

# The allocator's message, with its closing parenthetical, which is the part that says which wall
# was hit. Everything before it describes the request; only these three numbers describe the chip.
# Public: it is tt-metal's format and not ours, so every instrument that reads a run log reads it
# from here rather than transcribing it and rotting on somebody else's release schedule.
ALLOC_REFUSAL = re.compile(
    r"Not enough space to allocate (?P<req>\d+) B (?P<space>DRAM|L1) buffer across "
    r"(?P<banks>\d+) banks, where each bank needs to store (?P<per_bank>\d+) B, but bank size "
    r"is (?P<bank_size>\d+) B\s*\(allocated: (?P<allocated>\d+) B, free: (?P<free>\d+) B, "
    r"largest free block: (?P<largest>\d+) B\)")


def _mib(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.1f} MiB"


def is_alloc_refusal(exc: BaseException) -> bool:
    """Whether this exception is the device allocator refusing, and not any other failure.

    The one predicate every retry path shares, so a circular-buffer throw or a shape error is
    never quietly re-run through a fallback meant for an out-of-memory. Reads the same message
    ``describe_device_oom`` renders, so the two can never disagree about what a refusal is.
    """
    return describe_device_oom(str(exc)) is not None


# What the allocator's own numbers say happened. One decision, two renderings: the sentence a user
# reads (``describe_device_oom``) and the mechanism a ceiling row records (``classify_device_oom``).
# Written once because they have to agree -- a refusal a ladder records as fragmentation and the
# guard describes as a full chip is two answers to one question.
_SHAPE = "shape"            # the request does not fit an EMPTY bank
_FULL = "full"              # it does not fit the free bytes
_FRAGMENTED = "fragmented"  # it fits the free bytes but not any single free block


def _last_refusal(text: str) -> dict | None:
    """The LAST allocator refusal in ``text`` as typed numbers, or None if there is none.

    The last and not the first: several paths in this engine catch a refusal and retry with a
    smaller block, so an early one is routinely not the one that ended the run.
    """
    hits = list(ALLOC_REFUSAL.finditer(text))
    if not hits:
        return None
    return {k: (v if k == "space" else int(v)) for k, v in hits[-1].groupdict().items()}


def _refusal_kind(g: dict) -> str | None:
    """Which of the three walls a parsed refusal hit. None when the numbers describe no refusal."""
    per_bank = g["per_bank"]
    if per_bank > g["bank_size"]:
        return _SHAPE
    if per_bank > g["free"]:
        return _FULL
    if per_bank > g["largest"]:
        return _FRAGMENTED
    return None


def classify_device_oom(text: str) -> str | None:
    """Which of ``MECHANISMS`` an allocator refusal in ``text`` names, or None if it holds none.

    The vocabulary a CEILINGS row is written in, so a ladder that measures a rung and a row that
    publishes the ceiling name the same wall and nobody translates between them by hand.

    It reads the allocator's NUMBERS, which is the whole point. A refusal message carries the word
    DRAM (or L1) and the phrase "largest free block" in the same sentence, so a list of substring
    patterns can only ever report which memory it was and never which wall: whichever arm is listed
    first wins every time, and the other is unreachable. That is not hypothetical, it is what
    ``capacity_gate.classify`` did until this existed.

    ``FRAGMENTATION`` is not split by space. An L1 refusal comes back with ``free`` equal to
    ``largest`` (nothing to coalesce, see ``L1_BUDGET``), so the fragmented arm is a DRAM answer by
    construction rather than by assumption.
    """
    g = _last_refusal(text)
    if g is None:
        return None
    kind = _refusal_kind(g)
    if kind is None:
        return None
    if kind == _FRAGMENTED:
        return FRAGMENTATION
    return DRAM if g["space"] == "DRAM" else L1_BUDGET


def describe_device_oom(text: str) -> str | None:
    """One sentence for an allocator refusal, or None if `text` is not one.

    A user who asks for a size the chip cannot serve currently gets a C++ assertion line
    (``TT_FATAL @ .../bank_manager.cpp:439: false``) followed by twenty backtrace frames, and the
    one sentence that says what actually happened is buried in the middle. The assertion line is
    the least informative thing in the message: it names a file in tt-metal and the literal word
    "false".

    The three cases below are genuinely different problems and want different answers, so the
    sentence names which one it is instead of saying "out of memory" three ways:

      * the request does not fit an EMPTY bank -- one oversized tensor, and no amount of freeing
        would have helped. A smaller input is the only lever.
      * it fits the free bytes but not any single free block -- fragmentation. The chip has the
        room and cannot hand it over in one piece.
      * it does not fit the free bytes -- the chip is full. Something else resident has to go.

    The LAST refusal in the text, not the first: several paths in this engine catch a refusal and
    retry with a smaller block, so an early one is routinely not the one that ended the run.
    """
    g = _last_refusal(text)
    if g is None:
        return None
    space, per_bank, kind = g["space"], g["per_bank"], _refusal_kind(g)
    if kind == _SHAPE:
        why = (f"one allocation of {_mib(g['req'])} needs {_mib(per_bank)} in each of "
               f"{g['banks']} {space} banks and a bank holds {_mib(g['bank_size'])}. No chip "
               f"state would have served it, so this is the shape and not the load")
    elif kind == _FULL:
        why = (f"{_mib(g['req'])} was requested, needing {_mib(per_bank)} per {space} bank, and "
               f"only {_mib(g['free'])} is free. The chip is full")
    elif kind == _FRAGMENTED:
        why = (f"{_mib(g['req'])} was requested, needing {_mib(per_bank)} per {space} bank. "
               f"{_mib(g['free'])} is free but the largest single free block is "
               f"{_mib(g['largest'])}, so the room exists and cannot be handed over in one "
               f"piece: this is fragmentation, not a full chip")
    else:
        return None      # the numbers do not describe a refusal; say nothing rather than guess
    return f"out of device {space}: {why}."
