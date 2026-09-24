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
exactly as it always was, which matters more than it looks: a ladder walked WITH a ligand
(openbind's old 960 carried a 35-atom one) has a token wall above its residue cap, and letting that
wall speak for a ligand-free input would quietly raise a published ceiling nobody re-walked.

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
# motif PLUS designed residues, while PXDesign's 960 is TARGET residues only, with the designed
# binder on top and outside the number. A guard that compared a total against a target-only cap
# would refuse correct work on one model and pass oversized work on the other. Each row names its
# denominator, each model has a sizer that produces that denominator, and the guard test asserts the
# two agree -- so the slip is a test failure rather than a subtle wrong answer.
RESIDUES = "residues"              # polymer residues in the input, summed over chains and copies
DESIGN_TOTAL = "design_total"      # motif + designed, i.e. everything the model tokenises
DESIGN_TARGET = "design_target"    # the conditioned target only; the binder is extra
MAX_SEQUENCE = "max_sequence"      # residues in the LONGEST single sequence, not their sum
TARGET_ATOMS = "target_atoms"      # heavy atoms in the conditioned target; the binder is extra
COUNTS = (RESIDUES, DESIGN_TOTAL, DESIGN_TARGET, MAX_SEQUENCE, TARGET_ATOMS)

# The four residue denominators above are four ways of counting residues and they compare, loosely,
# with each other. TARGET_ATOMS does not: a deposited protein carries about 8 heavy atoms per
# residue, so 14786 atoms and 14786 residues differ by an order of magnitude. The dimension is what
# keeps the refusal message honest -- ``models_accepting`` may only offer a model whose cap is in
# the same dimension as the number being refused, or a 1200-residue refusal would name BoltzGen as
# having room because its cap reads 14786.
_COUNT_DIMENSION = {
    RESIDUES: "residues",
    DESIGN_TOTAL: "residues",
    DESIGN_TARGET: "residues",
    MAX_SEQUENCE: "residues",
    TARGET_ATOMS: "atoms",
}

# How each denominator reads in a refusal. A message that just said "residues" for all three would
# leave a design user unable to tell which number of theirs is too big.
_COUNT_NAMES = {
    RESIDUES: "residues",
    DESIGN_TOTAL: "residues (motif + designed)",
    DESIGN_TARGET: "target residues",
    MAX_SEQUENCE: "residues in its longest sequence",
    TARGET_ATOMS: "atoms in the target",
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
            residues=1536, pass_at=1536, fail_at=1664, binds=RUNTIME, mechanism=FRAGMENTATION,
            msa_rows=16384,
            evidence="1664 is where the fold stops being usable, walked 2026-09-23/24 on origin/main "
                     "c5b346679 (mgx-bigalloc merged), j10glx02 card 16, guard off, --host_threads "
                     "2, 8192 rows (ws:mgx-ceilings, perf/mgxceil). The residue pair, 1664^2 x 384 "
                     "x 2 B = 2126512128 B, is refused as one allocation (177209344 B per bank "
                     "against a 134360608 B largest block with 361496608 B free: fragmentation), "
                     "so every chunked pair op joins its blocks on the host "
                     "(tenstorrent.py host_acc_after_refusal -> _acc_concat). py-spy put 99-100 "
                     "percent of the main thread there and a trunk recycle took ~20 min, against "
                     "at most 9 min at 1536 (5223.9 s for the whole fold). After 7897.1 s the "
                     "trunk was on recycle 6 of 10, past the 6436 s watchdog JapanFold's "
                     "wk/mgx-platform-cap gives a 1664 predict (1500 s x (1664/1024)^3), and the fold was stopped with SIGINT so ttnn closed the chip cleanly. AICLK median 1000 MHz over 749 "
                     "samples, host load 44-63 on 64 cores. 1792 was at recycle 2 of 10 after "
                     "5536.2 s (card 19) and 1920 fails outright after 93.1 s on the trunk "
                     "recycle pair add (protenix.py trunk, 2831155200 B, 225 MiB per bank, 98.8 "
                     "MiB largest). So the cap stays 1536 and the reason above it is run time, not "
                     "a crash; TT_BIO_SIZE_LIMIT=0 runs 1664 for anyone prepared to wait hours. "
                     "1536 residues fold at 8192 alignment rows on the j10glx02 Galaxy, 2026-09-23 "
                     "(ws:mgx-bigalloc, perf/whceil/ladder.py, chip 20, tt-bio c8f75a9f0): "
                     "cdk2x2_1536_d8192 with all 10 trunk recycles, PASS in 5223.9 s at AICLK "
                     "1000 MHz sampled DURING the fold, under a host load that voids the time for "
                     "speed. Nine DRAM refusals were absorbed on the way, the largest 6948913152 B "
                     "(the refiner's 3008-token structural pair), and the trunk held a flat 4.88 GiB "
                     "per pairformer block. The Blackhole trunk freeze at recycle 9/10 does not "
                     "occur here. The walls this replaces were whole tensors now streamed or "
                     "row-blocked in shared code: the triangle-attention projections at 1088 "
                     "(909115392 B, a token gate fitted on a 128-channel pair), the alignment "
                     "feature uploaded whole, the updated MSA kept on device, OuterProductMean's "
                     "join beside a live pair, the refiner's padded pair copy, every pairformer "
                     "update beside its full-size input, and the diffusion's pair conditioning and "
                     "DiT biases built whole. Accuracy: cdk2x2_1024 is byte-identical to the "
                     "engine before these changes (CIF md5 b08a39e5), and 2ad6_1280 (a real "
                     "1280-token complex, 15115 rows) scores 0.28 A CA-RMSD / lDDT 0.9983 against "
                     "the upstream reference, which has one seed there. 1536 has no reference yet; "
                     "its fold keeps the backbone except one stretched 14-residue segment "
                     "(586-599, CA-CA up to 5.5 A, pLDDT 48-56) on a junction of the tiled "
                     "fixture, whose repeats have no defined arrangement. The previous row, 1024 "
                     "on GWH02 at the same depth with 1088 failing on the j10glx02 Galaxy, is "
                     "what these fixes moved. AT 16384 ROWS, the most the featurizer keeps "
                     "(ws:mgx-msa-depth, perf/mgx_msa_depth, chip 18, tt-bio 2f7e0606c): "
                     "cdk2_1536_d16384 PASS in 5951 s at AICLK 1000 MHz on 563 samples taken "
                     "DURING the fold, ten refusals absorbed, msa_depth 16384 in results.json, "
                     "CA-CA median 3.83 A with 98.0 % in band, so this row holds at every depth a "
                     "user reaches. "
                     "Re-passed on origin/main 6ea518246 (after sdpa-mask 94e7a002c), j10glx02, "
                     "2026-09-24, guard off, --host_threads 2, AICLK median 1000 MHz sampled during "
                     "the fold: 1536 at 16384 rows PASS in 5402.8 s (card 8, msa_depth 16384, pLDDT "
                     "0.725), above the 5062 s cube-scaled platform watchdog at 1536 on a loaded host",
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
            residues=1536, pass_at=1536, fail_at=1664, binds=RUNTIME, mechanism=FRAGMENTATION,
            msa_rows=16384,
            evidence="its OWN 1664 rung, walked 2026-09-23/24 on origin/main a102dfb5c (the "
                     "bigalloc engine; c5b346679 after it changes nothing opendde runs), j10glx02 "
                     "card 14, guard off, --host_threads 2, 8192 rows (ws:mgx-ceilings, "
                     "perf/mgxceil). The same 2126512128 B residue pair as opendde is refused as "
                     "one allocation (fragmentation), every chunked pair op joins its blocks on "
                     "the host, and after 9037.5 s the trunk was on recycle 7 of 10, past the "
                     "same 6436 s watchdog at 1664; stopped with SIGINT for a clean "
                     "close. AICLK median 1000 MHz over 857 samples, host load 43-63 on 64 cores. "
                     "Run time, not a crash, is what binds above 1536. "
                     "Its OWN 1536 rung at 8192 alignment rows on the j10glx02 Galaxy, 2026-09-23, "
                     "not inherited from opendde by architecture argument (ws:mgx-bigalloc, "
                     "perf/whceil/ladder.py, chip 18, tt-bio c8f75a9f0): cdk2x2_1536_d8192 with all "
                     "10 trunk recycles, PASS in 5366.1 s at AICLK 1000 MHz on 497 of 501 samples "
                     "taken DURING the fold, under a host load that voids the time for speed. Nine "
                     "DRAM refusals absorbed, the largest 6948913152 B (the refiner's 3008-token "
                     "structural pair); the diffusion held 8.45 GiB of 12. The backbone has no "
                     "break (CA-CA median 3.84 A, 97.9 % in band). The two checkpoints share every "
                     "tensor shape, so the walls this replaces are opendde's: the 1088 failure "
                     "recorded here on 2026-09-11 (the triangle-attention gate output, "
                     "tokens^2 x 384 x 2 B, after a 2424307712 B OuterProductMean refusal was "
                     "survived) and the whole tensors after it, now streamed or row-blocked in "
                     "shared code. 1536 has no upstream reference yet; 7aqx_1024 has one for this "
                     "checkpoint. The size ladder's own cdk2x2_1280 rung (single sequence, 6 steps), which "
                     "crashed after the trunk on an L1 clash before 72ea47311, folds on 598f158a8 "
                     "(chip 30, 2005.9 s, AICLK median 1000 MHz). The previous row was 1024 (3/3 byte-identical folds, 2026-09-08) "
                     "with 1088 failing. AT 16384 ROWS, the most the featurizer keeps (ws:mgx-msa-depth, "
                     "perf/mgx_msa_depth, chip 23, tt-bio b08951fd6): cdk2_1024_d16384 PASS in 1779 s "
                     "at AICLK 1000 MHz sampled DURING the fold, three DRAM refusals absorbed; "
                     "cdk2_1536_d16384 PASS in 5891 s at AICLK 1000 MHz on 556 samples taken DURING "
                     "the fold (chip 12, tt-bio b9cdecce1), ten refusals absorbed, msa_depth 16384 "
                     "in results.json, CA-CA median 3.85 A with 98.0 % in band. "
                     "Re-passed on origin/main 6ea518246 (after sdpa-mask 94e7a002c), j10glx02, "
                     "2026-09-24, guard off, --host_threads 2, AICLK median 1000 MHz sampled during "
                     "the fold: 1536 at 16384 rows PASS in 5342.0 s (card 16, msa_depth 16384, pLDDT "
                     "0.701), above the 5062 s cube-scaled platform watchdog at 1536 on a loaded host",
        ),
    },
    "openfold3": {
        "wormhole_b0": Ceiling(
            residues=1664, pass_at=1664, fail_at=1792, binds=MEMORY, mechanism=FRAGMENTATION,
            msa_rows=16384,
            evidence="1792 x 14190 re-failed on origin/main a102dfb5c (mgx-bigalloc merged; the "
                     "later c5b346679 and 720b43a5d do not change this path's allocations), j10glx02 "
                     "card 19, 2026-09-23, guard off, --host_threads 2, after 1018.1 s: bigalloc "
                     "moved the wall out of triangle attention into the diffusion transformer's "
                     "attention scores (openfold3_diffusion_transformer.py:216 scale_add), one "
                     "205520896 B request, 17129472 B per bank against a 17128832 B largest block "
                     "on a chip 95.8 percent full (43.4 MiB free): fragmentation on a full chip. "
                     "AICLK median 1000 MHz. Before that, "
                     "walked 2026-09-23 on origin/main a2a70b160 (combos merge: template features "
                     "stay on the host), j10glx02 card 19, guard off, --host_threads 2, apo CDK2 "
                     "tiled at 14190 alignment rows (ws:mgx-ceilings, perf/mgxceil). 1664 x 14190 "
                     "PASS in 3625.5 s, complex pLDDT 0.49. 1792 x 14190 fails after 1372.1 s in the "
                     "MSA stack's triangle attention (openfold3_msa_embedder.py:141 -> "
                     "TriangleAttention, the fp32 softmax tail after host_acc_after_refusal had "
                     "already absorbed an 8220835840 B refusal): one 822083584 B request, 65.3 MiB "
                     "per bank against a 63.0 MiB largest block with 202.8 MiB free, so "
                     "fragmentation. AICLK median 1000 MHz sampled during both. The 1664 wall "
                     "below, 512 B short in the diffusion transformer, is what the combos merge "
                     "removed. At 16384 alignment rows, the most the featurizer ever keeps, 1536 folds "
                     "(ws:mgx-msa-depth, chip 9, tt-bio 3f10800fd, 2026-09-23): cdk2_1536_d16384 PASS "
                     "in 2599.8 s at AICLK 1000 MHz sampled DURING the fold, 4 refusals recovered; "
                     "and 1664 x 16384 PASS on origin/main 811ac1316 (ws:mgx-ceilings, card 13, "
                     "2026-09-24, guard off, --host_threads 2, msa_depth 16384 in results.json) in "
                     "1158.4 s at AICLK median 1000 MHz over 112 samples DURING the fold, pLDDT "
                     "0.408, CA-CA median 3.83 A with 99.0 percent in band, no non-adjacent CA "
                     "pair under 3 A. 1792 failed at the shallower 14190, so it fails here too. "
                     "Earlier: 1536 residues fold at 14190 alignment rows on the j10glx02 Galaxy, "
                     "2026-09-23 (ws:mgx-bigalloc, perf/whceil/ladder.py, chip 9, tt-bio 296f5fcea): "
                     "cdk2x2_1536_d14190, PASS in 1917.9 s at AICLK 1000 MHz sampled DURING the fold, "
                     "under a host load that voids the time for speed. 1536 x 8192 rows folds too "
                     "(chip 4, 7bdc54533). The walls this replaces were four whole tensors, each "
                     "now streamed or row-blocked in shared code: the MSA transition's joined "
                     "result uploaded whole at 1088 x 14191 (1976295424 B), the diffusion "
                     "conditioning pair at 1216 (1703411712 B), the noisy-position layer norm "
                     "at 1536 (1207959552 B), and at 1536 x 14190 the deep MSA chunk list "
                     "resident through the pair stack (2.79 GB). The MSA representation now "
                     "waits on the host and passes through the chip a depth chunk at a time. "
                     "Accuracy above the old wall: 2ad6_1280 (a real 1280-token complex, 14743 "
                     "rows) scores 0.509 / 0.431 A CA-RMSD against the upstream reference's two "
                     "seeds, whose own floor is 0.379 A, and its CIF is byte-identical across "
                     "296f5fcea. At 1024 on 7aqx every chain moves less than the reference "
                     "between its own seeds (OuterProductMean's chunked depth sum reassociates "
                     "in bf16). Re-walked on origin/main cec7979b1 (ws:mgx-ceilings, perf/mgxceil, "
                     "card 12, 2026-09-23, guard off, --host_threads 2): 1536 x 14190 PASS in "
                     "2641.7 s, 1664 x 14190 fails after 2873.0 s in the diffusion transformer's "
                     "scale_add (openfold3_diffusion_transformer.py -> eltwise_fusion.py), one "
                     "177209344 B request, 14770176 B per bank, with 58276320 B free and a "
                     "largest block 512 B too small: fragmentation. AICLK median 1000 MHz during "
                     "both. The previous row, 1024 on GWH02 at the same depth "
                     "(ws:ceiling-openfold3-1024, ws:ceiling-1024-integration-and-gate), stays "
                     "valid below: every rung 640-1024 folded with an intact backbone",
        ),
    },
    "openbind": {
        "wormhole_b0": Ceiling(
            residues=1664, pass_at=1664, fail_at=1792, binds=MEMORY, mechanism=FRAGMENTATION,
            msa_rows=16384, ladder_ligand_atoms=0,
            evidence="walked on origin/main a102dfb5c (mgx-bigalloc merged; c5b346679 and 720b43a5d "
                     "do not change this path's allocations), j10glx02 card 19, 2026-09-23, guard "
                     "off, --host_threads 2, 14190 rows: 1664 PASS again in 1013.0 s, and 1792 "
                     "fails after 1185.2 s on openfold3's 1792 wall, the diffusion transformer's "
                     "scale_add (openfold3_diffusion_transformer.py:216), one 205520896 B request, "
                     "17129472 B per bank against a 17128832 B largest block on a chip 95.8 percent "
                     "full: fragmentation on a full chip. AICLK median 1000 MHz during both. "
                     "Earlier, 1664 residues fold apo at 14190 alignment rows on origin/main a2a70b160 "
                     "(ws:mgx-ceilings, perf/mgxceil, j10glx02 card 19, 2026-09-23, guard off, "
                     "--host_threads 2): PASS in 2909.6 s, AICLK median 1000 MHz over 278 samples "
                     "during the fold, pLDDT 0.442 (0.451 at 1536), no CA-CA break, no non-adjacent "
                     "CA pair under 3 A, 4 refused allocations recovered. The a2a70b160 tree keeps "
                     "OF3/openbind template features on the host; the bigalloc merge after it only "
                     "adds refusal recovery on this path. "
                     "1536 residues fold apo at 14190 alignment rows on the j10glx02 Galaxy, "
                     "2026-09-23 (ws:mgx-bigalloc, perf/whceil/ladder.py, chip 17, tt-bio 296f5fcea): "
                     "cdk2x2_1536_d14190, PASS in 1914.2 s at AICLK 1000 MHz sampled DURING the fold, "
                     "under a host load that voids the time for speed. 1536 x 8192 rows folds too "
                     "(chip 19, 7bdc54533). It shares openfold3's stack and the same four fixes "
                     "moved it: before them it failed at 1152 x 14190 on the MSA transition's "
                     "joined upload (2031058944 B) and at 1216 x 8192 on the diffusion "
                     "conditioning pair (1703411712 B). Accuracy above the old wall: 2ad6_1280 "
                     "scores 0.414 A / lDDT 0.9924 against the nearer upstream reference seed and "
                     "1.564 A against the other; the reference's own seed floor is 1.509 A. "
                     "The rungs were apo, so the wall is on TOKENS and ladder_ligand_atoms=0 "
                     "says so: a ligand's heavy atoms count against 1536 and an input past it is "
                     "refused at submission. The previous cap, 960 residues walked with CCD STU "
                     "(35 atoms) on GWH02 at e9cb5b70 (ws:ceiling-1024-integration-and-gate), "
                     "folded 768/896/960 with intact backbones and stays valid below. "
                     "Re-walked on origin/main cec7979b1 (ws:mgx-ceilings, perf/mgxceil, card 20, "
                     "2026-09-23, guard off, --host_threads 2): 1536 x 14190 PASS in 2250.8 s, "
                     "1664 x 14190 fails after 2561.9 s on the same diffusion-transformer "
                     "request as openfold3, 177209344 B, 14770176 B per bank, with 58366432 B "
                     "free on a 94.6 percent full chip and a largest block 512 B too small: "
                     "fragmentation. AICLK median 1000 MHz during both. "
                                          "AT 16384 ROWS, the most the vendored featurizer keeps, 2026-09-23 "
                     "(ws:mgx-msa-depth, perf/mgx_msa_depth, chip 29, tt-bio 3f10800fd): "
                     "cdk2_1536_d16384 PASS in 2383 s at AICLK 1000 MHz sampled DURING the fold, "
                     "four DRAM refusals absorbed. 1664 x 16384 PASS on origin/main 811ac1316 "
                     "(ws:mgx-ceilings, card 14, 2026-09-24, guard off, --host_threads 2, msa_depth "
                     "16384 in results.json) in 1225.6 s at AICLK median 1000 MHz over 118 samples "
                     "DURING the fold, pLDDT 0.443, CA-CA median 3.84 A with 97.2 percent in band; "
                     "1792 failed at the shallower 14190, so it fails here too",
        ),
    },
    "rf3": {
        "wormhole_b0": Ceiling(
            residues=1600, pass_at=1600, fail_at=1664, binds=MEMORY, mechanism=FRAGMENTATION,
            msa_rows=8192,
            evidence="walked 2026-09-23 on origin/main cec7979b1 (pair-residency merged), "
                     "j10glx02 card 29, guard off, --host_threads 2, apo CDK2 tiled with 8192 "
                     "alignment rows (ws:mgx-ceilings, perf/mgxceil). rf3 subsamples the "
                     "alignment to 1024 rows (rf3/featurize.py), so this row speaks for any depth "
                     "of 1024 or more. 1600 folds in 1159.6 s and 1664 fails after 1256.3 s in the "
                     "diffusion atom encoder's trunk-pair permute (rf3/diffusion_atom_encoder.py "
                     "_trunk_pair), one 1427898368 B request, 118992896 B per bank, on a chip 85.5 "
                     "percent full with 155415968 B free and a 59496448 B largest block: "
                     "fragmentation. AICLK median 1000 MHz sampled during both folds, host load "
                     "56-67 on 64 cores. 1664 re-fails on origin/main c5b346679 (bigalloc and "
                     "embed-scale merged), card 29, 2026-09-24, after 1192.6 s at the same request "
                     "and site, 85.5 percent full with an 82092064 B largest block, AICLK median "
                     "1000 MHz over 114 samples. The rungs below were walked on 8906d35a0 (1024 554.0 s to "
                     "1536 952.3 s, pLDDT 0.77-0.78, no clash), where 1600 still failed in the "
                     "ending-node triangle attention's pair transpose (655360000 B, 139.0 MiB free, "
                     "46.9 MiB largest block); pair-residency removed that wall. The row was 1095 "
                     "LADDER_TOP before that, from a 2026-09-07 GWH02 ladder that stopped there; "
                     "its 627 wall was the materialised fp32-softmax triangle attention and the "
                     "confidence head's one-row layer norm, and TT_BIO_RF3_TEMPLATE_FUSED_SDPA=0, "
                     "TT_BIO_RF3_MSA_FUSED_SDPA=0 or TT_BIO_RF3_GLN_ROW_FOLD=0 still restore that "
                     "route and that wall. "
                     "Re-passed on origin/main 6ea518246 (after sdpa-mask 94e7a002c), j10glx02, "
                     "2026-09-24, guard off, --host_threads 2, AICLK median 1000 MHz sampled during "
                     "the fold: 1600 PASS in 1057.3 s (card 3, pLDDT 0.771, no CA-CA break)",
        ),
    },
    "protenix-v2": {
        "wormhole_b0": Ceiling(
            residues=2048, pass_at=2048, fail_at=None, binds=LADDER_TOP,
            mechanism=NO_FAILURE, msa_rows=16384, ladder_ligand_atoms=0,
            evidence="2048, the top rung, folds at 16384 rows on origin/main 6ea518246, j10glx02 card "
                     "16, 2026-09-24, guard off, --host_threads 2 (ws:mgx-ceilings, perf/mgxceil, "
                     "msa_depth 16384 in results.json): PASS in 4832.6 s, AICLK median 1000 MHz over "
                     "466 samples during the fold, pLDDT 0.699, no CA-CA break, 46 refused "
                     "allocations recovered, the largest 8589934592 B. On the same tree 1920 at 8192 "
                     "rows folds in 3929.0 s (card 6, 1000 MHz, pLDDT 0.730), so the 8192-row OPM "
                     "failure at 1920 below is gone: mgx-msa-depth (a239d22da) re-runs a refused OPM "
                     "as host-tiled depth chunks. On 811ac1316 (no protenix-v2 default-path change "
                     "to 6ea518246: trace regions are opt-in, the trimul block narrows only from 2560 "
                     "tokens), card 16, 16384 rows, 1000 MHz: 1664 in 3079.8 s, 1792 in 3340.3 s, "
                     "1920 in 7266.5 s (45 refusals recovered, pLDDT 0.713). Above 1792 the trunk "
                     "joins pair blocks on host after a refusal, so wall time follows which "
                     "allocations were refused, not size alone. "
                     "1792 folds on origin/main a102dfb5c (mgx-bigalloc merged; the later "
                     "c5b346679 adds only ESMC/SaProt masks and perf/, neither reaches protenix-v2), "
                     "j10glx02 card 29, 2026-09-23, guard off, --host_threads 2, 8192 rows: PASS "
                     "in 2623.4 s, AICLK median 1000 MHz over 251 samples during the fold, pLDDT "
                     "0.714, no CA-CA break, 21 refused allocations recovered by the row-blocked "
                     "pair path. 1920 fails on the same tree and card after 168.1 s in the MSA "
                     "module's OuterProductMean (protenix.py _msa -> tenstorrent.py "
                     "outer_product_mean, after a 1006632960 B refusal was absorbed): one "
                     "125829120 B request, 10487040 B per bank against a 10485760 B largest block "
                     "on a chip 96.8 percent full (32.3 MiB free), so fragmentation on a full "
                     "chip. AICLK 1000 MHz. "
                     "Before bigalloc the row was 1664/1792: walked 2026-09-23 on origin/main cec7979b1 (pair-residency merged), "
                     "j10glx02, guard off, --host_threads 2, apo CDK2 tiled with 8192 "
                     "alignment rows, the --max_msa_seqs default (ws:mgx-ceilings, "
                     "perf/mgxceil). 1024 folds in 913.5 s, 1152 in 1292.5 s, 1280 in 1486.7 s, "
                     "1408 in 1881.7 s (card 3), 1536 in 2046.8 s and 1664 in 2609.9 s (card 9), "
                     "AICLK median 1000 MHz sampled during every fold, at host load 52-214 on 64 "
                     "cores. 1792 fails after 2692.9 s (card 17, same tree, 1000 MHz) in the diffusion "
                     "pair conditioning (protenix.py _diffusion_pair_cond), which asks for "
                     "3288334336 B, 261.3 MiB per bank, with 514.5 MiB free and a 219.3 MiB "
                     "largest block: fragmentation, not a full chip. The earlier row "
                     "(1024, fail 1095) was walked at 8832 rows, deeper than the default allows; "
                     "the 2026-09-11 whglx ladder at 8192 rows already folded 1152 on the tree "
                     "before pair-residency. The wall is on TOKENS (the failing tensor scales "
                     "with tokens x rows), so ladder_ligand_atoms=0 counts a ligand against it. "
                     "AT 16384 ROWS, the pool cap tt_bio/protenix_data.py now applies as upstream "
                     "does, 2026-09-23 on the j10glx02 Galaxy (ws:mgx-msa-depth, perf/mgx_msa_depth, "
                     "chip 0, tt-bio b08951fd6): cdk2_1024_d16384 PASS in 1277 s at AICLK 1000 MHz "
                     "sampled DURING the fold, two DRAM refusals absorbed; 1280 and 1536 fold at "
                     "16384 too (1803 s, 2574 s, chip 2, 3f10800fd). Before the block-0 OPM "
                     "fallback the 1280-token 2ad6 fold at 14743 rows died on a 160 MiB OPM depth "
                     "slice with a 10.0 MiB largest free block. The cap was then re-walked at 16384 (first sentence)",
        ),
    },
    "rfd3": {
        "wormhole_b0": Ceiling(
            residues=1536, pass_at=1536, fail_at=None, binds=LADDER_TOP,
            mechanism=NO_FAILURE, counts=DESIGN_TOTAL,
            evidence="1536 total residues design on whglx/j10glx02 card 15, 2026-09-23, "
                     "ws:mgx-design-ceiling: 1536 residues / 12085 heavy atoms in 7470.0 s at AICLK median 1000 MHz "
                     "sampled DURING the rung (n=1247), through the shipped `tt-bio design --model rfd3 --from_pdb` "
                     "CLI at the platform's 100 diffusion steps, contig A1-1008,B1-172,100 over a two-chain crop of "
                     "perf/bhdesign/targets/big_1831.cif. NO SPEED CLAIM: j10glx02 carried load 325-743 on 64 cores "
                     "while every MGX row shared it, which makes pxdesign 512 read 9.0x its GWH02 time at the same "
                     "clock, so these seconds are host contention and scripts/speed_bar.py cannot see it. "
                     "THIS ROW DEPENDS ON the _calibrate_linear phase-split: before it, 1536 threw "
                     "TT_FATAL Out of Memory on a 2717908992 B DRAM buffer at model.py:570, the RANDOM-operand "
                     "screen's reference output, because calibration held 2|out| + |x| + |w| of scratch alongside "
                     "the model (226 MB per bank against 204 MB free, so not fragmentation). With the screens split "
                     "into two phases neither reference coexists with the other and the same shape calibrates to "
                     "completion. The fix is inert where it is not needed: the 1024 design is BYTE-IDENTICAL "
                     "(md5 443c5ada) across the pre-fix and post-fix code on two different cards, 9 and 11, which is "
                     "the invariant calibration rests on -- it only ever returns a config it proved bitwise equal to "
                     "the default. Structure was SCORED through perf/wh-correctness/check_structure.py, the same "
                     "code release_gate.py's GEOMETRY leg imports: at 1536 the designed 100-residue binder is "
                     "COMPACT, Rg 13.37 A against ~12.66 A expected (its 97-residue core 12.20 A against 12.51 A), "
                     "3 clashes in 12085 atoms, clash_frac 0.00025, well inside the 0.016-0.051 band the rungs below "
                     "carry. The checker's chain-B `fail` is its chain assignment, not the design: the contig writes "
                     "the 428-residue target fragment AND the binder into chain B, so it scores two bodies 134.63 A "
                     "apart as one chain, giving Rg 3.10x and a break at 427->428. Three further gaps (428->429, "
                     "429->430, 430->431) are stray N-terminal binder residues, the same class this row already "
                     "records at 768 and 832. 1024 re-measured on this box scores 0 breaks, step median 3.801 A, "
                     "clash_frac 0.02267. LADDER_TOP because nothing above 1536 has been walked, and 1536 is the "
                     "MGX charter's target, not a wall the model hit. The platform fence is separate and LOWER: "
                     "LIMITS[\"max_residues\"] = 1024 in aiand-bio/japanfold/catalog.py is what a user meets, and no "
                     "engine rung reaches a customer until that moves. The 2026-09-08 GWH02 ladder this row replaces "
                     "is kept below because its bit-exactness and residency arithmetic still hold: "
                     "state/rfd3-swiglu-dram-resident-768-wh-verify.md, its own ladder measured 2026-09-08 on GWH02 "
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
            residues=1536, pass_at=1536, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            counts=DESIGN_TARGET,
            evidence="walked past its old 960 top to 1536 on 2026-09-23 on the whglx Galaxy "
                     "j10glx02 card 31 (ws:mgx-design-ceiling, perf/mgxdesign/walk.py over "
                     "perf/bhdesign/ladder.py, one rung per subprocess through the shipped CLI at "
                     "the platform's own --n_step 400, one design, an 80-residue binder at every "
                     "rung, ONE target for all of them -- perf/bhdesign/targets/big_1831.cif, "
                     "chain A 1008 + chain B 823 -- so a rung differs from its neighbour only in "
                     "size). 1536 and 1280 conditioned target residues both design, 1187.3 s and "
                     "580.7 s of fold time at an AICLK median of 1000 MHz sampled DURING the run, "
                     "and 512 designs on card 4 in 638.9 s. Nothing above 1536 was tried, so this "
                     "is the top of the ladder and not a wall. THE VERDICT IS THE ARTIFACT: each "
                     "rung's CIF carries an 80-residue / 321-atom binder AND conditioned_tokens "
                     "equal to the target asked for, which is what separates a real rung from a "
                     "run that quietly conditioned on the 1008-residue chain-A crop. NO SPEED "
                     "CLAIM comes from these rungs and none should be read into them: j10glx02 "
                     "carried a load average of 563-743 on 64 cores while five MGX rows fanned "
                     "out over 27 chips, and the 512 rung is 9.0x the 70.7 s the same size takes "
                     "on a quiet GWH02 at the same n_step and the same 1000 MHz. Coverage is "
                     "load-insensitive and stands; the timing is an artifact of the host. "
                     "STILL NOTE THE DENOMINATOR: counts=DESIGN_TARGET ignores binder_length "
                     "while the known wall above is on TOKENS -- 1664 (ws:ceiling-pxdesign, one "
                     "1.42 GB pair-transition buffer). 1536 + 80 is 1616, just under it, so a "
                     "1536-residue target with a 160-residue binder is NOT covered by this row. "
                     "The platform caps the sum at 1024 either way, so nothing it dispatches "
                     "today can reach any of this. What this replaced, and why that move was not "
                     "new headroom either: "
                     "perf/pxdesign/targets/laczc_960_b64.yaml -- 960 conditioned target residues "
                     "plus a 64-residue binder, 1024 tokens, the platform's shipped defaults "
                     "(4 designs, n_step 200, seed 42). Walked twice on the serving Galaxy "
                     "UF-EV-A13-GWH02, both times a clean pass with no device refusal: 2026-09-08 "
                     "on 4fbc152f in 210.3 s (ws:japanfold-pxdesign-1024-uncap) and re-measured "
                     "2026-09-19 on 72596df4f in 164 s with fit 0.114-0.136 A and the live "
                     "engine.pin d29d9a823 alongside it (ws:cov-stale-pxdesign-whgalaxy). "
                     "The binders come back geometrically clean -- 64 residues, 257 atoms, zero "
                     "backbone breaks, 100 % of Ca-Ca steps in band at 3.79-3.80 A, zero clashes. "
                     "WHY THIS ROW MOVED, and it is not new headroom: the platform has advertised "
                     "and enforced target+binder <= 1024 since 2026-09-08 (japanfold/limits.py) "
                     "and japanfold/size_evidence.py has recorded 1024 tokens proven since the "
                     "same day, while this row stayed at 768. Because the platform shells out to "
                     "`tt-bio design` and sets no TT_BIO_SIZE_LIMIT, every pxdesign job above 768 "
                     "TARGET residues was accepted by the service and then refused here, measured "
                     "2026-09-19 through the serving engine's own CLI. 768 was a LADDER TOP from "
                     "a fixture that could not reach higher (1DP0 chain A is 1011 residues), not "
                     "a wall. The 2026-08-29 ladder it came from still stands and is the quiet-box "
                     "reference the whglx rungs above are 9x off: 128 aa in 62.0 s, 256 in 50.8 s, "
                     "512 in 70.7 s, 768 in 99.9 s at n_step 400, one design",
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
    "boltz2": {
        "wormhole_b0": Ceiling(
            residues=1920, pass_at=1920, fail_at=2048, binds=MEMORY, mechanism=FRAGMENTATION,
            msa_rows=8192,
            evidence="re-walked 2026-09-23 after the combos merge freed the trunk's staged inputs "
                     "before diffusion (ws:mgx-ceilings, perf/mgxceil, j10glx02 card 16, guard off, "
                     "--host_threads 2, apo CDK2 tiled with 8192 alignment rows). On a2a70b160 1792 "
                     "folds in 813.6 s (pLDDT 0.77, no CA-CA gap over 4.2 A), 1920 in 880.7 s, and "
                     "2048 fails after 1172.9 s in the diffusion cache's token-bias permute "
                     "(tenstorrent.py _populate_diffusion_cache): one 3221225472 B tensor, 256.0 MiB "
                     "per bank against a 246.8 MiB largest block with 562.7 MiB free, so "
                     "fragmentation. On a102dfb5c (bigalloc merged) 1920 folds again in 914.0 s, "
                     "pLDDT 0.77. AICLK median 1000 MHz sampled during every fold. "
                     "Earlier: walked 2026-09-23 on origin/main 8906d35a0, j10glx02 card 31, guard off, "
                     "--host_threads 2, apo CDK2 tiled with 8192 alignment rows "
                     "(ws:mgx-ceilings, perf/mgxceil). 1024 folds in 380.5 s, 1152 in 428.4 s, "
                     "1280 in 456.9 s, 1408 in 556.5 s, 1536 in 595.4 s and 1664 in 678.1 s, "
                     "AICLK median 1000 MHz sampled during every fold. 1792 fails in the "
                     "confidence module's relative-position gather (tenstorrent.py pair_gather "
                     "via RelPosGather), which asks for one 1792 x 1792 x 128 bf16 pair tensor, "
                     "822083584 B or 65.3 MiB per bank, on a chip 91.9 percent full with 83.0 MiB "
                     "free and a 60.1 MiB largest block. The 2026-09-11 ladder on the same box "
                     "found the same two sizes. Until this row it was unmeasured and never "
                     "refused, so 1792 and above was admitted and died on the chip. Re-folded on "
                     "52c621bc2 (narrow-q on by default, card 29): 1664 still folds, 612.3 s at "
                     "1000 MHz. Re-walked after pair-residency: on cec7979b1 1792 still fails "
                     "(card 18, 741.3 s), now earlier, in the diffusion cache "
                     "(tenstorrent.py _populate_diffusion_cache), 2466250752 B or 196.0 MiB per "
                     "bank with 428.6 MiB free and a 178.0 MiB largest block; on 1b423e9e4, the "
                     "same boltz2 code, 1664 folds in 701.8 s (card 16), both at 1000 MHz. "
                     "Re-passed on origin/main 6ea518246 (after sdpa-mask 94e7a002c), j10glx02, "
                     "2026-09-24, guard off, --host_threads 2, AICLK median 1000 MHz sampled during "
                     "the fold: 1920 at 8192 rows PASS in 804.2 s (card 3, pLDDT 0.775, no CA-CA "
                     "break). 2048 fails on the same tree and card after 1007.9 s at the same "
                     "site, the diffusion cache's token-bias staging (tenstorrent.py _stage_token_bias), "
                     "3221225472 B or 256 MiB per bank against a 246.8 MiB largest block with 562.7 MiB free",
        ),
    },
    "esmfold2": {
        "wormhole_b0": Ceiling(
            residues=1664, pass_at=1664, fail_at=1792, binds=MEMORY, mechanism=DRAM,
            msa_rows=16384, ladder_ligand_atoms=0,
            evidence="walked 2026-09-23 on origin/main 1b423e9e4 (esmfold2 folds its MSA, a "
                     "fresh <=1024-row subsample per trunk loop), j10glx02, guard off, "
                     "--host_threads 2, --fast as Wormhole forces, apo CDK2 tiled "
                     "(ws:mgx-ceilings, perf/mgxceil), AICLK median 1000 MHz sampled during "
                     "every fold. Two arms, and the row is the tighter of them, which here is "
                     "the same number. With the 8192-row alignment (the default): 1024 1343.9 s, "
                     "1152 1538.5 s, 1280 1606.2 s, 1408 2163.8 s (card 14), 1664 3100.5 s "
                     "(card 16, mean CA pLDDT 87.2). Single-sequence: 1408 1559.1 s, 1536 "
                     "1842.3 s, 1664 2196.0 s (card 0, pLDDT 88.3). 1792 fails in both, three "
                     "times (MSA card 19 after 817.3 s, single-sequence card 0 after 364.9 s) "
                     "on the same request: the pair FFN output (esmc.py _row_blocked), "
                     "1644167168 B or 130.7 MiB per bank with the chip 99.4 percent full and "
                     "6.4 MiB free, residency rather than one oversized block. The encoder "
                     "sees at most 1024 rows per loop, so the row speaks for any alignment "
                     "depth. The wall is on TOKENS (the 09-09 cocrystal pair showed a ligand "
                     "costs exactly the tokens it adds), so ladder_ligand_atoms=0 counts a "
                     "ligand against the 1664. Was 1024/1056 from the 2026-09-09 GWH02 and "
                     "09-11 j10glx02 single-sequence ladders, walked before pair-residency and "
                     "bigalloc reached this trunk. Re-checked on cc908c377 (diffusion conditioning "
                     "prepared once per fold), card 29: 1792 single-sequence still fails on the "
                     "same request after 255.0 s, but now with 137.7 MiB free per bank and a "
                     "122.2 MiB largest block, so the merge turned a full chip into a "
                     "fragmented one 8.5 MiB short of the wall. Re-walked single-sequence on origin/main "
                     "c5b346679 (embed-scale: the LM now passes SDPA a mask), card 29, "
                     "2026-09-24: 1664 PASS in 2066.3 s, 1792 fails after 317.5 s on the same "
                     "request with the chip 99.4 percent full and 6.4 MiB free, residency again. "
                     "AICLK median 1000 MHz during both. "
                     "WITH AN ALIGNMENT, 2026-09-23 on the j10glx02 Galaxy (ws:mgx-msa-depth, "
                     "perf/mgx_msa_depth, tt-bio 5d1f961c0): 1024 x 8192 rows PASS in 904 s (chip "
                     "20) and 1024 x 16384 rows, upstream's own depth, PASS in 1120 s (chip 14), "
                     "both at AICLK 1000 MHz sampled DURING the fold. Re-measured after the MSA "
                     "encoder took upstream's fresh 1024-row subsample per trunk loop (tt-bio "
                     "2f7e0606c, chip 16): 1024 x 16384 PASS in 1426 s at AICLK 1000 MHz DURING, "
                     "one L1 refusal absorbed; the encoder never holds more than 1024 rows. "
                     "Re-passed on origin/main 6ea518246 (after sdpa-mask 94e7a002c), j10glx02, "
                     "2026-09-24, guard off, --host_threads 2, AICLK median 1000 MHz sampled during "
                     "the fold: 1664 with a 16384-row alignment PASS in 2350.3 s (card 20, msa true, "
                     "pLDDT 0.874) and single-sequence PASS in 1968.2 s (card 3, pLDDT 0.884), with "
                     "the same CA-CA gap count as the earlier passes",
        ),
    },
    "esmfold2-fast": {
        "wormhole_b0": Ceiling(
            residues=1664, pass_at=1664, fail_at=1792, binds=MEMORY, mechanism=DRAM,
            msa_rows=0, ladder_ligand_atoms=0,
            evidence="walked 2026-09-23 on origin/main 1b423e9e4 (esmfold2 MSA and LM-dropout "
                     "merge), j10glx02 card 15, guard off, --host_threads 2, --fast as Wormhole "
                     "forces, apo CDK2 tiled (ws:mgx-ceilings, perf/mgxceil). This checkpoint has "
                     "no MSA encoder, so the rungs are single-sequence whatever the input carries. "
                     "1536 folds in 1149.8 s and 1664 in 1281.0 s, AICLK median 1000 MHz sampled "
                     "during every fold. 1792 fails after 264.6 s in the pair FFN "
                     "(esmc.py _row_blocked): the 1 x 1792 x 1792 x 256 bf16 output is "
                     "1644167168 B, 130.7 MiB per bank, and after the row-blocked retry the chip "
                     "is 99.0 percent full with 10.5 MiB free, so it is residency and not one "
                     "oversized block. A repeat on card 16 fails at the same request with the "
                     "same free bytes, in 308.6 s. The same code before that merge "
                     "(cec7979b1, card 0) "
                     "folded 1024 through 1792 (1792 in 1105.2 s) and failed 1920 on "
                     "fragmentation, so the merge costs this model the 1792 rung. The rungs "
                     "were apo and the wall is on TOKENS (same trunk and tokenisation as "
                     "esmfold2), so ladder_ligand_atoms=0 counts a ligand against the 1664. "
                     "Was 1152/1248 from the 2026-09-09 and 09-11 ladders, walked before "
                     "pair-residency and bigalloc reached this trunk. Re-checked on cc908c377 "
                     "(diffusion conditioning prepared once per fold), card 29: 1792 still fails "
                     "on the same request after 270.2 s, now with 141.8 MiB free per bank but no "
                     "block large enough, fragmentation rather than residency. Re-walked on origin/main "
                     "c5b346679, card 29, 2026-09-23: 1664 PASS in 1214.2 s, 1792 fails after "
                     "328.3 s on the same request with the chip 99.0 percent full and 10.5 MiB "
                     "free, residency again. AICLK median 1000 MHz during both. "
                     "Re-passed on origin/main 6ea518246 (after sdpa-mask 94e7a002c), j10glx02, "
                     "2026-09-24, guard off, --host_threads 2, AICLK median 1000 MHz sampled during "
                     "the fold: 1664 PASS in 1095.2 s (card 20, pLDDT 0.891, no CA-CA break)",
        ),
    },
    "protenix-v1": {
        "wormhole_b0": Ceiling(
            residues=2048, pass_at=2048, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            msa_rows=16384,
            evidence="At 16384 alignment rows, the pool depth tt_bio/protenix_data.py reads as "
                     "upstream does, on origin/main 811ac1316, j10glx02 card 19, 2026-09-24, guard "
                     "off, --host_threads 2 (ws:mgx-ceilings, perf/mgxceil, msa_depth 16384 in "
                     "results.json): 1792 PASS in 965.8 s, 1920 in 1064.7 s and 2048 in 1270.2 s, "
                     "AICLK median 1000 MHz sampled during each fold (93, 103, 123 samples), 4-7 "
                     "refused allocations recovered per fold. 2048 at 16384: pLDDT 48.7, 6 CA-CA gaps "
                     "over 4.2 A, 96.6 percent of CA-CA in band, the same band as at 8192. "
                     "At 8192 rows: 2048 folds on origin/main c5b346679 (mgx-bigalloc parks the DiT pair bias, "
                     "the allocation that failed 2048 below), j10glx02 card 29, 2026-09-24, guard "
                     "off, --host_threads 2, 8192 rows: PASS in 1021.0 s, AICLK median 1000 MHz "
                     "over 97 samples during the fold, 2 refused allocations recovered (the largest "
                     "4294967296 B). pLDDT 45.3, 6 CA-CA gaps over 4.2 A and 7 non-adjacent CA pairs "
                     "under 3 A on the tiled apo fixture, inside the band the published rungs carry "
                     "(1536: 48.2, 0 and 23; 1920: 47.2, 17 and 19). LADDER_TOP because 2048 is "
                     "the top of this row's walk; nothing above it was run. Before bigalloc: "
                     "walked 2026-09-23 on origin/main cec7979b1 (pair-residency merged), "
                     "j10glx02 card 31, guard off, --host_threads 2, apo CDK2 tiled with 8192 "
                     "alignment rows (ws:mgx-ceilings, perf/mgxceil). 1664 folds in 738.6 s, "
                     "1792 in 960.3 s and 1920 in 999.0 s, AICLK median 1000 MHz sampled during "
                     "every fold, at host load 54-86 on 64 cores. 2048 fails after 958.8 s in the "
                     "diffusion DiT block biases (protenix.py _dit_block_biases, tenstorrent.py "
                     "compute_bias), one 536870912 B request, 44740608 B per bank, with 54913440 B "
                     "free and a 23117344 B largest block: fragmentation. The rungs below 1664 "
                     "were walked on 8906d35a0 (1024 281.8 s to 1536 554.0 s, card 29), where "
                     "1664 still failed in the diffusion conditioner's pair transition (1417674752 "
                     "B, more than the chip had free); pair-residency removed that wall. Until "
                     "these rows it was unmeasured and never refused",
        ),
    },
    "nesso1": {
        "wormhole_b0": _unmeasured(
            "no ceiling, and unusually this is a positive result rather than an untested gap: the "
            "range is measured on GWH02 from 128 aa (28.3 s) through 1024 aa (72.0 s), and 1152 "
            "still scores (catalog.py). Nothing in that ladder failed, so there is nothing to "
            "refuse on"),
    },
    "boltzgen": {
        # The only atom-denominated rows in this table. Not a stylistic choice: the wall is in the
        # trunk Pairformer's triangle attention, which is sized by the target's atoms, and atoms per
        # residue vary with composition -- these ladders carry 8.06-8.08 per residue against a
        # synthetic backbone's ~4, so a residue cap read off them would be wrong by a factor of two
        # on the other kind of target. TARGET_ATOMS and scan_boltzgen_target_atoms exist for these
        # two rows and nothing else.
        "blackhole": _unmeasured(
            "no ceiling published on Blackhole, and this row is a RECORD rather than a gap: on qb1 "
            "p150a a 2100-residue / 16948-atom target designs an 80-residue binder in 805.3 s "
            "(task bh-boltzgen-sdpa-circbuf-p3, 2026-09-10), which is 2.1x the 8095 atoms the "
            "first Blackhole pass could reach with the largest single chain on hand; 1831 residues "
            "/ 14786 atoms is 692.0 s, 3.2x faster than the same target on a Galaxy chip. What "
            "keeps 16948 off a published row is the ladder above it, not the unit: those rungs "
            "were read while the trimul in-projection wedged the card on a sub-tile slice above "
            "2048 padded tokens (_TRIMUL_MIN_CHUNK), so a rung that \'ran out its 3000 s budget "
            "with no allocator throw\' was a spinning card and bounds nothing. Publishing 16948 as "
            "a ladder top would refuse everything above it on the strength of a ladder whose top "
            "rungs measured a wedge -- re-walk it post-fix first",
            TARGET_ATOMS),
        "wormhole_b0": Ceiling(
            residues=14786, pass_at=14786, fail_at=None, binds=LADDER_TOP, mechanism=NO_FAILURE,
            counts=TARGET_ATOMS,
            evidence=
                "its own Wormhole Galaxy ladder, walked 2026-09-11 on j10glx02 chips 5-8 (task "
                "wh-seqlen-design-embed, perf/bhdesign/ladder.py: one rung per subprocess through "
                "the shipped CLI at shipped settings, an 80-residue binder at every rung, verdict "
                "read off the design\'s own CIF and not off the exit code). Six rungs, 3225 / 4671 "
                "/ 6180 / 8225 / 10482 / 14786 target atoms, all design and none failed; nothing "
                "above 14786 was tried, so the cap is the top of the ladder and not a wall. "
                "14786 atoms (1831 target residues) takes 2242.9 s on chip 7; 10482 reproduces on "
                "two chips (1336.7 s on 8, 1842.4 s on 6) and so does 8225 (575.7 s on 8, 801.0 s "
                "on 5), so the top rung is one chip and the two below it are two. This SUPERSEDES "
                "the 3158-4651 atom band wh-design-models-l1-budget-and-size-caps recorded: that "
                "band was an L1 wall read at a chunk width this ladder does not use, and 14786 is "
                "3.2x the top of it. The four smallest rungs are reproducible off the fixtures "
                "without a device -- tests/test_size_limits.py holds the sizer to them. "
                "THE ATOM AXIS COVERS A 1536-RESIDUE TARGET, measured rather than inferred from "
                "the 8.08 atoms per residue this fixture carries: 2026-09-23 on j10glx02 card 17 "
                "(ws:mgx-design-ceiling) a 1536-residue crop of big_1831.cif -- 12405 target "
                "atoms, 84 % of this row's top -- designs an 80-residue binder in 2896.1 s at an "
                "AICLK median of 1000 MHz sampled DURING the run, artifact chains A=80 binder + "
                "B=1008 + C=528. That time is not comparable to the 2242.9 s above it: the box "
                "carried a load average of 563-743 on 64 cores. It confirms a cell this row "
                "already covers and does not move the cap. FITTING IS NOT THE SAME AS DESIGNING "
                "WELL, and this row is the atom axis only: the 1536 design's geometry is worse "
                "than the same path's at 512. Both rungs ran `--steps design` off the same crop "
                "with the same 80-residue binder, and the 512 control (card 9, 892.6 s, 1000 MHz "
                "DURING) comes back with 1 marginal contact at 1.999 A, clash_frac 0.00021 and no "
                "`fail`, against 20 heavy-atom clashes, clash_frac 0.0015 -- 7.1x per atom -- a "
                "worst contact of 0.879 A and a `fail` at 1536. Mean confidence is low at BOTH "
                "sizes (0.121 and 0.040), so that half is the partial pipeline rather than the "
                "size. scRMSD through the full pipeline is what settles whether the clashes "
                "matter; until it lands, read this row as a fit, not as a quality claim"),
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
                     ligand_atoms: int = 0, counts: str = RESIDUES, *,
                     fast: bool = False) -> list[str]:
    """Models with a MEASURED ceiling that admits this input, so a refusal can point somewhere
    instead of only saying no.

    Only measured rows, and that keeps the list a promise we can keep: a model whose ladder nobody
    walked might well fold the input, but sending someone to it is advertising an untested size.
    A ligand-bearing input narrows it further, from ``capabilities.CAPABILITY`` rather than a
    second list here: OpenFold3 has plenty of room at these sizes and refuses a ligand by name, so
    naming it would send a cocrystal to a model that cannot take one.

    ``counts`` is the denominator the caller's number is in, and only rows in the same DIMENSION
    can answer (see ``_COUNT_DIMENSION``). Without it an atom-denominated row reads as enormous
    against a residue count and gets offered as the roomy alternative to every refusal.
    """
    arch = arch if arch is not None else current_arch()
    if ligand_atoms > 0:
        from .capabilities import CAPABILITY, HONOURED
        folds_ligand = {m for m, row in CAPABILITY.items() if row.get("ligand") == HONOURED}
    want = _COUNT_DIMENSION[counts]
    out = []
    for name in sorted(CEILINGS):
        if name == exclude or (ligand_atoms > 0 and name not in folds_ligand):
            continue
        c = ceiling(name, arch)
        if _COUNT_DIMENSION[c.counts] != want:
            continue
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
    # The same model in its other weight dtype is the first answer, ahead of any other model: on a
    # weight-bound row (esmc-6b on Wormhole, 1968 in bf16 against 8192 in block-fp8) it is the
    # only way to keep the model the user chose, and the TT_BIO_SIZE_LIMIT=0 escape below runs
    # straight into the allocator.
    sib = CEILINGS.get(model, {}).get(arch, _NO_ROW).fast
    fast_hint = (f" {model} with --fast (block-fp8 weights) is measured to handle "
                 f"{sib.residues} {_COUNT_NAMES[sib.counts]} on {arch}, so rerun with --fast."
                 if not fast and sib is not None
                 and not _verdict(model, sib, residues, ligand_atoms)[0] else "")
    alts = models_accepting(residues, arch, exclude=model, ligand_atoms=ligand_atoms,
                            counts=c.counts, fast=fast)
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
           "the largest size below the first one measured to run for hours instead of minutes"
           if c.binds == RUNTIME else
           "the largest size below the first measured failure")
    raise SizeTooLargeError(
        f"{where} has {had}, and {model} is measured to handle at "
        f"most {limit} on {arch}{depth} -- {top}.{ligand_note}{fast_hint}{hint}"
        f" If you have reason to think this input is roomier than the ladder that set the limit "
        f"(a single-sequence run of an MSA-dependent model is), set TT_BIO_SIZE_LIMIT=0 to run it "
        f"anyway."
    )


# ---------------------------------------------------------------------------------------------
# The size of an input, read off the file. No device, no CCD, no weights.
#
# THE SIZER CONTRACT is ``sizer(text, path=None) -> int``. It was ``sizer(text) -> int``, and that
# was enough while every denominator could be read out of the input text itself. BoltzGen's cannot:
# its ceiling is in ATOMS, and the atoms live in the structure file its ``file:`` entity points at,
# by a path that is relative to the spec. So the sizer needs to know where the spec came from.
#
# `path` is optional and every sizer takes it whether or not it reads it, rather than the table
# carrying a per-model "needs a path" flag and `check_input` having two ways to call a sizer. One
# call shape is one thing that can drift; two is a branch, and the branch would be on the model.
# A sizer called without a path still returns a number where it can and 0 where it cannot.
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


def scan_residues(text: str, path=None) -> int:
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


def scan_rfd3_total(text: str, path=None) -> int:
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


# --- The structure a design spec POINTS at ----------------------------------------------------
# Two of the design models are sized on something no amount of reading the spec text can produce:
# PXDesign conditions on whole chains of a structure file when the chain carries no `crop`, and
# BoltzGen's ceiling is denominated in the target's ATOMS. Both need the file the spec names, which
# is why the sizer contract carries a path. One reader serves both, returning residues AND atoms
# per chain, so the two denominators come off the same parse instead of two copies of it.


def _resolve(name, base: Path) -> Path:
    """A spec-relative file name to a path, the way the engines themselves resolve one.

    Beside the spec if that exists, otherwise against the working directory -- the rule
    ``pxdesign.inputs.read_design_yaml`` already applies, because upstream resolves against the
    cwd while a committed fixture wants to resolve beside itself, and both have to work.
    """
    p = Path(str(name)).expanduser()
    if p.is_absolute():
        return p
    beside = base / p
    return beside if beside.exists() else p


def _structure_text(path: Path) -> str:
    """One structure file as text, gzipped or not. "" if it cannot be read, which refuses nothing.

    Gzip because the repo's own PXDesign fixture is a ``.cif.gz``; a reader that only took plain
    text would size the shipped quick-start at 0 and look like it was working.
    """
    try:
        if path.suffix == ".gz":
            import gzip
            return gzip.decompress(path.read_bytes()).decode("utf-8", "replace")
        return path.read_text(errors="replace")
    except Exception:
        return ""


_HYDROGEN = ("H", "D")


def structure_chains(path) -> dict:
    """``{chain id: (residues, heavy atoms)}`` off one structure file's own ATOM records. {} if none.

    mmCIF or PDB, and no parser dependency: this runs before a device is opened, on a host that may
    have no CCD library and no gemmi, and it must cost milliseconds. The counts it produces are the
    ones the ladders were recorded in -- ``perf/bhdesign/ladder.py:cif_stats`` counts exactly these
    rows, which is what makes the BoltzGen row's atom numbers reproducible from the fixture files.

    Three decisions that are all the conservative one, because over-counting refuses work the chip
    can do:

    *Heavy atoms only*, matching the ``GetAtomicNum() > 1`` filter the featurizers apply when they
    lay one token per atom. A deposited structure with hydrogens would otherwise score about twice
    its token count -- and the repo's PXDesign fixture is exactly that file, 1248 hydrogens kept on
    purpose.

    *The first model only.* An NMR ensemble carries the same atoms 20 times over and folds once.

    *Chain ids are ``label_asym_id``*, the id both consumers name: BoltzGen's ``parse_mmcif`` names
    a chain by its gemmi subchain, which IS the label asym id, and PXDesign's schema says
    ``target.chains.<label_asym_id>`` out loud. A chain id the file does not carry scores 0 and
    refuses nothing; the engine's own parser then rejects it by name, which is the better error.
    """
    text = _structure_text(Path(path))
    if not text:
        return {}
    rows = _cif_atoms(text) if "_atom_site." in text else _pdb_atoms(text)
    per: dict = {}
    for chain, res, element in rows:
        if element.upper() in _HYDROGEN:
            continue
        cell = per.setdefault(chain, [set(), 0])
        cell[0].add(res)
        cell[1] += 1
    return {c: (len(seen), atoms) for c, (seen, atoms) in per.items()}


def _cif_atoms(text: str):
    """``(chain, residue key, element)`` per mmCIF ATOM/HETATM row of the first model.

    The column order is READ, never assumed: biotite writes ``_atom_site.id`` last where the RCSB
    order has it second, so a positional parser reads a residue number as a chain id on one of the
    two families and silently produces a plausible wrong count.
    """
    cols: list = []
    idx: dict = {}
    model = None
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("_atom_site."):
            cols.append(st.split(".", 1)[1].split()[0])
            idx = {}
        elif st.startswith(("ATOM", "HETATM")):
            if not idx:
                idx = {n: cols.index(n) for n in
                       ("label_asym_id", "auth_asym_id", "label_seq_id", "auth_seq_id",
                        "type_symbol", "pdbx_PDB_model_num", "pdbx_PDB_ins_code")
                       if n in cols}
            f = st.split()
            if len(f) != len(cols):
                continue
            if "pdbx_PDB_model_num" in idx:
                m = f[idx["pdbx_PDB_model_num"]]
                if model is None:
                    model = m
                elif m != model:
                    continue
            chain = f[idx["label_asym_id"]] if "label_asym_id" in idx else (
                f[idx["auth_asym_id"]] if "auth_asym_id" in idx else "?")
            # `label_seq_id` is `.` for every non-polymer atom, so a chain of ligands would collapse
            # to one residue on it alone. The author numbering is what separates them.
            seq = f[idx["label_seq_id"]] if "label_seq_id" in idx else "."
            if seq in (".", "?") and "auth_seq_id" in idx:
                seq = f[idx["auth_seq_id"]]
            ins = f[idx["pdbx_PDB_ins_code"]] if "pdbx_PDB_ins_code" in idx else ""
            yield chain, (seq, ins), f[idx["type_symbol"]] if "type_symbol" in idx else ""


def _pdb_atoms(text: str):
    """The same, off PDB fixed columns. Stops at the first ``ENDMDL``: one model is enough."""
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            return
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip()
        if not element:
            # No element column. The atom NAME's first non-digit carries it -- ' HB2' is hydrogen,
            # and a file without element columns is exactly the one whose hydrogens must still drop.
            element = next((ch for ch in line[12:16] if ch.isalpha()), "")
        yield line[21], (line[22:27].strip(), ""), element


def scan_pxdesign_target(text: str, path=None) -> int:
    """Conditioned TARGET residues in one PXDesign target YAML (DESIGN_TARGET). 0 if unsizable.

    Counted from the per-chain ``crop`` ranges where the spec gives them, and off the structure
    file where it does not: a chain with no ``crop`` conditions on the whole chain, and the only
    place its length is written down is ``target.file``. That case used to return 0 and refuse
    nothing, because the sizer had the spec text and no way to reach the file -- with a path in
    hand it is a read rather than a guess, and the permissive hole closes. It stays 0 where the
    file is not on this host or does not carry the chain the spec names. The binder is deliberately
    excluded -- this model's ladder was walked in target residues with the binder held at 80, so
    counting it in would compare against the wrong denominator.
    """
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    target = data.get("target") if isinstance(data.get("target"), dict) else None
    chains = (target or {}).get("chains")
    if not isinstance(chains, dict):
        return 0
    base = Path(path).expanduser().parent if path else Path(".")
    on_disk = None
    total = 0
    for cid, body in chains.items():
        crop = (body or {}).get("crop") if isinstance(body, dict) else None
        if isinstance(crop, str):
            crop = [crop]
        if not isinstance(crop, (list, tuple)):
            if on_disk is None:
                on_disk = structure_chains(_resolve(target.get("file") or "", base))
            whole = on_disk.get(str(cid), (0, 0))[0]
            if not whole:
                return 0      # nothing on disk to read it off, so the spec is still unsizable
            total += whole
            continue
        for rng in crop:
            m = re.match(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$", str(rng))
            if not m:
                return 0
            lo, hi = int(m.group(1)), int(m.group(2))
            total += max(0, hi - lo + 1)
    return total


# A BoltzGen designed chain is given as a LENGTH -- `80`, or `80..120` to sample one per design --
# where a target chain is given as a real sequence or as a structure file. That is what separates
# the two, and it is the only thing that does.
_LENGTH_SPEC = re.compile(r"^\d+(?:\.\.\d+)?$")


def _file_entity_atoms(body, base: Path, hop: bool = True):
    """Heavy atoms one BoltzGen ``file:`` entity contributes. None if it cannot be sized.

    None and 0 are different answers and the caller treats them differently: 0 is "this entity adds
    no atoms", None is "this spec has a target whose size I do not know", which makes the whole
    spec unsizable and refuses nothing.
    """
    if not isinstance(body, dict):
        return None
    raw = body.get("path")
    if isinstance(raw, list) or (isinstance(raw, str) and Path(raw).suffix in (".yaml", ".yml")):
        # A `path:` that names YAML is an indirection to another spec's `file:` body, re-anchored on
        # the directory of the file it names -- one hop, which is exactly what the engine's own
        # `parse_file` does before it falls through. A LIST is a list of such specs that the engine
        # picks one of at random, so the largest is the only safe reading, the same rule
        # `_seq_residues` applies to a `low..high` binder.
        if not hop:
            return None
        best = 0
        for one in (raw if isinstance(raw, list) else [raw]):
            q = _resolve(one, base)
            try:
                import yaml
                nested = (yaml.safe_load(_structure_text(q)) or {}).get("file")
            except Exception:
                return None
            n = _file_entity_atoms(nested, q.parent, hop=False)
            if n is None:
                return None
            best = max(best, n)
        return best
    if not isinstance(raw, str) or not raw.strip():
        return None
    # `exclude` and `include_proximity` REMOVE part of what `include` names, by residue and by
    # distance, and neither can be resolved from chain ids. Counting the named chains whole would
    # over-count, and an over-count refuses work the chip can do, so the spec is simply unsizable.
    if body.get("exclude") or body.get("include_proximity"):
        return None
    chains = structure_chains(_resolve(raw, base))
    if not chains:
        return None
    include = body.get("include", "all")
    if include is None or include == "all":
        return sum(atoms for _, atoms in chains.values())
    if not isinstance(include, list):
        return None
    total = 0
    for item in include:
        chain = (item or {}).get("chain") if isinstance(item, dict) else None
        if not isinstance(chain, dict) or "id" not in chain:
            return None
        # A `smiles:` on an include entry supplies a ligand TEMPLATE for a chain that must already
        # be in the file -- the engine raises if it is not -- so its atoms are counted off the file
        # like every other chain's. Adding the SMILES would count them twice.
        total += chains.get(str(chain["id"]), (0, 0))[1]
    return total


def scan_boltzgen_target_atoms(text: str, path=None) -> int:
    """Heavy atoms in the TARGET of one BoltzGen design spec (TARGET_ATOMS). 0 if unsizable.

    BoltzGen is the only model in this table whose ceiling is not denominated in residues. Its wall
    is in the trunk Pairformer's triangle attention, which is sized by the target's atoms, and atoms
    per residue vary with composition -- a deposited protein carries about 8, a synthetic backbone
    about 4 -- so converting the ladder to residues would be a guess dressed as a measurement.
    Which is why the sizer contract carries a path: the atoms are in the structure file the spec's
    ``file:`` entity points at, by a name relative to the spec, and nothing in the spec text counts
    them.

    The designed binder is deliberately outside the number, exactly as PXDesign's target-only
    denominator is: every rung of the ladder that set the cap held the binder at 80 residues, so
    counting it in would compare against a denominator nobody measured.

    Returns 0 rather than a guess wherever the spec names a target this cannot count -- a chain
    given as a sequence instead of a file, an `exclude` that removes part of one, a structure file
    that is not on this host. A guard that invented a number there would refuse real work, and that
    is worse than not refusing at all.
    """
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    entries = data.get("entities")
    if not isinstance(entries, list):
        return 0
    base = Path(path).expanduser().parent if path else Path(".")
    total = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        for key, body in e.items():
            k = str(key).lower()
            if k == "file":
                n = _file_entity_atoms(body, base)
                if n is None:
                    return 0
                total += n
            elif k in _POLYMER_KEYS:
                seq = (body or {}).get("sequence") if isinstance(body, dict) else None
                if not _LENGTH_SPEC.match(str(seq).strip()):
                    return 0      # a target given as a sequence: no atoms to read, so no refusal
    return total


def scan_longest_sequence(text: str, path=None) -> int:
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
    "boltzgen": (TARGET_ATOMS, scan_boltzgen_target_atoms, _DESIGN_SUFFIXES),
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
            # The path as well as the text: a design spec's target lives in a file it names
            # RELATIVE to itself, so a sizer that only ever saw the text could not reach it.
            n = sizer(q.read_text(), q)
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
