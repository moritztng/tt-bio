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
``ttnn.get_arch_name()``. The measured ceilings below are all Wormhole, from the GWH02 Galaxy pool,
and they do not transfer: a Blackhole p150a has 2.7x the DRAM of a Galaxy chip and a 13x10 grid
against 8x9, and OpenDDE -- capped at 544 on Wormhole -- folded every rung to 1024 aa on a p150a
(``state/sizes-recheck-opendde.md``). So there are no Blackhole rows, and on Blackhole this module
refuses nothing. That is the honest state, not an oversight; a fabricated BH number would be exactly
the failure the arch key exists to prevent.

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
likely true of the other ``dram_msa`` rows and is NOT true of the ``l1_clash`` rows, where the
throw comes from a static circular-buffer layout in the structural refiner and has nothing to do
with alignment depth.

Rather than guess a second set of numbers, the table keeps the measured (MSA-on) ceiling and
``TT_BIO_SIZE_LIMIT=0`` turns any refusal into a warning -- see ``enforced()``. A single-sequence
ladder for the MSA-dependent models would let these become two rows instead of one, and until
somebody walks it, guessing which configuration a user is in would be inventing a ceiling.

THE UNITS ARE RESIDUES, AND THAT IS A CHOICE
--------------------------------------------
Every ceiling below was measured by walking residue counts, so residues is what this table can
honestly express -- the model-internal expansion is already folded into the measured number.
OpenDDE's structural axis is ``Ns = 2*n_res - n_gly``, roughly 1.9x the residue count, and ligand
atoms add tokens on top of the polymer; none of that needs restating here, because a ladder walked
in residues already paid for it. What this table therefore CANNOT see is an input whose token count
is unusual for its residue count -- a short polymer carrying a large ligand. Converting these to
token-denominated ceilings would need the ladders re-walked on the token axis, which nobody has
done; asserting a token cap from a residue ladder would be a units substitution, not a measurement.
Said plainly so the gap is a known one.
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
LADDER_TOP = "ladder_top"  # nothing failed; this is simply the largest size proven
UNMEASURED = "unmeasured"  # nobody walked a ladder. Never refuses.
BINDS = (MEMORY, RUNTIME, LADDER_TOP, UNMEASURED)

# The named failure mechanism, so a row says WHY and not just WHERE. Raising a ceiling means
# attacking one of these, and they do not yield to the same fix -- chunking is the answer for a DRAM
# wall and useless against an L1 clash, where the throw lands at a consumer's program creation with
# DRAM nowhere near full.
L1_CLASH = "l1_clash"              # L1 static circular buffers overlap the tensor. Not monotonic.
DRAM = "dram"                      # a single allocation the chip cannot serve
DRAM_MSA = "dram_msa"              # DRAM, in the MSA track, growing with tokens and depth
FRAGMENTATION = "fragmentation"    # enough free DRAM, no block big enough
NO_FAILURE = "none"                # nothing broke
UNKNOWN = "unknown"                # not diagnosed
MECHANISMS = (L1_CLASH, DRAM, DRAM_MSA, FRAGMENTATION, NO_FAILURE, UNKNOWN)

# WHAT THE NUMBER COUNTS. Not decoration: the two design models were measured in DIFFERENT
# denominators, and holding one against the other would be a silent unit substitution. RFD3's 490 is
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

    @property
    def measured(self) -> bool:
        return self.binds != UNMEASURED


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
        "wormhole_b0": Ceiling(
            residues=544, pass_at=544, fail_at=576, binds=MEMORY, mechanism=L1_CLASH,
            evidence="catalog.py, measured 2026-08-16/17 through the live API on the GWH02 "
                     "Galaxy: 512 and 544 fold, 576 throws an L1 static-CB clash, 608 folds, 640 "
                     "throws. Pass/fail is NOT monotonic in residue count, so 544 is the largest "
                     "size below the FIRST failure and not the largest that folds. The throw's own "
                     "addresses reproduced identically in two worker processes a day apart -- L1 "
                     "buffer at 352256 against a static CB region ending at 382240, 29984 B short "
                     "(opendde-wh-crash-set-cap-nondeterministic)",
        ),
    },
    "opendde-abag": {
        "wormhole_b0": Ceiling(
            residues=544, pass_at=544, fail_at=576, binds=MEMORY, mechanism=L1_CLASH,
            evidence="catalog.py, its OWN ladder measured 2026-08-17, not inherited from opendde: "
                     "512 and 544 fold, 576 throws, 608 and 640 fold. Same first failure, same cap",
        ),
    },
    "openfold3": {
        "wormhole_b0": Ceiling(
            residues=576, pass_at=576, fail_at=614, binds=MEMORY, mechanism=DRAM_MSA, msa_rows=14190,
            evidence="catalog.py, measured 2026-08-17 on GWH02, tt-bio 6329f8ef, real ColabFold "
                     "alignments cut to length. The MSA track runs out of DRAM. Depth was varied "
                     "and the ladder walked at both: at 14190 rows 448/512/544/576 all fold; at "
                     "8138 rows 592 folds and 614 dies asking 2.01 GB against 207 MB free. 576 is "
                     "the largest size proven at the deepest alignment this pipeline has produced. "
                     "Single-sequence is far roomier (768 folds in 301 s) but is not the default",
        ),
    },
    "openbind": {
        "wormhole_b0": Ceiling(
            residues=576, pass_at=576, fail_at=614, binds=MEMORY, mechanism=DRAM_MSA, msa_rows=14190,
            evidence="inherits openfold3's ladder by construction rather than by assumption: same "
                     "OF3Trunk, same MSA track, and openfold3_fold.py asserts the two BUCKET_MULTIPLE "
                     "rows agree. Ligand atoms add tokens on top of the polymer, which this "
                     "residue-denominated cap covers but does not separately measure",
        ),
    },
    "rf3": {
        "wormhole_b0": Ceiling(
            residues=627, pass_at=627, fail_at=630, binds=MEMORY, mechanism=DRAM_MSA, msa_rows=29017,
            evidence="catalog.py: sixteen consecutive sizes fold (375 through 627) and then 630 "
                     "dies, as do 640 twice, 650, 656, 716, 796, 891, 980 and 1095. The wall is "
                     "clean and monotonic here, and it is not a depth artefact -- the deepest "
                     "alignment in the band (612 aa, 29017 rows) folds while the shallower 630 "
                     "(22936 rows) does not. TWO different DRAM failures sit above the cap: 630-656 "
                     "die late on a request as small as 103 MB with DRAM already 99% full (the "
                     "allocation-COUNT shape of of3-1024aa-oom-allocation-count-not-size), while "
                     "from 716 up a single 6.2-8.6 GB buffer is asked of a 12 GB chip. Only the "
                     "lower one yields to depth chunking, so fixing it reaches ~716 and not 1024",
        ),
    },
    "protenix-v2": {
        "wormhole_b0": Ceiling(
            residues=980, pass_at=980, fail_at=1095, binds=MEMORY, mechanism=DRAM,
            evidence="catalog.py, measured 2026-08-11 (tree d0ff69b2, warm MSA, platform flags): "
                     "980 aa folds in 1072 s, 1095 aa OOMs on DRAM. Memory binds the failure, but "
                     "runtime is close behind on the platform, whose budget is 1200 s -- a tt-bio "
                     "CLI user has no such budget and only the OOM applies. 1024 sits in an "
                     "UNTESTED gap between the two and one ladder rung would settle it",
        ),
    },
    "rfd3": {
        "wormhole_b0": Ceiling(
            residues=490, pass_at=490, fail_at=UNRECORDED, binds=MEMORY, mechanism=FRAGMENTATION,
            counts=DESIGN_TOTAL,
            evidence="wh-design-models-l1-budget-and-size-caps, measured on this Galaxy: 390 TARGET "
                     "residues, 490 total including the designed regions, 4373 atoms. The wall is "
                     "fragmentation and not capacity -- 2.04-2.37x the needed space is free while "
                     "the largest block is 0.89-0.93x of the request. Fails hard, not slowly. The "
                     "failing size above 490 was witnessed but never written down, which is why "
                     "this row's negative control is UNRECORDED rather than a number",
        ),
    },
    "pxdesign": {
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
                     "so much higher than any folding model's",
        ),
    },
    # --- No measured ceiling. Never refused. ---------------------------------------------------
    "boltz2": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE)},
    "esmfold2": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE)},
    "esmfold2-fast": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE)},
    "protenix-v1": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE)},
    "nesso1": {
        "wormhole_b0": _unmeasured(
            "no ceiling, and unusually this is a positive result rather than an untested gap: the "
            "range is measured on GWH02 from 128 aa (28.3 s) through 1024 aa (72.0 s), and 1152 "
            "still scores (catalog.py). Nothing in that ladder failed, so there is nothing to "
            "refuse on"),
    },
    "boltzgen": {
        "wormhole_b0": _unmeasured(
            "its measured cap is NOT residue-denominated and so cannot be expressed as a row here: "
            "wh-design-models-l1-budget-and-size-caps puts it between 3158 and 4651 ATOMS, in the "
            "trunk Pairformer's triangle attention, at shipped settings -- the same 8x9 per-core L1 "
            "wall that killed that task's chunk-width lever. Atoms per residue vary with "
            "composition, so converting that to residues would be a guess. Refusing on it needs an "
            "atom-denominated dimension this table does not yet carry"),
    },
    "esmc-300m": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE, MAX_SEQUENCE)},
    "esmc-600m": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE, MAX_SEQUENCE)},
    "saprot-35m": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE, MAX_SEQUENCE)},
    "saprot-650m": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE, MAX_SEQUENCE)},
    "saprot-1.3b": {"wormhole_b0": _unmeasured(_INHERITS_DEMO_FENCE, MAX_SEQUENCE)},
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


def ceiling(model: str, arch: str | None = None) -> Ceiling:
    """The ceiling for `model` on `arch`. Unknown model or unknown arch -> UNMEASURED, never a cap.

    Defaulting an unknown pair to "no limit" rather than to some other model's number is deliberate:
    every way of guessing is a way of refusing a fold the engine can do.
    """
    arch = arch if arch is not None else current_arch()
    if arch is None:
        return _NO_ROW
    return CEILINGS.get(model, {}).get(arch, _NO_ROW)


def models_accepting(residues: int, arch: str | None = None, exclude: str | None = None) -> list[str]:
    """Models with a MEASURED ceiling that admits this many residues, so a refusal can point
    somewhere instead of only saying no.

    Only measured rows, and that keeps the list a promise we can keep: a model whose ladder nobody
    walked might well fold the input, but sending someone to it is advertising an untested size.
    """
    arch = arch if arch is not None else current_arch()
    out = []
    for name in sorted(CEILINGS):
        if name == exclude:
            continue
        c = ceiling(name, arch)
        if c.measured and c.residues is not None and c.residues >= residues:
            out.append(name)
    return out


def check(model: str, residues: int, *, arch: str | None = None, where: str = "This input") -> None:
    """Refuse `residues` on `model` if a MEASURED ceiling says it will not fold. Otherwise silent.

    Raises ``SizeTooLargeError``. The message names the model, the dimension, the value, the limit
    and the architecture, because a refusal that omits the architecture is unactionable -- the same
    input at the same size folds on Blackhole for several of these models.
    """
    arch = arch if arch is not None else current_arch()
    c = ceiling(model, arch)
    if not c.measured or c.residues is None or residues <= c.residues:
        return
    if not enforced():
        import warnings
        warnings.warn(
            f"{where} has {residues} {_COUNT_NAMES[c.counts]}, above {model}'s measured limit of "
            f"{c.residues} on {arch}. TT_BIO_SIZE_LIMIT=0 is set, so this runs anyway and may "
            f"fail on the device.", stacklevel=2)
        return
    alts = models_accepting(residues, arch, exclude=model)
    hint = (f" Models with a measured ceiling above {residues} on this hardware: "
            f"{', '.join(alts)}." if alts else
            " No model has a measured ceiling this high on this hardware; try a smaller "
            "construct or domain.")
    depth = (f" (measured with alignments up to {c.msa_rows} rows)" if c.msa_rows else "")
    top = ("the largest size proven on a ladder that never failed above it"
           if c.binds == LADDER_TOP else
           "the largest size below the first measured failure")
    raise SizeTooLargeError(
        f"{where} has {residues} {_COUNT_NAMES[c.counts]}, and {model} is measured to handle at "
        f"most {c.residues} on {arch}{depth} -- {top}.{hint}"
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


def check_input(data, model: str, *, arch: str | None = None) -> None:
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
    text = str(data).strip()
    p = Path(text).expanduser()
    try:
        exists = p.exists()
    except OSError:      # an over-long "path" that is really a pasted sequence
        exists = False
    if not exists:
        if _BARE_SEQUENCE.match(text):
            check(model, len(text), arch=arch, where="The input sequence")
        return
    files = sorted(q for q in (p.glob("*") if p.is_dir() else [p])
                   if q.suffix.lower() in suffixes)
    for q in files:
        try:
            n = sizer(q.read_text())
        except Exception:
            continue
        if n:
            check(model, n, arch=arch, where=f"'{q.name}'")


def shipped_models() -> set:
    """Every name reachable from a CLI ``--model`` choice, discovered from main.py's own tuples.

    One implementation, in token_axis, which carries the incident that produced the discovery
    rule. This module had a byte-identical copy whose docstring already deferred to that one --
    and a guard against "a model reached the CLI without a row here" is worth less in two
    copies, because fixing the rule in one leaves the other still wrong.
    """
    from tt_bio.token_axis import shipped_models as _shipped
    return _shipped()
