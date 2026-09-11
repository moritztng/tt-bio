#!/usr/bin/env python3
"""Capacity gate: can every shipped model ALLOCATE and COMPLETE at 1536 tokens on this card?

THIS IS NOT A PARITY GATE AND CANNOT SUBSTITUTE FOR ONE. It answers one question -- does the shape
fit and does the pipeline finish -- and says nothing about whether the numbers are right. Its
fixture is a tandem-repeated chain whose structure is meaningless by construction, and it is meant
to run on cards (pc card 0 among them) that are known to miscompute matmuls. Correctness is
scripts/full_parity_gate.py's job, against real targets with cached references. A model that clears
this gate and fails that one has shipped a torn structure, which is worse than an OOM, not better.

WHY IT EXISTS. Three production failures this month were all the same hole: the platform's coverage
matrix sizes each cell at the model's own advertised ceiling, so while the ceilings were 576/576/627
it folded 576 and correctly passed. Raising every ceiling to 1024 moved the gate's target to 1024
and nothing re-ran it, so the failures surfaced in real traffic at 40-50%. Nothing in tt-bio tested
a size axis at all: full_parity_gate.py uses fixed targets, release_gate.py's ladder tops out at 768
single-sequence, and the only max-residues coverage anywhere ran against the live API on Wormhole.

THE TWO FAILURE CLASSES, AND WHY ONE TIER CANNOT SEE BOTH

  Class A, one oversized tensor. Shape-determined, so it appears on the FIRST execution of the op:
  OuterProductMean's 2 GiB single-shot z, the 8790736896 B DRAM buffer behind all three production
  failures, the 1.86 GB [depth, tokens, c_m] that still refuses at 960. One block exposes every one
  of these.

  Class B, cumulative residency and fragmentation. RF3 at 630-656 tokens dies LATE on a request as
  small as 103 MB with DRAM already 99% full: the named allocation is the last straw, not the
  problem. A single block cannot see this, because the residency has not been built up yet.
  OpenFold3's second wall is the same shape from the other side -- an L1 refusal inside the
  diffusion transformer, retried rather than fatal, so it presents as a STALL and not an error. A
  gate that only watches for exceptions scores that as a pass.

So a layer-subset screen is necessary and not sufficient, and this runs both:

  TIER 1, screen. One block per stack (scripts/capacity_hook.py truncates every block list to its
  first element) plus one diffusion step, at the target token count. Seconds per model. A FAIL here
  is definitive: a shape that cannot allocate once cannot allocate ever, so Tier 2 is skipped.
  A PASS here is NOT A VERDICT and is never reported as one -- see _screen.

  TIER 2, residency. The full pipeline at the target size. Minutes per model. Catches Class B and
  the diffusion-side L1 wall. Carries a STALL DETECTOR, because that failure mode is a hang.

EFFICIENCY THAT COSTS NO COVERAGE, all of it taken: a committed MSA fixture instead of an alignment
search (device memory does not care how the alignment was found); target-FIRST rather than
laddering up, so a passing model costs one run and only a failure pays for a bisect;
diffusion_samples=1; and one model per card over --workers.

EFFICIENCY THAT COSTS COVERAGE is a named decision, written into the report, never silent. Every
reduction a run applies lands in report["reductions"] and in the printed summary. Two are hard
rules: the diffusion/structure stage is never skipped (OpenFold3's remaining wall lives inside it),
and MSA depth is never reduced for an MSA model without saying so, because for the OF3 family the
failing tensor scales with tokens x rows -- at 14190 rows OpenFold3 folds 576 and dies at 614,
while single-sequence it folds 768 in 301 s.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import capacity_fixture                                                    # noqa: E402
from tt_bio import device_lease                                             # noqa: E402
from tt_bio import size_limits as sl                                        # noqa: E402

# ---------------------------------------------------------------------------------------------
# THE BAR
# ---------------------------------------------------------------------------------------------
#: The bar is 1536 TOKENS. The token axis buckets to a multiple of 32 and an unmasked tail is a
#: ~72x error, so the number tested has to be the number the hardware sees: 1536 is exactly
#: 48 x 32 and needs no padding at all, where a "1500" bar would pad to 1504 and report a size
#: 4 tokens smaller than what actually ran. Blackhole-only -- 1536 is out of reach on a 12 GiB
#: Wormhole chip for several of these models, so nothing here is a Wormhole statement.
TOKEN_BAR = 1536
TOKEN_BUCKET = 32
assert TOKEN_BAR % TOKEN_BUCKET == 0

#: Bisect rungs, used ONLY after a failure, to report where the real ceiling sits. A passing model
#: never runs any of these. All bucket-aligned for the same reason the bar is.
BISECT_RUNGS = (1408, 1280, 1024, 896, 768, 640, 512)

#: No forward progress for this long is a FAIL, not a wait. OpenFold3's diffusion-side L1 refusal
#: is retried rather than raised and sat for 2289 s at diffusion step 0; a gate that waits for an
#: exception scores that green.
#:
#: Progress is the CLI's own structured event stream (TT_BIO_PROGRESS_CAPTURE, the same stream the
#: UX-regression guard reads), with log growth as a fallback. It is deliberately not CPU time: the
#: OF3 stall is a RETRY loop, so it burns CPU while going nowhere.
STALL_S = 900
#: Before the first progress event, though, silence is normal and long. The JIT kernel cache is
#: keyed by shape, so the first run at a new token count compiles every kernel from scratch with
#: no output at all: measured on esmfold2 at 1504 tokens, 7.5 minutes of CPU and zero log bytes.
#: Applying the post-first-event threshold from t=0 would score that cold compile as a hang.
WARMUP_S = 2700
#: Total wall-clock ceiling per run. A 1536-token deep-MSA fold is minutes, not an hour.
RUN_TIMEOUT_S = 5400
#: Host RAM headroom below which a death is recorded as HOST_OOM and not as a device wall. A
#: 1536-token deep-MSA fold can OOM the HOST, which is a different failure and must not be
#: reported as a capacity ceiling.
HOST_RAM_FLOOR_MB = 700

VERDICTS = ("PASS", "FAIL", "STALL", "HOST_OOM", "NO_WEIGHTS", "CONTENDED", "CARD_DIRTY",
            "ERROR", "SKIPPED", "GATE_BUG")

#: The Tier 1 hook wraps __init__ on every class in every non-vendored tt_bio module, and on one
#: model that wrapping breaks construction outright: boltz2 died with "Boltz2.__init__() missing 5
#: required positional arguments" 8 s in, which the gate scored FAIL -- a capacity verdict invented
#: by the gate's own instrument. Without the hook the same fixture gets 172 s into the model. A
#: model the gate cannot even build has not failed the bar, so this is GATE_BUG and it is loud.
_HOOK_BROKE = re.compile(
    r"__init__\(\) missing \d+ required positional argument"
    r"|__init__\(\) takes \d+ positional argument", re.I)

#: The SECOND way the instrument broke a model, found by fixing the first. The depth cut leaves a
#: strict `load_state_dict` staring at the removed blocks' weights, and `_no_weights` below reads
#: "Unexpected key(s) in state_dict" as an unusable CHECKPOINT -- so a cut the gate made itself
#: came back as a fact about the artifact. `capacity_hook` now drops exactly the indices it
#: removed, but that repair needs an `nn.Module` container to hang off, so the misattribution is
#: also cut off here: an unexpected key naming a block index THIS RUN removed is the gate's bug.
_UNEXPECTED_KEY = re.compile(r"Unexpected key\(s\) in state_dict", re.I)


def screen_reduction_is_unsafe(leg: dict) -> bool:
    """True when a screen FAIL cannot be attributed to the bar, because the screen's own depth cut
    is a candidate cause.

    Tier 1's founding claim was that truncation "can only reduce what runs, so a screen can miss a
    Class B failure but cannot invent one". That is false, and boltz2 disproved it three separate
    ways: the wrapper defeated a signature-filtering loader, the cut left a strict state_dict load
    with orphaned weights, and -- once both were fixed -- `Module.__call__` in tt_bio/tenstorrent.py
    slices a per-layer bias tensor as `z.shape[1] // len(self.layers)`, so cutting 3 layers to 1
    turned a 4-head slice into a 12-head one: "The size of tensor a (4) must match the size of
    tensor b (12)". A block stack's LENGTH can be load-bearing for tensors outside the stack.

    The discriminator is the mechanism. A real capacity failure names one: rf3's is an allocator
    refusal for a single 18530435072 B DRAM buffer. A shape mismatch names none. So a FAIL with no
    capacity mechanism, on a leg where the cut was actually applied, is not scored -- the cell
    falls through to the un-truncated residency run, which reduces nothing and is always a valid
    measurement. It costs one slow run for the model whose fast path does not work, and nothing at
    all for every model whose does.
    """
    return bool(leg.get("verdict") == "FAIL" and not leg.get("mechanism")
                and leg.get("stacks_truncated"))


def _hook_cut_these_weights(tail: str, truncated: list) -> bool:
    """True when the unexpected keys name a block index the screen's own truncation removed.

    Deliberately narrow: it wants the recorded attribute AND an index inside the recorded depth,
    both from this run's own hook record. A checkpoint that is genuinely wrong for the module
    still reports NO_WEIGHTS, because its unexpected keys do not sit under a stack the gate cut.
    """
    if not tail or not truncated or not _UNEXPECTED_KEY.search(tail):
        return False
    for _qualname, attr, n in truncated:
        for i in range(1, min(int(n), 4096)):
            if f".{attr}.{i}." in tail:
                return True
    return False

# ---------------------------------------------------------------------------------------------
# THE MODEL LIST -- DERIVED, NEVER HARDCODED
# ---------------------------------------------------------------------------------------------
# A hand-typed list is exactly how a new port gets shipped with no capacity coverage; that class
# has bitten this repo repeatedly (release_gate.py's SIZE_LADDER tuple, perf_regression.py's SPECS,
# the platform's own model list). So the roster comes from main.py's own `*_MODELS` tuples via
# size_limits.shipped_models(), and `coverage_gaps()` FAILS the gate when a shipped model is
# neither runnable here nor carrying a written exemption. A new model appears automatically.


def _verbs() -> dict[str, str]:
    """model id -> the CLI verb that runs it, read off main.py's tuples."""
    from tt_bio import main as m
    out = {}
    for verb, models in (("predict", m.PREDICT_MODELS), ("embed", m.EMBED_MODELS),
                         ("saprot", m.SAPROT_MODELS), ("design", m.DESIGN_MODELS),
                         ("affinity", m.AFFINITY_MODELS)):
        for name in models:
            out[name] = verb
    return out


#: Models this gate cannot drive, each with the reason written down. A reason is not a pass: it
#: records what is NOT covered so the gap is readable, the same discipline as
#: release_gate.SIZE_LADDER_EXEMPT. Anything here is reported SKIPPED with its reason.
EXEMPT = {
    "boltzgen": "design, not a fold: its input is a target plus a binder spec and its measured "
                "cap is atom-denominated (between 3158 and 4651 atoms in the trunk Pairformer's "
                "triangle attention), which a token bar cannot express. Needs an atom-denominated "
                "cell of its own.",
    "rfd3":     "design, not a fold: sized on DESIGN_TOTAL (motif plus designed) from a contig "
                "spec, so the 1536-token fixture here is not a valid input. Its wall is "
                "fragmentation rather than capacity and wants its own cell.",
    "pxdesign": "design, not a fold: sized on DESIGN_TARGET from a target STRUCTURE, so it needs a "
                "1536-residue PDB rather than a sequence. The shipped ladder's own fixture source "
                "(1DP0 chain A, 1011 residues) cannot reach the bar either.",
    "nesso1":   "affinity, and the roster's only such model: it is handed a protein AND a ligand, "
                "and this gate's fixture is polymer-only by construction, so its batch builder "
                "refuses the input outright with 'No protein or ligand tokens found in the batch'. "
                "Measured, not assumed: the weights load (539 tensors) and the un-truncated "
                "residency leg still gets no valid input, so no size was tested at any bar. Needs "
                "a ligand-bearing cell, which is a fixture change. p1 recorded NO_WEIGHTS here "
                "and blamed the checkpoint; that was the gate's own depth cut, not the artifact.",
    "saprot-1.3b": "structure-aware embeddings, and the checkpoint is not in the local weights "
                   "cache on this host. saprot-35m and saprot-650m carry the same code path at "
                   "the bar; this one is a weights gap, not a code gap.",
}


def roster() -> list[str]:
    return sorted(sl.shipped_models())


def coverage_gaps() -> list[str]:
    """Shipped models that are neither runnable by this gate nor exempted in writing."""
    verbs = _verbs()
    return sorted(m for m in roster()
                  if m not in EXEMPT and verbs.get(m) not in ("predict", "embed", "saprot",
                                                              "affinity"))


def runnable() -> list[str]:
    """The models this gate is expected to hold a measured cell for: the roster, less the written
    exemptions, less anything it has no verb to run."""
    gaps = set(coverage_gaps())
    return [m for m in roster() if m not in EXEMPT and m not in gaps]


def baseline_gaps() -> list[str]:
    """Runnable models with no recorded cell in `docs/capacity_gate_baseline.json`.

    The roster guard next door asks whether a model is COVERED by the gate. This asks whether the
    coverage actually left a record behind, which is a different question and was the one nobody
    was asking: the baseline sat at 8 of 16 cells through two passes and every test stayed green.
    boltz2's PASS at 1536 was measured, written up in prose, and never recorded -- and its report
    lived in gitignored scratch inside a worktree fleet hygiene later removed, so the claim
    outlived its evidence. A verdict not folded into the baseline when it is measured is lost.

    A cell the card never produced counts as a gap rather than as coverage: see `nothing_ran`.
    Absence is reported loudly here, which is exactly why it beats a cell nothing downstream can
    tell apart from a measured one.
    """
    per_card = read_baseline()
    if not per_card:
        return runnable()
    cells_of = {card: (blk.get("cells") or {}) for card, blk in per_card.items()}
    return [f"{card}/{m}" for card in sorted(cells_of)
            for m in runnable()
            if m not in cells_of[card] or nothing_ran(cells_of[card][m])]


def baseline_stale() -> list[str]:
    """Recorded cells measured against a DIFFERENT ceiling table than the one this tree ships.

    Per cell, not per file. The file-level stamp this replaces was a false green waiting to
    happen: `record_baseline` writes the CURRENT fingerprint whether the run measured one model
    or twenty, so one `--models boltz2 --record` would have re-certified twelve cells nobody had
    re-measured and turned the trigger green. And those twelve really were another engine's
    numbers -- the ceilings that moved on 2026-09-07 arrived with 800 changed lines of
    `tt_bio/tenstorrent.py`, the RF3 triangle-attention rewrite and OpenFold3's MSA embedder, all
    of which move DRAM at 1536 tokens.

    Naming the stale ones is also what makes the sweep resumable. It is hours of card time, it has
    to run in stages, and a stage that re-measured four models should be able to show it.
    """
    fp = ceilings_fingerprint()
    return sorted(f"{card}/{m}"
                  for card, blk in read_baseline().items()
                  for m, c in (blk.get("cells") or {}).items()
                  if (c or {}).get("ceilings_fingerprint") != fp)


# ---------------------------------------------------------------------------------------------
# PER-MODEL CELLS -- THE BAR IS IN TOKENS, RESIDUES ARE DERIVED
# ---------------------------------------------------------------------------------------------
# Ligand atoms are tokens on top of the residue count, so OpenBind at 1536 RESIDUES is more than
# 1536 tokens and a residue-defined bar silently under-tests exactly the model that failed hardest
# in production. The bar is therefore in tokens and each cell derives its own residue count:
#
#     residues = TOKEN_BAR - ligand_tokens
#
# The gate's fixture is polymer-only and every residue in it is a standard amino acid, so for a
# ligand-free cell tokens == residues exactly (AF3-style tokenisation gives one token per standard
# residue). That is a property of THIS fixture, stated so the next person does not generalise it:
# a cell that adds a ligand must set ligand_tokens, and the ligand-token path is otherwise
# UNCOVERED here (named in report["reductions"]).


class Cell:
    """One model's capacity cell: how to drive it, and at what size and depth."""

    def __init__(self, model, verb, *, ligand_tokens=0, depth=None, recycling=None,
                 msa=True, reason_depth=None):
        self.model, self.verb = model, verb
        self.ligand_tokens = ligand_tokens
        self.depth = depth                 # None -> the committed source's full depth
        self.recycling = recycling         # None -> the model's production default
        self.msa = msa
        self.reason_depth = reason_depth

    def residues(self, tokens: int) -> int:
        return tokens - self.ligand_tokens

    def padded(self, tokens: int) -> int:
        """What the hardware actually sees, after this model's own token bucketing."""
        try:
            from tt_bio import token_axis
            return int(token_axis.bucketed_width(tokens, token_axis.bucket_multiple(self.model)))
        except Exception:
            return tokens


def cells(models: list[str], *, depth=None, recycling=None) -> list[Cell]:
    """Build a cell per model. `depth`/`recycling` are run-wide named reductions."""
    verbs, out = _verbs(), []
    from tt_bio.main import MSA_DEFAULT_MODELS
    for m in models:
        verb = verbs.get(m)
        # embed/saprot take a bare sequence and have no MSA track at all, so a depth reduction is
        # not a reduction for them and recycling does not exist.
        msa = verb == "predict" and m in MSA_DEFAULT_MODELS
        out.append(Cell(m, verb, depth=depth if msa else None,
                        recycling=recycling if verb == "predict" else None, msa=msa))
    return out


# ---------------------------------------------------------------------------------------------
# BOARD AND BANK GEOMETRY -- READ OUT OF THE ALLOCATOR, NOT A SPEC SHEET
# ---------------------------------------------------------------------------------------------
# "Blackhole" is not one board. p150a (pc, qb1) and p300c (qb2) differ, and an interleaved
# allocation is refused on BANK SIZE and not total capacity. Measured on pc's p150a: 8 DRAM banks
# of 4278190016 B, 31.875 GiB, against Wormhole's 12 x ~1 GiB. L1 bank count is host-specific too
# -- pc runs custom 130-core firmware where a stock part presents 140 -- so both are read from the
# live allocator. Without this the numbers are not comparable across hosts.

_GEOM_PROBE = r"""
import json, ttnn
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
d = ttnn.open_device(device_id=0)
o = {"arch": ttnn.get_arch_name(),
     "grid": str(d.compute_with_storage_grid_size())}
for tag, bt in (("dram", ttnn.BufferType.DRAM), ("l1", ttnn.BufferType.L1)):
    v = ttnn.get_memory_view(d, bt)
    o[tag] = {"banks": int(v.num_banks),
              "bytes_per_bank": int(v.total_bytes_per_bank),
              "largest_contiguous_free_per_bank":
                  int(v.largest_contiguous_bytes_free_per_bank)}
    o[tag]["total_bytes"] = o[tag]["banks"] * o[tag]["bytes_per_bank"]
ttnn.close_device(d)
print("CAPGATE_GEOM " + json.dumps(o))
"""


def board_type() -> str | None:
    """The canonical board_type, from tt-smi. p150a and p300c are both 'blackhole' to ttnn."""
    for exe in (Path.home() / ".local/bin/tt-smi", Path("/usr/local/bin/tt-smi")):
        if not exe.exists():
            continue
        try:
            r = subprocess.run([str(exe), "-s"], capture_output=True, text=True, timeout=60)
            info = json.loads(r.stdout)["device_info"]
            return info[0].get("board_info", {}).get("board_type")
        except Exception:
            continue
    return None


def geometry(worker) -> dict:
    """Open the card once, read arch + bank geometry off the allocator, close it."""
    rc, out, _ = worker.run([sys.executable, "-c", _GEOM_PROBE], timeout=600)
    for line in (out or "").splitlines():
        if line.startswith("CAPGATE_GEOM "):
            g = json.loads(line[len("CAPGATE_GEOM "):])
            g["board_type"] = board_type() if worker.is_local else None
            g["host"] = worker.host
            g["card"] = worker.card
            return g
    return {"error": f"geometry probe failed (rc={rc})", "host": worker.host, "card": worker.card}


# ---------------------------------------------------------------------------------------------
# RUNNING ONE CELL
# ---------------------------------------------------------------------------------------------
# The gate drives the real CLI. A capacity check that reimplemented the load-and-fold path would
# be testing its own reimplementation, and the failures it exists to catch live in the production
# path's allocation ORDER (ttnn.slice is not a view, and allocation order decides whether a block
# big enough exists).

#: Cheap host-side signals that a fold hit the device wall rather than anything else. Matched
#: against the run log so a verdict names its own mechanism instead of just "exited nonzero".
#: An allocator refusal is not in here on purpose: it is classified from its own figures by
#: `size_limits.classify_device_oom`, and a substring list cannot do that job. Every refusal
#: message carries the word DRAM (or L1) AND the phrase "largest free block" in one sentence, so
#: first-match-wins made whichever arm was listed first swallow the others: the "fragmentation" arm
#: that used to sit here was unreachable for any real log, and a fragmented chip was recorded as
#: plain "dram". Those are different walls with different fixes and `size_limits` gives them two
#: mechanism values for that reason.
#:
#: The mechanisms that mean THE ALLOCATOR REFUSED. A capacity ceiling is a statement about these
#: and nothing else: every other way a run can die is a statement about the model, the fixture or
#: the gate's own instrument.
ALLOC_MECHANISMS = ("dram", "l1")

MECHANISM_PATTERNS = (
    # Statically allocated circular buffers live in L1, not DRAM, and tt-metal words this
    # "grow to N B which is BEYOND max L1 size of M B" -- so the old pattern (".*exceed",
    # labelled "dram") could never match the message it was written for, and would have named the
    # wrong memory if it had. Quoted from a real nesso1 leg on a p150a. It carries no bank figures,
    # which is why it stays a pattern.
    (sl.L1_CLASH,   re.compile(r"circular buffers.*(?:beyond|exceed).*L1 size", re.I)),
    ("oom",         re.compile(r"Out of Memory|bad_alloc|std::bad_alloc", re.I)),
    # Not a capacity result at all: another process holds the card. Scoring this as FAIL would
    # publish a ceiling that was never measured -- and it is easy to hit, because a killed leg
    # whose spawned fold worker outlived the kill keeps the lease.
    ("contention",  re.compile(r"DeviceInUseError|device contention, nothing ran"
                               r"|is in use by", re.I)),
    # Same family as contention and the same consequence: no model code ran. But the aftermath of
    # a killed leg does not always announce itself as a busy device. On qb2's p300c it comes back
    # out of ttnn.open_device as a firmware-init throw (risc_firmware_initializer.cpp:1115,
    # "failed to initialize FW! Try resetting the board") or as a sysmem pin throw
    # (silicon_sysmem_manager.cpp:326, pin_or_map_iommu). Matching the PHASE rather than the error
    # is the whole point: tt_bio's worker prints this banner and re-raises whenever get_device()
    # fails, so the next open-time error nobody has seen yet lands on the arm that already exists.
    # It sits AFTER the allocator rows deliberately. If the allocator spoke in this log, that is
    # the stronger statement about the bar and it wins.
    ("card",          re.compile(r"device open failed", re.I)),
    # The ENGINE declining the size, not the hardware failing to hold it. tt_bio.size_limits
    # raises SizeTooLargeError when a request is above the model's measured ceiling for this
    # arch, and once CEILINGS grew blackhole rows (2026-09-10) that became reachable at this
    # gate's own 1536 bar: opendde and opendde-abag are capped at 1024 on blackhole because
    # 1536 was measured to FREEZE the trunk, which costs the card and the next job on it.
    # Classified LAST so a real allocator refusal above still wins, and classified at all so
    # the cell says which kind of wall it hit -- "FAIL, mechanism None" is the ambiguity that
    # let a broken bisect publish "the ceiling is below 512 tokens". A named mechanism also
    # keeps screen_reduction_is_unsafe from discarding these screens: a guard refusal is
    # decided before any block runs, so the depth cut cannot be its cause, and the refusals
    # really do bound the walk.
    ("size_guard",  re.compile(r"SizeTooLargeError|is measured to handle at most", re.I)),
)

#: Mechanisms that mean NO MODEL CODE RAN, and the verdict each one reports. Every retry and
#: recovery site reads this instead of comparing against one verdict string. A second string
#: sprayed across four call sites is exactly how "the card was gone" came to be scored as "the
#: model failed the bar" for three cells on qb2's p300c on 2026-09-10.
NOTHING_RAN = {"contention": "CONTENDED", "card": "CARD_DIRTY"}
NOTHING_RAN_VERDICTS = frozenset(NOTHING_RAN.values())


def nothing_ran(cell: dict | None) -> bool:
    """True when a recorded cell is one the card never produced: no model code ran at all.

    Finding 9 classified these legs; this is the same rule applied to the file they end up in.
    A CARD_DIRTY cell is not a weaker capacity result, it is the absence of one, so it must not
    read as coverage. It did: `baseline_gaps` asked only whether a cell was present, and
    `record_baseline`'s guard refused to overwrite a real cell with a non-measurement but
    happily created one where no cell existed yet -- which is exactly the case a re-measure of a
    dropped cell lands in. Recording opendde on 2026-09-10 took that branch and published
    CARD_DIRTY, and the coverage check went green on a cell whose leg died at firmware init.

    Scoped to NOTHING_RAN rather than to UNDECIDED on purpose. An INCONCLUSIVE screen ran the
    model and is a legitimate thing to record; it is only barred from overwriting a stronger
    verdict, which `_would_lose_evidence` already handles and still does.
    """
    return bool(cell) and cell.get("verdict") in NOTHING_RAN_VERDICTS

#: Why the cell decided nothing, in the words of whatever stopped it.
NOTHING_RAN_REASON = {
    "CONTENDED": "another process held the card; nothing was measured",
    "CARD_DIRTY": "the worker never opened the device, so no model code ran and this size was "
                  "not measured. Reset the card (tt-smi -r) and re-run this cell.",
}


def classify(log_text: str) -> str | None:
    """The mechanism this log names, in `size_limits.MECHANISMS`' own words.

    The allocator's numbers first, because they are the only thing that separates a full chip from
    a fragmented one, and the engine already reads them for the sentence it shows a user. Reusing
    that means a cell recorded here and a CEILINGS row published from it cannot name the wall
    differently.
    """
    return sl.classify_device_oom(log_text) or next(
        (name for name, pat in MECHANISM_PATTERNS if pat.search(log_text)), None)


#: A thrown error, at the start of a line: python's own, or tt-metal's macros.
_FIRST_THROW = re.compile(r"^(?:\w*(?:Error|Exception):\s*\S.*|TT_(?:THROW|FATAL)\b.*)$")


def first_error(log_text: str) -> str | None:
    """The FIRST error the log threw, which for a leg that died in a spawned worker is the only
    line that says why.

    The 25-line tail cannot hold it. When the fold worker dies, tt-bio's outer error is
    "every local worker exited before the run finished ... The worker's own traceback above says
    why" -- and the outer click traceback is what fills those 25 lines, so `above` is exactly what
    gets dropped. Finding 9 cost seven leg logs read off disk to recover a line the report had
    already seen and discarded, and the work dir is scratch that hygiene eventually deletes.
    """
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if not _FIRST_THROW.match(line.strip()):
            continue
        out = line.strip()
        # tt-metal prints the human sentence on the line after a bare `info:`.
        window = lines[i + 1:i + 4]
        for j, nxt in enumerate(window):
            if nxt.strip() == "info:" and j + 1 < len(window):
                out += " | " + window[j + 1].strip()
                break
        return out[:400]
    return None


def tree_cpu_s(pid: int) -> float:
    """CPU seconds burned by a process and its children, from /proc. Used only to ANNOTATE a
    stall, never to decide one: the OF3 failure this gate watches for is a retry loop, so it burns
    CPU going nowhere, and treating CPU as progress would hide exactly that case. But a stall that
    was idle throughout and one that was computing throughout are different findings, and the
    report should say which it saw."""
    total, seen = 0.0, set()
    stack = [pid]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            f = Path(f"/proc/{p}/stat").read_text().rsplit(") ", 1)[1].split()
            total += (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
            stack += [int(c) for c in
                      Path(f"/proc/{p}/task/{p}/children").read_text().split()]
        except (OSError, IndexError, ValueError):
            continue
    return total


def host_ram_free_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


class Worker:
    """One host:card slot. Mirrors scripts/full_parity_gate.py's vetted worker, whose --workers
    round-robin is exactly right for one-model-per-card, without importing 2500 lines of
    correctness harness to get it."""

    def __init__(self, host, card, is_local, remote_cwd=None, remote_python=None):
        self.host, self.card, self.is_local = host, card, is_local
        self.remote_cwd, self.remote_python = remote_cwd, remote_python

    def __repr__(self):
        return f"{self.host}:{self.card}"

    def env(self, extra: dict | None = None) -> dict:
        # TT_BIO_LEASE_CARDS is the card grant and tt-bio enforces it at the device open: an
        # UNPINNED open is refused too, because a process that can see four cards brings up all
        # four (UMD starts every visible chip, not just the one it computes on).
        e = {
            "TT_VISIBLE_DEVICES": str(self.card),
            "TT_BIO_LEASE_CARDS": str(self.card),
            # Inherit the fleet's holder identity when there is one: it is what the dispatcher's
            # running-task check reads, and overwriting it with a bare gate pid makes a live task
            # look idle. Only name ourselves when nothing else has.
            "TT_BIO_LEASE_HOLDER": os.environ.get("TT_BIO_LEASE_HOLDER")
                                   or f"capacity_gate:{os.getpid()}",
            # The env has tt_bio installed EDITABLE against the shared checkout, and running a
            # script puts scripts/ on sys.path[0] with cwd absent -- so `import tt_bio` silently
            # loads the stale shared tree. Every leg of this gate must score THIS worktree.
            "PYTHONPATH": str(REPO_ROOT),
        }
        e.update(extra or {})
        return e

    def cmd(self, argv: list[str], extra_env: dict | None = None) -> list[str]:
        env = self.env(extra_env)
        assign = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        work = self.remote_cwd or str(REPO_ROOT)
        if self.is_local:
            return ["sh", "-c", f"{assign} exec " + " ".join(shlex.quote(c) for c in argv)]
        remote = f"cd {shlex.quote(work)} && {assign} exec " + " ".join(
            shlex.quote(c) for c in argv)
        return ["ssh", "-o", "ConnectTimeout=5", self.host, remote]

    def run(self, argv, *, timeout, extra_env=None):
        p = subprocess.run(self.cmd(argv, extra_env), capture_output=True, text=True,
                           cwd=REPO_ROOT, timeout=timeout)
        return p.returncode, p.stdout, p.stderr

    def popen(self, argv, extra_env=None):
        """Same command, but streaming, so a caller can time what the child reaches and when.

        stderr is merged into stdout rather than captured separately: a ttnn process writes ~50
        lines of driver log per device open, and a pipe nobody drains fills up and blocks the very
        child whose progress is being timed.
        """
        return subprocess.Popen(self.cmd(argv, extra_env), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=REPO_ROOT)


HOST_ALIASES = {"qb1": "tt-quietbox", "qb2": "tt-quietbox2"}


def local_host() -> str:
    return (os.environ.get("HOSTNAME") or socket.gethostname()).split(".")[0]


def _is_local(host: str, this_host: str) -> bool:
    if host in ("localhost", "127.0.0.1", this_host):
        return True
    return HOST_ALIASES.get(host) == this_host or HOST_ALIASES.get(this_host) == host


def parse_workers(spec: str) -> list[Worker]:
    """'--workers host:card[:remote_cwd[:remote_python]][,...]'. A name can lie in both
    directions: an alias that exists only in one user's ssh config resolves nowhere else, and one
    that resolves to THIS box silently doubles a card's load instead of fanning out."""
    out, this = [], local_host()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        host, _, rest = part.partition(":")
        card, _, rest2 = rest.partition(":")
        cwd, _, py = rest2.partition(":")
        out.append(Worker(host, int(card or 0), _is_local(host, this), cwd or None, py or None))
    return out or [Worker(this, 0, True)]


# ---------------------------------------------------------------------------------------------
# THE SITECUSTOMIZE BOOTSTRAP
# ---------------------------------------------------------------------------------------------


#: Process groups of the legs running right now. A leg's fold gets its own session
#: (start_new_session=True below), which is what stops `killpg` from taking the gate down with
#: it -- and which also means a signal sent to the GATE never reaches the fold. Python runs no
#: `finally` on a default-handled SIGTERM, so the per-leg teardown was skipped whenever a wrapper
#: timed the gate out, and the fold survived with PPID 1, holding a card. Measured 2026-09-10:
#: `timeout 2100` on a ten-model sweep left an opendde 1536 screen on a card for 17 idle minutes,
#: and SIGKILLing it by hand left the chip needing a `tt-smi -r` before anything else would
#: dispatch. Module level and lock-guarded because the sweep runs one thread per card, so the
#: thread that owns a leg is not the thread a signal arrives on.
_LIVE_LEGS: set[int] = set()
_LIVE_LEGS_LOCK = threading.Lock()


def _track_leg(pgid: int) -> None:
    with _LIVE_LEGS_LOCK:
        _LIVE_LEGS.add(pgid)


def _untrack_leg(pgid: int) -> None:
    with _LIVE_LEGS_LOCK:
        _LIVE_LEGS.discard(pgid)


def reap_live_legs() -> list[int]:
    """SIGKILL every leg still running, from any thread. Returns the groups it signalled."""
    with _LIVE_LEGS_LOCK:
        pgids = sorted(_LIVE_LEGS)
        _LIVE_LEGS.clear()
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return pgids


def install_teardown(*, exit_now=None) -> None:
    """Make a SIGTERM or SIGHUP to the gate reach the fold it is running.

    `os._exit` after reaping rather than an orderly shutdown, on purpose: a raised exception
    only unwinds the thread the signal landed on, which is the one thread NOT running a leg.
    Nothing is lost by exiting hard -- the report is written after every cell, so a killed sweep
    keeps its finished cells and `--record-from` folds them in without touching a card.
    """
    exit_now = exit_now or os._exit
    atexit.register(reap_live_legs)

    def bail(signum, _frame):
        left = reap_live_legs()
        if left:
            print(f"\nsignal {signum}: killed {len(left)} leg(s) still on a card "
                  f"({', '.join(map(str, left))}). Finished cells are in the report; fold them "
                  f"in with --record-from.", flush=True)
        exit_now(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        try:
            signal.signal(sig, bail)
        except (ValueError, OSError):
            pass          # not the main thread, or no such signal here


def hook_dir(work: Path) -> Path:
    """Scratch dir holding the sitecustomize that arms the hook, and a copy of the hook itself.

    `predict` folds in a SPAWNED worker, so a patch installed in the launcher would never reach
    the process that opens the device. CPython imports `sitecustomize` in every interpreter it
    starts, and PYTHONPATH is inherited across spawn, so this reaches all of them.

    The hook is COPIED here rather than imported from scripts/, because it has to be importable
    from a directory the child is guaranteed to have. It was not: the child's PYTHONPATH carried
    the repo root and this dir but never repo/scripts, so `import capacity_hook` raised
    ModuleNotFoundError, sitecustomize's bare `except` swallowed it, and every "screen" silently
    ran at full depth for a whole campaign. Which is why the failure is now RECORDED and the gate
    refuses to call a leg a screen unless the hook says it installed.
    """
    d = work / "_hook"
    d.mkdir(parents=True, exist_ok=True)
    (d / "capacity_hook.py").write_text((Path(__file__).parent / "capacity_hook.py").read_text())
    (d / "sitecustomize.py").write_text(
        "# Generated by scripts/capacity_gate.py. Inert unless TT_BIO_CAPACITY_HOOK is set.\n"
        "import os\n"
        "if os.environ.get('TT_BIO_CAPACITY_HOOK'):\n"
        "    try:\n"
        "        import capacity_hook\n"
        "        capacity_hook.install()\n"
        "    except BaseException as exc:\n"
        "        # NEVER silent: a hook that did not install turns a screen into a full-depth run\n"
        "        # reported as a screen, and a residency peak into a number nobody measured.\n"
        "        out = os.environ.get('TT_BIO_CAPACITY_HOOK_OUT')\n"
        "        if out:\n"
        "            try:\n"
        "                with open('%s.install-failed.%d' % (out, os.getpid()), 'w') as fp:\n"
        "                    fp.write('%s: %s' % (type(exc).__name__, exc))\n"
        "            except OSError:\n"
        "                pass\n")
    return d


def hook_findings(out_prefix: Path) -> dict:
    """Merge what the hook reported from every interpreter in the run. The launcher patches and
    folds nothing, so its empty result must not overwrite the worker's real one."""
    failed = sorted(out_prefix.parent.glob(out_prefix.name + ".install-failed.*"))
    if failed:
        return {"install_failed": failed[0].read_text()[:200], "truncated": [],
                "instrumented": []}
    best = {}
    for p in sorted(out_prefix.parent.glob(out_prefix.name + ".*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        score = len(d.get("truncated", [])) + len(d.get("instrumented", []))
        if score >= len(best.get("truncated", [])) + len(best.get("instrumented", [])):
            if score or not best:
                best = d
    return best


# ---------------------------------------------------------------------------------------------
# ONE RUN
# ---------------------------------------------------------------------------------------------


def build_argv(cell: Cell, fixture: dict, out_dir: Path, *, tier: str) -> list[str]:
    """The CLI invocation for one cell. `tier` picks screen depth from residency depth."""
    py = [sys.executable, "-m", "tt_bio.main"]
    if cell.verb in ("embed", "saprot"):
        # No MSA track and no diffusion stage: one pass at the bar is the whole test. These verbs
        # want a FASTA, not predict's `sequences:` document -- handing them the latter parses to
        # nothing and the cell would pass having embedded zero residues.
        return py + [cell.verb, str(fixture["fasta"]), "--model", cell.model,
                     "--out_dir", str(out_dir)]
    if cell.verb == "affinity":
        return py + ["affinity", str(fixture["yaml"]), "--model", cell.model,
                     "--recycling_steps", "1" if tier == "screen" else "5",
                     "--out_dir", str(out_dir)]
    argv = py + ["predict", str(fixture["yaml"]), "--model", cell.model,
                 "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(out_dir)]
    if cell.msa:
        argv += ["--msa_dir", str(fixture["msa_dir"]), "--msa_cache_only"]
    else:
        argv += ["--single_sequence"]
    if tier == "screen":
        # One diffusion step and one recycle. The stage is NEVER skipped -- OpenFold3's remaining
        # wall lives inside it -- only shortened, and the shapes it allocates are step-independent.
        argv += ["--sampling_steps", "1", "--recycling_steps", "1"]
    elif cell.recycling is not None:
        argv += ["--recycling_steps", str(cell.recycling)]
    return argv


def execute(worker: Worker, argv: list[str], log: Path, *, mode: str,
            hook_out: Path, hookdir: Path, out_dir: Path,
            timeout=RUN_TIMEOUT_S, stall_s=STALL_S) -> dict:
    """Run one leg with a stall detector and a host-RAM watch.

    The stall detector is the reason this is not a plain subprocess.run: OpenFold3's diffusion-side
    L1 refusal is RETRIED rather than raised, so the process stays alive and busy and produces no
    output. Watching for an exception scores that as a pass; watching for forward progress does not.
    """
    events = log.with_suffix(".events.jsonl")
    events.unlink(missing_ok=True)
    beat = log.with_suffix(".beat")
    for old in beat.parent.glob(beat.name + ".*"):
        old.unlink(missing_ok=True)
    # THE OUTPUT DIRECTORY AND THE HOOK FILES TOO, and this one is a false green, not tidiness.
    # `tt_bio.main predict` skips a target whose results already exist, so a second leg at the same
    # (model, tokens) against a warm work dir returns rc=0 in seconds having folded nothing, prints
    # "All predictions complete", and `hook_findings` globs `<prefix>.*.json` and hands back the
    # PREVIOUS run's DRAM peak. Measured 2026-09-08 re-running the boltz2 cell: PASS in 2.6 s at
    # 1536 tokens carrying the earlier run's 5.78 GiB. With --record that banks a capacity cell
    # nobody measured. Every artifact this leg reads must be this leg's own.
    shutil.rmtree(out_dir, ignore_errors=True)
    for old in hook_out.parent.glob(hook_out.name + ".*"):
        old.unlink(missing_ok=True)
    env = {"TT_BIO_CAPACITY_HOOK": mode,
           # repo/scripts too: the gate's own sys.path tweak does not reach a spawned child.
           "TT_BIO_CAPACITY_HOOK_OUT": str(hook_out),
           "TT_BIO_CAPACITY_HOOK_BEAT": str(beat),
           "TT_BIO_PROGRESS_CAPTURE": str(events),
           "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": f"{hookdir}:{REPO_ROOT}:{REPO_ROOT / 'scripts'}"}
    ram_floor = host_ram_free_mb()
    t0 = time.monotonic()
    with open(log, "w") as fp:
        proc = subprocess.Popen(worker.cmd(argv, env), stdout=fp, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT, start_new_session=True)
    _track_leg(proc.pid)

    def progress() -> tuple[int, bool]:
        """(a monotonically growing progress counter, whether real work has started yet).

        Three signals summed. The heartbeat is the sharp one: the CLI's progress stream is per
        recycle, and at 1536 tokens ONE trunk block can run for minutes, so the coarse signal alone
        would read a legitimately grinding block as a hang.
        """
        n = events.stat().st_size if events.exists() else 0
        beats = sum(f.stat().st_size for f in beat.parent.glob(beat.name + ".*"))
        return (n + beats + (log.stat().st_size if log.exists() else 0)), (n > 0 or beats > 0)

    last, last_move, stalled, warm = -1, time.monotonic(), False, False
    cpu_at_quiet, cpu_now = tree_cpu_s(proc.pid), tree_cpu_s(proc.pid)
    try:
        while True:
            # Wait ON the process rather than sleeping and then asking: a plain sleep(5) means a
            # leg that exits early in a tick is not noticed for up to 5 s, and `wall` is taken
            # after the loop, so every leg was over-reported by 0-5 s. That is why the recorded
            # Tier 1 walls were all multiples of 5. The gate's own ~60 s/model budget verdict is
            # decided on this number, so a mean +2.5 s bias is not cosmetic.
            try:
                proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            ram_floor = min(ram_floor, host_ram_free_mb())
            cpu_now = tree_cpu_s(proc.pid)
            n, seen = progress()
            if n != last:
                last, last_move, cpu_at_quiet = n, time.monotonic(), cpu_now
            if seen and not warm:
                # First real progress event: the cold compile is behind us, so tighten up.
                warm, last_move = True, time.monotonic()
            now = time.monotonic()
            if now - last_move > (stall_s if warm else max(stall_s, WARMUP_S)):
                stalled = True
                break
            if now - t0 > timeout:
                break
        if proc.poll() is None:
            # The CLI spawns worker processes; killing only the outer pid orphans the engine.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=60)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        _untrack_leg(proc.pid)
    wall = time.monotonic() - t0
    text = log.read_text(errors="replace") if log.exists() else ""
    return {"rc": proc.returncode, "wall_s": round(wall, 1), "stalled": stalled,
            "warmed": warm, "quiet_s": round(time.monotonic() - last_move, 1),
            "progress_events": events.stat().st_size if events.exists() else 0,
            "block_calls": sum(f.stat().st_size for f in beat.parent.glob(beat.name + ".*")),
            # Evidence for a STALL: was it grinding or waiting? Annotation only, never the verdict.
            "cpu_s_while_quiet": round(max(0.0, cpu_now - cpu_at_quiet), 1),
            "host_ram_floor_mb": ram_floor, "mechanism": classify(text),
            "hook": hook_findings(hook_out),
            "first_error": first_error(text),
            "tail": "\n".join(text.splitlines()[-25:])}


#: Opening the device is NOT enough to prove the card is usable. A chip left dirty by a killed
#: fold opens fine and then hangs on the first program dispatch -- measured here: a SIGKILLed
#: large fold left card 0 in a state where the next run sat forever inside tt-bio's own
#: dispatch probe, all threads idle, with no error. So this probe dispatches and synchronizes.
#: CARD_OPEN separates the two phases. Everything before it is python starting, torch and ttnn
#: importing and the device opening -- host-side work whose cost moves with host load. Everything
#: after it is the dispatch, which is the part a wedged chip never completes.
_CARD_PROBE = (
    "import torch, ttnn\n"
    "from tt_bio.main import ensure_p300_mesh_descriptor\n"
    "ensure_p300_mesh_descriptor()\n"
    "d = ttnn.open_device(device_id=0)\n"
    "print('CARD_OPEN', flush=True)\n"
    "t = ttnn.from_torch(torch.zeros((32, 32), dtype=torch.bfloat16),\n"
    "                    layout=ttnn.TILE_LAYOUT, device=d)\n"
    "ttnn.add(t, t)\n"
    "ttnn.synchronize_device(d)\n"
    "ttnn.close_device(d)\n"
    "print('CARD_HEALTHY', flush=True)\n")

#: Seconds allowed for the DISPATCH phase alone, once the child has said CARD_OPEN.
#:
#: Measured on pc's p150a from three healthy probes this campaign logged, timed from ttnn's first
#: log line to "Closing user mode device drivers": 0.726 s, 0.663 s, 0.696 s. That whole window is
#: open + dispatch + synchronize + close, so the dispatch alone is well under a second, and this
#: budget is ~85x it.
#:
#: The budget cannot simply be tightened to the measured cost. A false "cannot dispatch" triggers
#: `tt-smi -r`, which on a p300c takes the BOARD PAIR down and can kill a sibling leg's card, so
#: the old single 300-420 s timeout was deliberately conservative. Splitting the phases is what
#: makes a short number safe: by the time it applies the child has already imported torch and ttnn
#: and opened the device, so none of the host-side variance the long timeout was covering is still
#: ahead of it. Wedge detection drops from up to 600 s to ~60 s, and the gate can tell "never
#: opened" (busy, contended, driver gone) from "opened and will not dispatch" (the dirty chip).
_DISPATCH_BUDGET_S = 60


class Probe(typing.NamedTuple):
    healthy: bool
    #: Where it got to: "done", "open" (never opened in time), "dispatch" (opened, then hung),
    #: "exit" (the process died without dispatching).
    phase: str
    seconds: float
    tail: str

    def __bool__(self) -> bool:
        return self.healthy

    def why(self) -> str:
        if self.healthy:
            return f"dispatches ({self.seconds:.1f}s)"
        return {
            "open": f"never opened the device within {self.seconds:.0f}s",
            "dispatch": (f"opened the device and then did not dispatch within "
                         f"{self.seconds:.0f}s, which is the dirty-chip signature"),
            "exit": f"the probe process exited without dispatching after {self.seconds:.1f}s",
        }.get(self.phase, self.phase)


def probe_card(worker: Worker, *, timeout=420, dispatch_budget=_DISPATCH_BUDGET_S) -> Probe:
    """Can this card open AND dispatch a program, and if not, which half failed?

    Zero processes is not proof of a clean chip. A killed leg leaves both possibilities: the lease
    still held by a spawned fold worker that outlived the kill, and a chip that accepts an open and
    then never dispatches. Both turn every leg after them into a spurious FAIL, which would publish
    a ceiling nobody walked -- so the gate checks before it believes a failure.
    """
    proc = worker.popen([sys.executable, "-u", "-c", _CARD_PROBE])
    seen: dict[str, float] = {}
    tail: list[str] = []

    def drain():
        for line in proc.stdout:                       # ends when the pipe closes
            tail.append(line.rstrip())
            del tail[:-25]
            for mark in ("CARD_OPEN", "CARD_HEALTHY"):
                if mark in line:
                    seen.setdefault(mark, time.monotonic())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    t0 = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if "CARD_HEALTHY" in seen:
                return Probe(True, "done", seen["CARD_HEALTHY"] - t0, "\n".join(tail))
            # Both conditions, and in this order: the reader has to have drained the pipe before
            # an exited process means the marker never came, or a probe that printed CARD_HEALTHY
            # and exited in the same breath races its own output and reads as a wedge.
            if proc.poll() is not None and not reader.is_alive():
                if "CARD_HEALTHY" in seen:
                    return Probe(True, "done", seen["CARD_HEALTHY"] - t0, "\n".join(tail))
                return Probe(False, "exit", now - t0, "\n".join(tail))
            if "CARD_OPEN" in seen:
                if now - seen["CARD_OPEN"] > dispatch_budget:
                    return Probe(False, "dispatch", now - seen["CARD_OPEN"], "\n".join(tail))
            elif now - t0 > timeout:
                return Probe(False, "open", now - t0, "\n".join(tail))
            time.sleep(0.2)
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass


def card_healthy(worker: Worker, *, timeout=420) -> bool:
    return bool(probe_card(worker, timeout=timeout))


def wait_for_card(worker: Worker, *, timeout=600) -> bool:
    """Block until the card is free and dispatching again."""
    deadline = time.monotonic() + timeout
    while True:
        if card_healthy(worker, timeout=300):
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(15)


#: Resetting a card is the ONE operation in this gate that can hurt somebody else's run, so it is
#: allowed only when this run owns the host outright. On a p300c a `tt-smi -r` resets the BOARD
#: PAIR and not the chip, so a reset issued for card 0 also takes down card 1 -- a card this gate
#: may not even be scheduling on.
def _may_reset(worker: Worker, workers: list[Worker]) -> bool:
    return worker.is_local and sum(1 for w in workers if w.host == worker.host) == 1


def co_tenant(worker: Worker) -> str | None:
    """Another live process holding this card's device lease, or None.

    THE reason this exists. "This run owns the host" was read off the gate's own --workers list,
    which says nothing about who else on the box is using the card. The fleet dispatcher can grant
    card 0 on pc to two workers at once, and it did, mid-campaign: tt-bio's own lease refused this
    gate's residency leg with "physical card 0 on pc is in use by worker:ceiling-rfd3 (pid ...)".
    That is handled -- it is a CONTENDED verdict and nothing is scored. What was NOT handled is
    what comes next: a contended card fails the health probe, a failed probe reads as a wedge, and
    a wedge gets `tt-smi -r`. The gate would have reset the chip out from under another worker's
    running job, which is the one action here that destroys somebody else's measurement.

    The lease file is authoritative and free to read, so it is consulted before the probe rather
    than inferred from it. A lease held by OUR OWN holder label is not a co-tenant: that is this
    gate's own leg, and a straggler of ours on a wedged chip is exactly the case a reset is for.
    """
    path = Path(device_lease.lease_dir()) / f"{worker.host}-card{worker.card}.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if meta.get("released") is not None:
        return None
    pid, holder = meta.get("pid"), meta.get("holder")
    if holder == os.environ.get("TT_BIO_LEASE_HOLDER"):
        return None
    try:
        os.kill(int(pid), 0)                 # signal 0: liveness, delivers nothing
    except (OSError, TypeError, ValueError):
        return None                          # holder is gone; a stale lease is not a co-tenant
    return f"{holder} (pid {pid})"


def recover_card(worker: Worker, workers: list[Worker]) -> tuple[bool, str]:
    """Bring a wedged card back, or say why not.

    This exists because provoking an out-of-memory refusal is THE JOB of this gate, and a
    device-side TT_FATAL can leave the chip accepting an open and then never dispatching. Without
    recovery the first model that legitimately fails the bar wedges the card and every model after
    it reads as a failure too, so a one-line real result would arrive wrapped in a cascade of
    invented ones. Polling cannot fix a wedge; only a reset can.
    """
    # Before the probe, not after: a co-tenant's card fails the probe for a reason that has
    # nothing to do with the chip, and the 300 s spent finding that out is 300 s in which the
    # answer was already sitting in the lease file.
    other = co_tenant(worker)
    if other:
        return False, (f"card {worker.card} is leased by {other}, so this gate must not reset it "
                       f"-- that would take the chip down under another job's running work. "
                       f"Nothing was measured here; re-run this cell when the card is free.")
    before = probe_card(worker, timeout=300)
    if before:
        return True, "still dispatching"
    if not _may_reset(worker, workers):
        return False, (f"card {before.why()} and this run does not own the host exclusively, "
                       f"so it must not reset (a reset takes the board pair down with it)")
    smi = os.path.expanduser("~/.local/bin/tt-smi")
    try:
        subprocess.run([smi, "-r", str(worker.card)], capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"reset failed to run: {type(exc).__name__}"
    after = probe_card(worker, timeout=420)
    if after:
        return True, f"{before.why()}; recovered by tt-smi -r {worker.card}"
    return False, f"{before.why()}; tt-smi -r {worker.card} did not restore dispatch: {after.why()}"


def _screen(worker, cell, fixture, work, hookdir) -> dict:
    """TIER 1. A FAIL here is definitive and short-circuits Tier 2. A pass is INCONCLUSIVE.

    That asymmetry is what makes the shortcut safe. The truncation can only reduce what runs, so
    it can miss a Class B failure but cannot invent one; a screen that passed therefore proves
    nothing and is never reported as PASS. Which means an over-permissive screen costs wall-clock
    and nothing else -- including the case where the hook matched no block list at all and the
    "screen" quietly ran full depth, which the hook records and the report prints.
    """
    # Keyed by SIZE, not just by model. A bisect screens seven rungs, and one shared path meant
    # each rung overwrote the last -- so the only surviving evidence for a reported ceiling was
    # its final rung, and every refusal that established the wall was gone. The residency leg was
    # already per-size; this makes the screen match it.
    tok = fixture["tokens_requested"]
    log = work / f"screen_{cell.model}_{tok}.log"
    hook_out = work / f"hook_screen_{cell.model}_{tok}"
    out_dir = work / f"out_screen_{cell.model}_{tok}"
    argv = build_argv(cell, fixture, out_dir, tier="screen")
    r = execute(worker, argv, log, mode="screen", hook_out=hook_out, hookdir=hookdir,
                out_dir=out_dir, timeout=RUN_TIMEOUT_S, stall_s=STALL_S)
    trunc = r["hook"].get("truncated") or []
    r["stacks_truncated"] = len(trunc)
    r["truncated"] = trunc[:40]
    # The screen's whole claim is "one block per stack". If the hook did not install or matched
    # nothing, this leg ran the FULL model and calling it a screen would misreport both what ran
    # and what a clean result means.
    r["hook_installed"] = "install_failed" not in r["hook"]
    if not r["hook_installed"]:
        r["note"] = (f"the block-truncation hook did not install "
                     f"({r['hook']['install_failed']}), so this leg ran at FULL depth. Not a "
                     f"screen.")
    elif not trunc:
        r["note"] = ("the hook installed but matched no block stack, so this leg ran at FULL "
                     "depth. Not a screen.")
    if r["rc"] == 0 and not r["stalled"]:
        r["verdict"] = "INCONCLUSIVE"          # deliberately not PASS
        return r
    if r["stalled"]:
        # A stall under truncation is not a definitive shape verdict; let Tier 2 decide.
        r["verdict"] = "INCONCLUSIVE"
        r["note"] = f"screen made no progress for {STALL_S}s; deferring to the residency run"
        return r
    if r["mechanism"] in NOTHING_RAN:
        r["verdict"] = NOTHING_RAN[r["mechanism"]]
        return r
    if r["hook_installed"] and (_HOOK_BROKE.search(r["tail"] or "")
                                or _hook_cut_these_weights(r["tail"], trunc)):
        r["verdict"] = "GATE_BUG"
        r["reason"] = ("the Tier 1 block-truncation hook broke this model's construction, so "
                       "nothing was measured. This is the gate's bug, not a failed bar.")
        return r
    if not r["mechanism"] and (r["host_ram_floor_mb"] < HOST_RAM_FLOOR_MB
                               or _host_killed(r["rc"], r["tail"])):
        # No device-side error and the worker died on a signal: the HOST ran out of memory. A
        # 1504-token fold that the kernel kills at 22.8 GiB anon-rss on a 30 GB box never reached
        # the device wall, and recording it as one publishes a ceiling nobody walked.
        r["verdict"] = "HOST_OOM"
        r["host_oom_evidence"] = _oom_killer_fired(r.get("model", ""))
        return r
    r["verdict"] = "FAIL"
    return r


def _residency(worker, cell, fixture, work, hookdir, tokens) -> dict:
    """TIER 2. The full pipeline at the target size. This is the only tier that can say PASS."""
    log = work / f"resid_{cell.model}_{tokens}.log"
    hook_out = work / f"hook_resid_{cell.model}_{tokens}"
    out_dir = work / f"out_resid_{cell.model}_{tokens}"
    argv = build_argv(cell, fixture, out_dir, tier="residency")
    r = execute(worker, argv, log, mode="residency", hook_out=hook_out, hookdir=hookdir,
                out_dir=out_dir)
    h = r["hook"]
    r["dram_peak_bytes"] = h.get("dram_peak_bytes")
    r["dram_total_bytes"] = h.get("dram_total_bytes")
    r["dram_largest_free_at_peak"] = h.get("dram_largest_free_at_peak")
    r["blocks_instrumented"] = len(h.get("instrumented") or [])
    if r["mechanism"] in NOTHING_RAN:
        r["verdict"] = NOTHING_RAN[r["mechanism"]]
    elif r["stalled"]:
        r["verdict"] = "STALL"
        r["stall_kind"] = ("compute-active" if r["cpu_s_while_quiet"] > 0.5 * STALL_S
                           else "idle")
    elif not r["mechanism"] and (r["host_ram_floor_mb"] < HOST_RAM_FLOOR_MB
                                 or _host_killed(r["rc"], r["tail"])):
        r["verdict"] = "HOST_OOM"
        r["host_oom_evidence"] = _oom_killer_fired(r.get("model", ""))
    elif r["rc"] == 0:
        r["verdict"] = "PASS"
    elif _input_rejected(r["tail"]):
        r["verdict"] = "BAD_FIXTURE"
    else:
        r["verdict"] = "FAIL"
    return r


#: fixture_for writes work/fixtures and work/msa under paths keyed by residue count and sequence
#: hash, and models sharing a residue count share the path -- so under the per-card fan-out two
#: threads would write one file while a third read it. Building is seconds against runs of
#: minutes, so it is simply serialised rather than made clever.
_FIXTURE_LOCK = threading.Lock()


def fixture_for(cell: Cell, tokens: int, work: Path, depth) -> dict:
    """Build (or reuse) the fixture for one cell at `tokens`, and wire its cached MSA.

    `predict --msa_cache_only` resolves an alignment as `<msa_dir>/<seq_hash(sequence)>.a3m`, so
    the generated a3m is filed under that name. No network, no search, and the SAME alignment
    every run.
    """
    res = cell.residues(tokens)
    with _FIXTURE_LOCK:
        f = capacity_fixture.build(res, work / "fixtures", depth=depth)
        if cell.msa:
            from tt_bio.cache import seq_hash
            seq = [l for l in f["yaml"].read_text().splitlines() if "sequence:" in l][0]
            seq = seq.split("sequence:")[1].strip()
            msa_dir = work / "msa"
            msa_dir.mkdir(parents=True, exist_ok=True)
            target = msa_dir / f"{seq_hash(seq)}.a3m"
            target.write_text(f["a3m"].read_text())
            f["msa_dir"] = msa_dir
            f["msa_file"] = target
    f["tokens_requested"] = tokens
    f["tokens_padded"] = cell.padded(tokens)
    f["residues"] = res
    return f


def run_cell(worker: Worker, cell: Cell, work: Path, hookdir: Path, *, depth,
             bisect: bool, recover=None) -> dict:
    """Screen at the bar, then the residency run, then bisect DOWN only if it failed.

    Target-first is the single biggest efficiency win here and it is the opposite of a ladder: a
    model that clears 1536 costs exactly one screen and one residency run, and nobody pays for
    640/768/896 to learn something the 1536 pass already proved.
    """
    rec = {"model": cell.model, "verb": cell.verb, "worker": repr(worker),
           "bar_tokens": TOKEN_BAR, "legs": []}
    if cell.model in EXEMPT:
        rec.update(verdict="SKIPPED", reason=EXEMPT[cell.model])
        return rec
    try:
        f = fixture_for(cell, TOKEN_BAR, work, depth)
    except Exception as exc:
        rec.update(verdict="ERROR", reason=f"fixture: {type(exc).__name__}: {exc}")
        return rec
    rec.update(tokens_requested=f["tokens_requested"], tokens_padded=f["tokens_padded"],
               residues=f["residues"], msa_rows_in_file=f["file_depth"] if cell.msa else 0,
               msa_rows_effective=f["effective_depth"] if cell.msa else 0)

    scr = _screen(worker, cell, f, work, hookdir)
    rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR))
    if scr["verdict"] in NOTHING_RAN_VERDICTS and wait_for_card(worker):
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR, retry=True))
    if scr["verdict"] in NOTHING_RAN_VERDICTS:
        rec.update(verdict=scr["verdict"], decided_by="screen", wall_s=scr["wall_s"],
                   reason=NOTHING_RAN_REASON[scr["verdict"]])
        return rec
    if scr["verdict"] in ("FAIL", "HOST_OOM") and not screen_reduction_is_unsafe(scr):
        # Definitive: a shape that cannot allocate once cannot allocate ever.
        rec.update(verdict=scr["verdict"], decided_by="screen",
                   mechanism=scr["mechanism"], wall_s=scr["wall_s"])
        if _no_weights(scr["tail"]):
            rec["verdict"] = "NO_WEIGHTS"
        elif bisect:
            rec["ceiling_tokens"] = _bisect(worker, cell, work, hookdir, depth, rec,
                                            recover=recover)
        return rec

    if screen_reduction_is_unsafe(scr):
        rec["screen_unsafe"] = True
        rec["note"] = ("the screen's depth cut is a candidate cause of its own FAIL "
                       f"({scr['stacks_truncated']} stacks cut, no capacity mechanism in the "
                       f"error), so it was not scored; this verdict comes from the un-truncated "
                       f"run. Tier 1 has no fast path for this model.")
    res = _residency(worker, cell, f, work, hookdir, TOKEN_BAR)
    rec["legs"].append(dict(res, tier="residency", tokens=TOKEN_BAR))
    # A STALL is only the MODEL's stall if the chip is still dispatching. A card left dirty by an
    # earlier killed fold accepts an open and then hangs, with all threads idle and no error --
    # indistinguishable from a model hang from the outside, and attributing it to the model would
    # publish a ceiling that is really a housekeeping bug.
    if res["verdict"] in (NOTHING_RAN_VERDICTS | {"STALL"}) and not card_healthy(worker):
        rec["legs"][-1]["card_unhealthy_after"] = True
        if wait_for_card(worker):
            res = _residency(worker, cell, f, work, hookdir, TOKEN_BAR)
            rec["legs"].append(dict(res, tier="residency", tokens=TOKEN_BAR, retry=True))
        else:
            rec.update(verdict="CARD_DIRTY", decided_by="residency", wall_s=res["wall_s"],
                       reason="the card stopped dispatching and did not recover; nothing was "
                              "measured. Reset it (tt-smi -r) and re-run this cell.")
            return rec
    elif res["verdict"] in NOTHING_RAN_VERDICTS and wait_for_card(worker):
        res = _residency(worker, cell, f, work, hookdir, TOKEN_BAR)
        rec["legs"].append(dict(res, tier="residency", tokens=TOKEN_BAR, retry=True))
    rec.update(verdict=res["verdict"], decided_by="residency", mechanism=res["mechanism"],
               wall_s=res["wall_s"], dram_peak_bytes=res["dram_peak_bytes"],
               dram_total_bytes=res["dram_total_bytes"],
               dram_largest_free_at_peak=res["dram_largest_free_at_peak"],
               blocks_instrumented=res["blocks_instrumented"],
               host_ram_floor_mb=res["host_ram_floor_mb"])
    if res["verdict"] != "PASS" and _no_weights(res["tail"]):
        rec["verdict"] = "NO_WEIGHTS"
    elif res["verdict"] in ("FAIL", "STALL") and bisect:
        rec["ceiling_tokens"] = _bisect(worker, cell, work, hookdir, depth, rec,
                                            recover=recover)
    return rec


_NO_WEIGHTS = re.compile(
    r"could not (be )?(download|fetch)|No such file or directory.*\.(pt|safetensors|ckpt)"
    r"|HFValidationError|RepositoryNotFound|GatedRepo|401 Client Error|Connection error"
    r"|Weights .* not found|missing weights"
    # An UNUSABLE checkpoint is the same non-result as an absent one: the model was never
    # constructed, so nothing was ever asked of the device. Measured on nesso1, whose artifact on
    # this host carries atom-encoder layers the module does not declare, so from_pretrained dies
    # in load_state_dict(strict=True) before a single tensor reaches the card.
    r"|Error\(s\) in loading state_dict|Unexpected key\(s\) in state_dict"
    r"|Missing key\(s\) in state_dict|size mismatch for ", re.I)


def _no_weights(tail: str) -> bool:
    """A model whose checkpoint is absent or unusable on this host has NOT failed the bar.
    Recording that as a capacity failure would be the same lie in the other direction as scoring
    it a pass: it publishes a ceiling nobody walked."""
    return bool(_NO_WEIGHTS.search(tail or ""))


#: A leg whose worker died on SIGKILL with no device-side error is the HOST out of memory, not the
#: card. Measured: esmfold2 at 1504 residues was killed by the kernel OOM killer at 22.8 GiB
#: anon-rss on this 30 GB host, and the launcher only ever saw "SpawnProcess-1 exit -9". The
#: RSS-floor sampler can miss it outright, because the kill happens between two samples.
_SIGKILLED = re.compile(r"exit -9\b|exit 137\b|Killed\b|SIGKILL")


def _host_killed(rc: int, tail: str) -> bool:
    return rc in (-9, 137) or bool(_SIGKILLED.search(tail or ""))


#: The model refusing the INPUT is not the card refusing the SIZE, and only one of those is a
#: capacity result. nesso1 is the roster's only `affinity` model and this gate's fixture is
#: polymer-only by construction, so its batch builder rejects the ligand-free input outright:
#: "No protein or ligand tokens found in the batch". Scored FAIL, that publishes a failed 1536
#: bar for a model that never received a valid input at ANY size -- the same lie as p1's defects
#: 7 and 8, where a host kill and an unusable checkpoint were read as a walked ceiling.
_BAD_INPUT = re.compile(
    r"No protein or ligand tokens found"
    r"|[Nn]o tokens found in the batch"
    r"|No sequences (?:found|provided)", re.I)


def _input_rejected(tail: str) -> bool:
    return bool(_BAD_INPUT.search(tail or ""))


def _oom_killer_fired(model: str) -> str | None:
    """The kernel's own record of the kill, so HOST_OOM is evidence and not an inference."""
    # kernel.dmesg_restrict=1 on this host, so the plain call fails with "Operation not
    # permitted" and the corroboration silently came back empty on every run. Fall back to a
    # non-interactive sudo, and stay best-effort: the SIGKILL itself is the primary signal.
    out = ""
    for argv in (["dmesg"], ["sudo", "-n", "dmesg"]):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if p.returncode == 0 and p.stdout:
            out = p.stdout
            break
    if not out:
        return None
    hits = [ln for ln in out.splitlines() if "Out of memory: Killed process" in ln]
    return hits[-1].strip()[-200:] if hits else None


def _bisect(worker, cell, work, hookdir, depth, rec, recover=None) -> int | None:
    """After a failure only: walk DOWN to the real ceiling, then refine it to the bucket.

    Two phases, because the two questions cost different amounts. The rung walk answers "what
    size actually COMPLETES", which needs a residency run and takes minutes. The refinement
    answers "where exactly is the wall", and for a Class A failure a SCREEN answers that
    definitively in seconds -- a shape that cannot allocate once cannot allocate ever. So the
    coarse rungs never get finer than they need to be, and the ceiling still comes back
    bucket-exact instead of "somewhere between 1280 and 1408".

    Both numbers are reported, never conflated: `ceiling_tokens` completed a full residency run,
    `alloc_ceiling_tokens` is the largest bucket-aligned size whose SHAPES allocate. The second
    is an upper bound on the first, because a clean screen cannot rule out a Class B failure.
    """
    def settle(leg: dict) -> None:
        """Re-probe the card after a fail-like rung, BEFORE the next one runs.

        p1's defect 6: rf3's allocator refusal is a TT_FATAL that leaves card 0 accepting a device
        open and then never dispatching, so the cell after it sits at 100% CPU inside tt_bio's own
        dispatch probe with no log line. p1 put the recovery after each CELL -- but a bisect
        provokes that refusal once per rung INSIDE one cell, which is the densest sequence of
        refusals this gate ever produces and had no recovery in it at all. Every rung below the
        first failing one was running on a card the rung above may have wedged, so the ceiling
        those rungs report is exactly the kind of number that gets published without being walked.
        """
        if recover is None or leg.get("verdict") not in (
                {"FAIL", "HOST_OOM", "STALL", "ERROR"} | NOTHING_RAN_VERDICTS):
            return
        ok, how = recover(worker)
        leg["card_after"] = how
        if not ok:
            leg["card_dirty_after"] = True

    lo = None                       # largest size that completed a residency run
    alloc_lo = None                 # largest size whose SHAPES allocate (a clean screen)
    alloc_hi = TOKEN_BAR            # smallest size whose shapes do NOT allocate
    walked = []                     # every rung actually screened, for the no-ceiling case
    resid_walked = []               # rungs the UN-TRUNCATED run actually reached
    screens_informative = True      # False once Tier 1 breaks this model by its own hand
    for rung in BISECT_RUNGS:
        try:
            f = fixture_for(cell, rung, work, depth)
        except Exception:
            continue
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=rung))
        if scr["verdict"] in NOTHING_RAN_VERDICTS:
            # The card was gone, so this rung says nothing about shapes. Recover and re-walk it
            # once; if it is still gone, the rung is not walked at all. Letting it fall through
            # set alloc_hi and published a wall at a size the allocator was never asked about --
            # measured on qb2's p300c 2026-09-10, where opendde (1280), opendde-abag (1344) and
            # protenix-v1 (1376) each got their ceiling from the rung above, whose worker died
            # inside ttnn.open_device after the previous leg was killed on the stall timeout.
            settle(rec["legs"][-1])
            scr = _screen(worker, cell, f, work, hookdir)
            rec["legs"].append(dict(scr, tier="screen", tokens=rung, retry=True))
            if scr["verdict"] in NOTHING_RAN_VERDICTS:
                continue
        walked.append(rung)
        if screen_reduction_is_unsafe(scr):
            # This rung learned NOTHING about allocation, so it may neither lower alloc_hi nor
            # raise alloc_lo. run_cell already applies exactly this test to the screen at the bar;
            # the bisect skipping it is how esmfold2 turned seven screens that all died of
            # `shape '[1, 1, 3, 1]' is invalid for input of size 81` into "shapes do not allocate
            # at any size walked ... the ceiling is below 512 tokens", while the same model folds
            # 512 in 48.7 s. That is the gate inventing a capacity verdict out of its own
            # instrument, which is the failure screen_reduction_is_unsafe exists to stop. Fall
            # through to the un-truncated run, which reduces nothing and always measures.
            rec["legs"][-1]["screen_unsafe"] = True
            screens_informative = False
        elif scr["verdict"] in ("FAIL", "HOST_OOM"):
            alloc_hi = rung
            settle(rec["legs"][-1])
            continue
        else:
            # The screen is clean, so the shapes allocate at this size. That is true regardless of
            # what the residency run below then does, and the two bounds are NOT the same bound: a
            # residency failure here must not lower the ALLOCATION ceiling, because the allocation
            # plainly succeeded. Conflating them discarded a measured screen result.
            alloc_lo = rung if alloc_lo is None else max(alloc_lo, rung)
        res = _residency(worker, cell, f, work, hookdir, rung)
        rec["legs"].append(dict(res, tier="residency", tokens=rung))
        resid_walked.append(rung)
        if res["verdict"] == "PASS":
            lo = rung
            break
        settle(rec["legs"][-1])

    if alloc_lo is None:
        # No screen on the walk came back clean. That IS the finding, and returning early without
        # recording it threw away the whole walk: seven rungs of card time came back as an empty
        # cell that reads as if the bisect had never run.
        #
        # WHY it came back dirty decides what may be said. A screen that failed on capacity
        # bounds the allocation ceiling; a screen that failed on Tier 1's own truncation bounds
        # nothing, and the residency runs are then the only measurement this cell has. Returning
        # `lo` rather than None matters for the same reason: with the truncation no longer
        # blocking them, those runs happen, and a completing ceiling they found must not be
        # discarded on the way out.
        #
        # BUT ONLY IF THE ALLOCATOR IS WHAT REFUSED, and only once truncation is ruled out. A
        # size wall is size-dependent by definition, so a walk where the allocator never spoke
        # has not found one, however many rungs it burned. Measured on esmfold2 on qb2's p300c,
        # 2026-09-10: the Tier 1 hook truncates block lists, which leaves a downstream reshape
        # inconsistent, so every rung from 1408 down to 512 died in 4-6 s with the identical
        # `shape '[1, 1, 3, 1]' is invalid for input of size 81` and the cell published "the
        # ceiling is below 512 tokens if there is one at all" -- for a model whose size ladder
        # folds 768 on that same card in 96.6 s. That case is `not screens_informative`, handled
        # below by falling through to the residency-only note instead of this guard.
        #
        # This is the THIRD way the instrument has broken a model (_HOOK_BROKE was the first,
        # _UNEXPECTED_KEY the second, both matched by message). A fourth message will not match
        # a fourth regex either, so the guard here is on the shape of the evidence instead: an
        # error that does not change with size is not evidence about size.
        if screens_informative:
            screens = [l for l in rec["legs"]
                       if l.get("tier") == "screen" and l.get("tokens") in walked]
            if walked and not any(l.get("mechanism") in ALLOC_MECHANISMS for l in screens):
                why = next((l.get("mechanism") for l in screens if l.get("mechanism")), None)
                rec["alloc_ceiling_tokens"] = None
                rec["alloc_ceiling_note"] = (
                    f"the bisect decided nothing. Every rung walked "
                    f"({', '.join(str(r) for r in walked)}) failed without the allocator ever "
                    f"refusing"
                    + (f" (mechanism {why})" if why else "")
                    + ", so the failure does not depend on size and is not a capacity wall. "
                      "Whatever killed these rungs has to be fixed before a ceiling can be read "
                      "off them.")
                return None
        rec["alloc_ceiling_tokens"] = None
        if not walked:
            rec["alloc_ceiling_note"] = "no rung could be built, so nothing was walked."
        elif not screens_informative:
            rec["alloc_ceiling_note"] = (
                f"no allocation bound from this walk: Tier 1 cannot build this model, so the "
                f"screens at {', '.join(str(r) for r in walked)} failed on the truncation and "
                f"not on capacity. Only the un-truncated runs at "
                f"{', '.join(str(r) for r in resid_walked) or 'no rung'} measured anything.")
        else:
            rec["alloc_ceiling_note"] = (
                f"shapes do not allocate at any size walked: {', '.join(str(r) for r in walked)}. "
                f"The ceiling is below {min(walked)} tokens if there is one at all.")
        return lo

    # Refine on screens, and only while screens still mean something for this model: a walk whose
    # screens die of the truncation cannot narrow anything, and looping on bounds it can never
    # move just spends card time to re-derive the same broken error.
    while screens_informative and alloc_hi - alloc_lo > TOKEN_BUCKET:
        mid = ((alloc_lo + alloc_hi) // 2 // TOKEN_BUCKET) * TOKEN_BUCKET
        if mid <= alloc_lo or mid >= alloc_hi:
            break
        try:
            f = fixture_for(cell, mid, work, depth)
        except Exception:
            break
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=mid, phase="refine"))
        if scr["verdict"] in NOTHING_RAN_VERDICTS:
            # Here it is worse than in the rung loop above: the else branch RAISES alloc_lo, so a
            # rung where no model code ran would be published as a size whose shapes allocate.
            # Recover and re-screen once, then stop: a walk that cannot reach the card cannot
            # refine, and the bounds it already has are the honest answer.
            settle(rec["legs"][-1])
            scr = _screen(worker, cell, f, work, hookdir)
            rec["legs"].append(dict(scr, tier="screen", tokens=mid, phase="refine", retry=True))
            if scr["verdict"] in NOTHING_RAN_VERDICTS:
                rec["refine_stopped"] = (
                    f"the card was gone at {mid} on two consecutive attempts, so the bound was "
                    f"not refined past {alloc_lo}")
                break
        if screen_reduction_is_unsafe(scr):
            rec["legs"][-1]["screen_unsafe"] = True
            screens_informative = False
        elif scr["verdict"] in ("FAIL", "HOST_OOM"):
            alloc_hi = mid
            settle(rec["legs"][-1])
        else:
            alloc_lo = mid
    rec["alloc_ceiling_tokens"] = alloc_lo
    rec["alloc_ceiling_note"] = (
        f"{alloc_lo} is the largest bucket-aligned size whose shapes ALLOCATE (screen); "
        f"{alloc_hi} is the smallest that does not. A clean screen cannot rule out a Class B "
        f"failure, so this is an upper bound on the completing ceiling, not a pass."
        + ("" if screens_informative else
           " NOT REFINED between those bounds: a screen on this walk failed on Tier 1's own "
           "truncation rather than on capacity, so the screens carry no information there."))

    # One residency run to try to promote the refined number to a completing ceiling. `lo` is
    # None when no rung completed, and that is exactly the case worth spending the run on.
    if lo is None or alloc_lo > lo:
        try:
            f = fixture_for(cell, alloc_lo, work, depth)
            res = _residency(worker, cell, f, work, hookdir, alloc_lo)
            rec["legs"].append(dict(res, tier="residency", tokens=alloc_lo, phase="refine"))
            if res["verdict"] == "PASS":
                lo = alloc_lo
        except Exception:
            pass
    return lo


# ---------------------------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------------------------


def reductions(depth, recycling, models) -> list[str]:
    """Every coverage reduction this run applied, by name. Silence here is the failure mode."""
    out = []
    if depth is not None:
        out.append(f"MSA depth truncated to {depth} rows. For the OF3 family the failing tensor "
                   f"scales with tokens x rows, so this REDUCES exposure to the exact wall that "
                   f"took OpenFold3 down at 614; the default is the committed source's full depth.")
    if recycling is not None:
        out.append(f"recycling_steps={recycling} instead of the production default. Peak tensor "
                   f"SHAPE is preserved (buffers are reused) but the number of alloc/free cycles "
                   f"drops, so Class B exposure is REDUCED. For RF3 recycling also drives MSA "
                   f"sampling (one i.i.d. sample per recycle), so it is not a free knob there.")
    out.append("diffusion_samples=1. Samples are independent and add no peak, so this costs no "
               "coverage.")
    out.append("The MSA is a committed alignment rather than a fresh search. Device memory does "
               "not care how the alignment was found; this costs no coverage.")
    out.append("Target-first: 1536 runs FIRST and a pass ends the cell. Only a failure pays for a "
               "downward bisect. Costs no coverage.")
    out.append("The fixture is polymer-only, so the LIGAND-TOKEN path is uncovered: no cell here "
               "sets ligand_tokens, and a model whose production input carries ligands is tested "
               "at 1536 polymer tokens only.")
    skipped = sorted(set(models) & set(EXEMPT))
    if skipped:
        out.append(f"Not driven by this gate at all: {', '.join(skipped)} (see EXEMPT for each "
                   f"reason). Reported SKIPPED, never PASS.")
    return out


def sweep(all_cells: list, workers: list, run_one, publish, retire=None) -> list:
    """ONE THREAD PER CARD, PULLING FROM A SHARED QUEUE. Returns the (index, cell) nobody ran.

    `--workers` used to round-robin the ASSIGNMENT and then run the cell inline, so four cards
    cost exactly what one card costs: `workers[i % len(workers)]` picked a different card each
    iteration and the loop still waited for it. The 19.9 min sweep would have stayed 19.9 min.
    Measured cell costs run 10 s to 175 s, so a static round-robin would also leave one card
    holding openfold3 + protenix-v2 while another finished saprot-35m and idled; a shared queue is
    both simpler and better balanced.

    A card `retire` cannot recover retires its own thread, and the cells still queued are taken by
    the live ones. Only what is left when every thread has gone comes back as unrun, which is the
    honest verdict for it: nothing was measured there.

    Module level with its collaborators injected, because the alternative is a closure inside
    main() that no test can reach without four cards and half an hour.
    """
    cellq: queue.Queue = queue.Queue()
    for item in enumerate(all_cells):
        cellq.put(item)

    def drain(w) -> None:
        while True:
            try:
                i, cell = cellq.get_nowait()
            except queue.Empty:
                return
            rec = run_one(w, cell)
            publish(i, rec)
            if retire is not None and retire(w, rec):
                return

    threads = [threading.Thread(target=drain, args=(w,), name=repr(w)) for w in workers]
    for t in threads:
        t.start()
    # Joined with a TIMEOUT rather than a bare join(), so the main thread returns to the eval
    # loop twice a second and any pending Python signal handler gets to run there.
    #
    # HARDENING, not a proven fix for a known bug. On 2026-09-10 a gate ten cells into a sweep
    # ignored two explicit SIGTERMs and had to be SIGKILLed from outside, wedging the card that
    # install_teardown() exists to protect; /proc/PID/status showed SIGTERM caught and the main
    # thread parked in futex_wait_queue, which points here. But that did NOT reproduce: a probe
    # that arms the same handler and sits in this same sweep dies on SIGTERM correctly, both with
    # and without ttnn imported (same SigCgt mask, 0000000100004003, as the process that hung).
    # So the cause of that hang is still unknown, and this change is kept only because an
    # interruptible wait is strictly better than an uninterruptible one and costs nothing.
    # Until it is root-caused, the reliable way to stop a sweep is still: SIGTERM, VERIFY it
    # died, then SIGKILL the gate and the fold's process group by explicit pid, then expect to
    # need a tt-smi -r on the card.
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.5)

    unrun = []
    while True:
        try:
            unrun.append(cellq.get_nowait())
        except queue.Empty:
            return unrun


def render(report: dict) -> str:
    g = report["geometry"]
    L = []
    L.append(f"CAPACITY GATE -- {report['bar_tokens']} tokens -- "
             f"{g.get('board_type') or '?'} / {g.get('arch') or '?'} on {g.get('host')}:"
             f"{g.get('card')}")
    if "dram" in g:
        d, l1 = g["dram"], g["l1"]
        L.append(f"  DRAM {d['banks']} banks x {d['bytes_per_bank']} B = "
                 f"{d['total_bytes'] / 2**30:.3f} GiB   "
                 f"L1 {l1['banks']} banks x {l1['bytes_per_bank']} B   grid {g.get('grid')}")
    L.append("  CAPACITY ONLY: allocates and completes. It does not and cannot check correctness "
             "-- that is scripts/full_parity_gate.py.")
    L.append("")
    hdr = f"  {'model':<14} {'verdict':<10} {'tok':>5} {'pad':>5} {'rows':>6} {'peak DRAM':>12} " \
          f"{'wall':>8}  by/mech"
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for r in report["results"]:
        peak = r.get("dram_peak_bytes")
        tot = r.get("dram_total_bytes")
        peak_s = "-" if not peak else (f"{peak / 2**30:.2f}G" +
                                       (f"/{100 * peak / tot:.0f}%" if tot else ""))
        L.append(f"  {r['model']:<14} {r['verdict']:<10} {r.get('tokens_requested', '-'):>5} "
                 f"{r.get('tokens_padded', '-'):>5} {r.get('msa_rows_effective', '-'):>6} "
                 f"{peak_s:>12} {str(r.get('wall_s', '-')):>8}  "
                 f"{r.get('decided_by', '-')}/{r.get('mechanism') or '-'}"
                 + (f"  ceiling~{r['ceiling_tokens']}" if r.get("ceiling_tokens") else "")
                 + (f"  [{r['reason'][:60]}]" if r.get("reason") else ""))
    L.append("")
    L.append("  Coverage reductions applied by this run:")
    for x in report["reductions"]:
        L.append(f"    - {x}")
    if report["coverage_gaps"]:
        L.append(f"  COVERAGE GAP: shipped models neither runnable nor exempted: "
                 f"{report['coverage_gaps']}")
    n = report["counts"]
    L.append("")
    L.append(f"  {n['PASS']} pass, {n['fail_like']} fail, {n['INCONCLUSIVE']} inconclusive, "
             f"{n['SKIPPED']} skipped, {n['NO_WEIGHTS']} no-weights, "
             f"{n['CONTENDED']} unmeasured of {len(report['results'])} cells")
    if n["INCONCLUSIVE"]:
        L.append("  INCONCLUSIVE is a Tier 1 result and is NOT a pass: one block per stack cannot "
                 "see the cumulative-residency class. Run without --tier screen for a verdict.")
    if n.get("BAD_FIXTURE"):
        L.append("  BAD_FIXTURE: the MODEL rejected the input, so nothing about the size was "
                 "measured. Not a failed bar -- this gate's fixture is polymer-only, and a model "
                 "needing a ligand never got a valid input at any size.")
    return "\n".join(L)


def main(argv=None) -> int:
    install_teardown()
    global STALL_S, TOKEN_BAR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=None,
                    help="comma-separated subset; default is every shipped model (derived)")
    ap.add_argument("--tokens", type=int, default=TOKEN_BAR,
                    help=f"the bar, in tokens; must be a multiple of {TOKEN_BUCKET}")
    ap.add_argument("--workers", default=None, help="host:card[,host:card...]")
    ap.add_argument("--work-dir", type=Path,
                    default=REPO_ROOT / "perf" / "capacity" / "work")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--card-type", default=None,
                    help="board type to file the recorded cells under (p150a, p300c, ...). Read "
                         "from tt-smi for a local worker; needed only for a remote one.")
    ap.add_argument("--depth", type=int, default=None,
                    help="truncate MSA depth (a NAMED reduction, written into the report)")
    ap.add_argument("--recycling", type=int, default=None,
                    help="override recycling_steps (a NAMED reduction)")
    ap.add_argument("--tier", choices=("screen", "both"), default="both",
                    help="'screen' runs Tier 1 only; it can never report PASS")
    ap.add_argument("--no-bisect", action="store_true",
                    help="do not walk down to find the real ceiling after a failure")
    ap.add_argument("--stall-s", type=int, default=STALL_S)
    ap.add_argument("--no-card-reset", action="store_true",
                    help="never tt-smi -r a wedged card; report the rest of the run CARD_DIRTY "
                         "instead. A reset takes the board pair down, so use this when anything "
                         "else on the host is in flight.")
    ap.add_argument("--list", action="store_true", help="print the derived roster and exit")
    ap.add_argument("--record", action="store_true",
                    help="write docs/capacity_gate_baseline.json, pinning the ceiling table this "
                         "run measured against. tests/test_capacity_gate.py fails until this is "
                         "re-recorded after any ceiling change.")
    ap.add_argument("--record-from", metavar="REPORT.JSON",
                    help="fold an ALREADY-COMPLETED run's report into the baseline and exit, "
                         "without opening a device. A full sweep at this bar takes hours and has "
                         "to run in stages, so without this the only way to record a finished "
                         "stage is to run it again.")
    a = ap.parse_args(argv)

    if a.record_from:
        return _record_from(Path(a.record_from))

    STALL_S = a.stall_s
    if a.tokens % TOKEN_BUCKET:
        print(f"--tokens {a.tokens} is not a multiple of {TOKEN_BUCKET}: it would pad to "
              f"{TOKEN_BUCKET * -(-a.tokens // TOKEN_BUCKET)} internally and the gate would "
              f"report a size the hardware never saw.", file=sys.stderr)
        return 2
    TOKEN_BAR = a.tokens

    models = [m.strip() for m in a.models.split(",")] if a.models else roster()
    if a.list:
        for m in models:
            print(f"{m:<16} {_verbs().get(m, '?'):<9} "
                  f"{'EXEMPT: ' + EXEMPT[m][:70] if m in EXEMPT else 'covered'}")
        print(f"\ncoverage gaps: {coverage_gaps() or 'none'}")
        return 0

    unknown = sorted(set(models) - set(roster()))
    if unknown:
        print(f"not shipped models: {unknown}", file=sys.stderr)
        return 2

    work = a.work_dir
    work.mkdir(parents=True, exist_ok=True)
    hookdir = hook_dir(work)
    workers = parse_workers(a.workers) if a.workers else [Worker(local_host(), 0, True)]

    # EVERY card, not just the first. A gate that vets workers[0] and fans out over four would
    # record every cell that landed on an unhealthy card 3 as a capacity failure -- the exact lie
    # the card 0 check exists to prevent, reintroduced by the fan-out.
    # A co-tenant is named before the probe runs. Its card cannot dispatch FOR US, but the chip is
    # fine and the operator's next move is to wait, not to reset -- and "cannot dispatch a trivial
    # program. Reset (tt-smi -r) and re-run" is a direct instruction to break the other job.
    busy = [f"{w!r} is leased by {t}" for w, t in ((w, co_tenant(w)) for w in workers) if t]
    if busy:
        print(f"{'; '.join(busy)}. Nothing was measured. Wait for the card, or point --workers "
              f"at a free one; do NOT reset it.", file=sys.stderr)
        return 4
    sick = [f"{w!r} {p.why()}" for w, p in ((w, probe_card(w)) for w in workers) if not p]
    if sick:
        print(f"{'; '.join(sick)}. Every leg landing there "
              f"would fail or hang and the gate would record capacity failures nobody walked. "
              f"Reset (tt-smi -r) and re-run.", file=sys.stderr)
        return 3

    report = {
        "bar_tokens": TOKEN_BAR, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                     capture_output=True, text=True).stdout.strip()),
        "workers": [repr(w) for w in workers],
        "geometry": dict(geometry(workers[0]),
                         **({"board_type": a.card_type} if a.card_type else {})),
        "coverage_gaps": coverage_gaps(),
        "reductions": reductions(a.depth, a.recycling, models),
        "results": [],
        "note": "CAPACITY ONLY. Allocates and completes. Says nothing about whether the output is "
                "correct; that is scripts/full_parity_gate.py, which this does not substitute for.",
    }
    # Flushed: a gate run is watched through a redirected log, where an unflushed header sits in
    # the buffer for the whole campaign and the board geometry it carries is what a reader needs
    # FIRST to know the numbers are comparable.
    print(render(dict(report, results=[], counts=dict.fromkeys(
        ("PASS", "fail_like", "SKIPPED", "NO_WEIGHTS", "INCONCLUSIVE", "CONTENDED",
         "BAD_FIXTURE"), 0))),
        flush=True)

    #: Cards this run has given up on: a wedge that a reset could not clear. Every cell still
    #: owed on such a card is reported CARD_DIRTY, because nothing was measured on it.
    dead: dict[str, str] = {}
    all_cells = list(cells(models, depth=a.depth, recycling=a.recycling))

    # ONE THREAD PER CARD, PULLING FROM A SHARED QUEUE.
    #
    # --workers used to round-robin the ASSIGNMENT and then run the cell inline, so four cards
    # cost exactly what one card costs: `w = workers[i % len(workers)]` picked a different card
    # each iteration and the loop still waited for it. The 19.9 min sweep would have stayed
    # 19.9 min. Measured cell costs run 10 s to 175 s, so a static round-robin would also leave
    # one card holding openfold3 + protenix-v2 while another finished saprot-35m and idled --
    # a shared queue is both simpler and better balanced.
    #
    # A card that cannot be recovered RETIRES its own thread and the remaining cells are taken by
    # the live ones. Only cells still queued when every thread has retired are CARD_DIRTY, which
    # is the honest verdict: nothing was measured on them.
    done: dict[int, dict] = {}
    lock = threading.Lock()

    def publish(i: int, r: dict) -> None:
        with lock:
            done[i] = r
            report["results"] = [done[k] for k in sorted(done)]
            _finish(report)
            (a.report or work / "report.json").write_text(
                json.dumps(report, indent=1, default=str))
            print(f"  -> {r['model']:<14} {r['verdict']:<10} {r.get('wall_s')}s "
                  f"{r.get('mechanism') or ''} {str(r.get('reason', ''))[:80]}", flush=True)

    def run_one(w: Worker, cell: Cell) -> dict:
        t0 = time.monotonic()
        try:
            r = run_cell(w, cell, work, hookdir, depth=a.depth, bisect=not a.no_bisect,
                         recover=None if a.no_card_reset
                         else lambda ww: recover_card(ww, workers)) \
                if a.tier == "both" else _screen_only(w, cell, work, hookdir, a.depth)
        except Exception as exc:
            r = {"model": cell.model, "verdict": "ERROR", "worker": repr(w),
                 "reason": f"{type(exc).__name__}: {exc}"}
        r.setdefault("wall_s", round(time.monotonic() - t0, 1))
        return r

    def retire(w: Worker, r: dict) -> bool:
        # A device-side fatal can leave the chip open-able but not dispatching, so the NEXT cell
        # would hang in tt-bio's dispatch probe and be recorded as this gate's own kind of
        # failure. Checked only after a failure, so a clean run pays nothing for it.
        if r["verdict"] not in ("FAIL", "STALL", "ERROR", "CARD_DIRTY") or a.no_card_reset:
            return False
        ok, how = recover_card(w, workers)
        r["card_after"] = how
        if not ok:
            with lock:
                dead[repr(w)] = how
            print(f"     {w}: {how} -- retiring this card", flush=True)
            return True
        if how != "still dispatching":
            print(f"     {w}: {how}", flush=True)
        return False

    for i, cell in sweep(all_cells, workers, run_one, publish, retire):
        publish(i, {"model": cell.model, "verdict": "CARD_DIRTY", "wall_s": 0.0,
                    "reason": f"not run: every card retired ({'; '.join(dead.values())})"})

    _finish(report)
    print(flush=True)
    print(render(report), flush=True)
    out = a.report or work / "report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nreport: {out}")
    if a.record:
        print(record_baseline(report, partial=bool(a.models)))
    # Said after recording, because that is the moment it is actionable: whatever is still listed
    # here has a verdict in somebody's prose and not in the committed baseline, and the run logs
    # that would back it up live in scratch. Not part of the exit code -- the gate's rc is about
    # capacity verdicts, and tests/test_capacity_gate.py is what fails on a gap.
    if (gaps := baseline_gaps()):
        print(f"\nBASELINE GAP: runnable models with no recorded cell: {gaps}"
              f"\n  Run the gate for them and --record, or --record-from a finished report.")
    if (stale := baseline_stale()):
        print(f"\nBASELINE STALE: cells measured against another ceiling table: {stale}"
              f"\n  The sweep runs in stages, so this is the remaining work, named.")
    ok = (report["counts"]["fail_like"] == 0 and not report["counts"]["GATE_BUG"]
          and not report["counts"]["BAD_FIXTURE"] and not report["coverage_gaps"])
    return 0 if ok else 1


BASELINE = REPO_ROOT / "docs" / "capacity_gate_baseline.json"

#: Bumped when the on-disk shape changes. 1 was a single flat `cells` block.
BASELINE_FORMAT = 2


def read_baseline() -> dict:
    """The baseline as ``{board_type: {..., "cells": {model: cell}}}``.

    THE FILE IS PER CARD, because the answer is. p150a and p300c are both "blackhole" to ttnn,
    and a gate that answers "does every model allocate and complete at 1536 tokens on this card"
    cannot file that answer under no card.

    Not because the L1 geometry differs. It does not: a stock p150a (qb1 card 3) and a p300c
    (qb2 card 3) both read 110 L1 banks on an (x=11,y=10) grid against identical DRAM. An earlier
    version of this docstring claimed 130 banks and 15.4 % more L1 for the p150a, from a
    measurement taken on pc, which runs custom 130-core firmware. The CARD_PROBE comment below
    already says every core count pc reports is a statement about pc.

    What differs is everything around the chip. A p300 is a board PAIR, so reset and dispatch
    granularity are two chips at a time, and a lone visible p300 chip is a CUSTOM topology that
    will not open at all without a 1x1 mesh-graph descriptor. The recorded verdicts differ too.

    Format 1 had one flat `cells` block and one file-level `geometry`, so it could hold exactly
    one card and never said which. It held p150a: the probes opened ttnn bare and could not open a
    p300c at all, so no other card could ever have been recorded into it. A format 1 file is read
    as the card its own geometry names, which is that card and no other.
    """
    if not BASELINE.exists():
        return {}
    try:
        data = json.loads(BASELINE.read_text())
    except ValueError:
        return {}
    if "cards" in data:
        return data["cards"]
    cells = data.get("cells") or {}
    card = (data.get("geometry") or {}).get("board_type")
    if not cells or not card:
        return {}
    keep = ("recorded", "host", "tree", "dirty_tree", "geometry", "reductions")
    return {card: {**{k: data[k] for k in keep if k in data}, "cells": cells}}


def ceilings_fingerprint() -> str:
    """A stable hash over every published ceiling, so moving any row is detectable.

    The published ceilings a user sees live in the serving platform, which tt-bio does not import.
    The engine's own copy is tt_bio.size_limits.CEILINGS, and that is what a ceiling change edits
    here, so that is what gets pinned.

    Every field that decides what a row ADMITS is in the hash, and `ladder_ligand_atoms` is one of
    them -- it converts a row's residue numbers into the token numbers a ligand-bearing input is
    checked against, so changing it changes the advertised size as surely as moving `residues`
    would. Evidence prose is not, on purpose: rewording a row is not a new measurement and should
    not cost a re-record.
    """
    import hashlib
    rows = []
    for model in sorted(sl.CEILINGS):
        for arch in sorted(sl.CEILINGS[model]):
            c = sl.CEILINGS[model][arch]
            rows.append([model, arch, c.residues, c.pass_at, c.binds, c.mechanism,
                         c.msa_rows, c.counts, c.ladder_ligand_atoms])
    return hashlib.sha256(json.dumps(rows, default=str).encode()).hexdigest()[:16]


# Verdicts that mean THIS RUN DECIDED NOTHING. INCONCLUSIVE is a clean Tier 1 screen, which by
# design is never a pass; CONTENDED and CARD_DIRTY are cells where the card was unavailable and
# no model code ran at all. Each is a legitimate thing to report, and none of them is evidence
# about the bar.
UNDECIDED = frozenset(("INCONCLUSIVE",)) | NOTHING_RAN_VERDICTS


def _would_lose_evidence(new: dict, old: dict | None) -> bool:
    """True when recording `new` over `old` replaces a measurement with the absence of one.

    The screen is cheap and the residency run is minutes to hours, so the cheap one is the one
    that gets re-run -- and a screen cell overwriting a Tier 2 PASS is a silent downgrade of the
    most expensive result in the file. The campaign shell knew this and worked around it by never
    passing --record to the warm sweep, which is a rule enforced by discipline rather than by the
    tool: one `--tier screen --record` erases every PASS in the baseline and prints success.
    """
    if not old or old.get("tokens_requested") not in (None, TOKEN_BAR):
        return False       # nothing to lose, or prior cell is another bar's and drops anyway
    return new.get("verdict") in UNDECIDED and old.get("verdict") not in UNDECIDED


def record_baseline(report: dict, *, partial: bool) -> str:
    """Merge this run's cells into docs/capacity_gate_baseline.json.

    A partial run (--models) updates only the cells it measured and leaves the rest standing, so
    re-measuring one model does not silently erase the others' recorded results.
    """
    card = (report.get("geometry") or {}).get("board_type")
    if not card:
        return ("NOT RECORDED: this run did not establish which board it measured, and the "
                "baseline is per card. `geometry()` reads board_type from tt-smi and only does "
                "so for a LOCAL worker, so a remote --workers leg lands here. Re-run the gate "
                "locally on the card, or pass --card-type. Filing the cells under the wrong "
                "board is how a p300c result ends up published as a p150a one.")
    per_card = read_baseline()
    prior = per_card.get(card, {})
    cells = prior.get("cells", {}) if partial else {}
    # A partial run must not carry cells measured at a DIFFERENT bar across into this baseline.
    # Measured: raising the bar 1504 -> 1536 and re-recording six cells left the two esmfold2
    # cells from the 1504 run inside a file stamped 1536, where they read as current evidence.
    # Pair tensors scale roughly quadratically, so a 1504 result is not a 1536 result.
    cells = {m: c for m, c in cells.items()
             if (c or {}).get("tokens_requested") in (None, TOKEN_BAR)}
    # A full-roster record replaces the file, so the prior cells have to be read from `prior`
    # rather than from `cells`, which is empty in that case. A screen sweep over the whole roster
    # is exactly the run that would wipe every PASS.
    before = (prior.get("cells") or {})
    fp = ceilings_fingerprint()
    # WHICH CARD each cell describes, per cell within this call, beyond the file's own per-card
    # bucket. `cards[card]["cells"]` already answers which board TYPE a cell belongs to -- that
    # is the schema change this file went through so a p150a run and a p300c run stop reading as
    # if whichever card recorded last owned every cell, which is what a flat `cells` dict keyed
    # by model alone produced, and did on 2026-09-10. `measured_on` is finer than that: a
    # `--workers` fan-out can record several cells in one call from different hosts, and only the
    # per-result worker label says which one actually ran a given model.
    geom = report.get("geometry") or {}
    kept, dropped = [], []
    for r in report["results"]:
        if _would_lose_evidence(r, before.get(r["model"])):
            kept.append(f"{r['model']} ({before[r['model']]['verdict']} kept over "
                        f"{r['verdict']})")
            cells[r["model"]] = before[r["model"]]
            continue
        # Nothing to keep and nothing to write: a leg where the card never opened produced no
        # cell at all. The branch above covers the case where a measurement already stands; this
        # one is the case where none does, and writing here is how a non-measurement becomes the
        # baseline's answer for the model.
        if nothing_ran(r):
            dropped.append(f"{r['model']} ({r['verdict']})")
            cells.pop(r["model"], None)
            continue
        cells[r["model"]] = {k: r.get(k) for k in
                             ("verdict", "tokens_requested", "tokens_padded", "residues",
                              "msa_rows_effective", "dram_peak_bytes", "dram_total_bytes",
                              "wall_s", "mechanism", "decided_by", "ceiling_tokens",
                              # Both ceiling numbers, because for a model that never completes a
                              # residency run at any rung, ceiling_tokens is None and the
                              # alloc ceiling is the ONLY number the bisect produced. Persisting
                              # just the first silently discards the bisect's whole result.
                              "alloc_ceiling_tokens", "alloc_ceiling_note", "reason")}
        # The ceiling table and the tree THIS cell was measured against, per cell. See
        # baseline_stale(): a file-level stamp let a one-model record re-certify every cell.
        cells[r["model"]]["ceilings_fingerprint"] = fp
        cells[r["model"]]["tree"] = report["tree"]
        cells[r["model"]]["measured_on"] = r.get("worker") or (
            f"{geom.get('host')}:{geom.get('card')}" if geom.get("host") else None)
    per_card[card] = {
        "recorded": report["started"],
        "host": (report.get("geometry") or {}).get("host"),
        "tree": report["tree"],
        "dirty_tree": report["dirty"],
        "geometry": report["geometry"],
        "reductions": report["reductions"],
        "cells": cells,
    }
    BASELINE.write_text(json.dumps({
        "format": BASELINE_FORMAT,
        "bar_tokens": report["bar_tokens"],
        "note": "CAPACITY ONLY: allocates and completes. Not a correctness record. Per board "
                "type, because p150a and p300c do not have the same L1 and a cell filed under "
                "no card is a cell about no card.",
        "cards": dict(sorted(per_card.items())),
    }, indent=1, default=str) + "\n")
    msg = f"recorded {BASELINE} ({card}: {len(cells)} cells, ceilings {fp})"
    # Said out loud. A cell that silently did not update is indistinguishable from one that did,
    # and the whole point of keeping it is that the stronger result cost card time.
    if kept:
        msg += ("\n  this run decided nothing for these, so the recorded result stands: "
                + "; ".join(kept))
    if dropped:
        msg += ("\n  the card never opened for these and nothing was recorded before, so no "
                "cell was written and the coverage check will report them missing: "
                + "; ".join(dropped))
    return msg


def _record_from(path: Path) -> int:
    """Fold a completed run's report into the baseline without opening a device.

    A full sweep at this bar is hours and a bisect alone can be hours, so the campaign runs in
    stages; re-running a stage just to reach `--record` would double the card time it cost.

    The bar guard is the point. `record_baseline` stamps the file with the report's OWN
    `bar_tokens` while keeping prior cells measured at `TOKEN_BAR`, so folding in a 1408 report
    would restamp a baseline full of 1536 cells as 1408 -- p1's defect 9 exactly, arriving
    through a new door. A bisect report is the likely input here and every rung below the bar is
    a different bar, so this is the one place that mistake is easy to make.
    """
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        print(f"--record-from {path}: cannot read a report out of it ({e})", file=sys.stderr)
        return 2
    if report.get("bar_tokens") != TOKEN_BAR:
        print(f"--record-from {path}: measured at bar {report.get('bar_tokens')}, and this tree's "
              f"bar is {TOKEN_BAR}. Recording it would stamp a baseline of {TOKEN_BAR} cells with "
              f"another bar's number. Re-run at {TOKEN_BAR} instead.", file=sys.stderr)
        return 2
    if not report.get("results"):
        print(f"--record-from {path}: no results in it, nothing to record.", file=sys.stderr)
        return 2
    measured = {r["model"] for r in report["results"]}
    # Partial unless this run actually covered the whole derived roster: a partial record leaves
    # the other cells standing, a full one replaces them.
    print(record_baseline(report, partial=measured != set(roster())))
    return 0


def _screen_only(worker, cell, work, hookdir, depth) -> dict:
    rec = {"model": cell.model, "verb": cell.verb, "worker": repr(worker), "legs": []}
    if cell.model in EXEMPT:
        rec.update(verdict="SKIPPED", reason=EXEMPT[cell.model])
        return rec
    f = fixture_for(cell, TOKEN_BAR, work, depth)
    rec.update(tokens_requested=f["tokens_requested"], tokens_padded=f["tokens_padded"],
               residues=f["residues"],
               msa_rows_effective=f["effective_depth"] if cell.msa else 0)
    scr = _screen(worker, cell, f, work, hookdir)
    rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR))
    if scr["verdict"] in NOTHING_RAN_VERDICTS and wait_for_card(worker):
        scr = _screen(worker, cell, f, work, hookdir)
        rec["legs"].append(dict(scr, tier="screen", tokens=TOKEN_BAR, retry=True))
    rec.update(verdict=scr["verdict"], decided_by="screen", mechanism=scr["mechanism"],
               wall_s=scr["wall_s"], stacks_truncated=scr.get("stacks_truncated"),
               block_calls=scr.get("block_calls"), note=scr.get("note"))
    # Not `!= "FAIL"`: an absent or unusable checkpoint is exactly what makes a leg exit
    # nonzero, so guarding this on "not already a FAIL" would skip every case it is for.
    if scr["verdict"] in ("FAIL", "HOST_OOM") and _no_weights(scr["tail"]):
        rec["verdict"] = "NO_WEIGHTS"
    elif screen_reduction_is_unsafe(scr):
        # --tier screen has no un-truncated leg to fall through to, so it must not report the
        # FAIL. Say which tier can decide it instead of scoring a bar nobody walked.
        rec.update(verdict="INCONCLUSIVE", screen_unsafe=True,
                   note="the screen's own depth cut is a candidate cause of this FAIL and there "
                        "is no capacity mechanism in the error, so Tier 1 cannot decide this "
                        "model. Run it with --tier both.")
    return rec


def _finish(report: dict) -> None:
    v = [r["verdict"] for r in report["results"]]
    report["counts"] = {
        "PASS": v.count("PASS"),
        "fail_like": sum(v.count(x) for x in ("FAIL", "STALL", "HOST_OOM", "ERROR")),
        "CONTENDED": v.count("CONTENDED") + v.count("CARD_DIRTY"),
        "SKIPPED": v.count("SKIPPED"),
        "NO_WEIGHTS": v.count("NO_WEIGHTS"),
        "INCONCLUSIVE": v.count("INCONCLUSIVE"),
        "GATE_BUG": v.count("GATE_BUG"),
        "BAD_FIXTURE": v.count("BAD_FIXTURE"),
    }


if __name__ == "__main__":
    sys.exit(main())
