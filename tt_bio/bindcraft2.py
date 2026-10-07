"""Run BindCraft 2's binder design loop with tt-bio's AlphaFold 2 Evoformer on a Tenstorrent card.

BindCraft 2 (https://github.com/PacesaLab/BindCraft2) designs binders by differentiating
AlphaFold 2 through a sequence. It takes its predictor as a parameter everywhere and constructs
one in a single place, so a backend is a class rather than a fork of the design loop. This module
is that class and the device trunk behind it::

    import bindcraft.campaign as campaign
    from tt_bio import bindcraft2

    with bindcraft2.campaign_predictor(card=0):
        campaign.run_campaign(settings, project, af2_weights=params, mpnn_weights=mpnn)

Everything between BindCraft 2's protein states and the trunk stays BindCraft 2's own code:
padding, templates, the input features, the structure module, the Kabsch alignment, the
confidence heads, the filters, the ranking and the step budget. Only AlphaFold 2's 48 Evoformer
blocks move to the card, which is where every O(L^3) op in the trunk lives.

``trunk="jax"`` is the control arm. Same class, same call path, same bucket rounding, BindCraft 2's
own trunk, so a device result is read against that rather than against a differently shaped
program.

A checkpoint the card does not hold folds on that same host trunk instead of stopping the
campaign, and the validation ensemble folds there by default: it is the instrument that decides
whether a design is accepted, and running it on card would put device numerics inside the
measurement that grades the device.

tt-bio neither ships BindCraft 2 nor serves it. Install it yourself; this module only binds to it
if it is importable.
"""
from __future__ import annotations

import collections
import contextlib
import functools
import json
import os
import pathlib
import re
import sys
import threading
from collections.abc import Mapping
from typing import Callable, Iterator

import numpy as np
import torch

from tt_bio import bcinputs, duotraj
#: Changing BindCraft 2's loss is a seam of its own (`docs/bindcraft2.md`, "Custom loss"), kept in
#: its own module because it touches BindCraft 2's loss registry and nothing on the card. It is
#: re-exported here so a user has one import: `bindcraft2.loss_terms`, `bindcraft2.terms`,
#: `bindcraft2.check_gradient`.
from tt_bio.bindcraft2_loss import (  # noqa: F401
    INTERMEDIATES, DeadGradient, LossTermError, NewTerm, Term, TermSignature, UnknownTerm,
    check_gradient, loss_term, loss_terms, synthetic_design, terms,
)

#: tt-bio's token axis buckets to 32, and rounding a design UP is faster than running it ragged:
#: the PD-L1 complex at 211 tokens costs 4.504 s on the trunk forward and the same design padded
#: to 224 costs 1.369 s. The splice pads, masks what it added and slices the result back.
TOKEN_BUCKET = 32

#: AlphaFold 2's Evoformer depth. The splice refuses a stack of any other length rather than
#: running a partial trunk.
EVOFORMER_BLOCKS = 48


def _pad32(n: int) -> int:
    return -(-n // TOKEN_BUCKET) * TOKEN_BUCKET


def design_tokens(settings: Mapping) -> int:
    """The padded token axis a campaign with these settings will run at, or 0 if it cannot be
    read.

    BindCraft 2 sizes its own prediction with `design_residue_count`, which parses the target and
    adds the longest binder, both padded to the campaign's bucket. Asking it rather than
    re-deriving the axis keeps the two in step, including for scaffolds, oligomers and cropped
    targets, where a count of our own would be wrong in a different way for each.

    Returns 0 rather than raising: this is only ever used to SIZE a default, and a default that
    cannot read the design falls back to one trajectory, which is BindCraft 2's own loop.
    """
    try:
        from bindcraft.protein_preparation import design_residue_count

        return int(design_residue_count(dict(settings)))
    except Exception:
        return 0


#: The largest token axis measured to complete a BindCraft 2 gradient round on one p150a
#: (34.226 GB of DRAM): 864 tokens, holding 31.39 GB resident with 2.84 GB free, once the fused
#: triangle attention serves 32 * p axes by padding up (`tenstorrent._tri_att_hifi_pad_up`).
#: Measured on qb1, 2026-09-30, `perf/b2p_ceiling/`, where the axis is the one the Evoformer seam
#: ran rather than `target_residues + binder` -- those differ, and an earlier value of 544 here
#: came from the arithmetic. It is quoted in a refusal as a reference point and is enforced
#: nowhere: the allocator decides, and a board with more DRAM has a different answer.
MEASURED_MAX_TOKENS_P150A = 864
#: The same card with the pad-up off (`TT_BIO_TRIATT_HIFI_PAD_UP=0`): 576 completes and 608
#: refuses, because the fused arm declines every call at 608 and the composed path asks for one
#: 3.596 GB score tensor. Measured on qb1, 2026-09-29, `state/bgx-size.md`, and again on a
#: p300c chip (same DRAM) on 2026-09-30, `perf/b2p_padup/out/bc2/off608`.
MEASURED_MAX_TOKENS_P150A_NO_PAD_UP = 576
#: The DRAM that figure was measured against, as the allocator reports it: 8 banks of
#: 4,278,190,016 B. A card with less refuses smaller complexes, so the 576 is no reference
#: point there -- which is why the Wormhole row below exists.
P150A_DRAM_BYTES = 8 * 4_278_190_016

#: The same measurement on one chip of a Wormhole Galaxy: 512 tokens completes a gradient round
#: and 544 refuses, in the Evoformer backward, with the card full at 12.756 GB of 12.885 GB.
#: Measured on dev .107 card 30, 2026-09-29, `state/bwx-bringup.md`. 576 refuses as well and
#: refuses with the fused triangle attention arm in place, which is what rules out 544 being a
#: hole in the L1 clash class rather than the top of the card: at 544 the arm declines and the
#: composed path holds 2.576 GB the fold would not otherwise need, so that rung alone could not
#: settle it. Quoted like the p150a figure and enforced nowhere: the allocator decides.
MEASURED_MAX_TOKENS_WH_GALAXY = 512
#: 12 banks of 1,073,741,792 B, as the allocator reports them.
WH_GALAXY_DRAM_BYTES = 12 * 1_073_741_792

#: Every board the ladder has actually been run on, by the DRAM the allocator reports: the
#: largest token axis measured to complete a gradient round on one. A refusal quotes the row
#: for the board in hand, because quoting another board's ceiling is how a user gets told their
#: fold should have fitted when on this card it never could.
_MEASURED_CEILINGS = (
    (P150A_DRAM_BYTES, MEASURED_MAX_TOKENS_P150A, "p150a"),
    (WH_GALAXY_DRAM_BYTES, MEASURED_MAX_TOKENS_WH_GALAXY, "Wormhole Galaxy chip"),
)

#: How the gradient round spends device memory, cheapest in time first.
#:
#: ``fast`` checkpoints each Evoformer, extra-MSA and template block and keeps everything else on
#: the card. Its peak is one block's recompute plus one pinned input per block.
#: ``lean`` also checkpoints each residual step inside a block (the outer product mean, the three
#: MSA steps, both triangle multiplications, both triangle attentions, the transition), so a
#: block's backward holds one step's tape instead of nine. It runs every block's forward once
#: more a round.
#: ``offload`` is ``lean`` with the pinned block inputs moved to host memory between the forward
#: and the backward: one download and one upload of each a round, and no pins on the card.
#: All three run the same ops on the same values; only what the card holds in between differs.
MEMORY_MODES = ("fast", "lean", "offload")

#: Device bytes a gradient round peaks at, per mode: a constant plus a count of bf16
#: `[N, N, 128]` pair tensors (256 B a token pair). `fast` is `bcw-census`'s account, which the
#: Wormhole and Blackhole ladders fit to 0.01-0.03 GB (`state/bcw/MEMORY.md`).
#: `lean` is fitted to a measured round: 544 tokens peaked at 7.527 GB on a Wormhole Galaxy
#: chip, which is 90 pair tensors, not the 95 the census projected.
#: `offload` is fitted to the 800-token round that COMPLETED: 8.952 GB is 50 pair tensors. The
#: earlier 768 figure (5.685 GB, 33 tensors) was what the round had reached when a separate L1
#: limit stopped it in the backward, so it was a floor and not a peak -- a refused round cannot
#: calibrate a mode, and this constant is what `auto` decides on.
_MODE_BASE_BYTES = 0.71e9
_MODE_PAIR_TENSORS = {"fast": 161, "lean": 90, "offload": 50}
#: The share of a card's DRAM a round can actually hold. A Wormhole Galaxy chip refused 544
#: tokens with a 12.701 GB resident frontier on 12.885 GB: past ~98 % the next pair-sized
#: buffer finds no contiguous room.
_MODE_USABLE = 0.978


def round_device_bytes(mode: str, padded: int) -> int:
    """What a gradient round at this token axis peaks at on the card in `mode`."""
    return int(_MODE_BASE_BYTES + _MODE_PAIR_TENSORS[mode] * 256 * int(padded) ** 2)


def memory_mode(requested: str, padded: int, card_bytes: int) -> str:
    """The mode a fold at `padded` tokens runs in. ``auto`` is the cheapest that fits the card,
    and the last one when none does, so a fold too big for every mode still tries the leanest
    and refuses with the allocator's numbers rather than a guess."""
    if requested != "auto":
        if requested not in MEMORY_MODES:
            raise ValueError(f"memory={requested!r}: expected 'auto' or one of {MEMORY_MODES}")
        return requested
    for mode in MEMORY_MODES:
        if padded <= max_tokens(mode, card_bytes):
            return mode
    return MEMORY_MODES[-1]


#: Per board and mode, the ladder that was actually run: the largest token axis measured to
#: COMPLETE a gradient round, and the smallest measured to REFUSE. Both halves are needed,
#: because only the pair brackets a ceiling. A completing axis on its own is a FLOOR on what the
#: mode holds, and the memory law alone is an estimate that can sit either side of the truth.
#:
#: The `offload` row is why this table exists. The law puts `offload` near 960 tokens on a
#: Wormhole Galaxy chip, and 928 was measured to refuse: it peaked at 11.197 GB with 1.687 GB
#: free and was then refused an 882 MB buffer, because the largest free block was 70 MB. The law
#: tracks the PEAK to within 0.07 GB across 768/800/832/864/896 -- what it cannot see is
#: fragmentation, and that is what ends the mode. So the law overshot by two buckets, in the one
#: sentence of a refusal the user acts on. Measured on dev .107 card 30, 2026-10-01, chain `wh7`.
#: 896 then refused on its third round twice out of two (65.3 MB largest free block against a
#: 68.5 MB per-bank need), while 864 completed five of five, so 864 is the top.
#: Measured at the v0.12.0 tree, `rel012-verify-wh`.
_MEASURED_MODE_LADDERS = {
    "Wormhole Galaxy chip": {
        # mode: (largest axis measured to complete, smallest measured to refuse or None)
        "fast": (512, 544),
        "lean": (544, 768),
        "offload": (864, 896),
    },
    # The law puts fast's top at 864 on a p150a, and fast refuses there at 704, 768 and 832 (704
    # and 832 fill the card in the backward, 768 fragments in the forward with 2.2 GB free), while
    # lean serves 768, 800, 832, 864 and 896. Fast is not monotone in the axis on this card: 736
    # and 864 each completed once in fast. A ladder with a refusal under it is not a ceiling, so
    # fast is held below its lowest refusal, where 576, 608, 640 and 672 all completed. No
    # `offload` row: on pc every offload rung was killed by the HOST for want of RAM, which is
    # not a device answer. Measured on pc card 0, 2026-10-01, `rel012-verify-bh`.
    "p150a": {
        "fast": (672, 704),
        "lean": (896, None),
    },
}


def mode_ceiling(mode: str, card_bytes: int):
    """``(tokens, measured)``: the largest axis to offer for `mode` on this card, and whether a
    fold of that size has actually completed one.

    Three cases, and the difference between them is what the refusal is allowed to claim:

    * the ladder bracketed the ceiling on ADJACENT buckets -- 864 completes, 896 refuses -- so
      the ceiling IS 864 and it is measured;
    * the ladder found a refusal further up, with a gap nobody ran. `lean` completes at 544 and
      refuses at 768, so the ceiling is somewhere in 544-736: the law's estimate is the best
      number inside that bracket, and 736 is the most that may be claimed whatever it says;
    * no ladder for this board and mode, so the law's estimate stands on its own.
    """
    law = TOKEN_BUCKET
    while round_device_bytes(mode, law + TOKEN_BUCKET) <= card_bytes * _MODE_USABLE:
        law += TOKEN_BUCKET
    board = _measured_board(card_bytes)
    rung = _MEASURED_MODE_LADDERS.get(board[2], {}).get(mode) if board else None
    if rung is None:
        return law, False
    completes, refuses = rung
    if refuses is None:
        return max(law, completes), law <= completes
    if refuses == completes + TOKEN_BUCKET:
        return completes, True
    return min(law, refuses - TOKEN_BUCKET), False


def max_tokens(mode: str, card_bytes: int) -> int:
    """The largest token axis to offer for `mode` on a card of `card_bytes`.

    Capped by a measured refusal where the ladder found one: the law is fitted to peak bytes and
    is blind to the fragmentation that actually stops `offload`, so where the two disagree the
    card's own answer wins.
    """
    return mode_ceiling(mode, card_bytes)[0]


#: tt-metal's allocator refusal, which carries every number a user needs and is buried under
#: forty lines of C++ backtrace by the time JAX has finished wrapping it. Per bank, except the
#: buffer size itself.
_ALLOCATOR_REFUSAL = re.compile(
    r"allocate (?P<want>\d+) B (?P<space>\w+) buffer across (?P<banks>\d+) banks, "
    r"where each bank needs to store (?P<per_bank>\d+) B, "
    r"but bank size is (?P<bank_size>\d+) B "
    r"\(allocated: (?P<allocated>\d+) B, free: (?P<free>\d+) B, "
    r"largest free block: (?P<largest>\d+) B\)")


#: Below this share of the card free, a refusal is a full card however the rest is scattered.
_FRAGMENTED_MIN_FREE = 0.05


def _gb(b: float) -> str:
    return f"{b / 1e9:.3f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"


def _measured_board(card_total: int):
    """The `_MEASURED_CEILINGS` row for the board in hand, or None for one nobody laddered.

    Matched on reported DRAM within 5 %, not on equality: the figure is a bank count times a
    bank size the allocator prints, and a board that reserves a little differently is still
    that board.
    """
    for dram, cap, name in _MEASURED_CEILINGS:
        if abs(card_total - dram) <= dram // 20:
            if cap == MEASURED_MAX_TOKENS_P150A and not _pad_up_on():
                cap = MEASURED_MAX_TOKENS_P150A_NO_PAD_UP
            return dram, cap, name
    return None


def _pad_up_on() -> bool:
    """Whether the fused triangle attention pads 32 * p axes up, which the p150a ceiling rests on."""
    from tt_bio import tenstorrent
    return tenstorrent._TRIATT_HIFI_PAD_UP_TILES > 0


def _size_aware_refusal(exc: BaseException, *, phase: str, n: int, padded: int,
                        mode: str = "fast"):
    """An allocator refusal rewritten to name the size that caused it, or None.

    What a researcher sees without this is a `JaxRuntimeError` wrapping ten Python frames, a
    tt-metal `TT_FATAL`, and forty lines of raw C++ hex, with the token axis, the memory it
    wanted and the size that would have fitted nowhere in it. Every number added here is read
    off the allocator's own line or off the fold in hand, so none of it can drift from the
    failure it describes.

    None means `exc` is not an allocator refusal and the caller must re-raise it unchanged: a
    wrapper that swallows the shape of an unrelated bug is worse than no wrapper.
    """
    hit = _ALLOCATOR_REFUSAL.search(str(exc))
    if hit is None:
        return None
    g = {k: int(v) for k, v in hit.groupdict().items() if v.isdigit()}
    space = hit.group("space")
    banks, want, per_bank = g["banks"], g["want"], g["per_bank"]
    free_total, largest, bank_size = g["free"] * banks, g["largest"], g["bank_size"]
    card_total, held = bank_size * banks, g["allocated"] * banks

    # Two refusals wear the same words and take different remedies. The card is full when the
    # free memory could not hold the request even in one piece, and also when what is free is
    # a sliver of the card: at Wormhole's 544-608 boundary the fold holds 98-99 % of the chip,
    # and whether the last 100-250 MB happens to cover the next request decided between "full"
    # and "not a full card" for the same state (`perf/b2p_wh/results/`, `perf/bwx_bringup/`).
    # Fragmentation is the diagnosis only when a real share of the card is free and scattered.
    fragmented = (free_total >= want and largest < per_bank
                  and free_total >= card_total * _FRAGMENTED_MIN_FREE)
    if fragmented:
        diagnosis = (
            f"This is fragmentation, not a full card: {_gb(free_total)} is free, which would "
            f"cover the {_gb(want)} request if it were in one piece, but the largest "
            f"contiguous block in a bank is {_gb(largest)} against the {_gb(per_bank)} that "
            f"bank needs.")
    elif free_total >= want:
        diagnosis = (
            f"The card is full: {_gb(held)} of {_gb(card_total)} is held by this fold, and the "
            f"{_gb(free_total)} left is in pieces of at most {_gb(largest)} a bank against the "
            f"{_gb(per_bank)} a bank this {_gb(want)} request needs.")
    else:
        diagnosis = (
            f"The card is full: {_gb(free_total)} free against a {_gb(want)} request, with "
            f"{_gb(held)} of {_gb(card_total)} already held by this fold.")

    # The ceiling to compare against is THIS board's, where one has been measured. A Wormhole
    # Galaxy chip stops at 512 and a p150a at 576, so a Wormhole user told the p150a number is
    # told their 544-token fold should have fitted, and sent looking for a phantom co-tenant.
    board = _measured_board(card_total)
    cap = board[1] if board else MEASURED_MAX_TOKENS_P150A
    board_name = board[2] if board else "p150a"
    # Those rows are `fast`-mode ladders. A fold refusing in a leaner mode has already passed
    # the fast ceiling -- 544 tokens run in `lean` on a chip whose measured row says 512 -- so
    # quoting the row would tell that user to drop to a size they are already above. A leaner
    # mode's ceiling is its own, and never below the measured row's.
    # Whether `cap` is still a MEASURED number. A mode's own ceiling comes from the memory law
    # fitted to the folds that mode was measured on, which is not the same thing as a fold of
    # that size having been run: on a Wormhole Galaxy chip the law puts `offload` near 960 and
    # the largest axis anyone has actually completed is 800. Calling the estimate "measured" in
    # a refusal would have the message overstate exactly the number the user is about to act on.
    cap_measured = True
    cap_mode = None
    if mode != MEMORY_MODES[0]:
        mode_cap, mode_cap_measured = mode_ceiling(mode, card_total)
        if mode_cap > cap:
            cap, cap_measured, cap_mode = mode_cap, mode_cap_measured, mode
    # A card nobody laddered and smaller than a p150a has no ceiling to compare against, so
    # a refusal there is always read as the size.
    unmeasured_smaller = board is None and card_total < P150A_DRAM_BYTES

    # Where the way down LANDS. One bucket down is the right answer only when that bucket is at
    # or under this board's ceiling. It often is not: a 608-token fold on a Wormhole Galaxy chip
    # was told to try 576, which refuses too, and a 736-token fold on a p150a was told to try
    # 704, which refuses twice over -- so the one actionable sentence in the message sent the
    # user to another failure and cost them the compile to find out. Measured at 544, 576 and
    # 608 on one Galaxy chip: `state/b2p-wh.md`. A board nobody laddered keeps the one-bucket
    # step, because `cap` there is a p150a's number and not this card's.
    landing = padded - TOKEN_BUCKET
    if board is not None and padded > cap:
        landing = min(landing, cap)
    drop = n - landing
    way_down = (
        f"{drop} residues off the binder takes this fold to {landing} tokens"
        if landing == padded - TOKEN_BUCKET else
        f"this fold has to lose {drop} residues to reach {landing} tokens -- the next bucket "
        f"down, {padded - TOKEN_BUCKET}, refuses on this board too, so {landing} is the size to "
        f"aim at and the binder on its own may not be long enough to give them up")
    # It opens a sentence in one branch and closes one in the other.
    sentence = way_down[0].upper() + way_down[1:] if way_down[0].isalpha() else way_down

    # The slower memory modes are the way to keep the whole complex, and a refusal that does not
    # name them leaves a researcher cropping a target the card could have held.
    later = MEMORY_MODES[MEMORY_MODES.index(mode) + 1:] if mode in MEMORY_MODES else ()
    roomier = [m for m in later if max_tokens(m, card_total) >= padded]
    if roomier:
        escape = (f"This fold ran in the {mode!r} memory mode. The {roomier[0]!r} mode should "
                  f"hold {padded} tokens on this card, slower: {_MODE_COST[roomier[0]]}. Pass "
                  f"memory={roomier[0]!r}, or leave memory='auto' and it is picked for you "
                  f"(docs/bindcraft2.md, 'Large complexes'). ")
    else:
        last_top, last_measured = mode_ceiling(MEMORY_MODES[-1], card_total)
        # "tops out NEAR 960" and "tops out AT 864" are different promises, and the user trims to
        # whichever number is in the sentence.
        tops = (f"tops out at {last_top} tokens on this card, the largest axis measured to "
                f"complete a gradient round in it" if last_measured else
                f"tops out near {last_top} tokens on this card")
        if mode != MEMORY_MODES[-1]:
            escape = f"Even the slowest memory mode, 'offload', {tops}. "
        else:
            escape = f"This fold already ran in the slowest memory mode, 'offload', which {tops}. "

    how_cap = ("the largest axis measured to complete a gradient round is" if cap_measured
               else "this mode should hold about")
    if padded > cap or unmeasured_smaller:
        reference = (
            (f"The largest axis measured to complete a gradient round on one {board_name} "
             f"({_gb(board[0])})"
             + (f" in the {cap_mode!r} mode" if cap_mode else "")
             + f" is {cap} tokens."
             if cap_measured else
             f"About {cap} tokens is what one {board_name} ({_gb(board[0])}) should hold in the "
             f"{mode!r} mode -- estimated from the memory folds in this mode were measured to "
             f"use, rather than a fold of that size anyone has run.")
            if board else
            f"This card has {_gb(card_total)}, less than the {_gb(P150A_DRAM_BYTES)} of the "
            f"p150a where {MEASURED_MAX_TOKENS_P150A} tokens is the largest axis measured to "
            f"complete a gradient round, so its own ceiling is lower.")
        action = "What to do: " + escape + (
            f"Otherwise run a smaller complex. The token axis is the complex BindCraft 2 "
            f"built, padded to a multiple of {TOKEN_BUCKET} -- it is LARGER than target "
            f"residues + binder length, so size the job off the {n} above and not off that "
            f"sum. {sentence}. {reference} Trimming the "
            f"target to the domain you are binding is the other lever and usually the bigger "
            f"one.")
    elif duotraj.GATE is not None:
        action = (
            f"What to do: {padded} tokens fits on a {board_name} with the card to itself "
            f"({how_cap} {cap}), so something else "
            f"is holding this card. Interleaved trajectories are the usual cause: pass "
            f"trajectories_per_card=1 to run BindCraft 2's own one-at-a-time loop. Otherwise "
            f"{way_down}.")
    else:
        # One trajectory already. Advising trajectories_per_card=1 here changes nothing, and a
        # p150a refused 768 and 832 in fast mode held alone (`state/rel012-verify-bh.md`).
        action = (
            f"What to do: {padded} tokens is within what a {board_name} holds "
            f"({how_cap} {cap}), and this fold was the only trajectory on the card. Another "
            f"process on this card is one cause (tt-smi lists it), but a fold can also refuse "
            f"under the measured ceiling on a card it holds alone. "
            + (escape if roomier else "") +
            f"Otherwise {way_down}.")

    return MemoryError(
        f"BindCraft 2 ran out of device memory in the Evoformer {phase} at {padded} tokens.\n"
        f"  complex   {n} residues, padded to {padded} tokens "
        f"(tt-bio buckets the token axis to {TOKEN_BUCKET})\n"
        f"  asked for {_gb(want)} in one {space} buffer "
        f"({_gb(per_bank)} in each of {banks} banks)\n"
        f"  free      {_gb(free_total)} of {_gb(card_total)}, largest contiguous block "
        f"{_gb(largest)}\n"
        f"{diagnosis}\n{action}\n"
        f"The allocator's own refusal follows.")


#: Every refusal `_refusal_names_the_size` has raised, newest last. A refusal raised inside a
#: `jax.pure_callback` reaches the caller only as a STRING inside a `JaxRuntimeError`, so the
#: object is kept here and handed back by `unwrap_device_refusal` rather than parsed back out of
#: a traceback. Bounded, because only the text of a refusal still in flight can be matched.
_REFUSALS_RAISED: "collections.deque[MemoryError]" = collections.deque(maxlen=8)
_REFUSALS_LOCK = threading.Lock()


@contextlib.contextmanager
def _refusal_names_the_size(phase: str, n: int, padded: int,
                            memory: "_Memory | None" = None) -> "Iterator[None]":
    """Name the token axis on the way out of a device seam. A no-op unless it refuses."""
    try:
        yield
    except Exception as exc:
        mode = memory.used.get(padded, "fast") if memory is not None else "fast"
        better = (_size_aware_refusal(exc, phase=phase, n=n, padded=padded, mode=mode)
                  or _l1_refusal_names_the_size(exc, phase=phase, n=n, padded=padded))
        if better is None:
            raise
        with _REFUSALS_LOCK:
            _REFUSALS_RAISED.append(better)
        raise better from exc


#: tt-metal's L1 refusal. Not an allocator refusal: it names core coordinates and a per-core
#: limit, and the card can be GB-free when it fires.
_L1_REFUSAL = re.compile(
    r"[Cc]ircular buffers on core range .*? grow to (?P<want>\d+) B which is beyond max L1 size "
    r"of (?P<limit>\d+) B")


def _l1_refusal_names_the_size(exc: BaseException, *, phase: str, n: int, padded: int):
    """An L1 circular-buffer refusal rewritten so it does not read as an out-of-memory, or None.

    The two are told apart by where the memory is: DRAM is the card and L1 is 1.5 MB inside each
    Tensix, so this one fires with the card gigabytes free and no amount of offloading or
    checkpointing touches it. A user who reads it as an OOM crops their target for nothing.

    It is a property of the TOKEN AXIS and not of the fold's size in bytes, and not even monotone
    in it: `in0_block_w` is the largest divisor of the axis in tiles that is at most 8, so a 768
    axis (24 tiles, divisible by 8) asks for the widest K block and refuses where 800 (25 tiles)
    and 832 (26) do not. Measured on one Wormhole Galaxy chip, 2026-10-01: 800 tokens completed a
    gradient round while 768 refused. So the way out is a different axis, up OR down, which is the
    opposite of the advice an OOM deserves -- and `tt_bio.autograd.bmm` already retries a narrower
    K block before any of this is reached, so arriving here means even the narrowest did not fit.
    """
    hit = _L1_REFUSAL.search(str(exc))
    if hit is None:
        return None
    want, limit = int(hit.group("want")), int(hit.group("limit"))
    return MemoryError(
        f"BindCraft 2 could not fit a kernel of the {phase} at {padded} tokens into a core's L1.\n"
        f"  complex   {n} residues, padded to {padded} tokens (tt-bio buckets the token axis "
        f"to {TOKEN_BUCKET})\n"
        f"  asked     {_gb(want)} of circular buffers against the {_gb(limit)} one Tensix has\n"
        f"This is NOT the card running out of memory -- L1 is per-core and the card's DRAM is "
        f"unrelated, so a slower memory mode will not help and cropping the target may not "
        f"either. It depends on the token axis in tiles rather than on the size: this axis "
        f"happens to ask for the widest contraction block. Another bucket, up or down, is likely "
        f"to run; {padded + TOKEN_BUCKET} is the one to try first.")


def unwrap_device_refusal(exc: BaseException) -> BaseException:
    """`exc` itself, or the clean refusal that `exc` is a stringified copy of.

    A refusal raised in the Evoformer BACKWARD does not reach the caller as it was raised. The
    backward runs inside a `jax.pure_callback`, and JAX turns any exception a callback raises
    into `JaxRuntimeError("INTERNAL: CpuCallback error calling callback: Traceback (most recent
    call last): ...")`. The refusal's text survives inside that string, but six frames of
    `jax/_src/callback.py` and `contextlib` arrive in front of it and `except MemoryError` no
    longer catches it -- so the size-aware refusal is there and the user reads JAX internals
    first.

    Which seam refuses is a property of the BOARD, which is why this is needed at all. On a
    Wormhole Galaxy chip the card saturates in the backward, so every refusal above 512 tokens
    is wrapped; on a p150a the 576-token ceiling is passed in a forward that runs outside the
    callback and the same mistake arrives clean (`state/b2p-wh.md`). Without this the two boards
    give a researcher who made one mistake two different answers, and only one of them is
    readable.

    Matching is by the identity of the text, not a parse of it: the refusal object raised at the
    seam is kept in `_REFUSALS_RAISED`, and `exc` is replaced only when it carries that exact
    message. Anything else is returned untouched -- a wrapper that guesses at the shape of an
    unrelated bug is worse than no wrapper.
    """
    if isinstance(exc, MemoryError):
        return exc
    text = str(exc)
    with _REFUSALS_LOCK:
        raised = list(_REFUSALS_RAISED)
    for refusal in reversed(raised):
        if str(refusal) in text:
            return refusal
    return exc


@contextlib.contextmanager
def refusals_unwrapped() -> "Iterator[None]":
    """Let a device refusal out of this block as the `MemoryError` it was raised as.

    Wrapped around the user-facing entries, so `except MemoryError` holds on both boards no
    matter which seam ran out of room. See `unwrap_device_refusal`.
    """
    try:
        yield
    except BaseException as exc:                                          # noqa: BLE001
        better = unwrap_device_refusal(exc)
        if better is exc:
            raise
        raise better from exc


def _fused_hifi_counts() -> "tuple[int, int]":
    """`(served, declined)` for the fused triangle attention, or `(0, 0)` off the device path."""
    try:
        from tt_bio import tenstorrent
        s = tenstorrent.TRIATT_FUSED_HIFI_STATS
        return int(s["served"]), int(s["declined"])
    except Exception:
        return 0, 0


def pin_card(card: int | str) -> None:
    """Pin this process to one physical card.

    ttnn reads ``TT_VISIBLE_DEVICES`` when it is imported and brings up every card it names, so
    the choice has to be made before that import and cannot be changed afterwards. Calling this
    once ttnn is loaded raises rather than pretending the argument was honoured.
    """
    want = str(card)
    have = os.environ.get("TT_VISIBLE_DEVICES")
    if have == want:
        return
    if "ttnn" in sys.modules:
        raise RuntimeError(
            f"ttnn is already imported with TT_VISIBLE_DEVICES={have!r}, so card={card} cannot "
            f"be honoured. Pin the card before importing ttnn, or start the process with "
            f"TT_VISIBLE_DEVICES={card} in its environment.")
    os.environ["TT_VISIBLE_DEVICES"] = want


# ----------------------------------------------------------------- the checkpoints, on card


def _is_multimer(path: pathlib.Path) -> bool:
    """Whether an AlphaFold 2 npz is a `multimer_v3` checkpoint, read off the file itself.

    The two families need a different weight remap and a different `AF2Model`, and loading one
    as the other raises a `KeyError` from inside the remap rather than returning a wrong model.
    The tell is the relative encoding: multimer_v3 embeds residue offsets through
    `~_relative_encoding/position_activations`, the monomer through `pair_activiations`
    (AlphaFold's own spelling). Reading the array names rather than the file name keeps a
    renamed or re-exported checkpoint from being loaded as the wrong family.

    The shipped `examples/pdl1.json` designs on all five `multimer_v3` checkpoints, so for
    BindCraft 2 this is the common case rather than the exotic one.
    """
    with np.load(path, allow_pickle=False) as npz:
        return any("~_relative_encoding" in name for name in npz.files)


class _Trunk:
    """One AlphaFold 2 checkpoint's Evoformer blocks on the card, under tt-bio's tape."""

    def __init__(self, path: pathlib.Path, *, template: bool = False):
        import ttnn

        from tt_bio import autograd, taped_ttnn
        from tt_bio.af2 import load_af2_device_model
        from tt_bio.af2_weights import load_af2_state_dict

        self.ttnn, self.ag, self.taped = ttnn, autograd, taped_ttnn
        self.multimer = _is_multimer(path)
        # `template` costs two more c=64 pair blocks of weights per checkpoint and is only
        # wanted by `TemplateOnDevice`, so it stays off unless that swap is armed: five trunks
        # already sit near the allocator's limit at n=288 (`TrunkPool.resident`).
        self.model = load_af2_device_model(load_af2_state_dict(str(path),
                                                               multimer=self.multimer),
                                           template=template, multimer=self.multimer,
                                           trunk_dtype=torch.bfloat16)
        self.device = self.model._device
        self.blocks = len(self.model.device_evoformer)
        self.extra_blocks = len(self.model.device_extra_msa)
        self.template_blocks = len(self.model.device_template)

    def up(self, t: torch.Tensor):
        return self.ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),
                                    layout=self.ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=self.ttnn.bfloat16)

    def down(self, t, shape) -> torch.Tensor:
        x = torch.Tensor(self.ttnn.to_torch(t)).float()
        while x.dim() > len(shape) and x.shape[0] == 1:
            x = x.squeeze(0)
        if tuple(x.shape) != tuple(shape):
            raise ValueError(f"the trunk returned {tuple(x.shape)}, expected {tuple(shape)}")
        return x

    def leaf(self, t: torch.Tensor):
        return self.ag.Tensor(self.up(t), requires_grad=True)

    def sync(self) -> None:
        self.ttnn.synchronize_device(self.device)

    def card_bytes(self) -> int:
        """This chip's DRAM as the allocator reports it."""
        mv = self.ttnn.get_memory_view(self.device, self.ttnn.BufferType.DRAM)
        return int(mv.total_bytes_per_bank) * int(mv.num_banks)

    def arm(self, mode: str) -> bool:
        """Set every block of this trunk to `mode` (see `MEMORY_MODES`). True when the
        checkpoint pins go to host. Called at every seam, forward and backward, because the
        blocks are shared by every trajectory on the card and only one seam runs at a time."""
        run = self.ag.substep if mode != "fast" else None
        model = self.model
        for block in (*model.device_evoformer, *model.device_extra_msa, *model.device_template):
            block.step_runner = run
        return mode == "offload"

    def evoformer(self, m, z, msa_mask, pair_masks, recompute: bool, offload: bool = False):
        for block in self.model.device_evoformer:
            if recompute:
                m, z = self.ag.checkpoint(
                    lambda a, b, blk=block: blk(a, b, msa_mask, *pair_masks), m, z,
                    offload=offload)
            else:
                m, z = block(m, z, msa_mask, *pair_masks)
        return m, z

    def extra_msa(self, z, pair_masks, recompute: bool, offload: bool = False):
        """The extra-MSA stack's four pair blocks, `pair -> pair`, left on card throughout.

        `AF2DeviceModel.extra_msa_stack` is this same loop with the dead MSA track written for
        its taps and the pair brought back to host at the end. Here the pair stays on card and
        under the tape, which is what makes it differentiable.

        The MSA track is not computed and none is owed. BindCraft 2 feeds `extra_msa` as a
        single zero row under an all-zero `extra_msa_mask` (`bindcraft/af2.py:134`), so the MSA
        track reaches `pair` only through an outer product mean that collapses to
        `proj_o.bias / eps` -- which is exactly the `opm_constant` injected here.
        """
        model = self.model

        # Built per execution, never captured. `_residual` DEALLOCATES its `update` operand
        # (`tt_bio/af2.py::_residual`), and under `recompute` the checkpoint runs this same
        # closure a second time in the backward, so a constant hoisted out of the loop body is
        # a freed buffer by then and `ttnn.typecast` raises "Buffer is not allocated". The
        # Evoformer loop above has no such operand, which is why only this stack broke.
        def const(index):
            return model._up(model.opm_constant[index].reshape(1, 1, -1))

        for index, block in enumerate(model.device_extra_msa):
            if recompute:
                z = self.ag.checkpoint(
                    lambda t, blk=block, i=index: blk(blk._residual(t, const(i)), *pair_masks), z,
                    offload=offload)
            else:
                z = block(block._residual(z, const(index)), *pair_masks)
        return z

    def template_stack(self, z, pair_masks, recompute: bool, offload: bool = False):
        """The template embedder's two c=64 pair blocks, `act -> act`, left on card throughout.

        `AF2DeviceModel`'s fold path runs these same blocks through
        `AF2DeviceTemplatePairStack` under `torch.no_grad`. Here they run under tt-bio's tape,
        which is what a gradient round needs. No `opm_constant`: the template stack has no MSA
        track for an outer product mean to collapse, unlike `extra_msa` above.
        """
        for block in self.model.device_template:
            if recompute:
                z = self.ag.checkpoint(lambda t, blk=block: blk(t, *pair_masks), z,
                                       offload=offload)
            else:
                z = block(z, *pair_masks)
        return z

    def seed(self, t: torch.Tensor, like):
        """A cotangent in the root's own device shape."""
        shape = [int(d) for d in like.value.shape]
        return self.ttnn.from_torch(t.detach().reshape(shape).to(torch.bfloat16),
                                    layout=self.ttnn.TILE_LAYOUT, device=self.device,
                                    dtype=self.ttnn.bfloat16)

    def grad(self, leaf, shape) -> torch.Tensor:
        return self.down(leaf.grad, shape) if leaf.grad is not None else torch.zeros(shape)


class _Tapes:
    """The tapes a splice has banked, keyed by the trajectory that banked them.

    One splice object serves every trajectory in flight -- the swap is a process-wide patch
    of `modules.layer_stack`, so there is one of these per STACK and not one per trajectory
    -- and each trajectory drops its own superseded tapes on every recycle. A flat sweep
    would drop the neighbour's live tape between its forward and its backward, and the
    neighbour would fail with "no live tape for token N" at a point where nothing is wrong
    with it.

    The slot is PASSED IN, never read off the calling thread. The device seam does not run on
    the trajectory's thread: measured on this program, 36 seams landed on 11 different
    `Dummy-N` threads, XLA:CPU's own pool. The slot reaches here as a constant captured when
    the program was TRACED, which does happen on the trajectory's thread.

    It is "" for a process running one trajectory, so an un-interleaved run sweeps and banks
    exactly what it always did.
    """

    def __init__(self):
        self._live: dict[int, dict] = {}
        self._next = 0
        self._lock = threading.Lock()

    def sweep(self, slot: str) -> None:
        """Drop THIS trajectory's superseded tapes. AlphaFold 2 stops the gradient on every
        recycle but the last and JAX still routes all of them through the forward rule, so
        without this the superseded tapes accumulate at several GB an Evoformer block."""
        with self._lock:
            for token in [t for t, e in self._live.items() if e["slot"] == slot]:
                del self._live[token]

    def bank(self, entry: dict, slot: str) -> int:
        with self._lock:
            token, self._next = self._next, self._next + 1
            self._live[token] = {**entry, "slot": slot}
        return token

    def take(self, token) -> dict | None:
        with self._lock:
            return self._live.pop(int(token), None)

    def count(self, slot: str = "") -> int:
        """How many tapes trajectory `slot` has banked."""
        return sum(1 for e in self._live.values() if e["slot"] == slot)


class TrunkPool:
    """The AlphaFold 2 checkpoints a campaign may draw from, on card, selected by name.

    BindCraft 2 samples one design model per gradient step, so a device trunk pinned to one
    checkpoint cannot run a multi-model campaign. Checkpoints load lazily, on the first step that
    reaches one, and the pool keeps what it loads.

    ``resident`` caps how many stay on card and evicts least-recently-used. Five AF2 trunks is
    about 910 MB of weights, and holding all five brought a backward-pass allocator refusal
    forward at n=288 that ``resident=1`` ran past, so cap it if a long run dies in the allocator.
    Eviction drops the only Python reference and relies on ttnn freeing the weight buffers when
    it is collected; tt-bio has no model-level deallocate, and that release has not been read off
    the allocator directly.

    ``source`` is a directory of ``params_<name>.npz`` (the layout ``tt-bio weights --download
    af2ig`` writes, and the one BindCraft 2's own ``data_dir`` uses), a mapping of model name to
    parameters file, a single parameters file to use for every name, or None for tt-bio's own
    weights cache. A checkpoint the source cannot supply is recorded in ``absent`` rather than
    refused, and the predictor folds it on BindCraft 2's own JAX trunk.
    """

    def __init__(self, source=None, *, resident: int | None = None,
                 template: bool = False):
        self._source = source
        self.resident = int(resident) if resident else None
        #: Whether the trunks this pool loads also bring their template pair stack on card.
        #: Set by `predictor(template=True)`; see `_Trunk.__init__` for why it is not free.
        self.template = bool(template)
        self.paths: dict[str, pathlib.Path] = {}
        self.absent: dict[str, str] = {}
        self.selections: dict[str, int] = {}
        self._trunks: dict[str, _Trunk] = {}
        self._order: list[str] = []
        # BindCraft 2 drives this pool from TWO threads: `campaign.py:182` starts
        # `compile_next_length_bucket`, a daemon thread that calls `sequence_gradients(...,
        # compile_only=True)` on the same model while `run_trajectory` folds on the main thread.
        # So the trunk cache and the selection are shared state and move under a lock, and the
        # load is deferred to the thread that folds.
        self._lock = threading.RLock()
        # Keyed by `duotraj.slot()`: two interleaved trajectories select different
        # checkpoints and the selection happens on the trajectory's own thread, outside the
        # device seam, so a single `_current` would have whichever thread selected last
        # decide what the other one folds. "" is the only slot a single-trajectory process
        # has, so nothing changes for one.
        self._selected: dict[str, str] = {}
        if isinstance(source, Mapping):
            self.require(source)

    # ------------------------------------------------------------------ the checkpoint set

    def require(self, names) -> None:
        """Resolve every name in `names` against the source; record what does not resolve.

        Called with the pool BindCraft 2 actually asks for. A campaign builds a design model and
        a validation model separately and they draw from different pools, so the set grows.

        A name the source cannot supply lands in `absent` rather than raising. BindCraft 2 holds
        validation out of design on purpose, so the shipped `examples/pdl1.json` designs on all
        five multimer checkpoints and validates on the monomer pool, and refusing there is fatal
        to a campaign that has already done all of its work. What the card does not hold folds on
        BindCraft 2's own JAX trunk instead; the predictor decides that and says so.
        """
        pairs = names.items() if isinstance(names, Mapping) else ((n, None) for n in names)
        for name, given in pairs:
            if name in self.paths or name in self.absent:
                continue
            try:
                path = pathlib.Path(os.path.expanduser(str(
                    given if given is not None else self._path_for(name))))
            except (KeyError, FileNotFoundError) as why:
                self.absent[name] = str(why)
                continue
            if path.exists():
                self.paths[name] = path
            else:
                self.absent[name] = str(path)

    def _path_for(self, name: str) -> pathlib.Path:
        source = self._source
        if isinstance(source, Mapping):
            raise KeyError(f"{name!r} is not in the checkpoint mapping {sorted(source)}; the "
                           f"campaign draws from a model this pool was never given")
        if source is None:
            from tt_bio import weights
            source = weights.resolve("af2-params")
            if source is None:
                raise FileNotFoundError(
                    "no AlphaFold 2 parameters in tt-bio's weights cache; run "
                    "`tt-bio weights --download af2ig`, or pass checkpoints=<dir>")
        path = pathlib.Path(os.path.expanduser(str(source)))
        return path / f"params_{name}.npz" if path.is_dir() else path

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, name) -> bool:
        return name in self.paths

    def holds(self, name: str) -> bool:
        """Whether the card can fold `name`, i.e. whether the weights resolved."""
        return name in self.paths

    # ------------------------------------------------------------------ selection

    def use(self, name: str) -> None:
        """Make `name` the checkpoint the trunk runs.

        An unknown name is an error, not a default. Folding a design on a checkpoint BindCraft 2
        did not pick is invisible downstream, because the shapes agree and the loss still falls.

        Selecting does NOT bring the trunk on card. The load is deferred to `current`, i.e. to the
        thread that actually folds, because BindCraft 2 selects from its compile thread too and
        that call never executes: `bindcraft/af2.py:404` returns after `lower().compile()`, before
        the `pure_callback` runs. Loading here put `_Trunk` construction, and with it a device
        bring-up and the first `tt_bio` import, on a thread racing the main fold -- which aborted
        the process on `context_id ... is out of range` from
        `Device::init_command_queue_device_with_topology`.
        """
        if name not in self.paths:
            raise KeyError(f"{name!r} is not in the trunk pool {self.names}")
        with self._lock:
            self._selected[duotraj.slot()] = name
            self.selections[name] = self.selections.get(name, 0) + 1

    def _load(self, name: str) -> _Trunk:
        with self._lock:
            trunk = self._trunks.get(name)
            if trunk is None:
                trunk = self._trunks[name] = _Trunk(self.paths[name],
                                                    template=self.template)
            if name in self._order:
                self._order.remove(name)
            self._order.append(name)
            while len(self._order) > (self.resident or len(self.paths)):
                self._trunks.pop(self._order.pop(0), None)
            return trunk

    @property
    def selected(self) -> str | None:
        """The checkpoint THIS thread's trajectory last selected, or None. Only sound on the
        trajectory's own thread, i.e. in the route selection -- never at a device seam."""
        return self._selected.get(duotraj.slot())

    def trunk_for(self, slot: str) -> _Trunk:
        """The trunk trajectory `slot` selected, loading it if this is the first fold to reach
        it. Takes the slot rather than reading the thread, because a device seam runs on one of
        XLA:CPU's pool threads and not on the trajectory's."""
        name = self._selected.get(slot)
        if name is None:
            if not self.paths:
                raise RuntimeError("the trunk pool is empty; nothing has asked it for a model")
            with self._lock:
                self._selected.setdefault(slot, self.names[0])
            name = self._selected[slot]
        return self._load(name)

    @property
    def current(self) -> _Trunk:
        return self.trunk_for(duotraj.slot())


# ----------------------------------------------------------------- the Evoformer on card


class _Memory:
    """The memory mode the on-card stacks run a fold in, decided once per token axis.

    One per predictor, shared by the Evoformer, extra-MSA and template swaps, so the three
    stacks of one fold always agree. ``requested`` is ``auto`` or one of `MEMORY_MODES`.
    """

    def __init__(self, requested: str = "auto"):
        if requested != "auto" and requested not in MEMORY_MODES:
            raise ValueError(f"memory={requested!r}: expected 'auto' or one of {MEMORY_MODES}")
        self.requested = requested
        #: The mode each token axis ran in.
        self.used: dict[int, str] = {}
        self._card = 0

    def mode(self, trunk: "_Trunk", padded: int) -> str:
        got = self.used.get(padded)
        if got is None:
            self._card = self._card or trunk.card_bytes()
            got = self.used[padded] = memory_mode(self.requested, padded, self._card)
            if got != "fast":
                fast_top = max_tokens("fast", self._card)
                print(f"[tt_bio.bindcraft2] {padded} tokens runs in the {got!r} memory mode "
                      f"({'chosen because the fast mode fits this card only to ' + str(fast_top) + ' tokens' if self.requested == 'auto' else 'as requested'}): "
                      f"{_MODE_COST[got]}. See docs/bindcraft2.md, 'Large complexes'.",
                      file=sys.stderr, flush=True)
        return got


#: What each slower mode costs, in the words a refusal and the announcement use. Measured on a
#: Wormhole Galaxy chip, `perf/bcw_slowmode/`.
_MODE_COST = {
    "fast": "the default",
    "lean": "each gradient round runs every Evoformer block's forward once more",
    "offload": "each gradient round runs every Evoformer block's forward once more and moves "
               "the block inputs to host memory and back",
}


class EvoformerOnDevice:
    """BindCraft 2's Evoformer stack, run on card, differentiable.

    ``layer_stack`` is a ``jax.lax.scan``, so its body is traced once and a device call cannot be
    dropped into one block: the stack is replaced whole. The cut is ``(msa, pair)``.
    ``single_activations`` sits outside the stack and stays in JAX, so JAX differentiates that
    projection itself and the MSA cotangent arriving here is a real one.

    ``recompute`` is gradient checkpointing on the device blocks. It trades speed for memory and
    is on by default: an uncheckpointed 48-block tape does not fit at the sizes BindCraft 2 draws.
    """

    def __init__(self, pool: TrunkPool, *, blocks: int = EVOFORMER_BLOCKS,
                 recompute: bool = True, memory: "_Memory | str" = "auto"):
        self.pool = pool
        self.blocks = blocks
        self.recompute = recompute
        self.memory = memory if isinstance(memory, _Memory) else _Memory(memory)
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        #: True while `on_host` is open. Read where haiku TRACES, by the stack replacement.
        self.host_only = False
        #: Folds handed back to BindCraft 2's own trunk, per checkpoint name.
        self.host_folds: dict[str, int] = {}
        self._mask_dev: dict = {}
        self._pair_mask_dev: dict = {}
        self._tapes = _Tapes()
        #: Token axes whose fused-arm L1 fit has already been decided and reported.
        self._fused_checked: set = set()

    # ------------------------------------------------------------------ which side folds

    @contextlib.contextmanager
    def on_host(self, name: str = ""):
        """Fold inside this block on BindCraft 2's own JAX trunk, leaving the card idle.

        A checkpoint the card does not hold has to fold somewhere, and refusing is fatal rather
        than slow: on the shipped `examples/pdl1.json` all five multimer checkpoints are design
        models, so BindCraft 2 moves validation to the monomer pool (`campaign.py:77-83`) and a
        refusal reaches every acceptance after all five design stages have already passed.
        Standing the card down for those folds also makes the validation stage numerically
        BindCraft 2's own rather than an approximation of it, which is what an acceptance
        measurement wants.

        The switch is read where haiku traces, and the route is then baked into a compiled
        program that BindCraft 2 caches by model FAMILY, so it is safe only at family
        granularity. `TenstorrentAlphaFoldDesignModel._route_by_family` is what enforces that.
        """
        was = self.host_only
        self.host_only = True
        if name and name not in self.host_folds:
            print(f"[tt_bio.bindcraft2] {name} is not on card; folding it on BindCraft 2's own "
                  f"JAX trunk", flush=True)
        if name:
            self.host_folds[name] = self.host_folds.get(name, 0) + 1
        try:
            yield
        finally:
            self.host_only = was

    # ------------------------------------------------------------------ host <-> card

    @staticmethod
    def _pad(m, z, mask, pair_mask):
        """Pad the token axis to a multiple of 32 and mask what was added.

        BindCraft 2 samples a binder length per trajectory, so the complex is whatever it drew and
        is rarely a multiple of 32. Padding does strictly more arithmetic and is faster; the mask
        keeps it exact and the caller slices the result back.
        """
        n = z.shape[0]
        n32 = _pad32(n)
        if n32 == n:
            return m, z, mask, pair_mask, n
        pad = n32 - n
        m = torch.nn.functional.pad(m, (0, 0, 0, pad))
        z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
        return m, z, mask, pair_mask, n

    def _inputs(self, msa_np, pair_np, mask_np, pair_mask_np):
        as_t = lambda a: torch.from_numpy(np.asarray(a).copy()).float()  # noqa: E731
        return self._pad(as_t(msa_np), as_t(pair_np), as_t(mask_np), as_t(pair_mask_np))

    @staticmethod
    def _key(mask: torch.Tensor):
        """A cache key that is the mask itself, not its shape.

        Shape is not enough and the difference is not exotic. BindCraft 2 draws binder lengths in
        a range and the token axis rounds up, so two trajectories of different length routinely
        land on the same padded size: 200 residues and 211 both become 224, with 24 and 13 masked
        tokens respectively and the same `(1, 224)` mask shape. A shape-keyed cache hands the
        second trajectory the first one's mask and folds its padding as real residues.
        """
        return tuple(mask.shape), mask.numpy().tobytes()

    def _msa_mask(self, trunk, mask):
        """Upload per mask and cache: the binder length changes from trajectory to trajectory."""
        key = self._key(mask)
        got = self._mask_dev.get(key)
        if got is None:
            got = self._mask_dev[key] = trunk.up(mask)
        return got

    def _pair_masks(self, trunk, pair_mask):
        """`af2_pair_masks` for this mask: the pair track's multiply and its key bias.

        `(None, None)` on an all-ones mask. BindCraft 2 buckets the token axis and zeroes the pad
        out of `seq_mask`, so the pair mask is all ones only when the complex happens to land on
        a multiple of the bucket. Dropping it read 0.534 pLDDT against BindCraft 2's own 0.950 on
        a 115-residue natural target, and 0.9541 with it.
        """
        from tt_bio.af2 import af2_pair_masks
        key = self._key(pair_mask)
        got = self._pair_mask_dev.get(key)
        if got is None:
            got = self._pair_mask_dev[key] = af2_pair_masks(pair_mask, trunk.device)
        return got

    # ------------------------------------------------------------------ forward and backward

    def _trunk(self, slot: str) -> _Trunk:
        trunk = self.pool.trunk_for(slot)
        if trunk.blocks != self.blocks:
            raise ValueError(f"{self.pool._selected.get(slot)!r} holds {trunk.blocks} Evoformer "
                             f"blocks and this splice was built for {self.blocks}")
        return trunk

    def _primal(self, slot, msa_np, pair_np, mask_np, pair_mask_np):
        """No tape. `predict` is forward-only and a validation refold calls it once per model, so
        a primal that banks a tape is an out-of-memory bug."""
        # `_inputs` is numpy and torch padding and touches no card, so it runs OUTSIDE
        # `duotraj.card`. Everything after it does touch the card and runs inside.
        m, z, mask, pair_mask, n = self._inputs(msa_np, pair_np, mask_np, pair_mask_np)
        with _refusal_names_the_size("forward", n, z.shape[0]), \
                duotraj.card(slot, "evoformer._primal"):
            trunk = self._trunk(slot)
            mo, zo = trunk.evoformer(trunk.up(m), trunk.up(z), self._msa_mask(trunk, mask),
                                     self._pair_masks(trunk, pair_mask), recompute=False)
            trunk.sync()
            self.calls["primal"] += 1
            return (trunk.down(mo, tuple(m.shape))[:, :n].numpy(),
                    trunk.down(zo, tuple(z.shape))[:n, :n].numpy())

    def _taped(self, slot, msa_np, pair_np, mask_np, pair_mask_np):
        m, z, mask, pair_mask, n = self._inputs(msa_np, pair_np, mask_np, pair_mask_np)
        # The fused arm's L1 fit is a property of the token axis, so one forward at each new
        # axis decides it for every later fold at that size.
        watch = _fused_hifi_counts() if z.shape[0] not in self._fused_checked else None
        with _refusal_names_the_size("forward", n, z.shape[0], self.memory), \
                duotraj.card(slot, "evoformer._taped"):
            trunk = self._trunk(slot)
            mode = self.memory.mode(trunk, z.shape[0])
            offload = trunk.arm(mode)
            ml, zl = trunk.leaf(m), trunk.leaf(z)
            with trunk.taped.tape():
                mo, zo = trunk.evoformer(ml, zl, self._msa_mask(trunk, mask),
                                         self._pair_masks(trunk, pair_mask),
                                         recompute=self.recompute, offload=offload)
            trunk.sync()
        # AlphaFold 2 stops the gradient on every recycle but the last and JAX still routes all
        # of them through the forward rule, so the superseded tapes are dropped here. At several
        # GB an Evoformer block, keeping them is fatal within one trajectory.
            self._tapes.sweep(slot)
            trunk.ag.release_pins()
            token = self._tapes.bank({"roots": (mo, zo), "leaves": (ml, zl),
                                      "shapes": (tuple(m.shape), tuple(z.shape)), "n": n,
                                      "mode": mode},
                                     slot)
            self.calls["taped"] += 1
            out = (trunk.down(mo.value, tuple(m.shape))[:, :n].numpy(),
                   trunk.down(zo.value, tuple(z.shape))[:n, :n].numpy(), np.int32(token))
        if watch is not None:
            self._note_if_the_fused_arm_declined(z.shape[0], watch)
        return out

    def _note_if_the_fused_arm_declined(self, padded: int, before: "tuple[int, int]") -> None:
        """Say once, per token axis, when the fused triangle attention declined every call.

        The arm's circular buffers do not fit L1 at every token axis, and when it declines the
        composed path runs in its place and materialises the `[N,4,N,N]` fp32 scores. Nothing in
        a run says so today, so a size that degrades looks like a size that is merely large. It
        is not a small effect and it is not monotone in the size: measured on qb1 p150a, one
        axis holds 25.75 GB at 69.2 s a round where the next bucket up holds 14.23 GB at 49.4 s,
        so the smaller fold is the heavier and the slower one, and two different targets on two
        different chain counts decline identically at the same axis.

        The note states the cost it can COMPUTE -- the composed tensor is `16 * n**3` bytes and
        that is exact -- and does not quote the axes it was measured at. An earlier version did,
        and the sizes it named came from arithmetic over a target file rather than from the
        seam, so when that arithmetic turned out to be a bucket off at several targets the
        message contradicted itself: it reported declining at 544 and advised moving to 544. A
        message that hard-codes measured sizes inherits every labelling error upstream of it,
        and the arm's L1 fit is a kernel policy that is expected to move anyway.

        The counters are process-wide, so the window is one forward and the test is a delta over
        it. Interleaved trajectories can serve inside that window; that direction only silences
        the note, which is the safe way to be wrong about it.
        """
        self._fused_checked.add(padded)
        served, declined = _fused_hifi_counts()
        served, declined = served - before[0], declined - before[1]
        if served or not declined:
            return
        extra = 16 * padded ** 3          # [n,4,n,n] fp32, the term the fused arm never holds
        print(f"[tt_bio.bindcraft2] the fused triangle attention declined all {declined} calls "
              f"at {padded} tokens: its circular buffers do not fit L1 at this token axis, so "
              f"the composed path is running and holding the [{padded},4,{padded},{padded}] "
              f"fp32 scores -- {_gb(extra)} of device memory this fold would not otherwise "
              f"need, and a slower round with it. Whether the arm fits is a property of the "
              f"token axis and is NOT monotone in it, so a neighbouring binder length is often "
              f"much cheaper: try one 32-token bucket UP as well as one down.",
              file=sys.stderr)

    def _backward(self, slot, token, g_msa_np, g_pair_np):
        entry = self._tapes.take(token)
        if entry is None:
            raise RuntimeError(f"no live tape for token {int(token)}")
        mo, zo = entry["roots"]
        ml, zl = entry["leaves"]
        m_shape, z_shape = entry["shapes"]
        n = entry["n"]
        # Building the cotangent buffers is torch on the host; only the backward needs the card.
        gm, gz = torch.zeros(m_shape), torch.zeros(z_shape)
        gm[:, :n] = torch.from_numpy(np.asarray(g_msa_np).copy()).float()
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        with _refusal_names_the_size("backward", n, z_shape[0], self.memory), \
                duotraj.card(slot, "evoformer._backward"):
            trunk = self.pool.trunk_for(slot)
            trunk.arm(entry["mode"])
            trunk.ag.backward([mo, zo], [trunk.seed(gm, mo), trunk.seed(gz, zo)])
            trunk.sync()
            out = (trunk.grad(ml, m_shape)[:, :n].numpy(),
                   trunk.grad(zl, z_shape)[:n, :n].numpy())
            trunk.ag.release_pins()
        self.calls["backward"] += 1
        return out

    def live_tapes(self, slot: str = "") -> int:
        return self._tapes.count(slot)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self, slot: str = ""):
        """`(msa, pair, msa_mask, pair_mask) -> (msa, pair)`, differentiable in the first two.

        `slot` is the trajectory this PROGRAM belongs to, captured here and handed to every
        callback as a constant. It cannot be read at the seam: the seam runs on one of
        XLA:CPU's own pool threads (measured: 36 seams on 11 different `Dummy-N` threads), and
        tracing is the last point on the trajectory's own thread.

        Both masks are arguments rather than captured host arrays: BindCraft 2 draws a new binder
        length per trajectory, so their shapes change under us. Neither is differentiable, so the
        backward returns zeros for both.

        The callback works in float32 while the model around it runs bfloat16, and AlphaFold 2
        carries the pair representation through a `while_loop` whose carry types must match
        exactly, so every value handed back takes the dtype it arrived with.
        """
        import jax
        import jax.numpy as jnp

        def shapes(msa, pair):
            return (jax.ShapeDtypeStruct(msa.shape, jnp.float32),
                    jax.ShapeDtypeStruct(pair.shape, jnp.float32))

        @jax.custom_vjp
        def stack(msa, pair, mask, pair_mask):
            m, z = jax.pure_callback(functools.partial(self._primal, slot), shapes(msa, pair),
                                     msa.astype(jnp.float32), pair.astype(jnp.float32),
                                     mask.astype(jnp.float32), pair_mask.astype(jnp.float32))
            return m.astype(msa.dtype), z.astype(pair.dtype)

        def fwd(msa, pair, mask, pair_mask):
            m, z, token = jax.pure_callback(
                functools.partial(self._taped, slot),
                shapes(msa, pair) + (jax.ShapeDtypeStruct((), jnp.int32),),
                msa.astype(jnp.float32), pair.astype(jnp.float32), mask.astype(jnp.float32),
                pair_mask.astype(jnp.float32))
            return (m.astype(msa.dtype), z.astype(pair.dtype)), (token, mask, pair_mask)

        def bwd(res, cotangents):
            token, mask, pair_mask = res
            g_msa, g_pair = cotangents
            gm, gz = jax.pure_callback(
                functools.partial(self._backward, slot),
                (jax.ShapeDtypeStruct(g_msa.shape, jnp.float32),
                 jax.ShapeDtypeStruct(g_pair.shape, jnp.float32)),
                token, g_msa.astype(jnp.float32), g_pair.astype(jnp.float32))
            return (gm.astype(g_msa.dtype), gz.astype(g_pair.dtype),
                    jnp.zeros_like(mask), jnp.zeros_like(pair_mask))

        stack.defvjp(fwd, bwd)
        return stack


class ExtraMsaOnDevice:
    """BindCraft 2's extra-MSA stack, run on card, differentiable in `pair` alone.

    `EvoformerOnDevice` replaces the 48-block trunk; this replaces the 4-block extra-MSA stack
    that runs before it. The cut is `pair` alone: `modules.py:1530` reads only `pair` out of the
    stack, and the MSA track collapses to a constant under BindCraft 2's all-zero
    `extra_msa_mask`, so no gradient into `extra_msa` comes back and none is owed.

    `_check_mask` refuses any nonzero mask rather than folding it against the wrong constant, so
    a featurisation that ever carries a real extra MSA stops here instead of agreeing quietly.

    Its tapes live in their own registry. JAX runs the extra-MSA forward, then the Evoformer
    forward, then the two backwards in reverse, so a registry shared with `EvoformerOnDevice`
    would have the Evoformer's stale-tape sweep drop this stack's live tape before its backward.
    """

    def __init__(self, pool: TrunkPool, *, recompute: bool = True,
                 memory: "_Memory | str" = "auto"):
        self.pool = pool
        self.recompute = recompute
        self.memory = memory if isinstance(memory, _Memory) else _Memory(memory)
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        #: What the mask actually carried, so a refusal can be explained after the fact.
        self.mask_seen = {"calls": 0, "abs_max": 0.0}
        self.swapped: list[int] = []
        self._pair_mask_dev: dict = {}
        self._tapes = _Tapes()

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _pad(z, pair_mask):
        """The pair half of `EvoformerOnDevice._pad`, which is all this swap hands the card."""
        n = z.shape[0]
        n32 = _pad32(n)
        if n32 == n:
            return z, pair_mask, n
        pad = n32 - n
        z = torch.nn.functional.pad(z, (0, 0, 0, pad, 0, pad))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
        return z, pair_mask, n

    def _check_mask(self, extra_mask_np):
        a = np.asarray(extra_mask_np)
        self.mask_seen["calls"] += 1
        self.mask_seen["abs_max"] = max(self.mask_seen["abs_max"], float(np.abs(a).max()))
        if a.any():
            raise ValueError(
                f"extra_msa_mask carries {int((a != 0).sum())} nonzero entries; this swap "
                f"injects the outer product mean an all-zero mask collapses to and is wrong "
                f"for anything else")

    def _inputs(self, pair_np, extra_mask_np, pair_mask_np):
        self._check_mask(extra_mask_np)
        as_t = lambda a: torch.from_numpy(np.asarray(a).copy()).float()  # noqa: E731
        return self._pad(as_t(pair_np), as_t(pair_mask_np))

    def _trunk(self, slot: str) -> _Trunk:
        trunk = self.pool.trunk_for(slot)
        if trunk.extra_blocks == 0:
            raise ValueError(f"{self.pool._selected.get(slot)!r} holds no extra-MSA blocks "
                             f"on card")
        return trunk

    def _pair_masks(self, trunk, pair_mask):
        """`af2_pair_masks` for this mask, keyed on the mask itself for the reason
        `EvoformerOnDevice._key` gives: two binder lengths routinely pad to one shape."""
        from tt_bio.af2 import af2_pair_masks
        key = EvoformerOnDevice._key(pair_mask)
        got = self._pair_mask_dev.get(key)
        if got is None:
            got = self._pair_mask_dev[key] = af2_pair_masks(pair_mask, trunk.device)
        return got

    # ------------------------------------------------------------------ forward and backward

    def _primal(self, slot, pair_np, extra_mask_np, pair_mask_np):
        z, pair_mask, n = self._inputs(pair_np, extra_mask_np, pair_mask_np)
        with duotraj.card(slot, "extra_msa._primal"):
            trunk = self._trunk(slot)
            zo = trunk.extra_msa(trunk.up(z), self._pair_masks(trunk, pair_mask),
                                 recompute=False)
            trunk.sync()
            self.calls["primal"] += 1
            return trunk.down(zo, tuple(z.shape))[:n, :n].numpy()

    def _taped(self, slot, pair_np, extra_mask_np, pair_mask_np):
        z, pair_mask, n = self._inputs(pair_np, extra_mask_np, pair_mask_np)
        with duotraj.card(slot, "extra_msa._taped"):
            trunk = self._trunk(slot)
            mode = self.memory.mode(trunk, z.shape[0])
            offload = trunk.arm(mode)
            zl = trunk.leaf(z)
            with trunk.taped.tape():
                zo = trunk.extra_msa(zl, self._pair_masks(trunk, pair_mask),
                                     recompute=self.recompute, offload=offload)
            trunk.sync()
        # Every recycle but the last is stop_gradient'ed and still goes through the forward
        # rule, so the superseded tapes are dropped here -- `EvoformerOnDevice._taped`'s reason.
            self._tapes.sweep(slot)
            trunk.ag.release_pins()
            token = self._tapes.bank({"root": zo, "leaf": zl, "shape": tuple(z.shape), "n": n,
                                      "mode": mode}, slot)
            self.calls["taped"] += 1
            return (trunk.down(zo.value, tuple(z.shape))[:n, :n].numpy(), np.int32(token))

    def _backward(self, slot, token, g_pair_np):
        entry = self._tapes.take(token)
        if entry is None:
            raise RuntimeError(f"no live extra-MSA tape for token {int(token)}")
        shape, n = entry["shape"], entry["n"]
        gz = torch.zeros(shape)
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        with duotraj.card(slot, "extra_msa._backward"):
            trunk = self.pool.trunk_for(slot)
            trunk.arm(entry["mode"])
            trunk.ag.backward([entry["root"]], [trunk.seed(gz, entry["root"])])
            trunk.sync()
            out = trunk.grad(entry["leaf"], shape)[:n, :n].numpy()
            trunk.ag.release_pins()
        self.calls["backward"] += 1
        return out

    def live_tapes(self, slot: str = "") -> int:
        return self._tapes.count(slot)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self, slot: str = ""):
        """`(pair, extra_msa_mask, pair_mask) -> pair`, differentiable in `pair` alone.

        `slot` is baked in at TRACE time; see `EvoformerOnDevice.as_jax`."""
        import jax
        import jax.numpy as jnp

        def f32(pair):
            return jax.ShapeDtypeStruct(pair.shape, jnp.float32)

        def args(pair, extra_mask, pair_mask):
            return (pair.astype(jnp.float32), extra_mask.astype(jnp.float32),
                    pair_mask.astype(jnp.float32))

        @jax.custom_vjp
        def stack(pair, extra_mask, pair_mask):
            z = jax.pure_callback(functools.partial(self._primal, slot), f32(pair),
                                  *args(pair, extra_mask, pair_mask))
            return z.astype(pair.dtype)

        def fwd(pair, extra_mask, pair_mask):
            z, token = jax.pure_callback(
                functools.partial(self._taped, slot),
                (f32(pair), jax.ShapeDtypeStruct((), jnp.int32)),
                *args(pair, extra_mask, pair_mask))
            return z.astype(pair.dtype), (token, extra_mask, pair_mask)

        def bwd(res, g_pair):
            token, extra_mask, pair_mask = res
            gz = jax.pure_callback(functools.partial(self._backward, slot), f32(g_pair), token,
                                   g_pair.astype(jnp.float32))
            return (gz.astype(g_pair.dtype), jnp.zeros_like(extra_mask),
                    jnp.zeros_like(pair_mask))

        stack.defvjp(fwd, bwd)
        return stack


class TemplateOnDevice:
    """BindCraft 2's multimer template pair stack, run on card, differentiable in `act` alone.

    `EvoformerOnDevice` replaces the 48-block trunk and `ExtraMsaOnDevice` the 4-block
    extra-MSA stack; this replaces the two c=64 blocks inside the template embedder. The cut is
    the `template_stack((act, safe_subkey))` call in
    `modules_multimer.SingleTemplateEmbedding.__call__`, between `construct_input` and
    `output_layer_norm`: AlphaFold's own feature construction and output norm stay in JAX and
    only the blocks move, which is where the seconds are (2.549 + 1.211 s of the 4.57 s
    profiled, `state/perf10/bcx-HOSTMAP.md`).

    Gradient flows back into `act` and nowhere else. The template features are the design's
    target structure and are constant across a trajectory, and `pair_mask` is a mask.

    Its tapes live in their own registry, for the reason `ExtraMsaOnDevice` gives: the stacks
    run forward in sequence and backward in reverse, so a shared registry would have one
    stack's stale-tape sweep drop another's live tape before its backward.
    """

    def __init__(self, pool: TrunkPool, *, recompute: bool = True,
                 memory: "_Memory | str" = "auto"):
        self.pool = pool
        self.recompute = recompute
        self.memory = memory if isinstance(memory, _Memory) else _Memory(memory)
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        #: What the JAX side handed over, so an inert swap cannot read as a working one.
        self.seen = {"calls": 0, "n": None, "channels": None, "blocks_swapped": None}
        self._pair_mask_dev: dict = {}
        self._tapes = _Tapes()

    # ------------------------------------------------------------------ inputs

    @staticmethod
    def _pad(act, pair_mask):
        """`ExtraMsaOnDevice._pad`, at the template stack's own channel count."""
        n = act.shape[0]
        n32 = _pad32(n)
        if n32 == n:
            return act, pair_mask, n
        pad = n32 - n
        act = torch.nn.functional.pad(act, (0, 0, 0, pad, 0, pad))
        pair_mask = torch.nn.functional.pad(pair_mask, (0, pad, 0, pad))
        return act, pair_mask, n

    def _inputs(self, act_np, pair_mask_np):
        as_t = lambda a: torch.from_numpy(np.asarray(a).copy()).float()  # noqa: E731
        act, pair_mask, n = self._pad(as_t(act_np), as_t(pair_mask_np))
        self.seen["calls"] += 1
        self.seen["n"], self.seen["channels"] = n, int(act.shape[-1])
        return act, pair_mask, n

    def _trunk(self, slot: str) -> _Trunk:
        trunk = self.pool.trunk_for(slot)
        if trunk.template_blocks == 0:
            raise ValueError(
                f"{self.pool._selected.get(slot)!r} holds no template blocks on card; this "
                f"swap needs TrunkPool(template=True), which predictor(template=True) sets")
        return trunk

    def _pair_masks(self, trunk, pair_mask):
        key = EvoformerOnDevice._key(pair_mask)
        got = self._pair_mask_dev.get(key)
        if got is None:
            from tt_bio.af2 import af2_pair_masks
            got = self._pair_mask_dev[key] = af2_pair_masks(pair_mask, trunk.device)
        return got

    # ------------------------------------------------------------------ forward and backward

    def _primal(self, slot, act_np, pair_mask_np):
        act, pair_mask, n = self._inputs(act_np, pair_mask_np)
        with duotraj.card(slot, "template._primal"):
            trunk = self._trunk(slot)
            out = trunk.template_stack(trunk.up(act), self._pair_masks(trunk, pair_mask),
                                       recompute=False)
            trunk.sync()
            self.calls["primal"] += 1
            return trunk.down(out, tuple(act.shape))[:n, :n].numpy()

    def _taped(self, slot, act_np, pair_mask_np):
        act, pair_mask, n = self._inputs(act_np, pair_mask_np)
        with duotraj.card(slot, "template._taped"):
            trunk = self._trunk(slot)
            mode = self.memory.mode(trunk, act.shape[0])
            offload = trunk.arm(mode)
            leaf = trunk.leaf(act)
            with trunk.taped.tape():
                out = trunk.template_stack(leaf, self._pair_masks(trunk, pair_mask),
                                           recompute=self.recompute, offload=offload)
            trunk.sync()
        # Every recycle but the last is stop_gradient'ed and still goes through the forward
        # rule, so the superseded tapes are dropped here -- `ExtraMsaOnDevice._taped`'s reason.
            self._tapes.sweep(slot)
            trunk.ag.release_pins()
            token = self._tapes.bank({"root": out, "leaf": leaf, "shape": tuple(act.shape),
                                      "n": n, "mode": mode}, slot)
            self.calls["taped"] += 1
            return (trunk.down(out.value, tuple(act.shape))[:n, :n].numpy(),
                    np.int32(token))

    def _backward(self, slot, token, g_act_np):
        entry = self._tapes.take(token)
        if entry is None:
            raise RuntimeError(f"no live template tape for token {int(token)}")
        shape, n = entry["shape"], entry["n"]
        g = torch.zeros(shape)
        g[:n, :n] = torch.from_numpy(np.asarray(g_act_np).copy()).float()
        with duotraj.card(slot, "template._backward"):
            trunk = self.pool.trunk_for(slot)
            trunk.arm(entry["mode"])
            trunk.ag.backward([entry["root"]], [trunk.seed(g, entry["root"])])
            trunk.sync()
            out = trunk.grad(entry["leaf"], shape)[:n, :n].numpy()
            trunk.ag.release_pins()
        self.calls["backward"] += 1
        return out

    def live_tapes(self, slot: str = "") -> int:
        return self._tapes.count(slot)

    # ------------------------------------------------------------------ the JAX face

    def as_jax(self, slot: str = ""):
        """`(act, pair_mask) -> act`, differentiable in `act` alone.

        `slot` is baked in at TRACE time; see `EvoformerOnDevice.as_jax`."""
        import jax
        import jax.numpy as jnp

        def f32(act):
            return jax.ShapeDtypeStruct(act.shape, jnp.float32)

        def args(act, pair_mask):
            return act.astype(jnp.float32), pair_mask.astype(jnp.float32)

        @jax.custom_vjp
        def stack(act, pair_mask):
            out = jax.pure_callback(functools.partial(self._primal, slot), f32(act),
                                    *args(act, pair_mask))
            return out.astype(act.dtype)

        def fwd(act, pair_mask):
            out, token = jax.pure_callback(
                functools.partial(self._taped, slot),
                (f32(act), jax.ShapeDtypeStruct((), jnp.int32)),
                *args(act, pair_mask))
            return out.astype(act.dtype), (token, pair_mask)

        def bwd(res, g_act):
            token, pair_mask = res
            g = jax.pure_callback(functools.partial(self._backward, slot), f32(g_act), token,
                                  g_act.astype(jnp.float32))
            return g.astype(g_act.dtype), jnp.zeros_like(pair_mask)

        stack.defvjp(fwd, bwd)
        return stack


def _template_stack_mask(fn, depth: int = 0, seen=None):
    """`padding_mask_2d` off `template_iteration_fn`'s closure, however deep haiku wrapped it.

    `gc.use_remat` puts `hk.remat` around the function before `layer_stack` ever sees it, so at
    the seam the only free variable is remat's own `dec_stateful_fun`. Recursing is what
    `_free_variable` does for the extra-MSA masks, for the same reason.
    """
    if depth > 6 or not callable(fn):
        return None
    seen = seen if seen is not None else set()
    if id(fn) in seen:
        return None
    seen.add(id(fn))
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    inner = []
    for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:                      # a cell still being filled
            continue
        if name == "padding_mask_2d":
            return value
        inner.append(value)
    for value in inner:
        got = _template_stack_mask(value, depth + 1, seen)
        if got is not None:
            return got
    return None


@contextlib.contextmanager
def template_on_device(tmpl: "TemplateOnDevice | None"):
    """Route the multimer template pair stack through `tmpl` for the duration.

    The two blocks are `template_stack((act, safe_subkey))`, inline in
    `modules_multimer.SingleTemplateEmbedding.__call__`. Unlike `extra_msa_stack_fn` there is no
    closure to rewrite, so this swaps the module-global `layer_stack` for a shim and does it
    only while the template embedder is tracing -- the Evoformer builds its own layer stack
    through the same name and must keep AlphaFold's.
    """
    if tmpl is None:
        yield None
        return
    from bindcraft.af.alphafold.model import modules_multimer

    class _Shim:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def layer_stack(self, num_block):
            def build(fn):
                # The jax face is built HERE, at trace time, rather than once for the whole
                # block: this is the last point that runs on the trajectory's own thread, so
                # it is where the slot can be read and baked into the callbacks.
                device_stack = tmpl.as_jax(duotraj.slot())
                mask = _template_stack_mask(fn)
                if mask is None:
                    raise ValueError(
                        "padding_mask_2d is not reachable from the template stack's closure; "
                        "the splice point in SingleTemplateEmbedding.__call__ moved")
                tmpl.seen["blocks_swapped"] = int(num_block)

                def run(carry):
                    act, key = carry
                    return device_stack(act, mask), key

                return run

            return build

    original = modules_multimer.SingleTemplateEmbedding.__call__

    def patched(self, *args, **kwargs):
        saved = modules_multimer.layer_stack
        modules_multimer.layer_stack = _Shim(saved)
        try:
            return original(self, *args, **kwargs)
        finally:
            modules_multimer.layer_stack = saved

    modules_multimer.SingleTemplateEmbedding.__call__ = patched
    try:
        yield tmpl
    finally:
        modules_multimer.SingleTemplateEmbedding.__call__ = original


def find_evoformer_masks(fn):
    """Recover the Evoformer masks from the closure of AlphaFold 2's `evoformer_fn`.

    Replacing the whole `layer_stack` means never seeing the masks dict that
    `EmbeddingsAndEvoformer` builds and `evoformer_fn` closes over. `hk.remat` wraps that closure
    twice, so the walk is recursive rather than one `__wrapped__` hop.
    """
    return _free_variable(fn, "evoformer_masks", lambda v: isinstance(v, dict) and "msa" in v)


#: Evoformer sub-layer dropout rates AlphaFold 2 ships (`config.py:158-216`): the MSA row
#: attention residual, and the triangle/pair residuals. `dropout_wrapper` (`modules.py:46`)
#: applies these to each sub-layer residual BEFORE the skip-add, not to the block output, so
#: nothing wrapping the stack from outside can reproduce them.
EVOFORMER_DROPOUT_RATES = (0.15, 0.25)


def find_evoformer_dropout(fn):
    """What dropout AlphaFold 2 would apply inside the stack `fn` is one block of.

    Returns `(use_dropout, rates)`, or `(None, ())` when the closure carries neither.
    `use_dropout` is `batch["use_dropout"]`, which `af2.py:401` threads in as a TRACED argument
    (`jnp.asarray(self.dropout)`), so it is an abstract value at trace time and cannot be
    branched on: the swap cannot decide that this particular fold happened to want no dropout.
    `rates` are static, read off `evoformer_iteration`'s config.
    """
    batch = _free_variable(fn, "batch",
                           lambda v: isinstance(v, dict) and "use_dropout" in v)
    use_dropout = None if batch is None else batch["use_dropout"]
    iteration = _free_variable(fn, "evoformer_iteration", lambda v: hasattr(v, "config"))
    rates = ()
    if iteration is not None:
        config = iteration.config
        found = set()
        for name in dir(config):
            try:
                sub = getattr(config, name)
            except Exception:
                continue
            rate = getattr(sub, "dropout_rate", None)
            if rate is not None and float(rate) != 0.0:
                found.add(float(rate))
        rates = tuple(sorted(found))
    return use_dropout, rates


#: The two spellings AlphaFold 2 gives the extra-MSA stack's per-block closure. The monomer
#: path (`modules.py:1517`) calls it `extra_msa_stack_fn`; the multimer path
#: (`modules_multimer.py:375`) calls it `extra_evoformer_fn`. The name is the only thing that
#: distinguishes this `layer_stack` call from the Evoformer's, so both have to be listed or the
#: swap is silently skipped on whichever tree is not named here.
EXTRA_MSA_FN_NAMES = ("extra_msa_stack_fn", "extra_evoformer_fn")


def find_extra_msa_masks(fn):
    """The two masks the extra-MSA stack uses, read off its closure.

    The multimer path builds them into one `extra_masks` dict
    (`modules_multimer.py:372`), shaped exactly like `evoformer_masks`. The monomer path has no
    dict to recover: `modules.py:1522` builds them inline from `batch` and `mask_2d`, so the
    walk falls back to those.
    """
    masks = _free_variable(fn, "extra_masks",
                           lambda v: isinstance(v, dict) and "msa" in v and "pair" in v)
    if masks is not None:
        return {"msa": masks["msa"], "pair": masks["pair"]}
    batch = _free_variable(fn, "batch", lambda v: isinstance(v, dict) and "extra_msa_mask" in v)
    mask_2d = _free_variable(fn, "mask_2d", lambda v: hasattr(v, "shape"))
    if batch is None or mask_2d is None:
        return None
    return {"msa": batch["extra_msa_mask"], "pair": mask_2d}


def _free_variable(fn, want, accept, depth: int = 0, seen=None):
    """The value `want` names in `fn`'s closure, followed through `hk.remat`'s wrappers."""
    import types
    seen = seen if seen is not None else set()
    if depth > 6 or not isinstance(fn, types.FunctionType) or id(fn) in seen:
        return None
    seen.add(id(fn))
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if name == want and accept(value):
            return value
        found = _free_variable(value, want, accept, depth + 1, seen)
        if found is not None:
            return found
    return None


#: The splice `evoformer_on_device` currently has installed, or None. The patch is on
#: `modules.layer_stack`, which is process-global, so a predictor has to be able to find it:
#: a `trunk="jax"` model built inside a live campaign would otherwise be spliced too.
_INSTALLED: EvoformerOnDevice | None = None


def installed() -> EvoformerOnDevice | None:
    """The splice currently patched into AlphaFold 2, or None if the trunk is BindCraft 2's."""
    return _INSTALLED


#: What `evoformer_on_device` does when the host stack would apply dropout and the device blocks
#: cannot. Overridable per run so an Evoformer-only benchmark, which has always been graded
#: dropout-free, does not have to thread the argument through every call site.
DROPOUT_POLICY_ENV = "TT_BIO_BC2_EVOFORMER_DROPOUT"


def _check_dropout(fn, policy: str):
    """Stop if the swap would silently drop AlphaFold 2's Evoformer dropout.

    The device blocks apply none. `dropout_wrapper` applies it to each sub-layer residual before
    the skip-add, so the swap replaces a stochastic program with a deterministic one wherever
    `use_dropout` is live. BindCraft 2 leaves it live for every gradient stage except `harden`
    (`trajectory.py:219`), which is most of a design trajectory, and `use_dropout` is traced, so
    this cannot be narrowed to the folds that actually wanted it.
    """
    policy = os.environ.get(DROPOUT_POLICY_ENV, policy)
    if policy not in ("refuse", "ignore"):
        raise ValueError(f"dropout policy must be 'refuse' or 'ignore', not {policy!r}")
    use_dropout, rates = find_evoformer_dropout(fn)
    if policy == "ignore" or use_dropout is None or not rates:
        return rates
    raise RuntimeError(
        "evoformer_fn carries use_dropout and AlphaFold 2 would apply dropout at rates "
        f"{rates} inside the stack, which the device blocks do not. Folding on card here is a "
        "different program from folding on host, and the difference is invisible in the output. "
        f"Pass dropout='ignore' (or set {DROPOUT_POLICY_ENV}=ignore) to accept a dropout-free "
        "trunk, or stand the stack down with EvoformerOnDevice.on_host.")


@contextlib.contextmanager
def evoformer_on_device(evo: EvoformerOnDevice,
                        extra_msa: "ExtraMsaOnDevice | None" = None,
                        dropout: str = "refuse"):
    """Swap AlphaFold 2's Evoformer stack for `evo` for the duration, and nothing else.

    `modules.py` calls `layer_stack` three times, twice for the template pair stack and once
    each for the extra-MSA and Evoformer stacks. The closures are distinguishable by name, so
    this patches the factory and swaps only the ones it is asked for.

    `extra_msa` is the second swap and is independent of the first: the default None leaves the
    extra-MSA stack in JAX, which is the program every Evoformer-only comparison was graded on.

    `dropout` decides what happens when AlphaFold 2 would have applied dropout inside the stack
    and the device blocks cannot. "refuse" stops; "ignore" swaps anyway and folds dropout-free.
    See `_check_dropout`.
    """
    from bindcraft.af.alphafold.model import modules

    real = modules.layer_stack.layer_stack
    swapped: list[int] = []

    def choose_extra(fn, made, num_layers):
        # Built at TRACE time so the slot is readable; see `EvoformerOnDevice.as_jax`.
        extra_stack = extra_msa.as_jax(duotraj.slot())
        blocks = extra_msa.pool.current.extra_blocks
        if num_layers != blocks:
            raise ValueError(f"extra_msa_stack_fn has {num_layers} blocks, tt-bio holds "
                             f"{blocks}")
        masks = find_extra_msa_masks(fn)
        if masks is None:
            raise RuntimeError(
                "batch/mask_2d not found in extra_msa_stack_fn's closure. The stack needs the "
                "extra-MSA and pair masks to fold a padded complex and guessing one is worse "
                "than stopping.")
        extra_msa.swapped.append(num_layers)

        def extra_on_device(x):
            activations, safe_key = x
            pair = extra_stack(activations["pair"], masks["msa"], masks["pair"])
            # The scan splits the key once per block and carries the first half on, so what
            # follows the stack sees the key it would have seen.
            for _ in range(num_layers):
                safe_key, _unused = safe_key.split()
            return {**activations, "pair": pair}, safe_key
        return extra_on_device

    def factory(num_layers, *args, **kwargs):
        made = real(num_layers, *args, **kwargs)

        def choose(fn):
            name = getattr(fn, "__name__", None)
            if extra_msa is not None and name in EXTRA_MSA_FN_NAMES:
                return choose_extra(fn, made, int(num_layers))
            if name != "evoformer_fn":
                return made(fn)
            if evo.host_only:
                # This fold runs on a checkpoint the card does not hold, or is the control arm
                # standing down. Hand back BindCraft 2's own stack; see `EvoformerOnDevice.on_host`.
                return made(fn)
            if int(num_layers) != evo.blocks:
                raise ValueError(f"evoformer_fn has {num_layers} blocks, tt-bio holds "
                                 f"{evo.blocks}")
            swapped.append(int(num_layers))
            device_stack = evo.as_jax(duotraj.slot())
            masks = find_evoformer_masks(fn)
            if masks is None:
                raise RuntimeError(
                    "evoformer_masks not found in evoformer_fn's closure. The trunk needs the "
                    "MSA and pair masks to fold a padded complex and guessing one is worse than "
                    "stopping.")
            _check_dropout(fn, dropout)

            def on_device(x):
                activations, safe_key = x
                msa, pair = device_stack(activations["msa"], activations["pair"],
                                         masks["msa"], masks["pair"])
                return {**activations, "msa": msa, "pair": pair}, safe_key
            return on_device
        return choose

    global _INSTALLED
    was, _INSTALLED = _INSTALLED, evo
    modules.layer_stack.layer_stack = factory
    try:
        yield swapped
    finally:
        modules.layer_stack.layer_stack = real
        _INSTALLED = was
    # Only on a clean exit, so a real failure inside the block is never replaced by this one.
    # A swap that is asked for and never taken runs the host stack and reports device numbers
    # for it, which is the one failure mode a counter cannot be read for after the fact.
    if extra_msa is not None and not extra_msa.swapped and swapped:
        raise RuntimeError(
            "extra_msa=True was asked for and the extra-MSA stack was never swapped: "
            f"the Evoformer was ({swapped} blocks), so the model was traced and only this "
            f"stack was missed. Its per-block closure is named none of {EXTRA_MSA_FN_NAMES}.")


# ----------------------------------------------------------------- the predictor


_MODEL_CLASS = None


def design_model_class():
    """`AlphaFoldDesignModel` with its Evoformer trunk selectable, built on first use.

    The class is built lazily because it subclasses BindCraft 2's, and tt-bio does not depend on
    BindCraft 2.
    """
    global _MODEL_CLASS
    if _MODEL_CLASS is not None:
        return _MODEL_CLASS
    try:
        from bindcraft.af2 import AlphaFoldDesignModel
    except ImportError as exc:
        raise ImportError(
            "BindCraft 2 is not importable. tt-bio does not ship it: install it from "
            "https://github.com/PacesaLab/BindCraft2 and put it on sys.path.") from exc
    print("[tt_bio.bindcraft2] BindCraft 2 is the authors' software under their own licence, which "
          "restricts hosting it for others: https://github.com/PacesaLab/BindCraft2/blob/main/LICENSE",
          file=sys.stderr, flush=True)

    class TenstorrentAlphaFoldDesignModel(AlphaFoldDesignModel):
        """BindCraft 2's design model with tt-bio's Evoformer trunk on a Tenstorrent card.

        `trunk="jax"` keeps BindCraft 2's own trunk and is the control arm: same class, same call
        path, same bucket rounding, so a device result is compared against this rather than
        against a differently shaped program.
        """

        def __init__(self, *args, trunk: str = "device", pool: TrunkPool | None = None,
                     **kwargs):
            if trunk not in ("jax", "device"):
                raise ValueError(f"trunk must be 'jax' or 'device', not {trunk!r}")
            if trunk == "device" and pool is None:
                raise ValueError("trunk='device' needs a TrunkPool to fold on")
            super().__init__(*args, **kwargs)
            self.trunk = trunk
            self.pool = pool
            #: Checkpoint name -> "device" or "jax", decided once, per family.
            self.routes: dict[str, str] = {}
            if pool is not None:
                pool.require(sorted(set(self.presets) | set(self.models)))
                if not pool.paths:
                    # Not a route: an empty pool means the source is wrong, and a campaign that
                    # runs entirely on the host while the caller believes it is on a card
                    # misattributes every number it produces. A pool that holds SOMETHING and
                    # is missing this model's checkpoints is the ordinary case, and falls back.
                    raise FileNotFoundError(
                        "trunk='device' but no AlphaFold 2 parameters resolved: " +
                        ", ".join(f"{n} at {w}" for n, w in sorted(pool.absent.items())) +
                        ". Pass checkpoints=<directory or {name: file}> to name them, or run "
                        "`tt-bio weights --download af2ig` for the monomer checkpoint.")
                self.routes = self._route_by_family()

        def _route_by_family(self) -> dict[str, str]:
            """Decide which folds go on card per model FAMILY, not per checkpoint.

            The route is read where haiku traces and is then baked into a compiled program that
            BindCraft 2 caches keyed on the family (`af2.py:271` for `predict`, `:331` for
            `sequence_gradients`). `alphafold_model_family` collapses all five multimer
            checkpoints to `('multimer',)`, and `model_1_ptm` and `model_2_ptm` to one monomer
            family because their `CONFIG_DIFFS` entries are identical. So two checkpoints of one
            family cannot take different routes: the second would silently reuse the first one's
            compiled program and fold on the wrong trunk, which nothing downstream can see
            because the shapes agree and the loss still falls.

            A family therefore goes on card only if the card holds EVERY checkpoint this model
            can draw from it. Splitting the shipped campaign the way it actually splits, five
            multimer design models on card and the monomer validation pool on host, is a split
            along family lines and is safe; three of five multimer checkpoints is not, and this
            sends all five to the host rather than fold two of them wrong.
            """
            members: dict[tuple, list[str]] = {}
            for name in sorted(set(self.presets) | set(self.models)):
                # An unknown name has no family and cannot collide, so give it its own.
                members.setdefault(self.model_families.get(name, ("?", name)), []).append(name)
            return {name: ("device" if all(self.pool.holds(n) for n in names) else "jax")
                    for names in members.values() for name in names}

        def _pick(self, model):
            """The checkpoint this call folds on, and the `model` argument to hand `super()`.

            `predict` and `sequence_gradients` resolve `model` themselves, and for `model=None`
            that does not look a name up, it SAMPLES one and splits `self.key` doing it. So
            resolving a second time here draws a second, independent name: the card runs one
            checkpoint's 48 Evoformer blocks while the embedder, the template stack, the
            structure module and the heads around them come from another. Nothing downstream can
            see it, because the shapes agree and the loss still falls.

            A multi-model pool therefore resolves once here and passes the name on. A pool of one
            reads its single name without resolving and passes `None` straight through, because
            sampling from a pool of one still splits `self.key` and the control arm's stream has
            to stay identical for a device result to be read against it.
            """
            if model is not None:
                return self._resolve_model_name(model), model
            if len(self.models) == 1:
                return self.models[0], None
            picked = self._resolve_model_name(None)
            return picked, picked

        @contextlib.contextmanager
        def _route(self, name):
            """Fold this call on the card or on BindCraft 2's own trunk, and say which.

            `name=None` is the control arm standing down. `evoformer_on_device` patches
            `modules.layer_stack` for the whole process, so a `trunk="jax"` model built inside a
            live campaign is spliced too unless it says otherwise, and a validation ensemble is
            exactly that model.
            """
            evo = installed()
            if evo is None:
                yield  # nothing is spliced, every fold is already BindCraft 2's own
                return
            if name is not None and self.routes.get(name, "device") == "device":
                self.pool.use(name)
                yield
                return
            with evo.on_host(name or ""):
                yield

        def predict(self, protein_states, model=None, *args, **kwargs):
            if self.trunk == "jax":
                with self._route(None):
                    return super().predict(protein_states, model, *args, **kwargs)
            name, passed = self._pick(model)
            with self._route(name):
                return super().predict(protein_states, passed, *args, **kwargs)

        def sequence_gradients(self, protein_states, losses, model=None, *args, **kwargs):
            # One gradient round of one trajectory, on that trajectory's own thread. It is the
            # only point in the design loop that is both, which is what `run_campaign` starts
            # the next interleaved trajectory on. A no-op unless trajectories are interleaved.
            _note_progress()
            duotraj.round_entered()
            if self.trunk == "jax":
                with self._route(None):
                    return super().sequence_gradients(protein_states, losses, model,
                                                      *args, **kwargs)
            name, passed = self._pick(model)
            with self._route(name):
                return super().sequence_gradients(protein_states, losses, passed,
                                                  *args, **kwargs)

    _MODEL_CLASS = TenstorrentAlphaFoldDesignModel
    return _MODEL_CLASS


def _factory(*, trunk: str, pool: TrunkPool | None, evoformer: EvoformerOnDevice | None = None,
             extra_msa: "ExtraMsaOnDevice | None" = None,
             template: "TemplateOnDevice | None" = None, exact: bool = False):
    cls = design_model_class()

    def build(*args, **kwargs):
        return cls(*args, trunk=trunk, pool=pool, **kwargs)

    build.trunk = trunk
    build.pool = pool
    build.evoformer = evoformer
    #: The extra-MSA swap, or None when the stack stayed in BindCraft 2's JAX. Its `calls`,
    #: `swapped` and `mask_seen` counters are how a caller checks the on-card path ran.
    build.extra_msa = extra_msa
    #: The template pair-stack swap, or None when it stayed in BindCraft 2's JAX. Its `calls`
    #: and `seen` counters are how a caller checks the on-card path ran.
    build.template = template
    #: Whether the tape this factory's models open runs softmax and layer norm exact. Inert on
    #: `trunk="jax"`, which opens no tt-bio tape at all.
    build.exact = exact
    #: What `fast_round` armed for this factory's models, or None.
    build.fast = None
    return build


#: The gradient levers the device round is measured with (`state/perf10/bcx-p10-devtop.md`),
#: as `(module, object, attribute, env var, value)`. Every one defaults off in tt-bio, so the
#: other models that open a tape run without them; `fast_round` turns them on for the duration
#: of a BindCraft 2 predictor only. A named env var still wins, so an A/B can take one lever out.
#: The rows without one are defaults their module reads live against its own env var already.
_FAST_ROUND = (
    ("mm_layout", None, "MM_LAYOUT", "TT_BIO_MM_LAYOUT", True),
    ("reblock_permute", None, "TAPED_MOVE", "TT_BIO_TAPED_CHANNEL_MOVE", True),
    ("rne_add", None, "WIDEN_ADD", "TT_BIO_WIDEN_ADD", True),
    ("taped_ttnn", None, "QKV_GRAD_JOIN", "TT_BIO_QKV_GRAD_JOIN", True),
    ("triatt_bw", None, "FUSED", "TT_BIO_TRIATT_BW_FUSED", True),
    ("triatt_bw", None, "QKV_PACKED", "TT_BIO_TRIATT_BW_QKV_PACKED", True),
    ("triatt_bw", None, "EXP_21F", "TT_BIO_TRIATT_BW_EXP_21F", True),
    ("autograd", None, "FANIN_CAST_FUSED", "TT_BIO_FANIN_CAST_FUSED", True),
    ("pair_transpose", None, "PAIR_TRANSPOSE_FUSED", "TT_BIO_PAIR_TRANSPOSE_FUSED", True),
    ("reblock_permute", None, "GATED_GRAD_PACKED", "TT_BIO_GATED_GRAD_PACKED", True),
    ("lnbw", None, "FUSED", "TT_BIO_LNBW_FUSED", True),
    ("reblock_permute", None, "GATED_BW_FUSED", "TT_BIO_GATED_BW_FUSED", True),
    ("gate_bw", None, "GATE_BW_FUSED", "TT_BIO_GATE_BW_FUSED", True),
    ("lead_sum", None, "LEAD_SUM_FUSED", "TT_BIO_LEAD_SUM_FUSED", True),
    ("pair_mm", None, "PAIR_MM_FUSED", "TT_BIO_PAIR_MM", True),
    ("ops", None, "NOGRAD_IS_INFERENCE", "TT_BIO_NOGRAD_INFERENCE", True),
    ("tenstorrent", None, "_TRIATT_FUSED_HIFI", "TT_BIO_TRIATT_FUSED_HIFI", True),
    ("af2", "AF2PairBlock", "rne_kernel", None, True),
    ("af2", "AF2PairBlock", "tri_att_g_in_matmul", "TT_BIO_AF2_G_BIAS_IN_MATMUL", True),
    ("af2", "AF2MaskedOuterProductMean", "rows_in_k", "TT_BIO_AF2_OPM_ROWS_IN_K", True),
    ("taped_ttnn", None, "TAPED_KERNELS_DEFAULT", None,
     "tri_att_sdpa_hifi,rne_add,reblock_permute_gated,pair_transpose,triatt_qkv_heads"),
    # OpenFold3 training turned the fp32 softmax backward on by default; the round above was
    # measured and graded (1.051x of the bf16 control) with it off, so it stays off here.
    ("autograd", None, "SOFTMAX_BW_FP32", "TT_BIO_SOFTMAX_BW_FP32", False),
)

#: Rows graded on Blackhole only (qb2 p300c, `state/bcp-evo.md`). On Wormhole `fast_round` leaves
#: them at the value it found, or for the taped-kernel list at `_FAST_ROUND_WORMHOLE_KERNELS`, until
#: each has a float64 grade and a round on a Wormhole chip. A named env var still wins.
_BLACKHOLE_ONLY = frozenset({
    "QKV_PACKED", "EXP_21F", "FANIN_CAST_FUSED", "PAIR_TRANSPOSE_FUSED", "GATED_GRAD_PACKED",
    "GATED_BW_FUSED", "GATE_BW_FUSED", "LEAD_SUM_FUSED", "PAIR_MM_FUSED", "NOGRAD_IS_INFERENCE",
    "tri_att_g_in_matmul", "TAPED_KERNELS_DEFAULT",
})
_FAST_ROUND_WORMHOLE_KERNELS = "tri_att_sdpa_hifi,rne_add"


@contextlib.contextmanager
def fast_round():
    """Arm `_FAST_ROUND` for the duration and put every value back on exit.

    Process-wide, not per thread: interleaved trajectories run on their own threads inside it
    and must all see the same program. Yields `{attribute: value}` as armed, for a stamp.
    """
    import importlib

    from tt_bio.envflags import env_flag

    from tt_bio import tenstorrent

    wormhole = tenstorrent.is_wormhole()
    saved = []
    try:
        for module, owner, attr, env, value in _FAST_ROUND:
            target = importlib.import_module(f"tt_bio.{module}")
            if owner:
                target = getattr(target, owner)
            saved.append((target, attr, getattr(target, attr)))
            if wormhole and attr in _BLACKHOLE_ONLY:
                value = (_FAST_ROUND_WORMHOLE_KERNELS if attr == "TAPED_KERNELS_DEFAULT"
                         else getattr(target, attr))
            setattr(target, attr, env_flag(env, value) if env else value)
        yield {attr: getattr(t, attr) for t, attr, _ in saved}
    finally:
        for target, attr, old in reversed(saved):
            setattr(target, attr, old)


@contextlib.contextmanager
def predictor(*, trunk: str = "device", card: int | str | None = None, checkpoints=None,
              resident: int | None = None, blocks: int = EVOFORMER_BLOCKS,
              recompute: bool = True,
              memory: str = "auto",
              extra_msa: bool = True,
              template: bool = True,
              exact: bool = False,
              fast: bool | None = None) -> Iterator[Callable[..., object]]:
    """Put tt-bio's Evoformer on card for the duration and yield a predictor factory.

    The factory takes BindCraft 2's own `AlphaFoldDesignModel` arguments (`presets`, `data_dir`,
    `num_recycle`, `length_bucket_size` and the rest) and returns a
    `DifferentiableProteinPredictor`::

        with bindcraft2.predictor(card=0) as build:
            model = build(presets="model_1_ptm", data_dir=params, length_bucket_size=32)
            gradients, prediction = model.sequence_gradients(protein_states, losses)

    `card` pins the chip and must be set before ttnn is imported; leave it None to accept
    whatever `TT_VISIBLE_DEVICES` already says. `checkpoints` is a `TrunkPool`, a directory of
    `params_<name>.npz`, a mapping of model name to file, a single file, or None for tt-bio's own
    weights cache. `resident` caps how many trunks stay on card at once.

    `extra_msa` additionally runs the 4-block extra-MSA stack on card and `template` the
    multimer template embedder's two c=64 pair blocks. Both default ON, because leaving them in
    BindCraft 2's JAX costs a round far more on the host than running them costs on the card:
    on a Wormhole Galaxy chip the round is 29.423 s with them in JAX against 16.267 s on card,
    1.8087x, host 17.188 -> 2.417 s against device 12.294 -> 13.856 (eight arms alternated at
    288 tokens, AICLK 1000 with 0 of 454 samples under it, `perf/bwx_perf/results/`). Blackhole
    agrees to 3 %: 37.675 against 20.220 s, 1.863x (`bcx-p10-resident`). That is one
    trajectory; two interleaved hide most of the host column, and there it is 1.1331x
    (15.776 -> 13.923 s on the same chip). Pass False to keep a
    comparison graded on the Evoformer alone on the program it was graded on. Read
    `build.extra_msa.calls` and `build.template.calls` to check the on-card paths ran.

    The template swap brings two more blocks of weights per checkpoint onto the card, so it is
    not free of allocator pressure, and `duotraj.auto_trajectories` prices the card from
    `free_device_bytes()` once a chip is open, which counts them. Monomer checkpoints are
    untouched: the swap is installed on `modules_multimer` only.

    `exact` runs softmax and layer norm on the host in float64 inside the tape, which
    reproduces AlphaFold 2's own gradient most closely. It is off by default because it is a
    diagnostic and it is expensive: one `sequence_gradients` call on a PD-L1 draw at n=192 takes
    479.59 s with it on against 19.285 s with it off, 24.87x (`perf/bcx_exact/ROUND_AB.json`).
    That is the gradient call, not the whole design round, which also carries BindCraft 2's own
    JAX work. What it buys is 1.1 % on the worst gradient tensor: at n=192 the MSA cotangent out
    of Evoformer block 1 sits 0.087998 from a float64 reference with it on and 0.088985 with it
    off, where bfloat16 alone already carries 0.075483 of that
    (`perf/bcx_exact/grade/VJP_on_n192.json` and `VJP_off_n192.json`). A PD-L1 campaign on the
    off setting accepts binders (`perf/bcx_exact/ACCEPT_GRADE.json`), which is the bar a design
    loop is graded on. Set `exact=True` to reproduce a training-style gradient bar, which
    BindCraft 2 does not have. Read `tt_bio.autograd.EXACT_SOFTMAX_STATS` to confirm which one
    ran.

    `memory` is how the round spends device memory, one of `MEMORY_MODES` or ``"auto"``, the
    default, which picks the fastest mode that fits this card at each fold's token axis. A
    Wormhole Galaxy chip runs ``fast`` to 512 tokens; past that, ``lean`` and ``offload`` trade
    seconds for room (`MEMORY_MODES` says how, docs/bindcraft2.md what they cost). Read
    `build.memory.used` for the mode each token axis ran in.

    `fast` arms the gradient levers the device round is measured with (`fast_round`) for the
    duration, and defaults to `not exact`: they change which kernels compute the round, not
    the work it does, and the exact tape is the arm they are graded against. `build.fast`
    holds what was armed.

    `trunk="jax"` opens no device and touches no card. It runs BindCraft 2's own trunk through
    this same class, which is the control arm every device result should be read against.

    With `trunk="device"`, a checkpoint whose weights the source cannot supply folds on that host
    trunk rather than raising, decided per model family and announced on the first such fold. If
    the source supplies nothing at all, building the model raises instead: an empty pool is a
    misconfiguration, not a route.
    """
    if trunk == "jax":
        yield _factory(trunk="jax", pool=None, exact=exact)
        return
    if card is not None:
        pin_card(card)
    # After `pin_card`: importing autograd imports ttnn, and a pin after that raises.
    from tt_bio import autograd
    from tt_bio.tenstorrent import get_device

    # Open the chip on this thread, not on whichever thread first reaches the card. UMD's
    # CHIP_IN_USE lock is a robust pthread mutex owned by the thread that opens the chip, and the
    # close at exit runs on the main thread. Left to the first device call, the open happens on
    # one of XLA's CPU pool threads (the device seams run there), the unlock at exit fails with
    # EPERM and the process aborts with exit 134 after the campaign has already finished.
    get_device()

    pool = checkpoints if isinstance(checkpoints, TrunkPool) else TrunkPool(
        checkpoints, resident=resident, template=template)
    if template and not pool.template:
        raise ValueError("predictor(template=True) needs a TrunkPool built with template=True; "
                         "the trunks it already loaded hold no template blocks on card")
    mem = _Memory(memory)
    evo = EvoformerOnDevice(pool, blocks=blocks, recompute=recompute, memory=mem)
    extra = ExtraMsaOnDevice(pool, recompute=recompute, memory=mem) if extra_msa else None
    tmpl = TemplateOnDevice(pool, recompute=recompute, memory=mem) if template else None
    # `_EXACT_TRAINING` is a process-wide stack, not thread-local, so this covers every tape
    # opened for the duration -- both `_taped` calls and the backward's recompute -- without
    # either swap having to know about it.
    fast = not exact if fast is None else fast
    # Outermost, so a refusal from any seam under it reaches the caller as the `MemoryError`
    # it was raised as even when JAX stringified it into a `JaxRuntimeError` on the way out.
    with refusals_unwrapped(), \
            autograd.exact_training(exact), evoformer_on_device(evo, extra), \
            template_on_device(tmpl), \
            (fast_round() if fast else contextlib.nullcontext()) as armed:
        build = _factory(trunk="device", pool=pool, evoformer=evo, extra_msa=extra,
                         template=tmpl, exact=exact)
        build.fast = armed
        build.memory = mem
        yield build


@contextlib.contextmanager
def campaign_predictor(*, validation: str = "jax",
                       **kwargs) -> Iterator[Callable[..., object]]:
    """`predictor`, with BindCraft 2's own predictor construction rebound to it.

    `campaign.py` builds `AlphaFoldDesignModel` in one place for the design model and one for the
    validation model, so rebinding that name is the whole integration::

        with bindcraft2.campaign_predictor(card=0):
            campaign.run_campaign(settings, project, af2_weights=params, mpnn_weights=mpnn)

    `validation` is the trunk the validation ensemble folds on and it defaults to `"jax"`,
    BindCraft 2's own. The validation ensemble is the instrument that decides whether a design is
    accepted, so folding it on card would put device numerics inside the measurement that grades
    the device; on host JAX that stage is bit-for-bit the reference's own and the thing under
    test stays the gradient loop. Any accepted count quoted as a result should come from this
    default, and should say so. Pass `validation="device"` to fold it on card as well, and
    re-measure the accepted count on that path before quoting it.

    The design model is the first predictor `run_campaign` builds and every later one is a
    validation ensemble, including the one `desperate_prediction_pools` rebuilds mid-campaign
    when validation has to move to another pool, so "every build after the first" is the rule.

    The design model's checkpoints come from one pool, so `resident` caps the card across it.

    Everything `predictor` takes passes through, `extra_msa` and `exact` included, and the swap
    it builds is re-exposed as `build.extra_msa` so a campaign can read its counters.
    """
    if validation not in ("jax", "device"):
        raise ValueError(f"validation must be 'jax' or 'device', not {validation!r}")
    with predictor(**kwargs) as build:
        from bindcraft import campaign
        control = None
        built = []
        # "The first build is the design model" is per TRAJECTORY, not per process: interleaved
        # trajectories each build their own design model on their own thread, and counting them
        # together would hand the second trajectory the host control arm as its design model.
        # A single-trajectory process has one slot and counts exactly as it did before.
        design_models = {}
        building = threading.Lock()

        def build_for_campaign(*args, **kw):
            nonlocal control
            with building:
                slot = duotraj.slot()
                nth = design_models[slot] = design_models.get(slot, 0) + 1
                if nth > 1 and validation == "jax" and build.trunk == "device":
                    if control is None:
                        control = _factory(trunk="jax", pool=None)
                    factory = control
                else:
                    factory = build
            made = factory(*args, **kw)
            built.append(made)
            return made

        build_for_campaign.trunk = build.trunk
        build_for_campaign.pool = build.pool
        build_for_campaign.evoformer = build.evoformer
        build_for_campaign.extra_msa = build.extra_msa
        build_for_campaign.template = build.template
        build_for_campaign.exact = build.exact
        build_for_campaign.fast = build.fast
        build_for_campaign.memory = getattr(build, "memory", None)
        build_for_campaign.validation = validation
        build_for_campaign.built = built

        real = campaign.AlphaFoldDesignModel
        campaign.AlphaFoldDesignModel = build_for_campaign
        try:
            yield build_for_campaign
        finally:
            campaign.AlphaFoldDesignModel = real


def stop_conditions(settings: Mapping) -> str:
    """How this campaign will end, in one line, before a card is opened.

    A campaign has two stop conditions and a researcher sets both: `number_of_final_designs`, the
    designs they want, and `max_trajectories`, what they will spend getting them. Whichever comes
    first ends it. Two of the ways of writing that are not what they look like, measured against
    the real accounting on 2026-09-30:

    * `max_trajectories=0` is **unbounded**, not "none" -- the same as leaving it out. Set to 0 by
      someone who meant "do not design", with designs still requested, it runs until enough
      designs are accepted, which on a hard target can be never, holding a leased chip the whole
      time. A soak is exactly where that gets discovered at 3 a.m.
    * a negative budget designs nothing at all and produces an empty folder, which reads like a
      crash rather than a setting.

    The count is also a floor rather than a quota: the stop condition is read when a trajectory is
    claimed, so interleaved arms already in flight can push the accepted total past the request.
    """
    budget = settings.get("max_trajectories")
    designs = settings.get("number_of_final_designs")
    if settings.get("trajectory_only"):
        return ""          # upstream already says a trajectory-only run accepts nothing
    try:
        budget = None if budget is None else int(budget)
        designs = None if designs is None else int(designs)
    except (TypeError, ValueError):
        return ""          # a malformed setting is preflight's to refuse, not this line's
    if budget is not None and budget < 0:
        return (f"[tt_bio.bindcraft2] max_trajectories={budget} designs NOTHING: no trajectory is "
                f"started and the campaign folder stays empty, which reads like a crash. Set it to "
                f"the number of trajectories you are willing to spend.")
    if designs is not None and designs <= 0:
        return (f"[tt_bio.bindcraft2] number_of_final_designs={designs} accepts NOTHING: the "
                f"campaign stops at its first claim. Set it to the number of designs you want.")
    if not budget:
        return (f"[tt_bio.bindcraft2] no trajectory budget "
                f"(max_trajectories={budget!r} means unbounded, not zero): this campaign runs "
                f"until {designs if designs is not None else 'the requested number of'} designs "
                f"pass the filters, however long that takes, and holds this chip until it does. "
                f"Set max_trajectories to bound the spend.")
    if designs is None:
        return (f"[tt_bio.bindcraft2] stops after {budget} trajectories.")
    return (f"[tt_bio.bindcraft2] stops at whichever comes first: {designs} accepted design"
            f"{'' if designs == 1 else 's'}, or {budget} trajector"
            f"{'y' if budget == 1 else 'ies'} spent. Interleaved arms already running can carry "
            f"the accepted count past {designs}, so it is a floor, not a quota.")


GATE_METRICS = (("min_monomer_plddt_", "monomer pLDDT"), ("min_plddt_", "pLDDT"),
                ("min_iptm_", "i_pTM"))
STAGE_ORDER = ("screen", "refine", "anneal", "harden", "mutate", "final")


def stage_gates(settings: Mapping) -> str:
    """The per-stage gates a trajectory can die on, which the campaign banner does not announce.

    A campaign prints one `filters ...` line, the acceptance filters, and a researcher reads it as
    the whole bar. It is not: a trajectory is also judged at the end of every design stage against
    a second set of thresholds carried in the campaign settings as `min_plddt_<stage>`,
    `min_iptm_<stage>` and `min_monomer_plddt_<stage>`, and missing one ends the trajectory there.
    That is charged against `max_trajectories` and shows up as a `terminated` count in
    `.campaign_state.json`.

    Measured on the live Wormhole soak leg on 2026-09-30, 2 of the first 4 trajectories ended this
    way, and what the run said was::

        rejected at refine design stage  i_pTM=0.79    pLDDT=0.59   due to [pLDDT]

    `pLDDT` appears in none of the seven filters the same run announced, so the reason names a bar
    the reader cannot find, cannot see the value of (0.6 here -- a 0.01 miss), and cannot move
    without knowing the setting behind it. This line names all three: stage, setting, threshold.

    The gates are read out of the settings by prefix rather than from a list kept here, because a
    list kept here goes stale silently: the first version of this named seven gates while the
    resolved pdl1 settings carried eleven, so it would have told a researcher that
    `min_iptm_final` and `min_iptm_anneal` -- both of which do end trajectories -- did not exist.
    """
    found = {}
    for key, value in settings.items():
        for prefix, metric in GATE_METRICS:
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            try:
                found[key] = (key[len(prefix):], metric, float(value))
            except (TypeError, ValueError):
                pass       # a malformed threshold is preflight's to refuse, not this line's
            break
    if not found:
        return ""

    def order(item):
        stage, metric, _ = item[1]
        rank = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
        return (rank, stage, metric)

    parts = [f"{stage} {metric} >= {value:g} ({key})"
             for key, (stage, metric, value) in sorted(found.items(), key=order)]
    metrics = sorted({metric for _, metric, _ in found.values()})
    return (f"[tt_bio.bindcraft2] a trajectory is also ended mid-design by the per-stage gates, "
            f"which the campaign's own `filters` line does not list: {', '.join(parts)}. A "
            f"trajectory that misses one is charged against the budget and counted under "
            f"`terminated`, and the rejection names the metric ({'/'.join(metrics)}), not the "
            f"setting.")


def print_stage_gates(settings: Mapping) -> str:
    line = stage_gates(settings)
    if line:
        print(line, flush=True)
    return line


def print_stop_conditions(settings: Mapping) -> str:
    line = stop_conditions(settings)
    if line:
        print(line, flush=True)
    return line


def print_resumption(project_folder: str, max_trajectories=None) -> str:
    """Say what a resumed campaign inherited, before it claims anything.

    A design campaign runs for hours and gets interrupted, and BindCraft 2 carries on into the
    same folder when `resume: true` is set. What it carries on from is not obvious, because the
    budget is spent when a trajectory is CLAIMED and its row is written only when it finishes:
    a campaign killed mid-trajectory has paid for work it has no row for, so it resumes one
    number further along than its own table, does not retry that recipe, and delivers one
    trajectory fewer than was asked for. And that memory lives in `.campaign_state.json`; delete
    that file and the counts are recovered from the tables instead, which cannot see the
    interrupted claim, so the same trajectory number is used twice across the two runs.

    None of that is wrong -- charging on the claim is what stops a trajectory that kills the
    process from being retried forever -- but all of it was silent. This prints it. Returns the
    line it printed, or "" for a fresh folder.
    """
    from bindcraft.campaign_output import (TRAJECTORY_STAGE, accepted_table, csv_row_count,
                                           stage_table)

    state_path = pathlib.Path(project_folder) / ".campaign_state.json"
    rows = csv_row_count(stage_table(project_folder, TRAJECTORY_STAGE))
    accepted_rows = csv_row_count(accepted_table(project_folder))
    if not state_path.exists() and not rows and not accepted_rows:
        return ""
    charged = accepted = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            charged, accepted = int(state.get("trajectories", 0)), int(state.get("accepted", 0))
        except (OSError, ValueError, TypeError):
            charged = accepted = None
    where = ("`.campaign_state.json`" if charged is not None
             else "the campaign tables, since `.campaign_state.json` is absent or unreadable")
    counted = charged if charged is not None else rows
    said = [f"[tt_bio.bindcraft2] resuming {project_folder}: {counted} "
            f"{'trajectory' if counted == 1 else 'trajectories'} already charged according to "
            f"{where}, {rows} in the trajectory table, "
            f"{accepted if accepted is not None else accepted_rows} accepted."]
    if charged is not None and charged > rows:
        missing = charged - rows
        said.append(f"{missing} claimed {'trajectory' if missing == 1 else 'trajectories'} "
                    f"never finished; {'it is' if missing == 1 else 'they are'} charged to the "
                    f"budget and will not be retried, so this campaign runs {missing} fewer "
                    f"than you asked for.")
        if max_trajectories:
            said.append(f"Raise max_trajectories to {int(max_trajectories) + missing} to get the "
                        f"{int(max_trajectories)} you wanted.")
    elif charged is None and rows:
        said.append("Recovered from the tables, so an interrupted trajectory's number is claimed "
                    "again; its recipe is still declined, so no design is repeated.")
    line = " ".join(said)
    print(line, flush=True)
    return line


@contextlib.contextmanager
def _one_campaign_not_n(campaign, trajectories: int):
    """Hold the two campaign-wide things N trajectories in one process would each do.

    `write_campaign_summary` reads the whole project and rewrites `summary.csv` through one
    fixed `summary.csv.partial`, and it takes no lock. N interleaved trajectories share a stop
    condition, so they reach it within milliseconds of each other and two of them writing that
    one partial file at once produce a summary that is neither. N worker PROCESSES have the same
    race and it is upstream's to fix; a process-wide lock is what this module can do about its
    own threads.

    `print_campaign_header` folds the first design trajectory to report the target it is about
    to run. It is the same campaign N times over, so it is printed once.

    The closing lines (`campaign stopped: ...`, `campaign done: ...`) are printed inline by
    `run_campaign` itself, gated on `design_worker_index() is None`, which is how BindCraft 2
    keeps N worker PROCESSES from each announcing the end. N threads share one environment, so
    all N pass that gate: the campaign was announced over twice while a trajectory was still
    printing stage lines. Those two call sites are the only readers of that name in
    `campaign.py`, so holding it back until the last arm arrives puts the footer last and once.
    """
    summary, header = campaign.write_campaign_summary, campaign.print_campaign_header
    worker_index = campaign.design_worker_index
    writing = threading.Lock()
    closing = threading.Lock()
    printed = []
    arrived = set()

    @functools.wraps(summary)
    def write_campaign_summary(*args, **kwargs):
        with writing:
            return summary(*args, **kwargs)

    @functools.wraps(header)
    def print_campaign_header(*args, **kwargs):
        with writing:
            if printed:
                return None
            printed.append(True)
        return header(*args, **kwargs)

    @functools.wraps(worker_index)
    def design_worker_index():
        if worker_index() is not None:
            return worker_index()  # A real worker process: upstream's gate already holds.
        with closing:
            arrived.add(threading.get_ident())
            return None if len(arrived) >= trajectories else 0

    campaign.write_campaign_summary = write_campaign_summary
    campaign.print_campaign_header = print_campaign_header
    campaign.design_worker_index = design_worker_index
    try:
        yield
    finally:
        campaign.write_campaign_summary = summary
        campaign.print_campaign_header = header
        campaign.design_worker_index = worker_index


#: Seconds with no gradient round begun AND nothing written under the project folder before a
#: campaign says it may be stuck. A round is 7-16 s; the slowest silent stretch a healthy
#: campaign has shown is a cold compile of a new length bucket plus the validation ensemble,
#: minutes, not tens of them. 45 minutes is far past both, so the line never fires on a campaign
#: that is merely slow -- and it only prints, it never stops anything.
STALL_WARN_S = float(os.environ.get("TT_BIO_CAMPAIGN_STALL_WARN_S", "2700"))

_LAST_PROGRESS = [0.0]


def _note_progress() -> None:
    import time
    _LAST_PROGRESS[0] = time.time()


def _newest_write(folder: str) -> float:
    newest = 0.0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                pass
    return newest


def stall_message(project_folder: str, quiet_s: float) -> str:
    return (f"[tt_bio.bindcraft2] no gradient round has started and nothing has been written under "
            f"{project_folder} for {quiet_s / 60:.0f} min. A chip that wedges mid-campaign does not "
            f"raise: the process just stops, holding the card. If this line repeats, the campaign "
            f"is not coming back on its own. Stop it (Ctrl-C or kill {os.getpid()}), reset the chip "
            f"if it stays unresponsive, and rerun on the same folder with resume=true: every design "
            f"accepted so far is kept, and the trajectory that was running is charged, not "
            f"repeated.")


@contextlib.contextmanager
def _stall_reporter(project_folder: str, warn_s: float | None = None, every: float = 60.0,
                    say: Callable[[str], None] | None = None):
    """Print once per silent stretch when a campaign has made no visible progress for `warn_s`.

    A chip that wedges after bring-up is the one sick-chip case nothing else catches. The
    bring-up probe is bounded (`tenstorrent._assert_local_dispatch`, 120 s); a busy chip is
    refused by its lease; an absent chip is refused by `visible_device_indices`. A wedge in
    round 400 of trajectory 15 blocks inside a C call with no timeout of its own, and the
    researcher is left looking at a log that simply stopped, with no way to tell a hang from a
    slow stage. Progress is either signal: a gradient round starting, or any file under the
    project folder changing, which covers MPNN, validation and acceptance, which run no rounds.
    """
    import time
    warn_s = STALL_WARN_S if warn_s is None else warn_s
    say = say or (lambda line: print(line, file=sys.stderr, flush=True))
    stop = threading.Event()
    _note_progress()

    def watch():
        warned_at = None
        while not stop.wait(every):
            last = max(_LAST_PROGRESS[0], _newest_write(project_folder))
            quiet = time.time() - last
            if quiet < warn_s:
                warned_at = None
            elif warned_at is None or quiet - warned_at >= warn_s:
                say(stall_message(project_folder, quiet))
                warned_at = quiet

    if warn_s <= 0:
        yield
        return
    watcher = threading.Thread(target=watch, name="bindcraft2:stall", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stop.set()


def run_campaign(settings: Mapping, project_folder: str, *,
                 trajectories_per_card: "int | str" = "auto",
                 stagger_timeout: float = 1800.0, **run_campaign_kwargs) -> int:
    with _stall_reporter(project_folder):
        return _run_campaign(settings, project_folder, trajectories_per_card=trajectories_per_card,
                             stagger_timeout=stagger_timeout, **run_campaign_kwargs)


def _run_campaign(settings: Mapping, project_folder: str, *,
                  trajectories_per_card: "int | str" = "auto",
                  stagger_timeout: float = 1800.0, **run_campaign_kwargs) -> int:
    """BindCraft 2's campaign loop, with as many trajectories on one card as the box can hold.

    Call it where you would call `campaign.run_campaign`, inside `campaign_predictor`::

        with bindcraft2.campaign_predictor(card=0, exact=False):
            bindcraft2.run_campaign(settings, project, af2_weights=params, mpnn_weights=mpnn)

    The default, `"auto"`, prices a trajectory at THIS design's token axis, reads free host
    memory (and the card, if one is open) BEFORE any thread starts, and takes the largest count
    that fits up to `duotraj.AUTO_CAP`, floored at 1. It prints the count and the reason it chose
    it. Both footprints grow with the square of the axis, so the count falls as the design grows:
    three trajectories of a 288-token design hold about 10 GB of the card and one of a 704-token
    design holds 19. A box whose free memory cannot be read, or a design whose axis cannot be
    read, gets 1: the two ways of guessing wrong do not cost the same, and the expensive one is a
    campaign that dies.

    `trajectories_per_card=1` is BindCraft 2's own call, unchanged: no threads, no gate, nothing
    in tt-bio behaves differently. Any explicit number is honoured exactly, including one this
    box cannot hold, which still raises `MemoryError` rather than being quietly lowered. Above 1
    it runs that many design trajectories on their own threads over one chip, each taking its own
    trajectory number out of the project's file-locked progress exactly as separate worker
    processes would.

    It is worth doing because a design round is a host column and a device column laid end to
    end, and one trajectory cannot overlap them: measured on a 288-token PD-L1 round, the card
    idles 2.6 s of every round with a host thread busy in all of it. Independent trajectories are
    the only work there is to fill it with. On one Blackhole chip the round goes from 9.06 s
    serial to 7.23 s at two trajectories and 6.76 s at three, amortised over the trajectories
    running (`state/perf10/bcx-p10-tritraj.md`).

    It costs memory on both sides, and both grow with the token axis:
    `duotraj.trajectory_bytes` and `duotraj.trajectory_host_bytes` are the estimates, fitted on
    a measured ladder from 288 to 704 tokens. `duotraj.refuse_if_it_will_not_fit` reads the box
    and the card before any thread starts and raises `MemoryError` naming what it wanted and what
    was free, because a campaign that dies is worse than a slower one.

    Trajectory i starts only once i-1 has its first gradient round behind it, so no two
    trajectories trace and compile at the same time; `stagger_timeout` bounds that wait.

    Returns the campaign's trajectory count. A trajectory that raises re-raises here once the
    others have finished, rather than leaving them orphaned on the card.
    """
    from bindcraft import campaign

    # Before a card is opened: a hotspot that names no residue of the target sets no flag, and
    # BindCraft 2 reads that as a campaign with no epitope rather than as a mistake.
    bcinputs.refuse_unusable_inputs(settings)
    print_resumption(project_folder, settings.get("max_trajectories"))
    print_stop_conditions(settings)
    print_stage_gates(settings)

    tokens = design_tokens(settings)
    if isinstance(trajectories_per_card, str):
        if trajectories_per_card != "auto":
            raise ValueError('trajectories_per_card must be a count or "auto", not '
                             f"{trajectories_per_card!r}")
        trajectories, why = duotraj.auto_trajectories(tokens)
        print(f"[tt_bio.bindcraft2] {trajectories} design "
              f"{'trajectory' if trajectories == 1 else 'trajectories'} on this card: {why}. "
              f"Pass trajectories_per_card to choose yourself; 1 is BindCraft 2's own loop.",
              flush=True)
    else:
        trajectories = int(trajectories_per_card)
    if trajectories < 1:
        raise ValueError("trajectories_per_card must be at least 1, not "
                         f"{trajectories_per_card!r}")
    if trajectories == 1:
        with refusals_unwrapped():
            return campaign.run_campaign(settings, project_folder, **run_campaign_kwargs)

    names = [f"t{i + 1}" for i in range(trajectories)]

    def one(i: int):
        waits_for = names[i - 1] if i else None

        def go():
            if waits_for is not None:
                cleared = duotraj.compile_round_cleared(waits_for)
                if not cleared.wait(stagger_timeout):
                    raise TimeoutError(
                        f"{names[i]}: {waits_for} did not clear its first gradient round in "
                        f"{stagger_timeout:.0f}s, so nothing was interleaved")
            try:
                return campaign.run_campaign(settings, project_folder, **run_campaign_kwargs)
            finally:
                # A trajectory that stops before its second round sets no event of its own, and
                # the next one would wait out the whole timeout for one that will never come.
                duotraj.compile_round_cleared(names[i]).set()
        return go

    with refusals_unwrapped(), \
            duotraj.interleave(trajectories=trajectories, tokens=tokens), \
            _one_campaign_not_n(campaign, trajectories):
        counted = duotraj.run([one(i) for i in range(trajectories)], names=names)
    # Each arm returns the trajectory count it read out of the shared campaign progress as it
    # left, so the last one out carries the whole campaign's. Summing would count that one file
    # N times.
    return max((n for n in counted if isinstance(n, int)), default=0)


run_campaign.__doc__ = _run_campaign.__doc__
