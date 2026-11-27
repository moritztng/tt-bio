"""``tt-bio train --model rfd4``: the Tier-0 entry for RFD4-Proteina.

RFD4's step is not the generic one. Upstream picks a winner among `k` noise candidates with a
grad-off pass through the production trunk and trains only the winner's last recycle, so the
step body lives in :mod:`tt_bio.train.rfd4_run` rather than in ``train.loop``. Everything
around it is the shared stack: the same `Folds`, `AdamW`, `Checkpointer`, `ProcessAxis` and
`GradientCoverage` an OpenFold3 run uses (G6).

What this module owns is the Tier-0 contract over that body: flag legality decided before a
device opens, DATA refused by name when it is missing or empty, ``status.json`` /
``progress.jsonl`` in ``--out``, and resume from the newest checkpoint in the run directory
when the same command is run again.

A flag the RFD4 step does not honour is REFUSED rather than ignored. `--lr` is the example that
forced it: the step takes upstream's rule at 1e-4, and silently accepting `--lr 3e-4` would
hand back a run whose status file disagreed with what it trained.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click

__all__ = ["train_rfd4", "refuse_unhonoured", "latest_checkpoint", "batches",
           "last_row", "NOT_HONOURED", "DATA_SUFFIX", "NO_DATA"]

#: Tier-0 flags the RFD4 step does not honour, and what to do instead. Checked against the
#: parameter SOURCE, so a default never trips one: only a flag the user actually typed.
NOT_HONOURED = {
    "lr": "--lr is not honoured by the RFD4 step, which takes upstream's rule at 1e-4 "
          "(tt_bio.train.rfd4_objective.update_rule). Drop the flag.",
    "warmup_steps": "--warmup-steps is not honoured: upstream's RFD4 recipe has none "
                    "(muon_warmup_steps and adam_warmup_steps are 0). Drop the flag.",
    "train_mode": "--train adapters is not supported for rfd4: the run trains the trunk's own "
                  "weights. Drop the flag.",
    "rank": "--rank is a LoRA flag; rfd4 trains weights. Drop the flag.",
    "alpha": "--alpha is a LoRA flag; rfd4 trains weights. Drop the flag.",
    "targets": "--target is a LoRA flag; rfd4 trains weights. Drop the flag.",
    "objective": "--objective is fixed for rfd4: the flow-matching objective "
                 "(tt_bio.train.rfd4_objective.rfd4_loss). Drop the flag.",
    "seed": "--seed is not honoured: an RFD4 step seeds its noise per step from the step "
            "index, so a run is reproducible without one. Drop the flag.",
}

#: The batch files a DATA directory holds, as `perf/r4t/loss_parity_reference.py` dumps them.
DATA_SUFFIX = ".npz"

#: DATA is required for rfd4. The shipped `rfd4_train_16.npz` fixture is a different thing --
#: upstream's collated features, which the objective and the parity tests read -- and it carries
#: none of what a step replays (the candidate noise, the recycle depth, upstream's winners), so
#: offering it as a default would hand back a KeyError instead of a run.
NO_DATA = ("--model rfd4 needs DATA: a directory of training batches, one .npz per example. "
           "Dump them from upstream's pipeline with perf/r4t/loss_parity_reference.py, which "
           "writes the features, x_1/x_t/t, the candidate noise and upstream's own winners "
           "and losses. docs/training.md has the recipe.")


def batches(data: Path) -> list:
    """The batch files in ``data``, sorted. Refuses a directory holding none, by name."""
    if data is None:
        raise click.UsageError(NO_DATA)
    found = sorted(p for p in Path(data).iterdir() if p.suffix == DATA_SUFFIX) \
        if Path(data).is_dir() else [Path(data)]
    if not found:
        raise click.UsageError(
            f"{data} holds no {DATA_SUFFIX} batch. An RFD4 training batch is what "
            f"perf/r4t/loss_parity_reference.py dumps: one file per example, carrying the "
            f"features, x_1/x_t/t and the candidate noise. docs/training.md has the recipe.")
    return found


def refuse_unhonoured(typed) -> None:
    """Refuse a flag the RFD4 step does not honour. ``typed`` names the flags the user actually
    passed, so an untouched default never trips one."""
    for name in sorted(set(typed) & set(NOT_HONOURED)):
        raise click.UsageError(NOT_HONOURED[name])


def train_rfd4(*, files, out_dir: Path, steps, chip_ids, checkpoint_every, tokens,
               status) -> None:
    """Run the RFD4 step loop on ``files`` under the Tier-0 contract."""
    argv = ["--batch", ",".join(str(f) for f in files), "--out", str(out_dir),
            "--steps", str(steps or len(files)), "--optimizer", "muon", "--exempt", "default"]
    if checkpoint_every:
        argv += ["--save-every", str(checkpoint_every)]
    if len(chip_ids) > 1:
        argv += ["--dp-cards", ",".join(str(c) for c in chip_ids)]
    elif chip_ids[0] != 0:
        os.environ["TT_VISIBLE_DEVICES"] = str(chip_ids[0])
    else:
        os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
    if tokens:
        argv += ["--n", str(tokens)]
    resume = latest_checkpoint(out_dir, chip_ids)
    if resume is not None:
        argv += ["--resume", str(resume)]
        click.echo(f"resuming from {resume}")

    from . import rfd4_run
    # A data-parallel run spawns one process per chip and the driver sees no step: the ranks
    # print theirs to OUT/rank<r>.log. Rank 0's last row is the run's, and it is read back
    # rather than guessed, so a DP status file says the same thing a one-chip one does.
    rc = rfd4_run.main(argv, on_step=None if len(chip_ids) > 1 else status.step)
    if rc:
        raise click.ClickException(f"the RFD4 step loop exited {rc}; its JSON rows are above "
                                   f"and in {out_dir}")
    if len(chip_ids) > 1:
        last = last_row(Path(out_dir) / "rank0.log")
        if last:
            status.step({**last, "s": last.get("step_s")})


def latest_checkpoint(out_dir: Path, chip_ids):
    """The newest checkpoint a previous run of this command left, or None.

    A data-parallel run checkpoints under ``rank0/``; every rank holds the same masters, so
    rank 0's file is the run's.
    """
    from .checkpoint import Checkpointer
    d = Path(out_dir) / "rank0" if len(chip_ids) > 1 else Path(out_dir)
    return Checkpointer(d, every=10**9).latest() if d.is_dir() else None


def last_row(log: Path) -> dict:
    """The last step row in a rank log, or ``{}``. Rows that are not steps (a checkpoint
    notice, a spike notice) are skipped rather than taken for one."""
    out = {}
    try:
        text = Path(log).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "step" in row and "loss" in row:
            out = row
    return out
