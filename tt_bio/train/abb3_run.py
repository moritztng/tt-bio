"""The ABodyBuilder3 reproduction run: crash-safe, auto-resuming, and data-parallel.

`train-b2-abb3-port` put the model on the card and measured the complete step. This module is
the thing that runs it for 193,512 steps on a host that does not stay up that long.

**Auto-resume is a build requirement here rather than an operational nicety, and the number is
measured.** qb2 took 5 watchdog resets in one day (2026-09-13, gaps down to 48 minutes) *while
running four cards at full tilt*, which is this run's load profile, so the schedule contains
roughly 9-47 interruptions and each one kills the process outright. Three consequences are built
in rather than documented:

* the checkpoint cadence is in MINUTES, not steps. What a reset costs is wall clock, so the
  quantity to bound is wall clock, and a step-count cadence silently changes what it caps
  whenever the step time moves -- which it did, from 14.472 s to 38.5 s, within one pass;
* ``max_seconds`` is per PROCESS and the run's cap is not. See ``tt_bio.train.deadline``: the
  5-day grant of ask 9115 is an epoch instant in the output directory, and each launch turns
  what is left of it into this process's budget. A cap measured from process start would be a
  cap on the gap between two watchdog resets, which is not a cap on anything;
* the optimizer state is not optional to restore. See ``abb3_checkpoint``;
* the checkpoint is **chip-count agnostic**. It holds the global batch's divisor, not the chip
  count, so a 2-chip run interrupted by a reset can resume on 1 chip when only one card is
  free. Refusing that would turn a card-availability problem into a lost run.

**The short final batch is KEPT, and that is pinned rather than chosen.** ``batches`` defaults to
dropping it, which would be the ordinary choice and is wrong here: upstream's released
checkpoint sits at ``global_step`` 193,512, and with their 8,395 training structures at batch 64
that is ``132 x 1466`` exactly, where dropping the partial batch gives ``131 x 1466 = 192046``.
So they keep it, and a run that drops it walks a different data order from the second epoch
onward -- with a loss curve that looks perfectly healthy the whole way.

**Data parallelism is one process per chip and the reduce is on the host.** A ttnn process that
can see four chips brings up all four, so each replica pins one card and the replicas cannot
share a device context -- ``ttnn.all_reduce`` is therefore unavailable and the exchange is
host-side. That costs nothing: the step already pulls every gradient to a float32 mirror before
the optimizer runs, so the 28.4 MB per rank per step is a transfer the step was paying anyway.

**The reduce is installed inside the optimizer, not hung off a callback.** ``tt_bio/train/mesh.py``
argues the position and the argument applies unchanged here: a forgotten reducer trains N
diverging replicas behind N loss curves that all fall, and there is no warning available at the
callback layer because a missing callback is indistinguishable from someone who meant one chip.
:class:`ReducedRAdam` wraps the recipe's own ``torch.optim.RAdam`` and is installed at
construction, so ``TrainStep.step()`` reduces whether or not the caller remembered -- and B2's
step file is not edited to achieve it.

**Each rank hashes its masters every step and the run fails unless all hashes agree.** That is
the only check with any power over the failure it addresses. It also covers the resume: every
rank loads the same file, so equality on the first post-resume step is a positive statement that
the resume restored all of them and not three of four.

**The learning rate follows the recipe's cosine, and it is driven by the GLOBAL STEP.**
``params.yaml``'s ``optimiser:`` block is RAdam at 5e-4 *plus* ``T_0: 50``, ``T_mult: 1`` and
``eta_min: 0``, which ``lightning_module.py:144`` turns into ``CosineAnnealingWarmRestarts``
stepped once per epoch. The first version of this run built a bare ``RAdam`` and nothing wrapped
its lr, so it held a constant 5e-4: 1.96x upstream's mean over a full schedule, and none of the
four troughs. :class:`CosineRestartsByStep` installs upstream's own scheduler class and samples
it at the FRACTIONAL epoch of the step about to run, ``(global_step - 1) / steps_per_epoch``.

Fractional rather than upstream's 50-value staircase for one reason that matters more here than
the smoothness does: the lr becomes a pure function of the global step, so a run that comes back
from a checkpoint lands on exactly the value the step index says and no scheduler state has to
survive in the checkpoint. A call-counted scheduler restarts at the top of the cosine on every
resume, and over a schedule containing 9 to 47 watchdog resets that is a different recipe every
time, with a perfectly healthy loss curve. The fractional form is also the closer of the two to
the continuous schedule in integral.

**And the lr and the gradient norm are in every history row.** Neither was logged before
2026-09-19, which is why the missing scheduler survived two readers: each inferred the lr from a
different file instead of reading it out of the run. ``schedule.json`` records the schedule's own
parameters beside them, and ``scripts/abb3_port/heartbeat.py`` re-evaluates the cosine from it and
fails on a row whose lr does not match.

**One recipe fact the provenance records on purpose.** ``batch_collapsed_periodicity`` is left at
``True``, which REPRODUCES the defect in upstream's ``supervised_chi_loss`` -- their einsum
output spec drops the batch ellipsis, so pi-periodicity was mostly disabled in the training that
produced the 2.714 A bar we are reproducing. PLAN.md §7h decided to reproduce it rather than fix
it: the bar was produced by training with the bug, so matching their conditions is the only way a
miss is interpretable. The flag is in the tree and it is OFF, and the provenance says so, because
a later reader who finds it would otherwise reasonably assume the reproduction used it.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import abb3_dataset, provenance
from .abb3_checkpoint import latest_checkpoint, load_run_state, save_run_state
from .cotenancy import CotenantSampler
from .hostreduce import HostReduce, master_hash
from .prefetch import host_stream
from .mesh import Mesh
from .sharding import batches, steps_per_epoch

__all__ = ["RunConfig", "ReducedRAdam", "CosineRestartsByStep", "run",
           "TRIPWIRE_FRACTIONS"]

#: r2's rule, kept as the run's falsifiability check: a folding model reaches ~90 % of its final
#: accuracy in the first few percent of its budget, so a validation curve that is flat at 3 % is
#: a broken run and it is knowable in days rather than weeks. 10 % is the confirmation point.
TRIPWIRE_FRACTIONS = (0.03, 0.10)


@dataclass
class RunConfig:
    """Everything that decides what the run IS, in one serialisable object.

    ``global_batch`` is pinned and never derived from the chip count -- it is the axis the
    published recipe fixes, and deriving it per box is how the same script becomes a different
    recipe on a different machine.
    """

    out_dir: Path
    steps: int = 193_512
    global_batch: int = 64
    micro_batch: int = 8
    tokens: int = 256
    seed: int = 0
    #: Wall clock between checkpoints. 30 minutes caps lost work at 30 minutes per reset and is
    #: ~444 checkpoints over the base schedule at ~102 MB each, so ~44 GB on a 3.0 TB disk.
    checkpoint_minutes: float = 30.0
    keep_checkpoints: int = 3
    rank: int = 0
    world: int = 1
    chips: tuple = (0,)
    rendezvous: Path = Path("/dev/shm/abb3-dp")
    #: Upstream's own defect, reproduced. See the module docstring and PLAN.md §7h.
    batch_collapsed_periodicity: bool = True
    val_every: int = 0
    log_every: int = 1
    max_seconds: float = 0.0
    #: How many steps of micro-batch building to run ahead of the card, on one worker thread.
    #: 0 is the serial build. See ``tt_bio.train.prefetch``; 1 covers a whole step of device
    #: work, which is the whole opportunity.
    prefetch: int = 0

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.rendezvous = Path(self.rendezvous)
        if self.global_batch % (self.world * self.micro_batch):
            raise ValueError(
                f"global batch {self.global_batch} does not split into {self.world} ranks of "
                f"whole {self.micro_batch}-sample micro-batches. The global batch is the axis "
                f"the recipe pins, so the fix is a different world or micro-batch, never a "
                f"rounded global batch")

    @property
    def accumulate(self) -> int:
        """The GLOBAL number of micro-batches per optimizer step, on every rank.

        Global rather than per-rank, and that is what keeps the cross-rank reduce a pure sum:
        each rank divides its own micro-batch losses by the global count, so summing the ranks
        gives the mean over the pinned global batch exactly. Dividing by the local count and
        then summing would scale the gradient by the chip count, which is a different learning
        rate on every box.
        """
        return self.global_batch // self.micro_batch

    @property
    def local_micro_batches(self) -> int:
        return self.accumulate // self.world

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items()}
        d["out_dir"] = str(self.out_dir)
        d["rendezvous"] = str(self.rendezvous)
        d["accumulate"] = self.accumulate
        d["local_micro_batches"] = self.local_micro_batches
        d["chi_periodicity_correction"] = "OFF -- reproduces upstream's defect, PLAN.md §7h"
        return d


class ReducedRAdam:
    """``torch.optim.RAdam`` with the cross-rank sum in front of it, and the hash behind it.

    Duck-types the one method ``TrainStep.step()`` calls, so it can be installed on an existing
    step object rather than requiring a change to the step's own file. Everything that makes a
    data-parallel run trustworthy therefore happens whether or not the caller remembered it:
    the gradients are summed across ranks before the update, and the masters are compared after
    it.

    The gradient is flattened into one contiguous float32 vector for the exchange. One 28.4 MB
    transfer beats ~450 small ones on any transport, and the flatten fixes the parameter order,
    which the bit-identical sum depends on.
    """

    def __init__(self, inner, mirrors, comm: HostReduce):
        self.inner, self.mirrors, self.comm = inner, list(mirrors), comm
        self.steps = 0
        self.last = {}
        self._sizes = [int(np.prod(m.shape)) for m in self.mirrors]
        self._offsets = np.cumsum([0] + self._sizes)

    @property
    def state(self):
        return self.inner.state

    @property
    def param_groups(self):
        return self.inner.param_groups

    def step(self, *a, **kw):
        self.steps += 1
        flat = self._flatten()
        if self.comm.world > 1:
            flat = self.comm.allreduce(flat, step=self.steps)
            self._scatter(flat)
        out = self.inner.step(*a, **kw)
        digest = master_hash([m.detach().cpu().numpy() for m in self.mirrors])
        self.comm.check_equal(digest, step=self.steps)
        # The norm of the gradient the update was actually taken from, after the reduce, so it
        # means the same thing on one chip and on two. Logged every step because a loss curve
        # alone cannot distinguish a run that has converged from one whose gradient has
        # vanished or blown up, and this run's history carried neither this nor the lr.
        self.last = {"master_digest": digest.hex(), "grad_norm": float(np.linalg.norm(flat))}
        return out

    def _flatten(self) -> np.ndarray:
        """Every gradient in one contiguous float32 vector, in a fixed parameter order.

        One 28.4 MB transfer beats ~450 small ones on any transport, and the fixed order is what
        the bit-identical cross-rank sum depends on. Built on one chip as well, where it costs a
        copy the norm needs anyway and keeps the one-chip and two-chip paths identical.
        """
        flat = np.empty(int(self._offsets[-1]), dtype=np.float32)
        for i, m in enumerate(self.mirrors):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            g = m.grad
            flat[lo:hi] = (np.zeros(hi - lo, dtype=np.float32) if g is None
                           else g.detach().cpu().numpy().astype(np.float32).ravel())
        return flat

    def _scatter(self, flat: np.ndarray) -> None:
        for i, m in enumerate(self.mirrors):
            lo, hi = self._offsets[i], self._offsets[i + 1]
            m.grad = torch.from_numpy(flat[lo:hi].reshape(tuple(m.shape)).copy())

    def zero_grad(self, *a, **kw):
        return self.inner.zero_grad(*a, **kw)


class CosineRestartsByStep:
    """Upstream's per-epoch cosine restarts, evaluated at the fractional epoch of a global step.

    ``torch.optim.lr_scheduler.CosineAnnealingWarmRestarts`` is used rather than re-derived, so
    the formula is upstream's own and cannot drift from it; what changes is only where it is
    sampled. See the module docstring for why the sampling point is the global step.

    ``schedule.json`` in the run directory is written by :meth:`write` and read back by
    :meth:`load`, which is how a reader checks a logged lr against the same class the run drove
    rather than against a second copy of the cosine.
    """

    def __init__(self, optimizer, *, steps_per_epoch: int, T_0: int, T_mult: int = 1,
                 eta_min: float = 0.0):
        if int(steps_per_epoch) < 1:
            raise ValueError(f"steps_per_epoch must be at least 1, got {steps_per_epoch}")
        self.steps_per_epoch = int(steps_per_epoch)
        self.inner = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(T_0), T_mult=int(T_mult), eta_min=float(eta_min))

    @property
    def spec(self) -> dict:
        return {"scheduler": "CosineAnnealingWarmRestarts",
                "lr": float(self.inner.base_lrs[0]), "T_0": int(self.inner.T_0),
                "T_mult": int(self.inner.T_mult), "eta_min": float(self.inner.eta_min),
                "steps_per_epoch": self.steps_per_epoch,
                "interval": "step, at the fractional epoch (global_step - 1)/steps_per_epoch"}

    def set_step(self, global_step: int) -> float:
        """Place the lr for the 1-based ``global_step`` about to run, and return it.

        Step 1 sits at fractional epoch 0, which is upstream's first epoch and the base lr.
        """
        self.inner.step(epoch=(int(global_step) - 1) / self.steps_per_epoch)
        return float(self.inner.optimizer.param_groups[0]["lr"])

    def write(self, out_dir) -> Path:
        path = Path(out_dir) / "schedule.json"
        path.write_text(json.dumps(self.spec, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, spec: dict) -> "CosineRestartsByStep":
        """Rebuild the schedule on a dummy parameter, for a reader with no model.

        The optimizer only exists to hold a parameter group for the cosine to write into, so
        one scalar is enough and nothing is ever stepped through it.
        """
        p = torch.zeros(1, requires_grad=True)
        return cls(torch.optim.RAdam([p], lr=float(spec["lr"])),
                   steps_per_epoch=int(spec["steps_per_epoch"]), T_0=int(spec["T_0"]),
                   T_mult=int(spec.get("T_mult", 1)), eta_min=float(spec.get("eta_min", 0.0)))


# ------------------------------------------------------------------------------- the loop


def run(step, dataset, cfg: RunConfig, *, resume: bool = True, on_step=None) -> dict:
    """Train ``step`` on ``dataset`` for ``cfg.steps`` optimizer steps, resuming if it can.

    ``step`` is a ``tt_bio.train.abodybuilder3_step.TrainStep`` built at ``cfg.accumulate``.
    ``dataset`` needs ``__len__`` and ``batch(indices) -> dict`` returning the sample
    dict the step reads. Featurisation stays the model's, per r3: a shared data layer across
    model families would have to model every family's cropping, which is the part that is
    genuinely different.

    Returns the history and the provenance. Writes ``history.jsonl`` as it goes, appended and
    flushed every step, because a run killed by a reset must leave its curve behind -- a
    history held in memory and written at the end is lost exactly when it is needed.
    """
    comm = HostReduce(cfg.rendezvous, cfg.rank, cfg.world)
    step.optimizer = ReducedRAdam(step.optimizer, step.mirror, comm)
    # Built BEFORE the resume, so the cosine's base lr is the recipe's own rather than whatever
    # a restored optimizer happens to be carrying in its parameter groups.
    sched = CosineRestartsByStep(
        step.optimizer.inner,
        steps_per_epoch=steps_per_epoch(len(dataset), cfg.global_batch,
                                        abb3_dataset.DROP_LAST),
        T_0=step.recipe["T_0"], T_mult=step.recipe["T_mult"],
        eta_min=step.recipe["eta_min"])
    ckpt_dir = cfg.out_dir / "checkpoints"
    start, history = 0, []
    if resume:
        found = latest_checkpoint(ckpt_dir)
        if found is not None:
            state = load_run_state(found, step)
            start = state.step
            step.optimizer.steps = start
            history = list(state.get("history_tail") or [])
            print(f"[rank {cfg.rank}] resumed from {found.name} at step {start}", flush=True)
        else:
            print(f"[rank {cfg.rank}] no checkpoint in {ckpt_dir}, starting from step 0",
                  flush=True)
    plan = itertools.islice(
        batches(len(dataset), global_batch=cfg.global_batch, steps=cfg.steps,
                seed=cfg.seed, drop_last=abb3_dataset.DROP_LAST,
                data_parallel=Mesh({"dp": list(cfg.chips)}).axis("dp")),
        start, None)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.rank == 0:
        sched.write(cfg.out_dir)
    log = open(cfg.out_dir / f"history-rank{cfg.rank}.jsonl", "a", buffering=1)
    tripwires = {int(f * cfg.steps) for f in TRIPWIRE_FRACTIONS}
    last_ckpt = time.monotonic()
    t_run = time.monotonic()
    # Sampled for the life of the run, not checked either side of it. A cotenant that arrives
    # and leaves inside the measurement is invisible to a snapshot, and on this box that is the
    # common case: a sibling worker takes and releases a card in 4-second leases.
    tenants = CotenantSampler()
    tenants.start()
    written: list = sorted(ckpt_dir.glob("step-*.safetensors")) if ckpt_dir.is_dir() else []
    cadence = cfg.checkpoint_minutes * 60.0
    # `TrainStep.loss_terms` accumulates per-term wall clock across the whole run, so the
    # per-step figure is a delta. Logged because the serial part of the step is almost entirely
    # the loss stage, and which TERM it is decides whether a lever touches the serial part or
    # the parallel part -- the distinction the whole optimisation queue is ranked on.
    prev_terms: dict = {}
    # `wall` times `step.step()` and nothing else, and the micro-batch build sits in FRONT of
    # it, so the cadence a watcher sees has always been larger than any row in the history --
    # 2.232 s a step larger on the first 5-day leg, 17.7 % of its wall clock, with no field
    # carrying it. These two close the loop: `data` is the build and the upload, `outer` is
    # every remaining microsecond, so t[n+1] - t[n] == data + wall + outer for n >= 1 exactly
    # and an unaccounted gap can no longer hide between two rows.
    t_outer = None
    stream = host_stream(plan, lambda b: [b.per_chip[cfg.rank][i:i + cfg.micro_batch]
                                          for i in range(0, len(b.per_chip[cfg.rank]),
                                                         cfg.micro_batch)],
                         dataset, cfg.prefetch)
    with provenance.during(seed=cfg.seed, config=cfg.as_dict()) as prov:
        while True:
            # `t_data` is stamped BEFORE the stream is pulled, so with a prefetch depth the wait
            # for the worker thread is charged to `data` and a win cannot be booked by moving
            # the cost to an unmeasured line.
            t_data = time.perf_counter()
            outer = 0.0 if t_outer is None else t_data - t_outer
            try:
                batch, hosts = next(stream)
            except StopIteration:
                break
            gs = batch.step + 1
            lr = sched.set_step(gs)
            micros = [dataset.upload(h) for h in hosts]
            t0 = time.perf_counter()
            parts, timing = step.step(micros)
            wall = time.perf_counter() - t0
            terms = {k: round(v - prev_terms.get(k, 0.0), 4)
                     for k, v in step.loss_terms.items()}
            prev_terms = dict(step.loss_terms)
            # The epoch stamp is what makes the history readable as a HEARTBEAT and not only as
            # a curve: `wall` is device time and says nothing about the gap a watchdog reset
            # leaves behind, and that gap is the thing a watcher has to be able to see.
            row = {"step": gs, "t": round(time.time(), 3), "wall": round(wall, 4),
                   "data": round(t0 - t_data, 4), "outer": round(outer, 4), **parts,
                   "lr": lr, "grad_norm": step.optimizer.last.get("grad_norm"),
                   "digest": step.optimizer.last.get("master_digest"),
                   "stages": timing.as_dict(), "loss_terms": terms}
            history.append(row)
            if gs % cfg.log_every == 0:
                log.write(json.dumps(row) + "\n")
            if on_step is not None:
                on_step(gs, row, step)
            # Cadence in wall clock, and only rank 0 writes: one file that every rank loads is
            # what makes the post-resume hash equality a fact rather than a hope. Four ranks
            # each writing their own would also be four chances to restore a different one.
            # A run that stops on the clock stops ON a checkpoint. Ask 9115 capped this leg at
            # 5 days of wall clock and called it a decision point rather than a truncation, so
            # the state at the cap is the deliverable -- losing up to a 30-minute cadence of it
            # because the cap landed between two saves throws away the one thing an extension
            # would resume from.
            out_of_time = bool(cfg.max_seconds) and time.monotonic() - t_run > cfg.max_seconds
            due = ((time.monotonic() - last_ckpt >= cadence) or gs == cfg.steps
                   or gs in tripwires or out_of_time)
            if due and cfg.rank == 0:
                path = save_run_state(
                    ckpt_dir / f"step-{gs:09d}.safetensors", step, global_step=gs,
                    metrics=parts, provenance=prov.as_dict(), history_tail=history[-50:])
                written.append(path)
                _prune(written, cfg.keep_checkpoints, tripwires)
                print(f"[rank 0] checkpoint {path.name} at step {gs}", flush=True)
            if due:
                last_ckpt = time.monotonic()
                # Every rank waits for the write, so a reset between rank 0's save and another
                # rank's next step cannot leave the ranks on different sides of a checkpoint.
                comm.allgather(f"ckpt-{gs:09d}", b"ok")
            if out_of_time:
                print(f"[rank {cfg.rank}] max_seconds reached at step {gs}", flush=True)
                break
            t_outer = time.perf_counter()
    log.close()
    co = tenants.stop()
    prov.config["cotenancy"] = co
    print(f"[rank {cfg.rank}] {tenants.summary()}", flush=True)
    return {"history": history, "provenance": prov, "comm": comm, "cotenancy": co,
            "steps_done": history[-1]["step"] if history else start}


def _prune(written: list, keep: int, protect: set) -> None:
    """Keep the newest ``keep``, and never delete a tripwire checkpoint.

    The newest is what a resume needs -- not the best-scoring one, which is a different
    question and the wrong file to restart from. The tripwire checkpoints are kept because the
    3 % and 10 % validation points are the run's falsifiability check and re-reaching step
    5,805 to re-evaluate it costs days.
    """
    while len(written) > keep:
        for i, p in enumerate(written):
            n = int(p.stem.split("-")[-1])
            if n not in protect:
                Path(p).unlink(missing_ok=True)
                written.pop(i)
                break
        else:
            return
