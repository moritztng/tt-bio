"""One process per chip, and the host-side all-reduce between them.

This is what makes a wide ``Mesh`` run instead of raise. The Tier-1 recipe holds one
replica's gradient because it is one process, so data parallelism is a process count before
it is anything else: :func:`drive` re-runs the calling program once per chip on the axis, and
in each of those runs :func:`reducer` turns the same axis into one that reduces across
processes rather than across a mesh device.

**Re-running the program, not shipping objects to it.** ``TT_VISIBLE_DEVICES`` is read at
``import ttnn``, so a fork cannot give a child a different chip than its parent -- the visible
set is already fixed by then -- and an unpinned open brings up every visible chip rather than
the one being computed on, which would have each rank holding the whole box. A rank therefore
has to be a fresh interpreter, and a fresh interpreter cannot be handed a live ``forward`` or
a dataset that holds a device. So the driver re-executes ``sys.argv`` with ``TT_BIO_DP_RANK``
set, exactly as ``torchrun`` does, and each rank builds its own forward and its own dataset
from its own arguments. The contract that buys is one line long: **the program must be
re-runnable up to the ``finetune`` call, and must not hold a card when it gets there.** Both
are checked before anything is spawned rather than trusted.

**The all-reduce goes through /dev/shm, deliberately not through the device.** The masters and
both Adam moments already live on the host because ``ttnn.moreh_adamw`` cannot hold an fp32
master, so the gradient has to cross PCIe to reach them whatever happens; an on-device
collective would move it to the device and back for nothing. Measured on two p150a by
``perf/train_d_dp`` at 1350 MHz: the exchange is 1.01 ms of a 0.264 s step at LoRA rank 8
(0.33 MB, 0.38 %) and 7.48 ms of a 0.329 s step at rank 128 (5.24 MB, 2.27 %).

**Sum, never mean.** The divisor is the global batch the caller pinned. Dividing by the chip
count here is the substitution that makes one recipe mean something different on a 2-chip box
than on a 4-chip one, which is what ``accelerate`` does at ``data_loader.py:347-348`` and what
:mod:`tt_bio.train.sharding` exists to refuse. The measured prototype this module promotes
averaged; that is the one thing about it that did not carry over. It was deleted in commit
81eaa6a6a and rebuilt against this launcher as ``perf/train_d_dp``.

**The ranks' masters are compared, not assumed equal.** Ranks start from one seed so their
weights are identical at step 0, and a reduce before every step keeps them identical after
it. :func:`drive` hashes every rank's masters at the end and requires exactly one distinct
hash. Without that a run whose replicas silently diverged would still produce a
respectable-looking throughput number, and every rank's loss curve would still fall, because
each rank's loss is real -- it is just a different model's loss.

**Each rank's chip is read back off the rank.** ``TT_VISIBLE_DEVICES=N`` does not open
``/dev/tenstorrent/N``: on qb1 the UMD ids run in PCI-BDF order and the kernel node numbers do
not, so ``TT_VISIBLE_DEVICES=1`` lands on node 2. Two rows on this fleet have lost a pass to
it, one of them sampling an idle neighbour's clock for a whole A/B. With one process per chip
the same mistake puts two ranks on one chip and leaves one idle, and it still shows a speedup.
So :func:`drive` resolves every rank's node from that rank's OWN ``/proc/<pid>/fd`` and
requires the set to be distinct and of size ``world``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

from .mesh import Axis

__all__ = ["drive", "driving", "inside", "rank", "world", "reducer", "replicas",
           "host_threads",
           "out_dir", "node", "master_sha", "report", "tick", "RANK_ENV", "WORLD_ENV",
           "RUN_ENV"]


RANK_ENV = "TT_BIO_DP_RANK"
WORLD_ENV = "TT_BIO_DP_WORLD"
RUN_ENV = "TT_BIO_DP_RUN"

#: Where the ranks exchange gradients. /dev/shm is RAM, so the exchange never touches a disk.
SHM_ROOT = "/dev/shm/tt-bio-dp"

#: How long a rank waits at a step's barrier before giving up. A peer that died holding the
#: barrier would otherwise hang the whole run silently, which is the failure a timeout turns
#: into a message naming the step and the rank.
BARRIER_TIMEOUT_S = 900.0

#: How many steps of exchanged gradients stay on /dev/shm. A rank deletes its own file from
#: two steps back, which is safe because the barrier one step back already proved every peer
#: read it, and it bounds the run at 2 x world x gradient bytes instead of steps x that.
_KEEP_STEPS = 2


def host_threads(world: int) -> int:
    """The host's physical cores divided across ``world`` ranks, at least one.

    Physical rather than logical: torch's intra-op pool is sized for compute, and two threads
    on one core contend for the same vector units rather than adding any. ``os.cpu_count()``
    reports the logical count, so it is halved when the machine reports SMT.
    """
    logical = os.cpu_count() or 1
    try:
        smt = int(Path("/sys/devices/system/cpu/smt/active").read_text().strip())
    except (OSError, ValueError):
        smt = 0
    cores = max(1, logical // 2) if smt else logical
    return max(1, cores // max(1, world))


def _sweep_shm() -> None:
    """Remove run directories whose driver is gone. /dev/shm is RAM on a shared box.

    Named ``<pid>-<unix time>``, so a directory whose pid is no longer alive belonged to a run
    that crashed or was killed before it could clean up, and nobody else is coming for it. A
    successful run removes its own; this is for the ones that did not get the chance.
    """
    root = Path(SHM_ROOT)
    if not root.is_dir():
        return
    for d in root.iterdir():
        try:
            pid = int(d.name.split("-", 1)[0])
        except (ValueError, IndexError):
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(d, ignore_errors=True)
        except PermissionError:
            pass


# --------------------------------------------------------------- where am I

def rank() -> int:
    """This process's rank, 0 when nothing launched it as one."""
    return int(os.environ.get(RANK_ENV, "0"))


def world() -> int:
    """How many processes are on the axis, 1 when this is not a data-parallel run."""
    return int(os.environ.get(WORLD_ENV, "1"))


def driving() -> bool:
    """True when this process would have to spawn the ranks rather than be one."""
    return RANK_ENV not in os.environ


def inside() -> bool:
    """True when this process IS a rank the launcher started."""
    return not driving()


def node() -> Optional[int]:
    """The ``/dev/tenstorrent`` node this process has open, read off its own fds.

    ``None`` before a device is opened. Not derived from ``TT_VISIBLE_DEVICES``, which names a
    UMD id and not a node -- see the module docstring.
    """
    from . import provenance
    ns = provenance.open_nodes()
    return ns[0] if ns else None


def out_dir(path) -> Path:
    """Where THIS rank writes its checkpoints.

    Rank 0 owns the directory it was given and every other rank gets a subdirectory of it.
    The masters are bit-identical across ranks, so one copy is the run's checkpoint; the
    point of keeping the others is that two processes writing one safetensors path is a
    truncated file, not a duplicate one.
    """
    p = Path(path)
    return p if rank() == 0 else p / f"rank{rank()}"


def master_sha(opt) -> str:
    """One sha256 over every fp32 master, in name order. The sync check's unit."""
    import numpy as np
    if not opt.master:
        return hashlib.sha256(b"").hexdigest()
    flat = np.concatenate([opt.master[n].ravel() for n in sorted(opt.master)])
    return hashlib.sha256(np.ascontiguousarray(flat, dtype=np.float32).tobytes()).hexdigest()


_MARK = [None]


def tick() -> Optional[float]:
    """Seconds since the previous call. ``None`` the first time, which is not a step.

    A training loop wants the per-step time in its own history rather than in a harness that
    wraps it, so this is one word inside the row the recipe already builds. The first call has
    no preceding mark and returns ``None``; that is also the step whose time carries the
    kernel compile on a cold cache, so every median taken off this is a steady-state number
    by construction instead of by a convention somebody has to remember.
    """
    now = time.perf_counter()
    prev, _MARK[0] = _MARK[0], now
    return None if prev is None else now - prev


def replicas(params: Dict[str, object]) -> dict:
    """This rank's own gradients, in the shape ``opt.step(replicas=...)`` wants.

    One entry per parameter that has a gradient, and the list under it holds exactly one
    tensor because this process holds exactly one replica. Empty on a single-process run,
    which is what makes ``opt.step(replicas=replicas(params))`` the same call at width 1 --
    the optimizer refuses a non-empty ``replicas`` without a wide axis, and an empty one is
    the honest single-chip case rather than a special one.
    """
    if world() == 1:
        return {}
    return {n: [t.grad] for n, t in params.items() if t.grad is not None}


# --------------------------------------------------------------- the axis, across processes

@dataclass(frozen=True)
class ProcessAxis(Axis):
    """The data-parallel axis as ONE rank sees it: its own gradient in, the global sum out.

    A :class:`~tt_bio.train.mesh.Axis` reduces across chips a single process holds, so its
    ``reduce`` takes one tensor per chip. A rank holds one chip, so this takes one tensor and
    the other ``world - 1`` arrive over /dev/shm. The width is the same number either way,
    which is what keeps the optimizer's ``UnreducedGradients`` guard meaningful here: a rank
    that forgets to pass its gradient still gets refused.

    ``reduce_all`` is the one that matters and it is overridden for the barrier, not for the
    bytes. A per-parameter loop would put one round trip and one barrier on every adapter
    tensor -- dozens per step -- so the whole parameter set goes in one message.
    """

    dp_rank: int = 0
    run: str = ""
    #: per-step timings, mutated in place because the dataclass is frozen. Three numbers per
    #: step and not one: ``publish``+``add`` is what the collective costs, while ``wait`` is
    #: mostly the slower peer's step tail. Summing them into one "comm" figure reports load
    #: imbalance as if it were transport, and on a fast step that reads as over 100 %.
    state: dict = field(default_factory=lambda: {
        "step": 0, "publish_s": [], "wait_s": [], "add_s": [], "bytes": 0})

    def reduce(self, tensors: Sequence) -> list:
        """One parameter on its own. Correct, and not what a step should call.

        Present so a ``ProcessAxis`` is a usable ``Axis`` rather than one with a hole in it.
        :meth:`reduce_all` is what :meth:`tt_bio.train.optim.AdamW.step` calls, and it pays
        one barrier for the whole parameter set instead of one per tensor.
        """
        return [self.reduce_all({"_": list(tensors)})["_"]]

    def reduce_all(self, per_param: Dict[str, Sequence]) -> dict:
        """Sum every parameter's gradient across the ranks. One message, one barrier.

        Flattens to one fp32 host vector in name order, exchanges it, and returns the sums
        ON THE HOST. fp32 on the wire because the masters are fp32 and the sum is what feeds
        them; summing in bf16 would round the gradient twice for no saving that matters
        against a step measured in tenths of a second.

        The sums stay on the host because that is where the optimizer reads them. Handing back
        a device tensor cost one PCIe write and one PCIe read per parameter and bought nothing:
        measured at 5.24 MB of adapter gradient, 54 ms of a 344 ms step, which was 16 % of the
        two-chip efficiency. Every rank adds the same files in rank order, so all of them
        compute the same fp32 sum bit for bit and their masters stay identical -- which is the
        invariant :func:`drive` checks, and it only holds because the order is fixed.
        """
        import numpy as np
        from .tensors import to_host

        if self.width == 1:
            return {n: list(v)[0] for n, v in per_param.items()}
        names = sorted(per_param)
        parts, meta = [], []
        for n in names:
            got = list(per_param[n])
            if len(got) != 1:
                raise ValueError(
                    f"parameter {n!r} came with {len(got)} gradients and this rank holds "
                    f"one replica, so it has exactly one to give. The other "
                    f"{self.width - 1} arrive over /dev/shm -- pass "
                    f"replicas=tt_bio.train.launcher.replicas(params)")
            g = got[0]
            a = np.ascontiguousarray(to_host(g), dtype=np.float32)
            meta.append((n, a.shape, g))
            parts.append(a.ravel())
        vec = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        total = self._exchange(vec)
        out, i = {}, 0
        for n, shape, _g in meta:
            k = int(np.prod(shape)) if shape else 1
            out[n] = total[i:i + k].reshape(shape)
            i += k
        return out

    # -- the collective itself

    def _exchange(self, vec):
        """Sum ``vec`` across every rank on the axis. Publish, barrier, add.

        The publish is a write to a temporary name and an ``os.replace``, so a peer polling
        the directory never sees a half-written file -- a reader that did would sum garbage
        into the gradient and the run would keep going.
        """
        import numpy as np
        step = self.state["step"] = self.state["step"] + 1
        d = Path(self.run)
        t0 = time.perf_counter()
        mine = d / f"s{step}_r{self.dp_rank}.npy"
        tmp = d / f"s{step}_r{self.dp_rank}.part"
        with open(tmp, "wb") as fh:
            np.save(fh, vec)
        os.replace(tmp, mine)
        t_pub = time.perf_counter()
        peers = [d / f"s{step}_r{r}.npy" for r in range(self.width)]
        while not all(p.exists() for p in peers):
            if time.perf_counter() - t_pub > BARRIER_TIMEOUT_S:
                here = sorted(p.name for p in peers if p.exists())
                raise TimeoutError(
                    f"rank {self.dp_rank} waited {BARRIER_TIMEOUT_S:.0f} s at step {step} for "
                    f"{self.width} gradients and has {here}. A rank that died holding the "
                    f"barrier stops the run here instead of hanging it")
            time.sleep(0.002)
        t_wait = time.perf_counter()
        acc = np.zeros_like(vec)
        for p in peers:
            acc += np.load(p)
        # Sum, not mean: see the module docstring.
        stale = d / f"s{step - _KEEP_STEPS}_r{self.dp_rank}.npy"
        stale.unlink(missing_ok=True)
        st = self.state
        st["publish_s"].append(t_pub - t0)
        st["wait_s"].append(t_wait - t_pub)
        st["add_s"].append(time.perf_counter() - t_wait)
        st["bytes"] = int(vec.nbytes)
        return acc


def reducer(axis: Axis) -> Axis:
    """``axis``, reduced across the launcher's processes. The plain axis at width 1.

    Returned unchanged on one chip so a single-chip run is the same program it was before
    this module existed, rather than the same program with a collective that no-ops.
    """
    if axis.width == 1:
        return axis
    run = os.environ.get(RUN_ENV)
    if not run:
        raise RuntimeError(
            f"axis {axis} is {axis.width} chips wide but this process is not one of the "
            f"launcher's ranks, so there is nobody to reduce with. Call the recipe through "
            f"tt_bio.train.finetune, which hands a wide axis to drive()")
    return ProcessAxis(name=axis.name, device_ids=axis.device_ids, mesh=axis.mesh,
                       dp_rank=rank(), run=run)


# --------------------------------------------------------------- the driver

def _relaunch_argv() -> list:
    """The command that re-runs this program, or raise saying why it cannot be re-run.

    ``-m pkg.mod`` is re-executed as ``-m pkg.mod`` rather than as the file it resolved to:
    running the file directly would change the package context and break its own imports.
    """
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    if spec is not None and spec.name:
        return [sys.executable, "-u", "-m", spec.name, *sys.argv[1:]]
    script = sys.argv[0] if sys.argv else ""
    if script and Path(script).is_file():
        return [sys.executable, "-u", str(Path(script).resolve()), *sys.argv[1:]]
    raise RuntimeError(
        f"a data-parallel run is one process per chip and this process cannot be re-run: "
        f"sys.argv[0] is {script!r}, which is not a file, and __main__ has no module spec. "
        f"That is what `python -c` and an interactive session look like. Put the call in a "
        f"script, or run it as `python -m your.module` -- each rank re-runs your program with "
        f"TT_BIO_DP_RANK set and builds its own forward and dataset, so there has to be a "
        f"program to re-run")


def _refuse_if_holding_a_card() -> None:
    """A driver must hold no chip when it spawns the ranks, and holding one is checkable.

    Not defensive: the ranks are fresh interpreters precisely because ``TT_VISIBLE_DEVICES``
    is read at ``import ttnn``, and a driver that already opened a card has both taken a chip
    a rank wants and proved that its dataset resolves a device before the launcher is
    reached. Checked against this process's own fds, which is the same evidence the NODES
    check uses.
    """
    from . import provenance
    held = provenance.open_nodes()
    if held:
        raise RuntimeError(
            f"this process already has /dev/tenstorrent/{held} open, so it cannot launch one "
            f"rank per chip: the rank would be a second opener of a card this process holds. "
            f"A data-parallel program must not touch a device before finetune() -- resolve "
            f"the dataset's `device` lazily, on first use, so it opens inside the rank that "
            f"computes on it")


def drive(axis: Axis, *, out_dir, steps: int, timeout_s: Optional[float] = None) -> dict:
    """Run the calling program once per chip on ``axis`` and return the aggregate.

    Returns the dict the Tier-1 recipe returns, so a wide axis and a narrow one come back the
    same shape. ``params`` and ``optimizer`` are empty here and that is the honest answer: the
    driver never opened a card, the ranks that did have exited, and their adapters are on disk
    under ``out_dir``. Everything a caller can still act on -- the loss history, the
    checkpoints, the provenance, and the DP record -- is present.
    """
    _refuse_if_holding_a_card()
    cmd = _relaunch_argv()
    ranks = list(axis.device_ids)
    n = len(ranks)
    _sweep_shm()
    run = Path(SHM_ROOT) / f"{os.getpid()}-{int(time.time())}"
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir(parents=True)
    t0 = time.perf_counter()
    procs = []
    try:
        for r, card in enumerate(ranks):
            env = dict(os.environ)
            env[RANK_ENV] = str(r)
            env[WORLD_ENV] = str(n)
            env[RUN_ENV] = str(run)
            # The UMD id this rank computes on. Every rank is pinned to exactly one, because
            # an unpinned open brings up every visible chip and each rank would hold the box.
            env["TT_VISIBLE_DEVICES"] = str(card)
            # The lease covers the whole axis: tt-bio refuses a card outside the grant at the
            # device open, and every rank's card has to be inside it.
            env["TT_BIO_LEASE_CARDS"] = ",".join(str(c) for c in ranks)
            # The host's cores, divided. Torch sizes its intra-op pool from the machine and has
            # no idea how many ranks share it, so every rank asks for the whole box and a
            # `world`-rank run oversubscribes by `world`. Measured on qb1 (16 physical cores,
            # ABodyBuilder3, four micro-batches a rank): four ranks at torch's default 16
            # threads spend 913.98 s of a 931.05 s step in `losses` and `host_backward`, at
            # load average 62 with no swap and 298 GB free; the same per-rank work with the
            # cores divided spends 2.50 s. The device stages do not move between the two.
            #
            # A caller who set OMP_NUM_THREADS meant it and keeps it -- that is the override,
            # and it is the same variable `tt_bio/runtime.py` already binds to both the
            # intra-op and the inter-op pool, so there is no second knob.
            env.setdefault("OMP_NUM_THREADS", str(host_threads(n)))
            log = open(run / f"rank{r}.log", "w")
            procs.append((r, card, subprocess.Popen(cmd, env=env, stdout=log,
                                                    stderr=subprocess.STDOUT), log))
        codes = []
        for r, card, p, log in procs:
            codes.append((r, p.wait(timeout=timeout_s)))
            log.close()
    finally:
        for _r, _c, p, log in procs:
            if p.poll() is None:
                p.kill()
            if not log.closed:
                log.close()
    bad = [(r, c) for r, c in codes if c]
    if bad:
        tails = "\n".join(
            f"--- rank {r} exited {c}, last 30 lines of {run}/rank{r}.log ---\n"
            + "".join((run / f"rank{r}.log").read_text(errors="replace").splitlines(True)[-30:])
            for r, c in bad)
        raise RuntimeError(f"{len(bad)} of {n} ranks failed: {bad}. Their logs and any "
                           f"results are left in {run}\n{tails}")
    res = []
    for r in range(n):
        f = run / f"result_r{r}.json"
        if not f.is_file():
            raise RuntimeError(
                f"rank {r} exited 0 but wrote no result to {f}. A rank reports through that "
                f"file, so a missing one means it never reached the end of the recipe. Its log "
                f"is beside it, in {run}")
        res.append(json.loads(f.read_text()))
    out = _aggregate(res, wall_s=time.perf_counter() - t0, steps=steps,
                     out_dir=out_dir, run=str(run))
    # Only once every check has passed. A failed run keeps its directory, because the rank logs
    # and the per-rank results in it are the whole diagnosis and the message above names the
    # path; the next driver sweeps it when its pid is gone.
    shutil.rmtree(run, ignore_errors=True)
    return out


def report(opt, params, history, checkpointer, prov, *, plan) -> None:
    """A rank's side of the contract: write what the driver aggregates, then return.

    A no-op when this process is not a rank, so the recipe calls it unconditionally. The
    masters' hash is taken here rather than in the driver because the driver has no device and
    no optimizer: the thing being compared only exists inside the rank that computed it.
    """
    if driving():
        return
    comm = dict(getattr(opt.data_parallel, "state", {}) or {})
    # Drop the first exchange with the first step: it carries the kernel compile on one side
    # and the peer's on the other, so its barrier is a startup number and not a step's.
    drop = lambda xs: list(xs)[1:] if len(xs or []) > 1 else list(xs or [])
    payload = {
        "rank": rank(), "world": world(),
        "visible": os.environ.get("TT_VISIBLE_DEVICES"),
        # The chip this rank ACTUALLY held, off its own fds. See the module docstring.
        "node": node(),
        "master_sha": master_sha(opt),
        "params": sorted(params),
        "history": history,
        "step_s": [h["s"] for h in history if h.get("s") is not None],
        "publish_s": drop(comm.get("publish_s")),
        "wait_s": drop(comm.get("wait_s")),
        "add_s": drop(comm.get("add_s")),
        "comm_bytes": comm.get("bytes", 0),
        "displacement": prov.config.get("displacement"),
        "provenance": prov.as_dict(),
        "plan": str(plan),
        "checkpoints": [{"path": str(w["path"]), "step": w["step"], "score": w["score"]}
                        for w in checkpointer.written],
    }
    d = Path(os.environ[RUN_ENV])
    tmp = d / f"result_r{rank()}.part"
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, d / f"result_r{rank()}.json")


def _aggregate(res: list, *, wall_s: float, steps: int, out_dir, run: str) -> dict:
    """Fold the ranks' reports into one run, and refuse the two failures that look fine.

    The sync check and the node check are both here, and both raise. A diverged run and two
    ranks on one chip produce a plausible throughput number and a falling loss curve, so
    neither is detectable from the numbers a user reads.
    """
    import numpy as np
    from .checkpoint import Checkpointer

    n = len(res)
    shas = sorted({r["master_sha"] for r in res})
    nodes = [r["node"] for r in res]
    mid = lambda xs: float(np.median(xs)) if xs else None
    per_rank = []
    for r in res:
        s = r["step_s"]
        # The collective's own cost is publish + add. The barrier wait beside it is mostly the
        # other rank's step tail, so it is reported separately rather than folded in.
        xfer = [a + b for a, b in zip(r["publish_s"], r["add_s"])]
        per_rank.append({
            "rank": r["rank"], "visible": r["visible"], "node": r["node"],
            "steps": len(s), "median_step_s": mid(s),
            "median_transfer_s": mid(xfer), "median_barrier_wait_s": mid(r["wait_s"]),
            "comm_bytes": r["comm_bytes"],
            "aiclk": r["provenance"].get("aiclk"),
            "loss_first": (r["history"][0]["loss"] if r["history"] else None),
            "loss_last": (r["history"][-1]["loss"] if r["history"] else None),
        })
    med = [p["median_step_s"] for p in per_rank if p["median_step_s"]]
    dp = {
        "world": n, "wall_s": wall_s, "nodes": nodes, "visible": [p["visible"] for p in per_rank],
        "distinct_master_sha": len(shas), "master_sha": shas[0] if len(shas) == 1 else shas,
        "sync_ok": len(shas) == 1,
        "median_step_s": mid(med),
        "comm_bytes": max(p["comm_bytes"] for p in per_rank) if per_rank else 0,
        "median_transfer_s": mid([p["median_transfer_s"] for p in per_rank
                                  if p["median_transfer_s"] is not None]),
        "median_barrier_wait_s": mid([p["median_barrier_wait_s"] for p in per_rank
                                      if p["median_barrier_wait_s"] is not None]),
        "per_rank": per_rank, "run": run, "params": res[0]["params"],
    }
    if len(shas) != 1:
        raise AssertionError(
            f"SYNC FAILED: {len(shas)} distinct master-weight hashes across {n} ranks, and "
            f"data parallelism requires exactly 1. The ranks trained {len(shas)} different "
            f"models. Every one of their loss curves falls, and the throughput number above "
            f"is real -- it is just not a number for the model anyone wanted.\n"
            f"  hashes: {[h[:16] for h in shas]}\n"
            f"  nodes:  {nodes}")
    if len(set(nodes)) != n or None in nodes:
        raise AssertionError(
            f"the {n} ranks held /dev/tenstorrent nodes {nodes}, which is not {n} distinct "
            f"chips. TT_VISIBLE_DEVICES names a UMD id and NOT a node number, so ranks can "
            f"share a chip while one sits idle, and the run still shows a speedup. Each node "
            f"here was read from that rank's own /proc/<pid>/fd, so this is what the ranks "
            f"held, not what they were told to hold.\n"
            f"  TT_VISIBLE_DEVICES per rank: {[p['visible'] for p in per_rank]}")
    # The loss a global batch saw is the mean over the equal-sized shards that made it up.
    hist = []
    for i in range(min(len(r["history"]) for r in res)):
        rows = [r["history"][i] for r in res]
        hist.append({"step": rows[0]["step"],
                     "loss": float(np.mean([x["loss"] for x in rows])),
                     "breakdown": rows[0]["breakdown"],
                     "grad_norm": rows[0]["grad_norm"], "lr": rows[0]["lr"],
                     "per_rank_loss": [x["loss"] for x in rows]})
    ck = Checkpointer(out_dir)
    ck.written = [{"path": Path(w["path"]), "step": w["step"], "score": w["score"]}
                  for w in res[0]["checkpoints"]]
    return {"history": hist, "params": {}, "optimizer": None, "checkpointer": ck,
            "provenance": _Record(res[0]["provenance"], dp), "plan": res[0]["plan"],
            "displacement": res[0]["displacement"], "dp": dp}


class _Record:
    """Rank 0's provenance as the driver can report it: a dict plus the DP record.

    A real :class:`~tt_bio.train.provenance.Provenance` samples a clock off the process that
    holds the card, and the driver holds none. Rather than build a hollow one, this carries
    the rank's own record verbatim and says which rank it came from.
    """

    def __init__(self, rank0: dict, dp: dict):
        self._d = dict(rank0)
        self._d["dp"] = dp
        self.config = self._d.setdefault("config", {})

    def as_dict(self) -> dict:
        return dict(self._d)

    def summary(self) -> str:
        clk = self._d.get("aiclk") or {}
        return (f"rank 0: seed {self._d.get('seed')}, {self._d.get('git_sha')}, "
                f"node {self._d.get('device_nodes')}, "
                f"{clk.get('median')} MHz median during over {clk.get('samples')} samples")
