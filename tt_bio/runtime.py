"""Tiny runtime primitives used by the scheduler and CLI.

PredictionJob is one input file the user asked us to predict; WorkerSlot is
one accelerator we can run it on. Discovery and detection helpers live here
so they can be reused from both the CLI and the worker subprocess.
"""

from __future__ import annotations

import glob
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path


INPUT_SUFFIXES = (".fa", ".fas", ".fasta", ".yml", ".yaml")

HOST_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def host_thread_cap(n_workers: int, host_threads: int | None = None) -> int:
    """Per-worker host thread budget for a process driving ``n_workers`` cards.

    Each worker's torch/OMP/BLAS pools otherwise default to ALL cores, so N
    co-resident workers spawn N*cores threads that thrash the CPU and collapse the
    host-side work (weight load, featurization, output) -- the multi-card slowdown.
    ``host_threads`` is this PROCESS's share of the host CPU, defaulting to every
    core. All-cores is right for one process driving the whole box, but wrong when an
    external launcher runs one single-card job per chip: each process then sees
    n_workers == 1 and claims all cores. Such a launcher passes
    ``cores // concurrent_jobs``.
    """
    budget = host_threads if host_threads and host_threads > 0 else (os.cpu_count() or 1)
    return max(1, budget // max(1, n_workers))


def host_thread_cap_env(n_workers: int, host_threads: int | None = None) -> dict[str, str]:
    """Thread-cap environment for a spawned per-card child.

    Explicit beats inherited: a passed ``host_threads`` overrides a pre-set env var
    (the launcher knows how many siblings it started), while the default only fills in
    what the operator left unset.
    """
    cap = str(host_thread_cap(n_workers, host_threads))
    return {var: cap for var in HOST_THREAD_VARS
            if host_threads or var not in os.environ}


def bind_host_threads() -> None:
    """Bind torch's thread pools to the OMP_NUM_THREADS cap set by the launcher.

    A fresh torch import honours the cap for the intra-op pool -- but not for the
    inter-op pool, which always sizes itself to cores/2 regardless. Bind both
    explicitly so a capped worker really holds only its share of the CPU. Nothing to
    do when the launcher set no cap.
    """
    cap = os.environ.get("OMP_NUM_THREADS")
    if not cap:
        return
    import torch

    torch.set_num_threads(int(cap))
    try:
        torch.set_num_interop_threads(int(cap))
    except RuntimeError:
        pass  # already started (only settable before the first parallel op)


@dataclass(frozen=True)
class PredictionJob:
    """One input target to predict."""

    id: str
    path: Path


@dataclass(frozen=True)
class WorkerSlot:
    """One execution slot that can run prediction jobs."""

    worker_id: str
    host: str
    accelerator: str
    device_id: int | str
    visible_devices: str | None = None
    logical_device_id: int = 0
    mesh_graph_descriptor: str | None = None

    @property
    def label(self) -> str:
        if self.accelerator == "tenstorrent":
            return f"{self.host}:tt{self.device_id}"
        return f"{self.host}:{self.accelerator}"


def discover_jobs(data: Path, structure_dir: Path, output_format: str, override: bool) -> list[PredictionJob]:
    """Discover runnable input files, applying resume semantics."""
    files = sorted(
        p for p in (data.glob("*") if data.is_dir() else [data])
        if p.suffix.lower() in INPUT_SUFFIXES
    )
    # A job is identified by its file stem (used for output filenames AND result/failure keys),
    # so two inputs sharing a stem — e.g. target.fasta and target.yaml — would silently overwrite
    # each other's output and merge into one result. Refuse the ambiguous case with a clear message
    # rather than lose a prediction. Checked on the full input set (before the resume filter).
    by_stem: dict[str, list[Path]] = {}
    for p in files:
        by_stem.setdefault(p.stem, []).append(p)
    dups = {stem: ps for stem, ps in by_stem.items() if len(ps) > 1}
    if dups:
        detail = "; ".join(f"'{stem}' <- {', '.join(pp.name for pp in ps)}"
                           for stem, ps in sorted(dups.items()))
        raise ValueError(
            "Input files share a name stem and would overwrite each other's output. "
            f"Rename so each input has a unique stem ({detail})."
        )
    if not override:
        files = [p for p in files if not (structure_dir / f"{p.stem}.{output_format}").exists()]
    return [PredictionJob(id=p.stem, path=p) for p in files]


def tt_dev_node_bdfs() -> dict[int, str]:
    """``/dev/tenstorrent/N`` -> that card's PCI BDF, from the tenstorrent sysfs class.

    The node number is the kernel driver's PROBE order. It is not the UMD device index
    and on a multi-root-complex box it is not even PCI order.
    """
    out: dict[int, str] = {}
    for entry in glob.glob("/sys/class/tenstorrent/tenstorrent!*/device"):
        node = int(os.path.basename(os.path.dirname(entry)).rsplit("!", 1)[-1])
        out[node] = os.path.basename(os.path.realpath(entry)).lower()
    return out


def _bdf_sort_key(bdf: str) -> tuple[int, ...]:
    """Numeric sort key for a PCI BDF, so ``0000:c1:00.0`` orders after ``0000:81:00.0``."""
    return tuple(int(part, 16) for part in re.split(r"[:.]", bdf) if part)


def tt_bdf_to_index() -> dict[str, int]:
    """PCI BDF -> UMD device index for every Tenstorrent card.

    UMD numbers chips by PCI enumeration order. That is NOT the ``/dev/tenstorrent/N``
    node number, which the kernel driver assigns in probe order, and the two differ on
    any box whose root complexes are probed out of PCI order.

    Measured on the Galaxy ``UF-EV-A13-GWH02`` (32 chips on four root complexes,
    2026-09-02): its nodes 0-7 are buses c1-c8, nodes 8-15 are 81-88, nodes 16-23 are
    01-08 and nodes 24-31 are 41-48. Cross-checking all 32 device-lease files (keyed on
    the UMD index the process pinned with ``TT_VISIBLE_DEVICES``) against ``lsof`` on
    every node gave UMD index k -> node f(k) with f a permutation that has NO fixed
    point: UMD 0 is node 16, UMD 7 is node 23, UMD 16 is node 8, UMD 24 is node 0. The
    permutation is exactly BDF-ascending rank, from two independent process families
    (the JapanFold pool and a sibling worker). Reading the node number as the UMD index,
    as this function used to, therefore named a different chip for all 32 cards.
    """
    return {bdf: i for i, bdf in enumerate(sorted(tt_dev_node_bdfs().values(),
                                                  key=_bdf_sort_key))}


def umd_index_to_dev_node() -> dict[int, int]:
    """UMD device index -> ``/dev/tenstorrent/N`` node number.

    The bridge every occupancy check needs: a lease and a ``TT_VISIBLE_DEVICES`` pin
    name a UMD index, while ``lsof`` and the fd-level collision live on the node. Check
    one and act on the other and you test the freeness of a chip you never open.
    """
    by_bdf = tt_bdf_to_index()
    return {by_bdf[bdf]: node for node, bdf in tt_dev_node_bdfs().items() if bdf in by_bdf}


def dev_node_to_umd_index() -> dict[int, int]:
    """``/dev/tenstorrent/N`` node number -> UMD device index."""
    return {node: idx for idx, node in umd_index_to_dev_node().items()}


def visible_device_indices(visible: str) -> list[int]:
    """Parse a ``TT_VISIBLE_DEVICES`` value into UMD device indices.

    ttnn's device open accepts either form per token: a UMD index (``0``) or a PCI
    BDF (``0000:01:00.0``). BDFs resolve through the tenstorrent sysfs class, whose
    per-card ``device`` symlink names the card's PCI address; a token that matches
    no card raises a ValueError naming the index form, so a typo fails here with a
    clear message instead of at device open (issue #11).
    """
    indices: list[int] = []
    bdfs: list[str] = []
    for tok in visible.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            indices.append(int(tok))
        except ValueError:
            bdfs.append(tok.lower())
    if bdfs:
        by_bdf = tt_bdf_to_index()
        for bdf in bdfs:
            match = next((idx for full, idx in by_bdf.items()
                          if full == bdf or full.endswith(":" + bdf)), None)
            if match is None:
                known = ", ".join(f"{idx} ({full})" for full, idx in sorted(by_bdf.items()))
                raise ValueError(
                    f"TT_VISIBLE_DEVICES entry '{bdf}' matches no Tenstorrent card "
                    f"(present: {known or 'none detected'}). Use the UMD index form, "
                    "e.g. TT_VISIBLE_DEVICES=0."
                )
            indices.append(match)
    return indices


def detect_tenstorrent_devices(device_ids: str | None, num_devices: int, max_workers: int) -> list[int]:
    """Return TT device IDs selected for this run without importing ttnn.

    An explicit ``device_ids`` is validated against the cards actually present under
    ``/dev/tenstorrent``: a request for a device that isn't there (a typo like ``--device_ids 7``
    on a two-card box) raises a clear error naming the available ids, instead of passing straight
    through and failing much later with an opaque low-level device-open error.
    """
    all_devices = sorted(int(p.rsplit("/", 1)[-1]) for p in glob.glob("/dev/tenstorrent/[0-9]*"))
    # Honor ambient TT_VISIBLE_DEVICES (the same env ttnn/tt-smi read at
    # device-open time): a job pinned to card N via the environment must fan out
    # only onto card N. Without this filter, a predict launched with
    # TT_VISIBLE_DEVICES=1 enumerated every physical card and spawned one worker
    # per card, so the card-0 worker wedged card 0 for every concurrent job on
    # the box. Explicit --device_ids still wins (validated below against the
    # ambient-visible set).
    visible = os.environ.get("TT_VISIBLE_DEVICES")
    if visible is not None:
        allowed = set(visible_device_indices(visible))
        all_devices = [d for d in all_devices if d in allowed]
    if device_ids:
        requested = [int(d.strip()) for d in device_ids.split(",") if d.strip()]
        missing = [d for d in requested if d not in all_devices]
        if missing:
            avail = ", ".join(map(str, all_devices)) if all_devices else "none detected"
            raise ValueError(
                f"Requested Tenstorrent device id(s) {missing} not available "
                f"(present: {avail}). Fix --device_ids, or leave it unset to use all cards."
            )
        devices = requested
    elif num_devices > 0:
        devices = all_devices[:num_devices]
    else:
        devices = all_devices
    return devices[:max_workers]


def build_local_workers(accelerator: str, jobs: list, devices: list[int]) -> list[WorkerSlot]:
    """Build worker slots for the local host (one per device, capped to jobs)."""
    host = socket.gethostname()
    if accelerator == "tenstorrent":
        return [
            WorkerSlot(
                worker_id=f"{host}:tt:{device}",
                host=host,
                accelerator=accelerator,
                device_id=device,
                visible_devices=str(device),
            )
            for device in devices[:len(jobs)]
        ]
    return [WorkerSlot(worker_id=f"{host}:{accelerator}:0", host=host,
                       accelerator=accelerator, device_id=0)]
