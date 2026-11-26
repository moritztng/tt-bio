"""Running a list of docking jobs: the one place a pose is produced and written.

Two sampler paths meet here. ``--no_model`` is upstream's seed-conformer baseline, needs no
checkpoint and no device, and runs today. The real reverse-diffusion rollout lives in
``tt_bio.diffdock.sampling`` and is not on this branch yet; until it lands, asking for it
says so in one line instead of failing somewhere inside a loader.

Failure isolation follows the input form (FEATURE-PLAN conflict 1): one complex on the
command line fails loudly with its reason, a CSV or a directory keeps going and writes
``failures.csv``, because one bad ligand must not end a 10,000-ligand screen.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tt_bio.diffdock import checkpoints, output, seed_pose
from tt_bio.diffdock.job import DockJob, DockOptions

log = logging.getLogger("tt_bio.dock")


class SamplerUnavailable(RuntimeError):
    pass


def dock(jobs: list[DockJob], options: DockOptions, out_dir, *, cache=None,
         isolate: bool = False, download: bool = False) -> dict:
    """Dock every job, writing upstream's layout under ``out_dir``. Returns a summary."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not options.no_model:
        # Resolve both checkpoints before the first complex: a missing weight file is not
        # a per-complex failure, and finding out after 9,999 rows is worse than before one.
        checkpoints.resolve("score", options.checkpoint, cache, download=download)
        if not options.no_confidence:
            checkpoints.resolve("confidence", options.confidence_checkpoint, cache,
                                download=download)

    done, failures = [], []
    for job in jobs:
        try:
            written = _dock_one(job, options, out_dir / job.name, cache=cache)
            done.append({"name": job.name, "files": [p.name for p in written]})
        except Exception as e:                       # noqa: BLE001 - reported, not swallowed
            if not isolate:
                raise
            log.warning("%s failed: %s", job.name, e)
            failures.append((job.name, f"{type(e).__name__}: {e}"))
            if len(failures) >= options.limit_failures:
                log.warning("stopping after %d failures (--limit_failures)", len(failures))
                break
    output.write_failures(out_dir, failures)
    return {"docked": done, "failed": failures, "out_dir": str(out_dir)}


def _dock_one(job: DockJob, options: DockOptions, write_dir: Path, *, cache=None):
    mol = seed_pose.load_ligand(job.ligand)
    receptor = _receptor_structure(job, write_dir)
    if options.no_model:
        centre = seed_pose.receptor_centre(receptor)
        poses = seed_pose.seed_poses(mol, centre, options, n=options.diffusion_samples)
        return output.write_poses(write_dir, mol, poses, confidences=None)
    raise SamplerUnavailable(
        "the DiffDock-L reverse diffusion is not on this branch yet; "
        "`--no_model` runs the seed-conformer baseline end to end today")


def _receptor_structure(job: DockJob, write_dir: Path) -> Path:
    """A structure on disk for this job's receptor, folding a bare sequence first.

    A sequence-only receptor is folded with tt-bio's own ESMFold2 rather than upstream's
    `esmfold_v1` import: same contract (a PDB in the complex's output directory), a better
    model, and it runs on the card (D-dfd-4's reuse rule applied to the folder).
    """
    if job.receptor_path is not None:
        path = Path(job.receptor_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"receptor {path} does not exist")
        return path
    raise SamplerUnavailable(
        f"{job.name}: folding a sequence-only receptor with esmfold2 needs a card and is "
        f"not wired on this branch; pass a receptor structure file")
