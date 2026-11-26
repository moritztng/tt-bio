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
        # The seed baseline writes no scores unless a confidence model is named outright:
        # `--no_model` is upstream's "no network at all" path and its output contract
        # (`confidence: null`) is dfd-cli's. Passing `--confidence_checkpoint` with it is
        # the one way to rank poses end to end on host torch before the rollout lands, and
        # it ranks seed conformers, which is a weak thing to rank -- the number it writes
        # is the confidence model's real output, not a docking result.
        confidences = None
        if options.confidence_checkpoint is not None:
            poses, confidences = rank_poses(poses, mol, receptor, options, cache=cache,
                                            name=job.name)
        return output.write_poses(write_dir, mol, poses, confidences=confidences)
    # When the rollout lands here, its poses go through the same one line:
    #     poses, confidences = rank_poses(poses, mol, receptor, options, cache=cache,
    #                                     name=job.name)
    # `rank_poses` honours `--no_confidence` itself, so there is no second flag check.
    raise SamplerUnavailable(
        "the DiffDock-L reverse diffusion is not on this branch yet; "
        "`--no_model` runs the seed-conformer baseline end to end today")


def rank_poses(poses, mol, receptor_path, options: DockOptions, *, cache=None,
               name: str = ""):
    """Order one complex's poses by the confidence network. Returns ``(poses, scores)``.

    This is the second network of DiffDock-L, and the reason `tt-bio dock` can report a
    top-1 pose at all: the sampler returns `--samples_per_complex` poses in no meaningful
    order, and ranking them is what turns that into an answer. `--no_confidence` keeps the
    sampling order and writes no scores, which is upstream's behaviour and the only path
    that needs neither the 19 MB checkpoint nor an ESM-2 pass.

    Frames: the poses arrive in the receptor's own coordinates. Both sides are centred on
    the receptor's CA centroid before the forward pass, as upstream's dataset does, and the
    returned poses are put back in the input frame -- the writer's job is to emit
    coordinates a viewer can overlay on the input PDB. The network reads positions only
    through edge vectors, so the centring moves no logit; it is there to keep this
    numerically on upstream's path rather than a translation away from it.
    """
    if options.no_confidence:
        return poses, None

    import numpy as np
    import torch

    from tt_bio.diffdock import confidence_model, esm, input as dfd_input
    from tt_bio.diffdock import ligand as dfd_ligand
    from tt_bio.diffdock import receptor as dfd_receptor
    from tt_bio.diffdock import structure

    ckpt, params = checkpoints.resolve("confidence", options.confidence_checkpoint, cache)
    model = confidence_model.load_checkpoint(ckpt, params.parent)

    struct = structure.parse_receptor(receptor_path)
    chains, _ = struct.chain_sequences()
    # The confidence checkpoint takes the same ESM-2 650M embedding the score network does
    # (its `lm_embedding_layer` is [ns, ns + 1280]); on this path it runs on host torch,
    # because the whole host rollout must work without a card.
    embeddings = esm.receptor_embeddings(chains, on_device=False)
    rec = dfd_receptor.build_receptor_graph(
        struct, lm_embeddings=embeddings, **dfd_input.CONFIDENCE_GRAPH)
    lig = dfd_ligand.build_ligand_graph(mol, remove_hs=True)

    centre = rec["pos"].mean(dim=0, keepdim=True)
    rec["pos"] = rec["pos"] - centre
    rec["atom"]["pos"] = rec["atom"]["pos"] - centre
    pos = torch.as_tensor(np.asarray(poses), dtype=torch.float32) - centre

    scores = confidence_model.score_poses(model, rec, lig, pos)
    ranked, scores = confidence_model.rank_poses(pos + centre, scores)
    log.info("%s: ranked %d poses, best confidence %.3f",
             name or "complex", len(scores), float(scores[0]))
    return ranked.numpy(), scores.numpy()


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
