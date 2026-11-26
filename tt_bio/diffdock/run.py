"""Running a list of docking jobs: the one place a pose is produced and written.

Two sampler paths meet here. ``--no_model`` is upstream's seed-conformer baseline, needs no
checkpoint and no device. The real path builds the complex graph, embeds the receptor with
ESM-2, randomises ``--samples`` starting poses, runs the reverse diffusion in
``tt_bio.diffdock.sampling``, ranks by confidence and writes upstream's layout. It reaches
the score network through ``tt_bio.diffdock.network.load``, the rollout's only seam; until
`dfd-parity`'s ``score_model`` lands that raises, and asking for the real sampler says so in
one line instead of failing somewhere inside a loader.

``--max_parallel_samples`` caps how many poses are scored in one device call, and halves
itself if the card refuses one. It is a memory knob and nothing else: it is applied to the
score function, where the poses are disjoint graphs and splitting them is bit-identical, and
not to the sampler, whose per-step noise is one draw over the whole batch -- chunking there
would be a different (equally valid) trajectory. See state/dfd/SAMPLING.md.

Failure isolation follows the input form (FEATURE-PLAN conflict 1): one complex on the
command line fails loudly with its reason, a CSV or a directory keeps going and writes
``failures.csv``, because one bad ligand must not end a 10,000-ligand screen.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tt_bio.diffdock import checkpoints, network, output, seed_pose
from tt_bio.diffdock.job import DockJob, DockOptions

log = logging.getLogger("tt_bio.dock")


class SamplerUnavailable(RuntimeError):
    pass


def dock(jobs: list[DockJob], options: DockOptions, out_dir, *, cache=None,
         isolate: bool = False, download: bool = False) -> dict:
    """Dock every job, writing upstream's layout under ``out_dir``. Returns a summary."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model = None
    if not options.no_model:
        # Resolve both checkpoints before the first complex: a missing weight file is not
        # a per-complex failure, and finding out after 9,999 rows is worse than before one.
        checkpoints.resolve("score", options.checkpoint, cache, download=download)
        if not options.no_confidence:
            checkpoints.resolve("confidence", options.confidence_checkpoint, cache,
                                download=download)
        # And the network itself once for the whole screen, not once per complex: 30 M
        # parameters, and a device path would otherwise re-open a card 10,000 times.
        try:
            model = network.load(options, cache=cache, download=download)
        except network.ScoreNetworkUnavailable as e:
            raise SamplerUnavailable(str(e)) from e

    done, failures = [], []
    for job in jobs:
        try:
            written = _dock_one(job, options, out_dir / job.name, cache=cache, model=model)
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


def _dock_one(job: DockJob, options: DockOptions, write_dir: Path, *, cache=None,
              model=None):
    receptor = _receptor_structure(job, write_dir)
    if options.no_model:
        mol = seed_pose.load_ligand(job.ligand)
        centre = seed_pose.receptor_centre(receptor)
        poses = seed_pose.seed_poses(mol, centre, options, n=options.diffusion_samples)
        return output.write_poses(write_dir, mol, poses, confidences=None)
    return _sample_one(job, options, write_dir, receptor, model, cache=cache)


def _sample_one(job: DockJob, options: DockOptions, write_dir: Path, receptor: Path,
                model, *, cache=None):
    """The real path: build the graph, roll the diffusion out, rank, write.

    The score network arrives through :mod:`tt_bio.diffdock.network`, the rollout's only
    seam, so the host reference and the card path run this same function.
    """
    import torch

    from tt_bio.diffdock import chunking, esm, input as dfd_input, neighbour_cap, sampling

    spec = dfd_input.ComplexSpec(name=job.name, protein_path=str(receptor),
                                 ligand_description=job.ligand)
    chains = dfd_input.chain_sequences_for_embedding(receptor)
    # The receptor embedding runs wherever the score network runs: a host-reference network
    # keeps the whole dock on the CPU, the card path embeds on the card (D-dfd-input-5's
    # reuse rule - this is `tt-bio predict --model esmc`'s encoder, not a second ESM-2).
    embeddings = esm.receptor_embeddings(chains, on_device=getattr(model, "on_device", True),
                                         cache_dir=cache)
    graph = dfd_input.build_complex_graph(spec, seed=options.seed, receptor_path=receptor,
                                          lm_embeddings=embeddings,
                                          all_atoms=model.config.all_atoms)
    # Say so when this ligand is dense enough that upstream's torch_cluster graph would be
    # truncated and ours is not (D-dfd-trace-1). One `cdist` on a ligand, once per complex.
    neighbour_cap.check_ligand_graph(graph.ligand["pos"], model.config.lig_max_radius,
                                     label=job.name, on_input=True)

    cfg = _sampling_config(options)
    bounds = sampling.SigmaBounds.from_config(model.config)
    noise = sampling.Noise(seed=options.seed)
    centre = None
    if options.pocket is not None:
        centre = torch.tensor(options.pocket, dtype=torch.float32) - graph.original_center[0]

    start = sampling.randomize_position(
        graph.ligand["pos"], receptor_pos=graph.receptor["pos"],
        n_samples=options.diffusion_samples, cfg=cfg, bounds=bounds, noise=noise,
        edge_index=graph.ligand["edge_index"].T[graph.ligand["edge_mask"]],
        mask_rotate=graph.ligand["mask_rotate"], no_torsion=model.config.no_torsion,
        centre=centre)
    # The score function, not the sampler, is what the pose batch is chunked through: the
    # poses are disjoint graphs there, so the chunking is exact, while the sampler's per-step
    # noise is one draw over the whole batch and chunking it would be a different trajectory.
    n_torsions = 0 if model.config.no_torsion else len(graph.ligand["mask_rotate"])
    score_fn = chunking.chunked_score_fn(model.score_fn(graph), options.max_parallel_samples,
                                         n_torsions=n_torsions,
                                         reset=getattr(model, "drop_trace", None))
    rollout = sampling.sample(
        score_fn, start, cfg=cfg, bounds=bounds, noise=noise,
        edge_index=graph.ligand["edge_index"].T[graph.ligand["edge_mask"]],
        mask_rotate=graph.ligand["mask_rotate"], no_torsion=model.config.no_torsion,
        crop_beyond_a=model.config.crop_beyond, trajectory=options.save_trajectory)

    from rdkit import Chem

    # The graph, and so every pose, is heavy-atom only; upstream writes the H-stripped mol.
    mol = Chem.RemoveHs(Chem.Mol(graph.mol), sanitize=False)
    poses = (rollout.poses + graph.original_center).numpy()
    confidences = None if options.no_confidence else model.confidence(graph, rollout.poses)
    order = list(range(len(poses)))
    if confidences is not None:
        confidences = torch.nan_to_num(torch.as_tensor(confidences).flatten().float(),
                                       nan=-1000.0)
        order = torch.argsort(confidences, descending=True).tolist()
        poses, confidences = poses[order], confidences[order].tolist()

    trajectories = None
    if options.save_trajectory and rollout.trajectory:
        frames = [(f + graph.original_center).numpy() for f in rollout.trajectory]
        trajectories = [output.trajectory_pdb(mol, [f[b] for f in frames]) for b in order]
    return output.write_poses(write_dir, mol, poses, confidences=confidences,
                              trajectories=trajectories)


def _sampling_config(options: DockOptions):
    """The sampler's config from the CLI's options, recipe defaults underneath."""
    from tt_bio.diffdock import sampling

    return sampling.SamplingConfig.from_recipe(
        options.recipe,
        inference_steps=options.sampling_steps,
        actual_steps=options.actual_steps,
        samples_per_complex=options.diffusion_samples,
        no_final_step_noise=options.no_final_step_noise,
        initial_noise_std_proportion=options.initial_noise_std,
        choose_residue=options.init_at_residue,
        ode=options.sampler == "ode",
        no_random=options.deterministic,
        pocket_knowledge=options.pocket is not None,
        pocket_cutoff=options.pocket_radius,
    )


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
