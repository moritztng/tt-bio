"""`tt-bio dock`'s real path, end to end on the CPU, with a stub score network.

The rollout is graded against upstream elsewhere (``test_diffdock_sampling_controls.py``,
bit-identical with the noise stream aligned). What this file covers is everything *around*
it: a real receptor PDB and a real ligand SDF go in, the complex graph is built, the ESM-2
columns are requested for the receptor's own chains, ``--samples`` poses are randomised,
rolled out, ranked by confidence and written in upstream's layout -- names, ranks,
``confidence.json`` and the ``--save_trajectory`` PDB.

The score network is stubbed because `dfd-parity` owns the real one, and it is stubbed *at
the seam* -- ``tt_bio.diffdock.network.load`` -- so landing the real network is a one-line
swap with this test unchanged. The stub's config carries only what the dock path reads off
a checkpoint, which is the seam's whole surface: the six sigma bounds, ``no_torsion``,
``crop_beyond`` and ``all_atoms``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tt_bio.diffdock import checkpoints, confidence_model, esm, network, output, run
from tt_bio.diffdock.job import DockJob, DockOptions

FIX = Path(__file__).parent / "fixtures" / "diffdock"
RECEPTOR = FIX / "6Z5Z_BDF_protein.pdb"
LIGAND = FIX / "6Z5Z_BDF_ligand.sdf"

STUB_CONFIG = SimpleNamespace(
    all_atoms=False, no_torsion=False, crop_beyond=20.0,
    tr_sigma_min=0.1, tr_sigma_max=19.0, rot_sigma_min=0.03, rot_sigma_max=1.55,
    tor_sigma_min=0.0314, tor_sigma_max=3.14,
    # The released recipe's ligand radius. `run.dock` checks the ligand against upstream's
    # 32-neighbour cap through it (tt_bio/diffdock/neighbour_cap.py).
    lig_max_radius=5.0)


class StubNetwork:
    """A score network with no weights: smooth, deterministic, the right shapes.

    It pulls poses towards the origin of the centred frame the way a trained network pulls
    them into the pocket. The shape is a Gaussian score, ``-x / sigma**2``, damped: the
    rollout multiplies it by ``g**2 * dt ~ 10 * sigma**2 * dt``, so an undamped or
    1/sigma-shaped stub oscillates and runs the poses out to 10^5 A -- the trap an O(1)
    stub sprang on an earlier parity harness, met again here.
    """

    config = STUB_CONFIG
    on_device = False

    def __init__(self):
        self.calls = []

    def score_fn(self, graph):
        n_tor = len(graph.ligand["mask_rotate"])

        def score(step):
            self.calls.append((step.step, step.crop_cutoff))
            centres = step.lig_pos.mean(dim=1)
            tr = -0.05 * centres / (step.tr_sigma ** 2 + 1.0)
            rot = torch.zeros_like(tr)
            tor = torch.zeros(step.lig_pos.shape[0] * n_tor)
            return tr, rot, tor

        return score

    def confidence(self, graph, poses):
        # Rank by how close the pose centre sits to the receptor centre: a stand-in that is
        # a real function of the pose, so the ranking is actually exercised.
        return -poses.mean(dim=1).norm(dim=1)


@pytest.fixture
def stubbed(monkeypatch):
    net = StubNetwork()
    monkeypatch.setattr(network, "load", lambda options, **kw: net)
    # The weight files are `dfd-parity`'s; the stub network needs none, so the pre-flight
    # resolve (dfd-cli's fail-fast, tested in test_diffdock_cli.py) stands aside here.
    monkeypatch.setattr(checkpoints, "resolve", lambda *a, **kw: None)
    seen = {}

    def embeddings(chains, **kw):
        seen["chains"] = list(chains)
        seen["on_device"] = kw.get("on_device")
        return [torch.zeros(len(c), 1280) for c in chains]

    monkeypatch.setattr(esm, "receptor_embeddings", embeddings)
    return net, seen


@pytest.fixture
def released_checkpoints():
    """The released v1.1 score and confidence checkpoints, or a skip.

    They are 124 MiB and not in the repo, so a host without them skips rather than
    downloading inside a test run.
    """
    try:
        root = checkpoints.models_root()
    except checkpoints.CheckpointMissing as e:
        pytest.skip(str(e))
    score, confidence = root / checkpoints.SCORE_DIR, root / checkpoints.CONFIDENCE_DIR
    if not (score / checkpoints.SCORE_CKPT).exists():
        pytest.skip(f"no score checkpoint under {root}")
    return score, confidence


def options(**kw):
    kw.setdefault("seed", 7)
    return DockOptions(diffusion_samples=3, sampling_steps=6, actual_steps=5, **kw)


def test_dock_writes_upstreams_layout(tmp_path, stubbed):
    """A real complex, the real pipeline, upstream's files."""
    net, seen = stubbed
    job = DockJob(name="6Z5Z_BDF", receptor_path=RECEPTOR, ligand=str(LIGAND))
    summary = run.dock([job], options(), tmp_path)

    out = tmp_path / "6Z5Z_BDF"
    names = sorted(p.name for p in out.iterdir())
    assert "rank1.sdf" in names, names
    ranked = [n for n in names if n.startswith("rank") and "_confidence" in n]
    assert len(ranked) == 3, names
    assert summary["failed"] == []

    rows = json.loads((out / "confidence.json").read_text())
    assert [r["rank"] for r in rows] == [1, 2, 3]
    confs = [r["confidence"] for r in rows]
    assert confs == sorted(confs, reverse=True), "poses must be written best first"

    # The ESM-2 columns were asked for the receptor's own chains, on the host.
    assert seen["chains"] and all(seen["chains"])
    assert seen["on_device"] is False

    # Every step called the network once, with the annealing crop radius.
    assert [c[0] for c in net.calls] == list(range(5))
    cutoffs = [c[1] for c in net.calls]
    assert cutoffs == sorted(cutoffs, reverse=True) and cutoffs[-1] < cutoffs[0]


def test_poses_are_written_in_the_crystal_frame(tmp_path, stubbed):
    """The receptor centroid the graph subtracted is added back before writing."""
    from rdkit import Chem

    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    run.dock([job], options(), tmp_path)
    pose = next(iter(Chem.SDMolSupplier(str(tmp_path / "c" / "rank1.sdf"), removeHs=False)))
    assert pose is not None
    pos = pose.GetConformer().GetPositions()

    rec = Chem.MolFromPDBFile(str(RECEPTOR), sanitize=False, removeHs=False)
    rec_pos = rec.GetConformer().GetPositions()
    # Not a docking-quality claim: only that the pose lands inside the receptor's own box,
    # which it cannot do if the centre was dropped or added twice.
    assert ((pos.min(axis=0) > rec_pos.min(axis=0) - 80).all()
            and (pos.max(axis=0) < rec_pos.max(axis=0) + 80).all())


def test_same_seed_same_output(tmp_path, stubbed):
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    run.dock([job], options(), tmp_path / "a")
    run.dock([job], options(), tmp_path / "b")
    a = (tmp_path / "a" / "c" / "rank1.sdf").read_text()
    b = (tmp_path / "b" / "c" / "rank1.sdf").read_text()
    assert a == b, "--seed must pin the whole dock, not just the rollout"

    run.dock([job], options(seed=8), tmp_path / "d")
    assert (tmp_path / "d" / "c" / "rank1.sdf").read_text() != a


def test_save_trajectory_writes_one_model_per_step(tmp_path, stubbed):
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    run.dock([job], options(save_trajectory=True), tmp_path)
    text = (tmp_path / "c" / "rank1_reverseprocess.pdb").read_text()
    assert text.count("MODEL ") == 6, "the randomised start plus one frame per step"
    assert text.count("ENDMDL") == 6 and text.rstrip().endswith("END")


def test_no_confidence_keeps_sample_order(tmp_path, stubbed):
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    run.dock([job], options(no_confidence=True), tmp_path)
    rows = json.loads((tmp_path / "c" / "confidence.json").read_text())
    assert [r["confidence"] for r in rows] == [None, None, None]
    assert [r["sdf"] for r in rows] == ["rank1.sdf", "rank2.sdf", "rank3.sdf"]


def test_missing_weights_name_the_command_that_fixes_them(tmp_path, monkeypatch):
    """With no checkpoint on the host, the dock stops before the first complex and says how."""
    monkeypatch.setattr(checkpoints.weights, "cached_path", lambda *a, **kw: None)
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    with pytest.raises(checkpoints.CheckpointMissing, match="tt-bio weights --download"):
        run.dock([job], options(), tmp_path)


def test_load_returns_a_score_network(released_checkpoints):
    """The seam is wired: `load` hands back the real network, not a NotImplemented."""
    score_dir, confidence_dir = released_checkpoints
    net = network.load(options(checkpoint=score_dir, confidence_checkpoint=confidence_dir))
    assert isinstance(net, network.ScoreNetwork)
    assert net.config.name and net.config.all_atoms is False
    assert net.confidence_model is not None
    assert sum(p.numel() for p in net.model.parameters()) == 30_222_796


def test_dock_with_the_released_weights_writes_finite_poses(tmp_path, released_checkpoints):
    """One real complex through the real network, on the CPU. Three steps, two poses:
    this proves the path runs and writes, not that it docks well -- the pose quality bar is
    `dfd-eval`'s and the score parity is the assembled-network matrix in
    `state/dfd/parity/score.json`."""
    score_dir, confidence_dir = released_checkpoints
    job = DockJob(name="6Z5Z_BDF", receptor_path=RECEPTOR, ligand=str(LIGAND))
    opts = DockOptions(seed=7, diffusion_samples=2, sampling_steps=3, actual_steps=3,
                       checkpoint=score_dir, confidence_checkpoint=confidence_dir)
    run.dock([job], opts, tmp_path)

    rows = json.loads((tmp_path / "6Z5Z_BDF" / "confidence.json").read_text())
    assert len(rows) == 2
    assert [r["sdf"].split("_")[0] for r in rows] == ["rank1", "rank2"]
    assert all(r["confidence"] is not None for r in rows)
    assert rows[0]["confidence"] >= rows[1]["confidence"]

    from rdkit import Chem

    for row in rows:
        mol = next(iter(Chem.SDMolSupplier(str(tmp_path / "6Z5Z_BDF" / row["sdf"]),
                                           removeHs=False, sanitize=False)))
        pos = mol.GetConformer().GetPositions()
        assert pos.shape[0] == mol.GetNumAtoms()
        assert bool((abs(pos) < 1e4).all()) and not bool((pos != pos).any())


def test_trajectory_pdb_is_parsable(tmp_path):
    from rdkit import Chem

    mol = next(iter(Chem.SDMolSupplier(str(LIGAND), removeHs=True)))
    pos = mol.GetConformer().GetPositions()
    text = output.trajectory_pdb(mol, [pos, pos + 1.0])
    first = text.split("ENDMDL")[0]
    again = Chem.MolFromPDBBlock(first, sanitize=False, removeHs=False)
    assert again is not None and again.GetNumAtoms() == mol.GetNumAtoms()


# --------------------------------------------------------------------- non-finite poses
# A pose the rollout cannot advance is a result, not a lost complex (D-dfd-sampling-9). The
# sampler-level behaviour is `tests/test_diffdock_nonfinite_poses.py`; what is here is what
# the dock path does with it: rank it last, keep the rest, and lose only the complex whose
# every pose went bad -- as a `failures.csv` row on a screen, loudly on one complex.

class BrokenNetwork(StubNetwork):
    """The stub, with the translation head overflowing on *rows* at step 1.

    1e38, not an infinity: an infinite score is *repaired* (`sampling.guard_scores`) and the
    pose carries on, which is upstream's behaviour and the first half of this fix. What makes
    a pose unusable is an update no repair can scale -- here the step overflows fp32.
    """

    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.complexes = 0

    def score_fn(self, graph):
        inner = super().score_fn(graph)
        # One network serves the whole screen, so break only the first complex it is asked
        # for: that is what makes "the other complexes are unaffected" testable at all.
        self.complexes += 1
        broken = self.complexes == 1

        def score(step):
            tr, rot, tor = inner(step)
            if broken and step.step == 1:
                tr = tr.clone()
                tr[self.rows] = 1e38
            return tr, rot, tor

        return score


@pytest.fixture
def broken(monkeypatch, stubbed):
    def install(rows):
        net = BrokenNetwork(rows)
        monkeypatch.setattr(network, "load", lambda options, **kw: net)
        return net

    return install


def test_one_failed_sample_is_not_written_and_the_rest_are(tmp_path, broken):
    broken([1])
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    summary = run.dock([job], options(), tmp_path)

    assert summary["failed"] == []
    assert summary["docked"][0]["failed_samples"] == 1
    rows = json.loads((tmp_path / "c" / "confidence.json").read_text())
    assert len(rows) == 2 and [r["rank"] for r in rows] == [1, 2]
    assert all(r["confidence"] > confidence_model.UNSCORABLE for r in rows)
    assert len(list((tmp_path / "c").glob("rank*_confidence*.sdf"))) == 2


def test_a_complex_whose_poses_all_fail_is_one_failures_row(tmp_path, broken):
    broken([0, 1, 2])
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    good = DockJob(name="d", receptor_path=RECEPTOR, ligand=str(LIGAND))
    summary = run.dock([job, good], options(), tmp_path, isolate=True)

    assert [name for name, _ in summary["failed"]] == ["c"]
    assert "PoseRolloutFailed" in summary["failed"][0][1]
    assert "c: all 3 samples failed" in summary["failed"][0][1]
    assert (tmp_path / "failures.csv").exists()
    # The other complex is untouched: one bad complex does not end a screen.
    assert [d["name"] for d in summary["docked"]] == ["d"]
    assert (tmp_path / "d" / "rank1.sdf").exists()


def test_a_single_complex_losing_every_pose_raises(tmp_path, broken):
    broken([0, 1, 2])
    job = DockJob(name="c", receptor_path=RECEPTOR, ligand=str(LIGAND))
    with pytest.raises(run.PoseRolloutFailed, match="c: all 3 samples failed"):
        run.dock([job], options(), tmp_path)

def test_dock_writes_a_heteroaromatic_ligand(tmp_path, stubbed):
    """A thiophene ligand reaches disk.

    The writer kekulises. Preparing the pose molecule with ``RemoveHs(..., sanitize=False)``
    leaves the aromatic flags without kekulisation, so ``SDWriter.write`` raised
    ``KekulizeException`` after the whole rollout had run and lost the complex, leaving a
    0-byte ``rank1.sdf``. It hit 273 of the 428 PoseBusters ligands (`dfd-eval`, the
    PoseBusters id below is the one it was found on).
    """
    job = DockJob(name="5S8I_2LY", receptor_path=RECEPTOR, ligand="CNC(=O)c1scc2c1OCCO2")
    summary = run.dock([job], options(), tmp_path, isolate=True)

    assert summary["failed"] == []
    rank1 = tmp_path / "5S8I_2LY" / "rank1.sdf"
    assert rank1.stat().st_size > 0
    from rdkit import Chem
    mol = next(Chem.SDMolSupplier(str(rank1), sanitize=True, removeHs=True), None)
    assert mol is not None and mol.GetNumAtoms() == 13
