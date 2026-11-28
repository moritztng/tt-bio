"""A sample whose score is finite but off the schedule is frozen, failed and never written.

The score network returns a finite 1e9 to 1e18 on one isolated call, about one sample of ten
on four of 428 PoseBusters complexes (7AVI_S2Q, 7TBU_S3P, 7X5N_5M5, 7ZCC_OGA), and upstream's
network returns the same number on the same input. `isfinite` passes it, the ligand moves
1e8 A or more, and the confidence model then scores a ligand with no receptor edge at about
-1.0, above every real pose. Three of the four came back with that as rank 1.

`sampling.off_schedule` is the one check: the score times sigma is how many sigmas from the
data the network claims the pose is, and past `sigma_max / sigma_min` no pose on the
schedule can be. Card-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tt_bio.diffdock import loop, sampling

BOUNDS = sampling.SigmaBounds(tr_sigma_min=0.1, tr_sigma_max=19.0, rot_sigma_min=0.03,
                              rot_sigma_max=1.55, tor_sigma_min=0.0314, tor_sigma_max=3.14)
FIXTURE = Path(__file__).parent / "fixtures/diffdock/7ZCC_OGA_blowup.npz"
BATCH, HOT, HOT_STEP = 10, 6, 2


def config(**kw):
    return sampling.SamplingConfig.from_recipe(inference_steps=6, actual_steps=5, **kw)


def ligand():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.2, 1.2, 0.0], [3.6, 1.1, 0.4]])
    return pos, torch.tensor([[1, 2]]), torch.tensor([[False, False, False, True]])


def healthy(step):
    """A plausible score: pulls each pose toward the origin, row-local, O(1) sigmas."""
    b = step.lig_pos.shape[0]
    return (-0.05 * step.lig_pos.mean(dim=1) / (step.tr_sigma ** 2 + 1.0),
            0.01 * step.lig_pos[:, 0], 0.02 * step.lig_pos[:, :, 0].sum(dim=1))


def run(score_fn, seed=5):
    pos, edge_index, mask_rotate = ligand()
    start = pos + torch.arange(BATCH, dtype=torch.float32)[:, None, None]
    return sampling.sample(score_fn, start, cfg=config(), bounds=BOUNDS,
                           noise=sampling.Noise(seed=seed), edge_index=edge_index,
                           mask_rotate=mask_rotate, trajectory=True)


def injected(head, value):
    def score_fn(step):
        heads = [h.clone() for h in healthy(step)]
        if step.step == HOT_STEP:
            heads[head].reshape(BATCH, -1)[HOT] = value
        return tuple(heads)
    return score_fn


@pytest.mark.parametrize("value", [1e15, -1e15, 1e9, float("nan"), float("inf")])
@pytest.mark.parametrize("head", [0, 1, 2])
def test_one_bad_sample_of_ten_is_frozen_and_the_nine_are_bit_identical(head, value):
    clean = run(healthy)
    out = run(injected(head, value))
    assert not bool(clean.failed.any())
    if np.isfinite(value):
        assert out.failed.tolist() == [row == HOT for row in range(BATCH)]
        # frozen at the pose it held before the bad step, for the rest of the rollout
        assert torch.equal(out.poses[HOT], clean.trajectory[HOT_STEP][HOT])
    else:
        # a non-finite score is upstream's repair (1 % of the head's mean), not a failure
        assert not bool(out.failed.any())
    others = [row for row in range(BATCH) if row != HOT]
    assert torch.equal(out.poses[others], clean.poses[others])
    assert torch.isfinite(out.poses).all()


def test_the_limit_is_the_schedules_dynamic_range_in_sigmas():
    for kind, lo, hi in (("tr", 0.1, 19.0), ("rot", 0.03, 1.55), ("tor", 0.0314, 3.14)):
        for sigma in (lo, 1.0, hi):
            assert BOUNDS.score_limit(kind, sigma) * sigma == pytest.approx(hi / lo)


def test_a_score_just_inside_the_limit_is_left_alone():
    def score_fn(step):
        heads = [h.clone() for h in healthy(step)]
        if step.step == HOT_STEP:
            heads[0][HOT] = 0.5 * BOUNDS.score_limit("tr", step.tr_sigma)
        return tuple(heads)
    assert not bool(run(score_fn).failed.any())


def test_every_sample_off_schedule_fails_every_sample():
    def score_fn(step):
        tr, rot, tor = healthy(step)
        return torch.full_like(tr, 1e15), rot, tor
    out = run(score_fn)
    assert bool(out.failed.all())
    assert torch.equal(out.poses, out.trajectory[0])


def test_the_dumped_7zcc_heads_are_off_schedule_at_every_step():
    """The real numbers: upstream's own tr 2.4e15 and rot 4.4e16 at step 17 of 7ZCC_OGA."""
    d = np.load(FIXTURE)
    tr = torch.from_numpy(d["upstream_tr"]).float().reshape(1, 3)
    rot = torch.from_numpy(d["upstream_rot"]).float().reshape(1, 3)
    cfg = sampling.SamplingConfig.from_recipe()
    sched = cfg.schedules(BOUNDS)[0]
    for t in sched[:cfg.steps]:
        tr_sigma, rot_sigma, tor_sigma = BOUNDS.sigmas(float(t), float(t), float(t))
        limits = [BOUNDS.score_limit(k, s) for k, s in
                  (("tr", tr_sigma), ("rot", rot_sigma), ("tor", tor_sigma))]
        assert bool(sampling.off_schedule((tr, torch.zeros(1, 3), None), limits))
        assert bool(sampling.off_schedule((torch.zeros(1, 3), rot, None), limits))
        assert float(tr.abs().max()) > 1e10 * limits[0]


def test_the_traced_loop_freezes_the_same_sample():
    pos, edge_index, mask_rotate = ligand()
    start = pos + torch.arange(BATCH, dtype=torch.float32)[:, None, None]
    cfg = config()
    table = loop.coefficient_table(cfg, BOUNDS, dtype=torch.float32)
    scheds = cfg.schedules(BOUNDS)
    bonds = [tuple(int(i) for i in e) for e in edge_index]

    def scores(hot):
        def score_fn(p, t_idx):
            sig = BOUNDS.sigmas(*(float(s[t_idx]) for s in scheds))
            tr, rot, tor = healthy(SimpleNamespace(lig_pos=p, tr_sigma=sig[0]))
            tr = tr.clone()
            if hot and t_idx == HOT_STEP:
                tr[HOT] = 1e15
            return tr, rot, tor.reshape(BATCH, 1)
        return score_fn

    def go(hot):
        g = torch.Generator().manual_seed(3)
        return loop.run_loop(start, scores(hot), table,
                             lambda t: (torch.randn(BATCH, 3, generator=g),
                                        torch.randn(BATCH, 3, generator=g),
                                        torch.randn(BATCH, 1, generator=g)),
                             bonds, mask_rotate)

    clean, _, _, clean_failed = go(False)
    out, _, _, failed = go(True)
    assert not bool(clean_failed.any())
    assert failed.tolist() == [row == HOT for row in range(BATCH)]
    others = [row for row in range(BATCH) if row != HOT]
    assert torch.equal(out[others], clean[others])
    assert float((out[HOT] - start[HOT]).abs().max()) < 1e3


# ---------------------------------------------------------------------------
# ranking and output: the zero-receptor-edge case
# ---------------------------------------------------------------------------

class _Model:
    """Stands in for `network.load`'s object: a healthy score, except one sample at one step,
    and a confidence that does what the real one does to a ligand with no receptor edge."""

    config = SimpleNamespace(all_atoms=False, lig_max_radius=5.0, no_torsion=False,
                             crop_beyond=None, tr_sigma_min=0.1, tr_sigma_max=19.0,
                             rot_sigma_min=0.03, rot_sigma_max=1.55, tor_sigma_min=0.0314,
                             tor_sigma_max=3.14)
    on_device = False

    def __init__(self, hot_rows):
        self.hot_rows = hot_rows

    def score_fn(self, graph):
        def fn(step):
            b = step.lig_pos.shape[0]
            tr = -0.05 * step.lig_pos.mean(dim=1) / (step.tr_sigma ** 2 + 1.0)
            if step.step == HOT_STEP:
                tr[self.hot_rows] = 1e15
            return tr, torch.zeros(b, 3), torch.zeros(b * graph.n_torsions)
        return fn

    def confidence(self, graph, poses):
        # Real poses score -1.3 to -4.4 on the four complexes; the zero-edge path about -1.0.
        return -1.5 - torch.arange(poses.shape[0], dtype=torch.float32) * 0.1


def _dock(tmp_path, monkeypatch, hot_rows, n=10):
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit.Chem import AllChem

    from tt_bio.diffdock import esm, input as dfd_input, neighbour_cap, run
    from tt_bio.diffdock.job import DockJob, DockOptions

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    heavy = Chem.RemoveHs(mol)
    graph = SimpleNamespace(
        mol=mol, n_torsions=0, original_center=torch.zeros(1, 3),
        ligand={"pos": torch.tensor(heavy.GetConformer().GetPositions(), dtype=torch.float32),
                "edge_index": torch.zeros(2, 0, dtype=torch.long),
                "edge_mask": torch.zeros(0, dtype=torch.bool),
                "mask_rotate": np.zeros((0, 3), dtype=bool)},
        receptor={"pos": torch.randn(20, 3, generator=torch.Generator().manual_seed(0)) * 8})
    monkeypatch.setattr(dfd_input, "chain_sequences_for_embedding", lambda p: [])
    monkeypatch.setattr(esm, "receptor_embeddings", lambda *a, **k: None)
    monkeypatch.setattr(dfd_input, "build_complex_graph", lambda *a, **k: graph)
    monkeypatch.setattr(neighbour_cap, "check_ligand_graph", lambda *a, **k: None)
    rec = tmp_path / "rec.pdb"
    rec.write_text("END\n")
    options = DockOptions(seed=42, diffusion_samples=n)
    out = tmp_path / f"out{len(hot_rows)}"
    job = DockJob(name="7ZCC_OGA", receptor_path=str(rec), receptor_sequence=None, ligand="CCO")
    return run.dock([job], options, out, model=_Model(hot_rows)), out / "7ZCC_OGA"


def _coords(path):
    from rdkit import Chem
    return Chem.MolFromMolFile(str(path), sanitize=False).GetConformer().GetPositions()


def test_a_failed_sample_is_never_written_or_ranked(tmp_path, monkeypatch):
    clean, clean_dir = _dock(tmp_path, monkeypatch, [])
    summary, out_dir = _dock(tmp_path, monkeypatch, [0])
    assert clean["docked"][0]["failed_samples"] == 0
    assert summary["docked"][0]["failed_samples"] == 1
    rows = json.loads((out_dir / "confidence.json").read_text())
    clean_rows = json.loads((clean_dir / "confidence.json").read_text())
    # Sample 0 had the best confidence; it failed, so the other nine move up one, unchanged.
    assert [r["rank"] for r in rows] == list(range(1, 10))
    assert [r["confidence"] for r in rows] == [r["confidence"] for r in clean_rows[1:]]
    for row, was in zip(rows, clean_rows[1:]):
        xyz = _coords(out_dir / row["sdf"])
        assert np.abs(xyz).max() < 1e3
        assert np.array_equal(xyz, _coords(clean_dir / was["sdf"]))
    assert len(list(out_dir.glob("rank*_confidence*.sdf"))) == 9


def test_a_complex_whose_samples_all_fail_is_an_error_naming_it(tmp_path, monkeypatch):
    from tt_bio.diffdock import run
    with pytest.raises(run.PoseRolloutFailed, match="7ZCC_OGA: all 10 samples failed"):
        _dock(tmp_path, monkeypatch, list(range(10)))
    assert not list(tmp_path.glob("out10/7ZCC_OGA/*.sdf"))
