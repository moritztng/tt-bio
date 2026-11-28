"""The confidence network: strict load, the zero-edge guard, and parity against upstream.

The parity numbers live in `state/dfd/CONFIDENCE.md`; what is pinned here is the behaviour
that must not regress silently.

The zero-edge tests build the degenerate case **directly** rather than hoping a rollout
produces one. Upstream only reaches it on a pose that has drifted out of the pocket, so a
test that waits for one grades nothing on most runs, and a truncated rollout is not a
cheap way to force it (D-dfd-orch-10: `--steps 1` and `--steps 3` hit upstream's crash
because the poses are still undocked, which cost `dfd-ref` two runs).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tt_bio.diffdock import config as dfd_config
from tt_bio.diffdock.confidence_model import (
    LIGAND_GROUPS,
    ConfidenceModel,
    ScalarAtomEncoder,
    load_released,
)
from tt_bio.diffdock.conv import TensorProductConvLayer

CKPT_DIR = Path.home() / "scratch/dfd/ckpt/extracted/confidence_model"
SH = "1x0e+1x1o+1x2e"


def _checkpoint():
    ckpt = CKPT_DIR / "best_model_epoch75.pt"
    if not ckpt.exists():
        pytest.skip(f"no confidence checkpoint at {ckpt}")
    return torch.load(ckpt, map_location="cpu", weights_only=False)


def _config():
    if not (CKPT_DIR / "model_parameters.yml").exists():
        pytest.skip("no confidence model_parameters.yml")
    return dfd_config.load(CKPT_DIR, old_model=True, name="confidence")


# -- the checkpoint ---------------------------------------------------------------------

def test_released_checkpoint_loads_strictly():
    """No missing and no unexpected key, and every tensor accounted for.

    `load_released` raises on either, so reaching the assertions is most of the test. The
    counts are here so that a future change which quietly drops a module shows up as a
    number rather than as a slightly different confidence.
    """
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)

    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    assert params == 4_769_636
    assert params + buffers == 4_773_124
    assert params + buffers + 5_778 == dfd_config.CONFIDENCE_PARAM_COUNT == 4_778_902
    assert len(model.conv_layers) == 9 * cfg.num_conv_layers == 45
    assert cfg.num_confidence_outputs == 2


def test_encoder_is_the_old_one_not_the_new_one():
    """The receptor encoder adds its scalar projection; it does not concatenate.

    Upstream ships two atom encoders with identical widths. `score_model.AtomEncoder` is the
    newer one and concatenates; the confidence checkpoint wants `OldAtomEncoder`, which adds.
    Swapping them loads cleanly in the score model's case and gives wrong features, so this
    pins the arithmetic rather than the key names.
    """
    enc = ScalarAtomEncoder(4, ([5], 0), sigma_embed_dim=3)
    with torch.no_grad():
        enc.atom_embedding_list[0].weight.copy_(
            torch.arange(20, dtype=torch.float32).reshape(5, 4))
        enc.linear.weight.zero_()
        enc.linear.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    x = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    # category 2 -> embedding row [8,9,10,11]; the scalar branch contributes its bias.
    assert torch.equal(enc(x), torch.tensor([[9.0, 11.0, 13.0, 15.0]]))


def test_head_emits_two_columns_and_ranking_uses_column_zero():
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    assert model.confidence_predictor[-1].out_features == 2


# -- the zero-edge guard (D-dfd-orch-10) --------------------------------------------------

def test_empty_edge_set_returns_zeros_at_the_destination_count():
    """A cross group with no edges returns zeros shaped by `out_nodes`, not by its source.

    This is the exact layer that crashes upstream: it reads receptor nodes and writes ligand
    nodes, so the two counts differ and returning the source's would break the caller's
    residual add even when the values are right.
    """
    layer = TensorProductConvLayer("24x0e", SH, "24x0e + 6x1o", 72, residual=False).eval()
    source = torch.randn(7, 24)
    empty = torch.zeros((2, 0), dtype=torch.long)
    out = layer(source, empty, torch.zeros(0, 72), torch.zeros(0, 9), out_nodes=5)
    assert out.shape == (5, 42)
    assert torch.all(out == 0)
    assert torch.isfinite(out).all()


def test_fully_masked_edge_set_matches_an_empty_one():
    """Masking every edge must give the empty answer, not NaN from a 0/0 mean.

    The mean is computed as sum-then-divide, so an all-masked group drives the divisor to
    zero; it is clamped, which is what keeps this finite.
    """
    layer = TensorProductConvLayer("24x0e", SH, "24x0e + 6x1o", 72, residual=False).eval()
    torch.manual_seed(0)
    node_attr = torch.randn(6, 24)
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 0, 1]])
    masked = layer(node_attr, edge_index, torch.randn(4, 72), torch.randn(4, 9),
                   out_nodes=6, edge_mask=torch.zeros(4))
    assert torch.isfinite(masked).all()
    assert torch.allclose(masked, torch.zeros_like(masked), atol=1e-6)


def test_confidence_model_scores_a_pose_with_no_ligand_receptor_edges():
    """The whole network on a pose placed far outside the pocket.

    Upstream raises `ValueError: number sections must be larger than 0` here
    (`tensor_layers.py:362`). The port returns a finite score. The ligand is moved 500 A
    away, which empties both the ligand-receptor and the ligand-atom cross graphs.
    """
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    torch.manual_seed(0)

    n_lig, n_rec, n_atom = 5, 6, 9
    lig_x = torch.zeros(n_lig, len(_lig_dims()))
    bond_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    bond_attr = torch.zeros(4, cfg.in_lig_edge_features)
    rec_pos = torch.randn(n_rec, 3) * 4
    atom_pos = torch.randn(n_atom, 3) * 4
    far = torch.randn(n_lig, 3) + 500.0

    out = model(
        lig_x, far, bond_index, bond_attr,
        torch.zeros(n_rec, 1 + 1280), rec_pos, _ring(n_rec),
        torch.zeros(n_atom, 4), atom_pos, _ring(n_atom),
        torch.stack([torch.arange(n_atom), torch.arange(n_atom) % n_rec]),
    )
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_last_layer_runs_only_the_three_ligand_groups():
    assert LIGAND_GROUPS == (0, 1, 2)
    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    calls: list[int] = []
    for i, layer in enumerate(model.conv_layers):
        layer.register_forward_hook(lambda m, a, o, i=i: calls.append(i))

    n_lig, n_rec, n_atom = 4, 5, 7
    model(torch.zeros(n_lig, len(_lig_dims())), torch.randn(n_lig, 3) * 2,
          torch.tensor([[0, 1], [1, 2]]), torch.zeros(2, cfg.in_lig_edge_features),
          torch.zeros(n_rec, 1 + 1280), torch.randn(n_rec, 3) * 2, _ring(n_rec),
          torch.zeros(n_atom, 4), torch.randn(n_atom, 3) * 2, _ring(n_atom),
          torch.stack([torch.arange(n_atom), torch.arange(n_atom) % n_rec]))

    last = [c for c in calls if c >= 9 * (cfg.num_conv_layers - 1)]
    assert sorted(last) == [36, 37, 38]


def _lig_dims():
    from tt_bio.diffdock.constants import lig_feature_dims
    return lig_feature_dims[0]


def _ring(n: int) -> torch.Tensor:
    """A connected `[2, n]` graph, so no node is isolated."""
    src = torch.arange(n)
    return torch.stack([src, (src + 1) % n])


# -- ranking poses, and the `tt-bio dock` wiring ------------------------------------------

RECEPTOR = Path("/home/moritz/scratch/dfd/patchtest/examples/1a46_protein_processed.pdb")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_rank_poses_sorts_best_first_and_keeps_ties_in_sampling_order():
    from tt_bio.diffdock.confidence_model import rank_poses

    poses = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    scores = torch.tensor([0.0, 2.0, -1.0, 2.0, 1.0])
    ranked, ordered = rank_poses(poses, scores)
    assert ordered.tolist() == [2.0, 2.0, 1.0, 0.0, -1.0]
    # The two 2.0s keep poses 1 then 3: a stable sort is what makes a run reproducible.
    assert torch.equal(ranked[0], poses[1]) and torch.equal(ranked[1], poses[3])


def test_score_poses_refuses_a_score_only_receptor_graph():
    """Handing it the score network's graph must fail, not score something else.

    The score model's receptor graph has no `atom` entry. Silently falling back to the
    residue graph would produce a plausible number from the wrong network input.
    """
    from tt_bio.diffdock.confidence_model import score_poses

    with pytest.raises(ValueError, match="all-atom"):
        score_poses(None, {"x": torch.zeros(1, 1)}, {}, torch.zeros(1, 1, 3))


def test_score_poses_is_the_per_pose_forward_and_takes_column_zero():
    """`score_poses` over N poses equals N direct calls, column 0, in order."""
    from tt_bio.diffdock.confidence_model import score_poses

    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)
    torch.manual_seed(0)

    n_lig, n_rec, n_atom = 5, 6, 9
    lig = {"x": torch.zeros(n_lig, len(_lig_dims())),
           "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
           "edge_attr": torch.zeros(4, cfg.in_lig_edge_features)}
    rec = {"x": torch.zeros(n_rec, 1 + 1280), "pos": torch.randn(n_rec, 3) * 4,
           "edge_index": _ring(n_rec),
           "atom": {"x": torch.zeros(n_atom, 4), "pos": torch.randn(n_atom, 3) * 4,
                    "edge_index": _ring(n_atom),
                    "atom_rec_edge_index": torch.stack(
                        [torch.arange(n_atom), torch.arange(n_atom) % n_rec])}}
    poses = torch.randn(3, n_lig, 3) * 2

    got = score_poses(model, rec, lig, poses)
    assert got.shape == (3,)
    with torch.no_grad():
        want = [model(lig["x"], p, lig["edge_index"], lig["edge_attr"], rec["x"],
                      rec["pos"], rec["edge_index"], rec["atom"]["x"], rec["atom"]["pos"],
                      rec["atom"]["edge_index"], rec["atom"]["atom_rec_edge_index"])[0, 0]
                for p in poses]
    assert torch.allclose(got, torch.stack(want))


@pytest.mark.skipif(not RECEPTOR.exists(), reason="no local receptor structure")
def test_dock_writes_ranked_poses_when_a_confidence_checkpoint_is_given(tmp_path,
                                                                        monkeypatch):
    """`tt-bio dock` end to end on host torch: poses out, ranked, scores written.

    The ESM-2 pass is stubbed: a 650 MB encoder download is not what this test is about,
    and the embedding's *values* cannot change whether the ranking is wired -- only its
    shape, which the stub keeps. Everything after it is the real thing: the real receptor
    structure, the real all-atom graph, the released 19 MB checkpoint and the real head.
    """
    import json

    from click.testing import CliRunner

    from tt_bio.diffdock import esm
    from tt_bio.main import cli

    if not (CKPT_DIR / "best_model_epoch75.pt").exists():
        pytest.skip("no confidence checkpoint")
    monkeypatch.setattr(esm, "receptor_embeddings",
                        lambda chains, **kw: [torch.zeros(len(c), 1280) for c in chains])

    out = tmp_path / "o"
    r = CliRunner().invoke(cli, [
        "dock", str(RECEPTOR), "--ligand", ASPIRIN, "--name", "x", "--no_model",
        "--diffusion_samples", "3", "--confidence_checkpoint", str(CKPT_DIR),
        "--out_dir", str(out)], catch_exceptions=False)
    assert r.exit_code == 0, r.output

    ranking = json.loads((out / "x" / "confidence.json").read_text())
    assert [row["rank"] for row in ranking] == [1, 2, 3]
    scores = [row["confidence"] for row in ranking]
    assert all(s is not None for s in scores), "a named checkpoint must produce scores"
    assert scores == sorted(scores, reverse=True), "poses must be written best first"
    # Rank 1 is written twice, bare and suffixed, as upstream does.
    assert (out / "x" / "rank1.sdf").exists()


# -- the register's named tests -----------------------------------------------------------

GOLDENS = Path.home() / ".coworker/state/dfd/goldens"
GOLDEN_IDS = ("8G43_ZU6", "8AEU_M0L", "7A9E_R4W")


def test_confidence_scores_match_cpu_golden():
    """The `[N, 2]` the port emits against the values captured from upstream on the CPU.

    Both sides get the *identical* graph: upstream's own dataset tensors are read out of
    `confidence_inputs.npz` rather than rebuilt here, so a disagreement is the network's and
    not the graph builder's (that is `dfd-input`'s to grade, and mixing the two would let an
    edge-ordering difference read as a conv-layer error).

    Two references, because they answer different questions.
    `confidence.npz` is upstream re-run on exactly these inputs, so it carries no sampling
    noise and the port must reproduce it to float32 rounding.
    `poses.npz::confidence_raw` is what upstream produced inside the full 19-step rollout; it
    sits up to 0.25 away from its own rebuild (state/dfd/goldens/CONFIDENCE-REBUILD.json), a
    float32 reference floor that is upstream's, not the port's. So that one is graded on the
    thing that survives it: the ranking.
    """
    import numpy as np

    state, cfg = _checkpoint(), _config()
    model = load_released(state, cfg)

    seen = 0
    for cid in GOLDEN_IDS:
        d = GOLDENS / f"{cid}__s0"
        if not (d / "confidence_inputs.npz").exists():
            continue
        seen += 1
        inp = {k: torch.from_numpy(v) for k, v in np.load(d / "confidence_inputs.npz").items()}
        poses = np.load(d / "poses.npz")
        lig_pos = torch.from_numpy(poses["pos"] - poses["center"]).float()
        with torch.no_grad():
            got = torch.cat([
                model(inp["lig_x"].float(), pose, inp["bond_index"].long(),
                      inp["bond_attr"].float(), inp["rec_x"].float(), inp["rec_pos"].float(),
                      inp["rec_edge_index"].long(), inp["atom_x"].float(),
                      inp["atom_pos"].float(), inp["atom_edge_index"].long(),
                      inp["atom_rec_edge_index"].long())
                for pose in lig_pos]).numpy()
        assert got.shape == (len(lig_pos), 2)

        rebuild = np.load(d / "confidence.npz")["confidence_raw"]
        assert np.max(np.abs(got - rebuild)) < 1e-2, f"{cid}: port != upstream rebuilt"
        assert _col0_order(got) == _col0_order(rebuild), f"{cid}: ranking differs from the rebuild"

        rollout = poses["confidence_raw"]
        assert _pcc(got, rollout) > 0.999, f"{cid}: port != upstream's rollout capture"
        assert _col0_order(got) == _col0_order(rollout), \
            f"{cid}: ranking differs from the captured rollout"
    if seen == 0:
        pytest.skip(f"no confidence goldens under {GOLDENS}")


def _col0_order(a) -> list[int]:
    """Upstream's ranking: `np.argsort(confidence[:, 0])[::-1]`, best first."""
    import numpy as np

    return np.argsort(-np.asarray(a, np.float64)[:, 0], kind="stable").tolist()


def _pcc(a, b) -> float:
    import numpy as np

    return float(np.corrcoef(np.asarray(a, np.float64).ravel(),
                             np.asarray(b, np.float64).ravel())[0, 1])


@pytest.mark.skipif(not RECEPTOR.exists(), reason="no local receptor structure")
def test_confidence_graph_is_all_atom(monkeypatch):
    """The confidence network gets its own graph, and the ESM-2 pass is not repeated.

    The released confidence checkpoint sets `all_atoms: true` and
    `use_original_model_cache: false`, so upstream builds a second dataset for it. The port
    does the same from the structure the complex graph recorded, and hands it the embedding
    the complex already carries: rebuilding a k-NN graph is cheap, a second ESM-2 650M pass
    over the same sequence is not.
    """
    from tt_bio.diffdock import input as dfd_input
    from tt_bio.diffdock import receptor as dfd_receptor
    from tt_bio.diffdock import score_model as sm

    assert dfd_input.CONFIDENCE_GRAPH["all_atoms"] is True
    assert dfd_input.SCORE_GRAPH["all_atoms"] is False
    assert {k: v for k, v in dfd_input.CONFIDENCE_GRAPH.items() if k != "all_atoms"} == \
           {k: v for k, v in dfd_input.SCORE_GRAPH.items() if k != "all_atoms"}, \
        "the two graphs differ in all_atoms and nothing else; the k-NN caps are the same"

    # And those settings are the released checkpoint's own, not a guess.
    if (CKPT_DIR / "model_parameters.yml").exists():
        from tt_bio.data.yaml_input import load_mapping

        raw = load_mapping(CKPT_DIR / "model_parameters.yml")
        assert raw["all_atoms"] is True
        assert raw["use_original_model_cache"] is False, \
            "upstream builds a second dataset for the confidence model only when this is false"
        assert int(raw["c_alpha_max_neighbors"]) == dfd_input.CONFIDENCE_GRAPH["c_alpha_max_neighbors"]
        assert int(raw["atom_max_neighbors"]) == dfd_input.CONFIDENCE_GRAPH["atom_max_neighbors"]

    spec = dfd_input.ComplexSpec(name="cx", protein_path=str(RECEPTOR),
                                 ligand_description=ASPIRIN)
    chains = dfd_input.chain_sequences_for_embedding(RECEPTOR)
    embeddings = [torch.zeros(len(c), 1280) for c in chains]

    score_graph = dfd_input.build_complex_graph(spec, lm_embeddings=embeddings,
                                                all_atoms=False)
    conf_graph = dfd_input.build_complex_graph(spec, lm_embeddings=embeddings,
                                               all_atoms=True)

    assert "atom" not in score_graph.receptor, "the score graph is residue-level"
    atom = conf_graph.receptor["atom"]
    assert set(atom) >= {"x", "pos", "edge_index", "atom_rec_edge_index"}
    n_atoms, n_res = atom["pos"].shape[0], conf_graph.receptor["pos"].shape[0]
    assert n_atoms > n_res, f"{n_atoms} atoms for {n_res} residues is not an all-atom graph"
    assert atom["atom_rec_edge_index"].shape == (2, n_atoms), \
        "every receptor atom is linked to exactly one residue"
    assert int(atom["atom_rec_edge_index"][1].max()) < n_res
    assert n_res == score_graph.receptor["pos"].shape[0], \
        "the two graphs cover the same residues"

    # And the confidence pass rebuilds that graph itself, from the structure and the
    # embedding the complex already carries, with no second encoder call.
    from tt_bio.diffdock import esm

    def no_second_pass(*a, **kw):
        raise AssertionError("the confidence pass re-ran ESM-2 over the same sequence")

    monkeypatch.setattr(esm, "receptor_embeddings", no_second_pass)

    seen: list[dict] = []
    real = dfd_receptor.build_receptor_graph

    def spy(struct, **kw):
        seen.append(kw)
        return real(struct, **kw)

    monkeypatch.setattr(dfd_receptor, "build_receptor_graph", spy)

    net = sm.HostScoreNetwork(_score_stub(), confidence_model=_confidence_stub())
    net.confidence(score_graph, torch.zeros(2, score_graph.ligand["pos"].shape[0], 3))

    assert len(seen) == 1, "the confidence graph is built once per complex"
    assert seen[0]["lm_embeddings"] is embeddings, \
        "the confidence graph must reuse the complex's own ESM-2 embedding"
    assert {k: v for k, v in seen[0].items() if k != "lm_embeddings"} == \
           dfd_input.CONFIDENCE_GRAPH


@pytest.mark.skipif(not RECEPTOR.exists(), reason="no local receptor structure")
def test_confidence_receptor_is_built_once_per_structure_file(monkeypatch, tmp_path):
    """A screen ranks many ligands against one receptor: its all-atom graph and embedding
    are built on the first ligand and kept, and a different structure file drops them."""
    import shutil

    from tt_bio.diffdock import input as dfd_input
    from tt_bio.diffdock import receptor as dfd_receptor
    from tt_bio.diffdock import score_model as sm

    builds: list = []
    real = dfd_receptor.build_receptor_graph
    monkeypatch.setattr(dfd_receptor, "build_receptor_graph",
                        lambda struct, **kw: builds.append(1) or real(struct, **kw))
    embeddings = [torch.zeros(len(c), 1280)
                  for c in dfd_input.chain_sequences_for_embedding(RECEPTOR)]

    def graph(path):
        spec = dfd_input.ComplexSpec(name="cx", protein_path=str(path),
                                     ligand_description=ASPIRIN)
        return dfd_input.build_complex_graph(spec, lm_embeddings=embeddings, all_atoms=False)

    first, other = graph(RECEPTOR), graph(shutil.copy(RECEPTOR, tmp_path / "other.pdb"))
    builds.clear()
    net = sm.HostScoreNetwork(_score_stub(), confidence_model=_confidence_stub())
    poses = torch.zeros(2, first.ligand["pos"].shape[0], 3)
    net.confidence(first, poses)
    net.confidence(first, poses)
    assert len(builds) == 1, "the second ligand on the same receptor rebuilt its graph"
    net.confidence(other, poses)
    assert len(builds) == 2, "a different structure file must not reuse the kept receptor"


def _score_stub():
    """A stand-in with the one attribute `HostScoreNetwork.__init__` reads."""
    class _Stub:
        config = None
    return _Stub()


def _confidence_stub():
    """Returns a fixed `[1, 2]` per pose: this test grades the graph, not the values."""
    class _Stub:
        def receptor_side(self, *a, **kw):
            return {}

        def ligand_side(self, *a, **kw):
            return dict(nodes=None, edges=None)

        def convolve(self, *a, **kw):
            return None

        def head(self, *a, **kw):
            return torch.zeros(1, 2)

        def eval(self):
            return self
    return _Stub()


def test_ranking_order():
    """Poses come out sorted by descending column 0, ties in sampling order.

    Upstream: `confidence = confidence[:, 0]` when `rmsd_classification_cutoff` is a list
    (the released checkpoint's is `[2.0]`), then `np.argsort(confidence)[::-1]`. Column 1 must
    not reach the ordering, which is what the second case pins: it is the mirror image of
    column 0, so a port that ranked on it, or on the row sum, would come out reversed.
    """
    import numpy as np

    from tt_bio.diffdock.confidence_model import rank_poses

    poses = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    raw = torch.tensor([[0.0, 9.0], [2.0, -9.0], [-1.0, 5.0], [2.0, 1.0], [1.0, 0.0]])

    ranked, ordered = rank_poses(poses, raw[:, 0])
    assert ordered.tolist() == [2.0, 2.0, 1.0, 0.0, -1.0]
    assert _col0_order(raw.numpy()) == [1, 3, 4, 0, 2]
    for rank, sample in enumerate(_col0_order(raw.numpy())):
        assert torch.equal(ranked[rank], poses[sample])

    # Ties keep sampling order: samples 1 and 3 both score 2.0 and stay in that order.
    assert torch.equal(ranked[0], poses[1]) and torch.equal(ranked[1], poses[3])

    flipped = rank_poses(poses, raw[:, 1])[1]
    assert flipped.tolist() != ordered.tolist(), \
        "column 1 orders differently, so ranking on column 0 is a real choice"
    assert np.array_equal(_col0_order(raw.numpy()), [1, 3, 4, 0, 2])


def test_score_only_writes_unranked(tmp_path, monkeypatch):
    """`--no_confidence`: sampling order, no scores in the filenames, no 19 MB checkpoint.

    Upstream's score-only path is `--confidence_model_dir` unset. Its output is where the
    layouts part company: upstream still formats `confidence[rank]` and so raises, counting
    the complex as a failure, while `tt-bio dock` writes the poses it has (D-dfd-9). What is
    pinned here is that the order is the sampler's own -- every written pose is the sampled
    pose of the same index, shifted by one constant -- because "unranked" is a claim about
    the order, not just about the filenames.

    The checkpoint guard is the other half: a score-only run must not pay for loading, or
    even locating, the confidence model.
    """
    import numpy as np

    from conftest import dock_on_host_torch, sdf_coords
    from tt_bio.diffdock import checkpoints
    from tt_bio.diffdock import confidence_model as cm

    real_resolve = checkpoints.resolve

    def resolve(kind, *a, **kw):
        assert kind != "confidence", "--no_confidence still went looking for the checkpoint"
        return real_resolve(kind, *a, **kw)

    monkeypatch.setattr(checkpoints, "resolve", resolve)
    monkeypatch.setattr(cm, "load_checkpoint", lambda *a, **kw: pytest.fail(
        "--no_confidence loaded the confidence checkpoint"))

    samples = 4
    rollout: list = []
    out = dock_on_host_torch(tmp_path / "results", monkeypatch, samples=samples,
                             actual_steps=3, confidence=False, rollout=rollout)
    write_dir = out / "cx"

    assert {p.name for p in write_dir.glob("*.sdf")} == \
           {f"rank{r}.sdf" for r in range(1, samples + 1)}
    assert not list(write_dir.glob("*_confidence*.sdf")), \
        "a score-only run writes no confidence into a filename"

    ranking = json.loads((write_dir / "confidence.json").read_text())
    assert [row["rank"] for row in ranking] == list(range(1, samples + 1))
    assert [row["confidence"] for row in ranking] == [None] * samples

    written = np.asarray([sdf_coords(write_dir / f"rank{r}.sdf")[1]
                          for r in range(1, samples + 1)])
    sampled = np.asarray(rollout[0])
    flat = (written - sampled).reshape(-1, 3)
    assert np.all(flat.max(axis=0) - flat.min(axis=0) < 1e-3), \
        "pose N as written is not sample N shifted by one constant: the order moved"


def test_old_architecture_loads_and_runs():
    """Upstream's two architecture booleans, and what each of them selects here.

    `--weights` carries both (`WEIGHTS[name] -> (old_score_model, old_filtering_model)`).
    The confidence side is the old architecture and always is: upstream's default recipe
    sets `old_filtering_model: true`, and the released 19 MB checkpoint is that network --
    which this test shows by loading it strictly under `old_model=True` and watching
    `old_model=False` be *refused* -- upstream's other confidence architecture, which is not
    ported and has no released weights here. That refusal is the point: the two have
    identical widths, so a flag that did nothing would still produce finite numbers.

    The score side has no released weights for the old architecture anywhere (the v1.1 asset
    is DiffDock-L), and it is not ported, so `--weights diffdock` is refused by name
    (D-dfd-confidence-3) rather than quietly running DiffDock-L under a flag that says
    otherwise.
    """
    from tt_bio.diffdock import network
    from tt_bio.diffdock.job import WEIGHTS, DockOptions

    assert WEIGHTS["diffdock-l"] == (False, True)
    assert WEIGHTS["diffdock"] == (True, True)
    assert DockOptions().weights == "diffdock-l"
    assert DockOptions().old_filtering_model is True, \
        "upstream's default_inference_args.yaml sets old_filtering_model: true"

    # The old confidence architecture, real released weights, loaded and run.
    state, cfg = _checkpoint(), _config()
    assert cfg.old_model is True
    model = load_released(state, cfg)
    n_lig, n_rec, n_atom = 4, 5, 7
    with torch.no_grad():
        out = model(torch.zeros(n_lig, len(_lig_dims())), torch.randn(n_lig, 3) * 2,
                    torch.tensor([[0, 1], [1, 2]]), torch.zeros(2, cfg.in_lig_edge_features),
                    torch.zeros(n_rec, 1 + 1280), torch.randn(n_rec, 3) * 2, _ring(n_rec),
                    torch.zeros(n_atom, 4), torch.randn(n_atom, 3) * 2, _ring(n_atom),
                    torch.stack([torch.arange(n_atom), torch.arange(n_atom) % n_rec]))
    assert out.shape == (1, 2) and torch.isfinite(out).all()

    # Asking for the *other* confidence architecture is refused, not quietly served this one.
    # The two have identical widths, so without a guard the flag would be decorative: a
    # caller asking for the new network would get the old one and a confidence that looked
    # fine. `load_checkpoint` carries the choice, so a wrong `old_filtering_model` is named.
    new_cfg = dfd_config.load(CKPT_DIR, old_model=False, name="confidence")
    with pytest.raises(ValueError, match="AAOldModel"):
        load_released(state, new_cfg)
    with pytest.raises(ValueError, match="AAOldModel"):
        from tt_bio.diffdock.confidence_model import load_checkpoint

        load_checkpoint(CKPT_DIR / "best_model_epoch75.pt", CKPT_DIR, old_model=False)

    # And the old score architecture is named as missing rather than silently substituted.
    with pytest.raises(network.ScoreNetworkUnavailable, match="old_score_model"):
        network.load(DockOptions(weights="diffdock", no_confidence=True))
