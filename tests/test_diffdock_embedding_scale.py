"""What the receptor's ESM-2 block has to look like for the score network to behave.

The score network takes the receptor's 1280-wide ESM-2 embedding as plain input features:
there is no normalisation between it and the first linear, and eleven residual convolutions
downstream of that. Feed it a *constant* where a language model would put a distribution and
the scalar channels compound layer by layer -- on the complex pinned here, conv layer 7 is at
6.8e2, layer 10 at 8.1e11, the centre head's final convolution at 5.2e18, and the norm the
head takes of that overflows fp32 to inf, which the final linear turns into NaN.

That is where the 2026-11-27 "NaN under torch 2.8, finite under torch 2.13" report came from
(D-dfd-orch-40). The torch version was a red herring: both builds return the same numbers for
the same input, and the venvs differ in rdkit and numpy, which pick the starting conformer.
What the two runs shared was a test stub that passed `torch.zeros(len(chain), 1280)` for the
embedding. Which side of fp32's ceiling such a run lands on is luck.

**Upstream does the same thing**, which is why the explosion is input and not a port defect.
Run unmodified DiffDock-L in the reference environment on this complex with the same zeroed
block (`scripts/diffdock_port/zero_esm_upstream.py`) and it returns NaN in 30 of its step-0 tr
and rot entries and raises the same `linalg.svd` error. It is complex-dependent, not universal:
on 7A9E_R4W a zeroed block leaves upstream at tr 0.1294049, and the port at 0.1294050 on
upstream's own graph, PCC 1.000000. Our graph for the complex below is byte-identical to
upstream's -- node features, both edge lists, positions, all maxdiff 0.

So this file pins both halves, with the released weights, on one real complex (1a46 + aspirin,
step 0 of a seed-0 rollout):

* at a realistic embedding scale the step-0 scores are small, and near upstream's own golden
  (`goldens/8G43_ZU6__s0/scores.npz` has tr absmax 7.6e-2, rot 1.07e-1 at step 0);
* a constant embedding is what explodes here, as it does for upstream, so no fixture can
  quietly reintroduce one and read the resulting NaN as a defect in the network.

Real ESM-2 statistics, measured on dfd-ref's captured upstream input for 8G43_ZU6
(`8G43_ZU6__score__rec_node_embedding__additional_features_embedder__c0.npz`): mean -0.0005,
std 0.272, absmax 8.27. The draw below uses that std.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tt_bio.diffdock import checkpoints, config as dfd_config, score_model  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures/diffdock/1a46_aspirin_step0.npz"
CKPT = Path.home() / "scratch/dfd/ckpt/extracted/score_model"

#: the standard deviation of a real ESM-2 650M final hidden state, see the module docstring
ESM2_STD = 0.272


def _released_model():
    if not CKPT.exists():
        pytest.skip(f"no score checkpoint at {CKPT}")
    ckpt, params = checkpoints.resolve("score", str(CKPT), None)
    cfg = dfd_config.load(params.parent, old_model=False, name="diffdock")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    return score_model.load_released(state, cfg).eval()


def _inputs(embedding):
    """The pinned complex's score-network arguments, with *embedding* as the ESM-2 block.

    Everything but the embedding comes out of a real `tt-bio dock` call: the graph builder's
    own node features, both edge lists and the step-0 sigmas. The fixture stores only the
    receptor's residue-type column, because the run that produced it had the embedding
    stubbed -- the 1280 columns are this function's job.
    """
    d = np.load(FIXTURE)
    rec_x = torch.cat([torch.from_numpy(d["rec_type"]).unsqueeze(1), embedding], dim=1)
    return (torch.from_numpy(d["lig_x"]).long(), torch.from_numpy(d["lig_pos"]),
            torch.from_numpy(d["bond_index"]).long(), torch.from_numpy(d["bond_attr"]),
            torch.from_numpy(d["edge_mask"]), rec_x, torch.from_numpy(d["rec_pos"]),
            torch.from_numpy(d["rec_edge_index"]).long(), float(d["t_tr"]),
            torch.from_numpy(d["tr_sigma"]), torch.from_numpy(d["rot_sigma"]),
            torch.from_numpy(d["tor_sigma"]))


def _n_receptor_rows():
    return int(np.load(FIXTURE)["rec_type"].shape[0])


def test_a_realistic_embedding_scale_gives_scores_the_size_upstreams_are():
    model = _released_model()
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(_n_receptor_rows(), 1280, generator=g) * ESM2_STD
    with torch.no_grad():
        tr, rot, tor = model(*_inputs(emb))[:3]

    for name, x in (("tr", tr), ("rot", rot), ("tor", tor)):
        assert torch.isfinite(x).all(), f"{name} is not finite: {x}"
    # upstream's step-0 golden is 7.6e-2 (tr) and 1.07e-1 (rot) on its own complex. A draw is
    # not that complex, so the bar is an order of magnitude, not a tolerance: anything under 1
    # is the right regime and the 1e17 this test exists for is fourteen decades away.
    assert tr.abs().max() < 1.0, f"tr absmax {float(tr.abs().max()):.3e}, expected O(0.1)"
    assert rot.abs().max() < 1.0, f"rot absmax {float(rot.abs().max()):.3e}, expected O(0.1)"


@pytest.mark.parametrize("constant", [0.0, 0.02])
def test_a_constant_embedding_is_what_drives_the_network_out_of_range(constant):
    """The stub, not the network: upstream explodes on this complex too, measured.

    If a future change makes the network tolerate a constant embedding, this test fails, and
    that is worth being told about -- it would mean the port had stopped matching upstream on
    the degenerate input, which is where `_repair_nans` and the sampler's guard are aimed.
    """
    model = _released_model()
    emb = torch.full((_n_receptor_rows(), 1280), constant)
    with torch.no_grad():
        tr, rot = model(*_inputs(emb))[:2]

    worst = max(float(tr.abs().max()), float(rot.abs().max()))
    assert not np.isfinite(worst) or worst > 1e6, (
        f"a constant {constant} embedding gave {worst:.3e}; this test records that a constant "
        f"embedding is out-of-distribution input, and the docking fixtures rely on that")
