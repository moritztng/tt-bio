"""The port matches upstream on the poses whose score reaches 1e15, not just on calm ones.

Four PoseBusters complexes (7AVI_S2Q, 7TBU_S3P, 7X5N_5M5, 7ZCC_OGA) end a dock in
``linalg.svd: ... non-finite`` because the tr/rot heads return a *finite* ~1e15 that no NaN
guard keys on. That looked like a parity break in this network. It is not: feeding upstream's
unmodified ``TensorProductScoreModel`` the port's own dumped inputs from the exact call that
blows up, **upstream returns the same magnitude**, and the port tracks it to PCC 1 - 3e-9.

| 7ZCC_OGA, step 17 | upstream | port |
|---|---|---|
| tr absmax | 2.44916e+15 | 2.44602e+15 |
| rot absmax | 4.37859e+16 | 4.37218e+16 |

So the overflow is the score model's own behaviour on a reachable pose, shared with upstream,
and the thing worth pinning is that the port stays faithful to upstream *in that regime* --
where a clamp, a rescale or a lost epsilon would be easy to introduce and would read as a
"fix". A magnitude clamp here would be a parity break, because upstream does not clamp.

The fixtures carry the verified-identical inputs plus **upstream's own** tr/rot/tor as the
golden, written by ``scripts/diffdock_port/blowup_upstream.py --golden-out`` after it checked
every input tensor against the dump (maxdiff 0; ``rec_x`` ~6e-06, which is ESM-2 noise from the
two sides having run the encoder on different hosts) and applied upstream's own ``crop_beyond``
at ``3 * tr_sigma + 20``. Host torch only, no device, and it runs under torch 2.8.

`state/dfd/BLOWUP-2026-11-27.md` carries the full measurement and the hand-back to
`dfd-sampling`, which owns keeping one overflowing sample from taking the batch's SVD down.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from tt_bio.diffdock import checkpoints, config as dfd_config, score_model  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures/diffdock"
CKPT = Path.home() / "scratch/dfd/ckpt/extracted/score_model"

#: (fixture, the head magnitudes upstream itself returns on it)
CASES = {
    "7ZCC_OGA_blowup": {"tr": 2.44916e15, "rot": 4.37859e16},
    "7ZCC_OGA_prev": {"tr": 2.45908e00, "rot": 4.22352e00},
}

#: PCC bar. The calm step agrees to 12 digits; in the overflow the two differ by fp32 noise
#: amplified by ~1e15, which costs a few digits and still leaves the heads parallel.
PCC_BAR = {"7ZCC_OGA_blowup": 0.99999, "7ZCC_OGA_prev": 0.9999999999}


def _released_model():
    if not CKPT.exists():
        pytest.skip(f"no score checkpoint at {CKPT}")
    ckpt, params = checkpoints.resolve("score", str(CKPT), None)
    cfg = dfd_config.load(params.parent, old_model=False, name="diffdock")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    return score_model.load_released(state, cfg).eval()


def _load(name):
    path = FIXTURES / f"{name}.npz"
    if not path.exists():
        pytest.skip(f"no fixture at {path}")
    return np.load(path)


def _run(model, d):
    args = (torch.from_numpy(d["lig_x"]).long(), torch.from_numpy(d["lig_pos"]),
            torch.from_numpy(d["bond_index"]).long(), torch.from_numpy(d["bond_attr"]),
            torch.from_numpy(d["edge_mask"]), torch.from_numpy(d["rec_x"]),
            torch.from_numpy(d["rec_pos"]), torch.from_numpy(d["rec_edge_index"]).long(),
            float(d["t_tr"]), torch.from_numpy(d["tr_sigma"]),
            torch.from_numpy(d["rot_sigma"]), torch.from_numpy(d["tor_sigma"]))
    with torch.no_grad():
        tr, rot, tor = model(*args)[:3]
    return {"tr": tr[0].double(), "rot": rot[0].double(), "tor": tor.double()}


def _pcc(a, b):
    a, b = torch.as_tensor(a).flatten().double(), torch.as_tensor(b).flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_tracks_upstream_into_the_overflow(name):
    """Same inputs, same answer as upstream, on the calm step and on the 1e15 one alike."""
    d = _load(name)
    out = _run(_released_model(), d)
    for head in ("tr", "rot"):
        golden = torch.from_numpy(d[f"upstream_{head}"]).double()
        got = out[head]
        assert torch.isfinite(got).all(), f"{name} {head}: the port went non-finite, upstream did not"
        pcc = _pcc(got, golden)
        assert pcc >= PCC_BAR[name], (
            f"{name} {head}: PCC {pcc:.12f} below {PCC_BAR[name]} against upstream's own output")
        ratio = float(got.abs().max() / golden.abs().max())
        assert 0.99 <= ratio <= 1.01, (
            f"{name} {head}: magnitude ratio {ratio:.6f} against upstream. PCC is "
            f"scale-invariant, so this is what catches a clamp or a lost scale factor")


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_overflow_is_upstreams_own_magnitude_and_is_not_clamped(name):
    """The 1e15 is real and stays. Clamping it would be a parity break, not a fix."""
    d = _load(name)
    out = _run(_released_model(), d)
    for head, expected in CASES[name].items():
        got = float(out[head].abs().max())
        assert got == pytest.approx(expected, rel=0.02), (
            f"{name} {head}: absmax {got:.6g}, upstream's own is {expected:.6g}")


def test_the_blow_up_step_carries_no_nan_or_inf_on_either_side():
    """What kills these complexes is a FINITE number, which is why no NaN guard sees it."""
    d = _load("7ZCC_OGA_blowup")
    out = _run(_released_model(), d)
    for head in ("tr", "rot", "tor"):
        assert torch.isfinite(out[head]).all(), f"{head} is non-finite"
        assert torch.isfinite(torch.from_numpy(d[f"upstream_{head}"])).all(), (
            f"upstream {head} is non-finite")
    assert float(out["tr"].abs().max()) > 1e12, (
        "this fixture exists because the head reaches 1e15 while staying finite")
