"""The atom key gather has to be scored against the MATRIX, not against the shapes.

No device. `boltz2.get_indexing_matrix` is built at the REAL window count and
`tenstorrent._populate_diffusion_cache` zero-pads it out to the atom bucket, so the cached
operand carries a fact no shape expresses: how many of its windows gather anything at all. A
fast path that reconstructs the gather from the query window, the key window and the matrix's
SHAPE cannot see that, and it fills the padded windows with real atoms where the model fills
them with zeros.

That is not hypothetical. `TT_BIO_ATOM_KEY_WINDOW` (`wk/b2z2-akw-ship`) does exactly this and
read 4.21875 max abs against the matrix at the 512 aa shape, on 13 of 140 windows, while every
fold-level check stayed green. `perf/b2z2_gather_audit/` holds the device scoring.

So this file pins two things:

  * the discriminating fixture -- a bucketed matrix under a LIVE atom tail. Build the matrix at
    the padded window count instead and the wrong construction is bit-exact, which is the fixture
    the first A/B used and the reason the defect shipped to a staged branch.
  * the invariant that keeps today's tree safe: every slot the matrix gathers nothing for is a
    slot the atom bias masks with -1e9, because that bias is gathered THROUGH the same matrix.
    Whoever changes how the atom bias is built is changing what protects this call site.

Any host-side gather decision the tree grows is picked up automatically by
`test_registered_gather_matches_the_matrix`, which runs whatever `tt_bio.tenstorrent` exposes.
"""
import pytest

torch = pytest.importorskip("torch")

from tt_bio.boltz2 import get_indexing_matrix                    # noqa: E402

W, H = 32, 128                     # ATOM_WINDOW, ATOM_DIM, imported below where ttnn is present
SHIFT = H // 2 - W // 2            # a window's keys start 48 atoms before it

# The 512 aa production shape: 4128 real atoms in 129 windows, bucketed to 4480 in 140.
WINDOWS_REAL, WINDOWS_PAD = 129, 140


def cached_matrix(windows_real=WINDOWS_REAL, windows_pad=WINDOWS_PAD):
    """What the runtime caches: built at the real count, zero-padded to the bucket."""
    ki = get_indexing_matrix(windows_real, W, H, torch.device("cpu"))
    return torch.nn.functional.pad(
        ki, (0, 8 * windows_pad - ki.shape[1], 0, 2 * windows_pad - ki.shape[0]))


def matrix_gather(s, ki):
    """`single_to_keys` on the cached matrix, in float64. This is the answer, by definition."""
    b, k, w, d = s.shape
    y = torch.einsum("bjid,jk->bkid", s.double().view(b, 2 * k, w // 2, d), ki.double())
    return y.reshape(b, k, H, d)


def geometric_window(s):
    """The same gather reconstructed from geometry alone: pad by 48, slice, concat.

    A transcription of the shape-gated construction, kept here so the test owns the thing it
    is discriminating against and does not depend on a branch.
    """
    b, k, w, d = s.shape
    flat = s.reshape(b, k * w, d)
    padded = torch.nn.functional.pad(flat, (0, 0, SHIFT, (k + H // w) * w - SHIFT - k * w))
    blocks = padded.view(b, k + H // w, w, d)
    return torch.cat([blocks[:, c:c + k] for c in range(H // w)], dim=2).double()


def live_tail_atoms(windows_pad=WINDOWS_PAD, dim=8, seed=0):
    """An atom sequence with values in the bucket's padded tail.

    Inside the diffusion transformer the pad rows are never zero: they go through linears with a
    bias and through layer norms before they reach this gather. A fixture that zeroes them tests
    a tensor the model does not have.
    """
    torch.manual_seed(seed)
    return torch.randn(1, windows_pad, W, dim)


def test_cached_matrix_carries_a_zero_tail_no_shape_expresses():
    ki = cached_matrix()
    live = (ki != 0).any(dim=0).view(WINDOWS_PAD, 8).any(dim=1)
    assert int(live.sum()) < WINDOWS_PAD, "no zero tail: this fixture cannot discriminate"
    assert bool(live[:int(live.sum())].all()), "the live windows must be a prefix"
    # Two matrices of the same shape, gathering for a different number of windows.
    assert cached_matrix(WINDOWS_PAD, WINDOWS_PAD).shape == ki.shape


def test_geometry_only_gather_is_wrong_under_a_bucketed_matrix():
    """The discriminating fixture. Windows `windows-2` and up pick up atoms the matrix zeroes."""
    s = live_tail_atoms()
    ki = cached_matrix()
    diff = (geometric_window(s) - matrix_gather(s, ki)).abs().amax(dim=(0, 2, 3))
    bad = (diff > 0).nonzero().flatten().tolist()
    assert bad == list(range(WINDOWS_REAL - 2, WINDOWS_PAD)), bad
    assert float(diff.max()) > 1.0


def test_the_same_construction_is_bit_exact_when_the_matrix_has_no_tail():
    """Why it shipped: build the matrix at the PADDED count and the wrong path reads 0.0."""
    s = live_tail_atoms()
    ki = cached_matrix(WINDOWS_PAD, WINDOWS_PAD)
    assert torch.equal(geometric_window(s), matrix_gather(s, ki))


def test_zeroing_the_atom_tail_does_not_hide_it_either():
    """A zeroed-pad fixture still misses 11 of the 13 windows: it is not the missing control."""
    s = live_tail_atoms()
    s.view(1, -1, s.shape[-1])[:, WINDOWS_REAL * W:, :] = 0.0
    diff = (geometric_window(s) - matrix_gather(s, cached_matrix())).abs().amax(dim=(0, 2, 3))
    assert (diff > 0).any()


def test_atom_bias_masks_every_slot_the_matrix_gathers_nothing_for():
    """What actually protects the call site today, stated as an invariant rather than assumed.

    `_populate_diffusion_cache` gathers the atom mask THROUGH the same matrix and turns every
    zero into -1e9, so a slot the matrix zeroes is a slot with no attention weight. That, not
    the fold's final slice, is why a wrong gather has not changed a structure. It holds only
    while the bias is built from this matrix; build it any other way and the gather's
    correctness is load-bearing again.
    """
    ki = cached_matrix()
    atom_mask = torch.zeros(1, WINDOWS_PAD, W, 1)
    atom_mask[:, :WINDOWS_REAL] = 1.0
    gathered = matrix_gather(atom_mask, ki).reshape(1, WINDOWS_PAD, H)
    dead_column = ~(ki != 0).any(dim=0)                       # slots the matrix gathers nothing for
    dead_slot = dead_column.view(WINDOWS_PAD, 8, 1).expand(WINDOWS_PAD, 8, W // 2)
    assert not gathered.reshape(WINDOWS_PAD, 8, W // 2).bool()[dead_slot].any()


def test_registered_gather_matches_the_matrix():
    """Whatever gather decision the tree exposes on the host must reproduce the matrix.

    Skips while the tree has none, which is the state of `main`. The moment an elision lands
    with a host-side window count, this scores it on the fixture above.
    """
    ttnn = pytest.importorskip("ttnn")                        # the module imports it at top level
    del ttnn
    import tt_bio.tenstorrent as T
    decide = getattr(T, "_atom_gather_shift_windows", None)
    if decide is None:
        pytest.skip("no host-side atom gather decision in this tree")
    ki = cached_matrix()
    windows = decide(ki)
    assert windows is not None, "the decision declined the shipped production matrix"
    live = (ki != 0).any(dim=0).view(WINDOWS_PAD, 8).any(dim=1)
    assert windows >= int(live.sum()), (windows, int(live.sum()))
    assert windows < WINDOWS_PAD, "a decision that reports the padded count cuts nothing"


@pytest.mark.device
def test_device_atom_gather_reproduces_the_matrix():
    """Score every DEVICE-side atom-gather elision in this tree on the discriminating fixture.

    The host test above cannot reach a construction that takes ttnn tensors and decides from
    their shapes, which is the exact shape of the defect: on `wk/b2z2-akw-ship` there is no host
    decision to score and the file skips. This is the leg the release gate was missing.

    Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> python3 -m pytest
    tests/test_atom_gather_operand.py -k device
    """
    ttnn = pytest.importorskip("ttnn")
    import tt_bio.tenstorrent as T

    elisions = {}
    if hasattr(T, "_atom_key_window"):
        elisions["_atom_key_window"] = lambda s_, ki_, ki_pt_: T._atom_key_window(s_, ki_)
    if hasattr(T, "_atom_shift_gather") and hasattr(T, "_atom_gather_shift_windows"):
        def _shift(s_, ki_, ki_pt_):
            windows = T._atom_gather_shift_windows(ki_pt_)
            return None if windows is None else T._atom_shift_gather(s_, windows)
        elisions["_atom_shift_gather"] = _shift
    if not elisions:
        pytest.skip("no device-side atom gather elision in this tree")

    s_pt = live_tail_atoms(dim=128).to(torch.bfloat16).float()
    ki_pt = cached_matrix()
    ref = matrix_gather(s_pt, ki_pt).float()

    dev = T.get_device()
    s = ttnn.from_torch(s_pt, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ki = ttnn.from_torch(ki_pt, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)
    for name, fn in elisions.items():
        out = fn(s, ki, ki_pt)
        if out is None:
            continue                      # declined this geometry: the caller keeps the matmul
        got = ttnn.to_torch(out).float()
        per_window = (got - ref).abs().amax(dim=(0, 2, 3))
        wrong = (per_window > 0).nonzero().flatten().tolist()
        assert not wrong, (
            f"{name} does not reproduce the cached matrix: max abs "
            f"{float(per_window.max())}, {len(wrong)} wrong windows, first {wrong[0]}")
