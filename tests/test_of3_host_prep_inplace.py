"""The host atom-pair legs (``ref_atom_embed``'s plm and ``atom_pair_completion``) are
computed in place for speed; they must equal the textbook three-term and MLP forms exactly."""
import pytest
import torch

from tt_bio import openfold3_host_prep as H


def _features(n_atom=300, n_pad=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = n_atom + n_pad
    mask = torch.ones(n)
    mask[n_atom:] = 0
    return {
        "ref_pos": torch.randn(n, 3, generator=g) * 5 * mask[:, None],
        "ref_charge": torch.randint(-1, 2, (n,), generator=g).float(),
        "ref_mask": mask.int(),
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, 119, (n,), generator=g), 119).int(),
        "ref_atom_name_chars": torch.nn.functional.one_hot(
            torch.randint(0, 64, (n, 4), generator=g), 64).int(),
        "ref_space_uid": (torch.arange(n) // 9).int(),
        "atom_mask": mask,
    }


def _weights(seed=1, bias=False):
    g = torch.Generator().manual_seed(seed)
    shapes = dict(linear_ref_pos=(128, 3), linear_ref_charge=(128, 1), linear_ref_mask=(128, 1),
                  linear_ref_element=(128, 119), linear_ref_atom_chars=(128, 256),
                  linear_ref_offset=(16, 3), linear_inv_sq_dists=(16, 1), linear_valid_mask=(16, 1))
    w = {f"{k}.weight": torch.randn(*s, generator=g) for k, s in shapes.items()}
    if bias:
        w |= {f"{k}.bias": torch.randn(s[0], generator=g) for k, s in shapes.items()}
    enc = {"linear_l.weight": torch.randn(16, 128, generator=g),
           "linear_m.weight": torch.randn(16, 128, generator=g)}
    enc |= {f"pair_mlp.{i}.weight": torch.randn(16, 16, generator=g) for i in (1, 3, 5)}
    return w, enc


def _plm_reference(w, f):
    dlm, vlm, inv = H.ref_atom_block_inputs(f, f["atom_mask"].float())
    return (H._lin(dlm, w, "linear_ref_offset") * vlm
            + H._lin(inv, w, "linear_inv_sq_dists") * vlm
            + H._lin(vlm, w, "linear_valid_mask") * vlm)


def _completion_reference(enc, cl, plm, atom_mask):
    cl_l, cl_m, pair_mask = H.convert_single_rep_to_blocks(
        ql=cl, n_query=H.N_QUERY, n_key=H.N_KEY, atom_mask=atom_mask)
    cl_lm = (H._lin(cl_l.relu().unsqueeze(-2), enc, "linear_l")
             + H._lin(cl_m.relu().unsqueeze(-3), enc, "linear_m")) * pair_mask.unsqueeze(-1)
    plm = plm + cl_lm
    h = plm
    for key in ("pair_mlp.1", "pair_mlp.3", "pair_mlp.5"):
        h = H._lin(h.relu(), enc, key)
    return (plm + h) * pair_mask.unsqueeze(-1)


@pytest.mark.parametrize("bias", [False, True])
def test_ref_atom_embed_plm_is_the_three_term_sum(bias):
    f = _features()
    w, _ = _weights(bias=bias)
    _, plm = H.ref_atom_embed(w, f)
    ref = _plm_reference(w, f)
    assert plm.shape == ref.shape and plm.dtype == ref.dtype
    assert torch.equal(plm, ref)


def test_atom_pair_completion_is_the_mlp_form():
    f = _features()
    w, enc = _weights()
    cl, plm = H.ref_atom_embed(w, f)
    keep = plm.clone()
    out = H.atom_pair_completion(enc, cl, plm, f["atom_mask"])
    assert torch.equal(plm, keep), "plm must not be modified"
    ref = _completion_reference(enc, cl, keep, f["atom_mask"])
    assert out.shape == ref.shape and torch.equal(out, ref)
