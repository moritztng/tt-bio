"""`ConfidenceHead.confidence_samples` returns, per sample, exactly what the per-sample host path did.

The device is stubbed: the pairformer is a fixed torch function of its inputs, so what is compared
is the host side that the overlap rearranged (z build, read-back order, heads). Runs on CPU.
"""
import torch
import torch.nn.functional as F

import tt_bio.protenix as P


class _Dev:
    """A device tensor stand-in: holds a bf16 torch tensor, records frees."""
    live = 0

    def __init__(self, t):
        self.t = t
        _Dev.live += 1


def _stub(monkeypatch):
    def from_torch(x, **kw):
        return _Dev(x.to(torch.bfloat16))

    def to_torch(d):
        return d.t

    def deallocate(d):
        _Dev.live -= 1

    def pairformer(pf, s, z, dev):
        return _Dev(torch.tanh(s.t.float()) * 2), _Dev((z.t.float() * 0.5 + 1).to(torch.bfloat16))

    monkeypatch.setattr(P.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(P.ttnn, "to_torch", to_torch)
    monkeypatch.setattr(P.ttnn, "deallocate", deallocate)
    monkeypatch.setattr(P, "bucketed_pairformer", pairformer)
    return pairformer


def _head(N, n_atom, c_s=449, c_z=32, nb=50, g=None):
    g = g or torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    h = P.ConfidenceHead.__new__(P.ConfidenceHead)
    h.dev, h.pf = None, None
    h._w = {
        "input_strunk_ln.weight": r(384), "input_strunk_ln.bias": r(384),
        "linear_no_bias_s1.weight": r(c_z, c_s), "linear_no_bias_s2.weight": r(c_z, c_s),
        "lower_bins": torch.linspace(3.25, 50.75, 39), "upper_bins": torch.cat([torch.linspace(4.5, 50.75, 38), torch.tensor([1e9])]),
        "linear_no_bias_d.weight": r(c_z, 39), "linear_no_bias_d_wo_onehot.weight": r(c_z, 1),
        "pae_ln.weight": r(c_z), "pae_ln.bias": r(c_z), "linear_no_bias_pae.weight": r(64, c_z),
        "pde_ln.weight": r(c_z), "pde_ln.bias": r(c_z), "linear_no_bias_pde.weight": r(64, c_z),
        "plddt_ln.weight": r(384), "plddt_ln.bias": r(384), "plddt_weight": r(24, 384, nb),
    }
    return h


def _reference(h, pairformer, s_inputs, s_trunk, z_trunk, coords, feats):
    """The per-sample host path as it was before `confidence_samples`."""
    g, b = h._g, h._bias
    N = s_trunk.shape[0]
    s_t = F.layer_norm(torch.clamp(s_trunk, -512, 512), (384,)) * g("input_strunk_ln.weight") + b("input_strunk_ln.bias")
    z = (z_trunk + F.linear(s_inputs, g("linear_no_bias_s1.weight")).unsqueeze(1)
         + F.linear(s_inputs, g("linear_no_bias_s2.weight")).unsqueeze(0))
    xr = coords.reshape(-1, 3)[feats["distogram_rep_atom_mask"].bool()]
    d = torch.cdist(xr, xr)
    oh = ((d.unsqueeze(-1) >= g("lower_bins")) & (d.unsqueeze(-1) < g("upper_bins"))).float()
    z = z + F.linear(oh, g("linear_no_bias_d.weight")) + F.linear(d.unsqueeze(-1), g("linear_no_bias_d_wo_onehot.weight"))
    T = lambda x: _Dev(x.float().to(torch.bfloat16))
    so, zo = pairformer(None, T(s_t.unsqueeze(0)), T(z.unsqueeze(0)), None)
    s_single = so.t.float().reshape(N, 384)
    zf = zo.t.float().reshape(N, N, -1)
    pae = F.linear(F.layer_norm(zf, (zf.shape[-1],)) * g("pae_ln.weight") + b("pae_ln.bias"), g("linear_no_bias_pae.weight"))
    pde = F.linear(F.layer_norm(zf + zf.transpose(0, 1), (zf.shape[-1],)) * g("pde_ln.weight") + b("pde_ln.bias"),
                   g("linear_no_bias_pde.weight"))
    a = s_single[feats["atom_to_token_idx"].long()]
    aln = F.layer_norm(a, (384,)) * g("plddt_ln.weight") + b("plddt_ln.bias")
    pl = torch.einsum("nc,ncb->nb", aln, g("plddt_weight")[feats["atom_to_tokatom_idx"].long()])
    return h._postprocess(pae, pde, pl, feats)


def test_confidence_samples_matches_per_sample_path(monkeypatch):
    pairformer = _stub(monkeypatch)
    torch.manual_seed(1)
    N, n_sample = 24, 4
    n_atom = 3 * N
    h = _head(N, n_atom)
    a2t = torch.arange(N).repeat_interleave(3)
    mask = (torch.arange(n_atom) % 3 == 1).float()                 # one representative atom per token
    feats = {"distogram_rep_atom_mask": mask, "atom_to_token_idx": a2t,
             "atom_to_tokatom_idx": torch.randint(0, 24, (n_atom,)),
             "asym_id": (torch.arange(N) >= N // 2).long()}
    s_inputs, s_trunk, z_trunk = torch.randn(N, 449), torch.randn(N, 384), torch.randn(N, N, 32)
    coords = torch.randn(n_sample, n_atom, 3) * 10
    _Dev.live = 0
    got = h.confidence_samples(s_inputs, s_trunk, z_trunk, list(coords), feats)
    assert len(got) == n_sample
    for k in range(n_sample):
        want = _reference(h, pairformer, s_inputs, s_trunk, z_trunk, coords[k], feats)
        assert want.keys() == got[k].keys()
        for key, v in want.items():
            if torch.is_tensor(v):
                assert torch.equal(v, got[k][key]), (k, key)
            else:
                assert v == got[k][key], (k, key)
    one = h.confidence(s_inputs, s_trunk, z_trunk, coords[0], feats)
    assert torch.equal(one["pae"], got[0]["pae"])
