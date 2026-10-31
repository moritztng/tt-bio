"""tt_bio.protenix_output vs Mol-1's own confidence summary and dumper. CPU only.

The golden (tests/data/mol1_io/confidence_golden.pt) is Mol-1 e6e64866's compute_full_data_and_summary on seeded
synthetic logits, made by scripts/mol1_io/confidence_goldens.py. Floats to 1e-5 (float32 reductions in the same
order), booleans and integers exact."""
import json
from pathlib import Path

import pytest
import torch

from tt_bio import protenix_output as po

GOLD = Path(__file__).parent / "data" / "mol1_io" / "confidence_golden.pt"


@pytest.mark.skipif(not GOLD.exists(), reason="confidence golden not generated yet")
@pytest.mark.parametrize("name", ["monomer", "dimer_ligand", "clash"])
def test_summary_matches_upstream(name):
    c = torch.load(GOLD, weights_only=False)[name]
    assert torch.allclose(po.contact_probs(c["distogram_logits"]), c["contact_probs"], atol=1e-6)
    sm, fd = po.summarize(c["pae_logits"], c["plddt_logits"], c["pde_logits"], c["contact_probs"],
                          c["token_asym_id"], c["token_has_frame"], c["atom_coordinate"], c["atom_to_token_idx"],
                          c["atom_is_polymer"], c["N_recycle"])
    for mine, ref in zip(sm, c["summary"]):
        assert set(mine) == set(ref)
        for k in ref:
            assert torch.allclose(torch.as_tensor(mine[k]).float(), torch.as_tensor(ref[k]).float(), atol=1e-5), k
    for mine, ref in zip(fd, c["full"]):
        for k in ("atom_plddt", "token_pair_pae", "token_pair_pde", "contact_probs"):
            assert torch.allclose(mine[k], ref[k], atol=1e-5), k
    if name == "clash":
        assert bool(sm[0]["has_clash"]) and not bool(sm[1]["has_clash"])


def _toy(n_s=3, chains=(4, 3)):
    g = torch.Generator().manual_seed(1)
    asym = torch.cat([torch.full((n,), i) for i, n in enumerate(chains)])
    nt = int(asym.numel())
    return dict(pae_logits=torch.randn(n_s, nt, nt, 64, generator=g), plddt_logits=torch.randn(n_s, nt, 50, generator=g),
                pde_logits=torch.randn(n_s, nt, nt, 64, generator=g), contact=torch.rand(nt, nt, generator=g),
                token_asym_id=asym, token_has_frame=torch.ones(nt, dtype=torch.long),
                coords=torch.randn(n_s, nt, 3, generator=g) * 20, atom_to_token_idx=torch.arange(nt),
                atom_is_polymer=torch.ones(nt, dtype=torch.long), n_recycle=20)


def test_contract_invariants():
    sm, _ = po.summarize(**_toy())
    for s in sm:
        assert s["chain_pair_iptm"].diagonal().abs().max() == 0 and s["chain_pair_gpde"].diagonal().abs().max() == 0
        assert 0 <= float(s["plddt"]) <= 100 and float(s["chain_plddt"].max()) <= 1
        assert torch.isclose(s["ranking_score"], 0.8 * s["iptm"] + 0.2 * s["ptm"] - 100 * s["has_clash"])
    mono, _ = po.summarize(**_toy(chains=(6,)))
    assert all(float(s["iptm"]) == 0 for s in mono)


def test_layout_and_ranking(tmp_path):
    sm, fd = po.summarize(**_toy())
    seen = []
    ranks = po.write(tmp_path, "job", 101, sm, fd, write_structure=lambda i, p, bf: seen.append((i, p.name, bf.max())),
                     need_atom_confidence=True)
    d = tmp_path / "job" / "seed_101" / "predictions"
    best = max(range(len(sm)), key=lambda i: float(sm[i]["ranking_score"]))
    assert ranks[best] == 0 and sorted(ranks) == [0, 1, 2]
    s0 = json.loads((d / "job_summary_confidence_sample_0.json").read_text())
    assert s0["ranking_score"] == pytest.approx(float(sm[best]["ranking_score"]))
    assert isinstance(s0["has_clash"], bool) and s0["num_recycles"] == 20
    full = json.loads((d / "job_full_data_sample_0.json").read_text())
    assert {"token_pair_pae", "token_pair_pde", "contact_probs", "atom_plddt", "atom_to_token_idx",
            "token_asym_id", "token_has_frame"} == set(full)
    assert all(abs(v - round(v, 2)) < 1e-6 for v in full["atom_plddt"])   # float32 rounded, as upstream
    assert sorted(n for _i, n, _b in seen) == [f"job_sample_{k}.cif" for k in range(3)]


class _LogitStub:
    """Protenix.fold with the network replaced: random coordinates and the confidence logits the engine hands the
    upstream writer (per-sample pae/pde/plddt logits, the fold's contact probabilities)."""

    def fold(self, feats, n_sample=1, return_logits=False, **_kw):
        g = torch.Generator().manual_seed(0)
        na, nt = feats["atom_to_token_idx"].shape[0], feats["restype"].shape[0]
        contact = torch.rand(nt, nt, generator=g)
        confs = [{"plddt": 0.5, "plddt_atom": torch.zeros(na), "pae": torch.zeros(nt, nt), "pde": torch.zeros(nt, nt)}
                 for _ in range(n_sample)]
        if return_logits:   # the engine only keeps them when the worker asks
            for c in confs:
                c.update(pae_logits=torch.randn(nt, nt, 64, generator=g), pde_logits=torch.randn(nt, nt, 64, generator=g),
                         plddt_logits=torch.randn(na, 50, generator=g), contact_probs=contact)
        return [torch.randn(na, 3, generator=g) * 10 for _ in range(n_sample)], confs


def test_a_mol1_fold_writes_upstreams_layout(tmp_path):
    from tt_bio import protenix_json
    from tt_bio.worker import _WorkerState

    job = {"name": "dimer", "sequences": [{"proteinChain": {"sequence": "MKTAYIAKQR", "count": 2}},
                                          {"ligand": {"ligand": "CCD_ATP", "count": 1}}]}
    src = tmp_path / "jobs.json"
    src.write_text(json.dumps([job]))
    yaml = sorted(protenix_json.stage(src, tmp_path / "staged").glob("dimer*.yaml"))[0]
    state = object.__new__(_WorkerState)
    state.model, state.accelerator = _LogitStub(), "cpu"
    res = tmp_path / "mol1_results_jobs"
    (res / "structures").mkdir(parents=True)
    cfg = {"model": "mol1", "msa_dir": str(tmp_path), "struct_dir": str(res / "structures"), "output_format": "cif",
           "recycling_steps": 20, "sampling_steps": 1, "diffusion_samples": 3, "single_sequence": True,
           "template_structures": str(tmp_path), "write_pae": True}
    state.predict_one(yaml, cfg)
    pred = res / "dimer" / "seed_101" / "predictions"
    names = {p.name for p in pred.iterdir()}
    assert names == {f"dimer_{part}sample_{k}.{ext}" for k in range(3)
                     for part, ext in (("", "cif"), ("summary_confidence_", "json"), ("full_data_", "json"))}
    s = [json.loads((pred / f"dimer_summary_confidence_sample_{k}.json").read_text()) for k in range(3)]
    assert [x["ranking_score"] for x in s] == sorted((x["ranking_score"] for x in s), reverse=True)
    assert s[0]["num_recycles"] == 20 and len(s[0]["chain_plddt"]) == 3 and 0 <= s[0]["plddt"] <= 100


def test_engine_heads_hand_the_writer_its_logits():
    """`_postprocess(keep_logits=True)` returns the logits it was given, and `distogram_logits` is upstream's
    DistogramHead (linear on z, plus its token-axis transpose)."""
    from types import SimpleNamespace

    from tt_bio.protenix import ConfidenceHead, Protenix

    g = torch.Generator().manual_seed(1)
    n, na = 6, 9
    feats = {"asym_id": torch.tensor([0, 0, 0, 1, 1, 1])}
    pae, pde, pl = torch.randn(n, n, 64, generator=g), torch.randn(n, n, 64, generator=g), torch.randn(na, 50, generator=g)
    head = object.__new__(ConfidenceHead)
    assert "pae_logits" not in head._postprocess(pae, pde, pl, feats)
    out = head._postprocess(pae, pde, pl, feats, keep_logits=True)
    assert all(torch.equal(out[k], v) for k, v in (("pae_logits", pae), ("pde_logits", pde), ("plddt_logits", pl)))
    z = torch.randn(n, n, 16, generator=g, dtype=torch.float64)
    w, b = torch.randn(64, 16, generator=g, dtype=torch.float64), torch.randn(64, generator=g, dtype=torch.float64)
    d = Protenix.distogram_logits(SimpleNamespace(_w={"distogram_head.linear.weight": w,
                                                      "distogram_head.linear.bias": b}), z)
    ref = z @ w.T + b
    assert torch.allclose(d.double(), ref + ref.transpose(0, 1), atol=1e-4)
