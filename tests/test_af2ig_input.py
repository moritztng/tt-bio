"""The af2ig route from a submission to the trunk, with no card and no 4 GB of parameters.

What is card-bound here is the trunk, and the trunk is already covered
(tests/test_af2_device_floor.py against docs/implementation-parity-data/af2ig-trunk-device.json).
What was missing was everything either side of it: the reader that turns a request into
`complex_features`' arguments, and the writer that turns atom37 back into a file. Both run on
host, so both are tested here, against the real featurizer -- a stub model stands in for the
trunk alone, and its output shapes are the ones `af2_reference.AF2Model` returns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tt_bio import af2ig  # noqa: E402
from tt_bio.af2_data import NUM_ATOM  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "af2ig" / "designed_complex.pdb"
EXAMPLE = REPO / "examples" / "af2_designed_complex.yaml"
BINDER = "LYRWIKSVDPSRPVQY"


def _yaml(tmp_path, body: str, name="job.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _job(tmp_path, *, sequence: str = BINDER, target: str | None = None) -> Path:
    target = target or f'  file: "{FIXTURE}"\n'
    return _yaml(tmp_path, f"target:\n{target}  chain: A\nbinder:\n  sequence: {sequence}\n"
                           f"  chain: B\n")


def test_the_shipped_example_reads(tmp_path):
    """The example is the copy-pasteable form of the request, so it is a test, not a doc."""
    spec = af2ig.read_af2ig_input(EXAMPLE)
    assert spec.binder_sequence == BINDER
    assert spec.target_chain == "A" and spec.binder_chain == "B"
    assert spec.structure.startswith("ATOM")


def test_an_inline_structure_and_a_file_give_the_same_submission(tmp_path):
    inline = af2ig.read_af2ig_input(EXAMPLE)
    onfile = af2ig.read_af2ig_input(_job(tmp_path))
    assert inline == onfile


def test_mmcif_is_accepted_because_that_is_what_a_design_job_returns(tmp_path):
    """`tt-bio design` writes CIFs. Refusing mmCIF here would break design -> score, which is
    the whole reason this model is exposed."""
    import io

    import biotite.structure.io.pdb as _pdb
    import biotite.structure.io.pdbx as _pdbx

    arr = _pdb.PDBFile.read(str(FIXTURE)).get_structure(model=1)
    cf = _pdbx.CIFFile()
    _pdbx.set_structure(cf, arr)
    buf = io.StringIO()
    cf.write(buf)
    spec = af2ig.read_af2ig_input(_job(tmp_path, target=f"  structure: |\n"
                                       + "".join(f"    {l}\n" for l in buf.getvalue().splitlines())))
    assert af2ig.token_count(spec) == af2ig.token_count(af2ig.read_af2ig_input(_job(tmp_path)))


@pytest.mark.parametrize("body,expect", [
    ("version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MKT\n",
     "target"),
    ('target:\n  file: "/nonexistent/t.pdb"\nbinder:\n  sequence: MKT\n', "does not exist"),
    ("target:\n  chain: A\nbinder:\n  sequence: MKT\n", "structure"),
    ("binder:\n  sequence: MKT\ntarget:\n  structure: |\n    ATOM\n", "chain id"),
])
def test_a_malformed_submission_is_a_sentence(tmp_path, body, expect):
    """Every one of these reaches a user before any weights load (main._refuse_unfoldable_jobs),
    so each has to say what to do, not where it crashed."""
    if expect == "chain id":
        body = body.replace("sequence: MKT", "sequence: MKT\n  chain: BB")
    with pytest.raises(ValueError, match=expect):
        af2ig.read_af2ig_input(_yaml(tmp_path, body))


def test_a_boltz2_yaml_names_the_input_af2ig_actually_wants(tmp_path):
    """The likeliest wrong request: a chain list, because that is what every other model takes."""
    with pytest.raises(ValueError) as e:
        af2ig.read_af2ig_input(_yaml(tmp_path, "version: 1\nsequences:\n  - protein:\n"
                                               "      id: A\n      sequence: MKTAYIAKQRQ\n"))
    msg = str(e.value)
    assert "structure" in msg and "binder" in msg and "cofolders" in msg


def test_a_binder_sequence_that_does_not_fit_its_backbone_is_refused(tmp_path):
    """AF2-IG places the sequence on the backbone it is handed, so a length mismatch is not a
    detail the featurizer should discover -- and the message must not leak a temp path."""
    spec = af2ig.read_af2ig_input(_job(tmp_path, sequence=BINDER[:-1]))
    with pytest.raises(ValueError) as e:
        af2ig.features(spec)
    assert "15 residues" in str(e.value) and "/tmp" not in str(e.value)


def test_non_standard_residues_are_refused(tmp_path):
    with pytest.raises(ValueError, match="non-standard"):
        af2ig.read_af2ig_input(_job(tmp_path, sequence=BINDER[:-1] + "X"))


def test_the_features_are_the_binder_protocol(tmp_path):
    """Not a re-test of complex_features (tests/test_af2_data.py owns that): the check is that
    the reader hands it the right two chains in the right order."""
    spec = af2ig.read_af2ig_input(EXAMPLE)
    feats = af2ig.features(spec)
    asym = feats["asym_id"]
    assert len(asym) == 48 and int((asym == 0).sum()) == 32
    assert feats["template_mask"].tolist() == [1.0]          # the initial guess is a template
    assert feats["msa_feat"].shape[0] == 1                   # single sequence, both chains


class _StubTrunk:
    """The trunk's output contract, and nothing else: enough to drive fold() on host.

    Each pass moves the previous pass's coordinates by a fixed offset, so the written
    structure reads back as OFFSET x the number of passes only if `prev` really was threaded.
    """

    OFFSET = 1.25

    def __init__(self, tokens: int, bins: int = 64):
        self.tokens, self.bins = tokens, bins
        self.calls = 0

    def __call__(self, feats, prev):
        self.calls += 1
        n = self.tokens
        logits = torch.zeros(n, 50)
        logits[:, 10] = 10.0                       # a sharp pLDDT peak in one bin
        pae = torch.zeros(n, n, self.bins)
        pae[..., 1] = 10.0
        return {"msa_first_row": torch.zeros(n, 256), "pair": torch.zeros(n, n, 128),
                "plddt_logits": logits, "pae_logits": pae,
                "pae_breaks": torch.linspace(0.0, 31.0, self.bins - 1),
                "structure": {"final_atom_positions":
                              prev["prev_pos"] + self.OFFSET,
                              "final_atom_mask": feats["atom37_atom_exists"]}}


def test_fold_threads_the_recycling_state_and_writes_the_complex(tmp_path):
    spec = af2ig.read_af2ig_input(EXAMPLE)
    feats = af2ig.features(spec)
    model = _StubTrunk(len(feats["residue_index"]))
    pred = af2ig.fold(model, spec, recycles=3)

    assert model.calls == 4, "recycles=3 is four forward passes, as af2_reference.run_recycles"
    # The last pass's coordinates are what is written, not the first.
    moved = feats["batch/all_atom_positions"] + _StubTrunk.OFFSET * 4
    arr = pred.atom_array
    assert pred.tokens == 48 and pred.binder_length == 16
    assert set(arr.chain_id) == {"A", "B"}
    assert int((arr.chain_id == "B").sum()) == int(
        (feats["atom37_atom_exists"][32:] > 0).sum())
    # Each written atom carries the coordinate of its own atom37 slot.
    exists = feats["atom37_atom_exists"] > 0
    expect = moved[exists]
    assert np.allclose(pred.coords, expect.astype(np.float32), atol=1e-5)
    # pLDDT on the PDB's 0-100 scale, one value per residue repeated over its atoms.
    assert pred.b_factors.min() > 0 and pred.b_factors.max() <= 100.0
    # The metric is rounded to 4 decimals for the results row; the column is not.
    assert pytest.approx(pred.metrics["plddt"] * 100, abs=5e-3) == float(pred.b_factors[0])


def test_the_written_numbering_is_the_one_the_input_carried(tmp_path):
    """The featurizer jumps residue_index by +50 at the chain break so the trunk sees two
    chains. A user reading the output back should see their own numbering."""
    spec = af2ig.read_af2ig_input(EXAMPLE)
    feats = af2ig.features(spec)
    pred = af2ig.fold(_StubTrunk(len(feats["residue_index"])), spec, recycles=0)
    arr = pred.atom_array
    assert sorted(set(arr.res_id[arr.chain_id == "A"])) == list(range(1, 33))
    assert sorted(set(arr.res_id[arr.chain_id == "B"])) == list(range(1, 17))


def test_the_metrics_are_the_interface_ones_a_design_is_judged_on():
    spec = af2ig.read_af2ig_input(EXAMPLE)
    pred = af2ig.fold(_StubTrunk(48), spec, recycles=0)
    assert set(pred.metrics) == {"plddt", "ptm", "iptm", "pae", "ipae", "interface_pae"}
    # interface_pae is the Angstrom reading of the normalised ipae, not a second measurement.
    from tt_bio.af2_confidence import PAE_MAX_ERROR_BIN
    assert pytest.approx(pred.metrics["ipae"] * PAE_MAX_ERROR_BIN, abs=1e-3) == \
        pred.metrics["interface_pae"]


def test_atom37_slots_that_do_not_exist_are_not_written():
    """A glycine has no CB and an alanine no OG: writing all 37 slots would put atoms at the
    origin for every residue and read as a wrecked structure."""
    spec = af2ig.read_af2ig_input(EXAMPLE)
    feats = af2ig.features(spec)
    pred = af2ig.fold(_StubTrunk(48), spec, recycles=0)
    assert pred.atom_array.array_length() < 48 * NUM_ATOM
    assert pred.atom_array.array_length() == int((feats["atom37_atom_exists"] > 0).sum())


# Merge condition of ask 10626. af2_data.initial_recycle_state(initial_guess=False) starts
# prev_pos at zero, and af2ig run that way is plain single-sequence AF2 under the af2ig name.
# These tests fail if a YAML key, a route parameter or a CLI option can reach that branch.

@pytest.mark.parametrize("where", ["top", "target", "binder"])
def test_initial_guess_is_not_a_yaml_key(tmp_path, where):
    body = EXAMPLE.read_text()
    if where == "top":
        body += "initial_guess: false\n"
    else:
        body = body.replace(f"\n{where}:\n", f"\n{where}:\n  initial_guess: false\n", 1)
    with pytest.raises(ValueError, match="initial_guess"):
        af2ig.read_af2ig_input(_yaml(tmp_path, body))


def test_no_route_function_or_cli_option_names_the_initial_guess():
    import dataclasses
    import inspect

    from tt_bio.main import cli

    params = [f.name for f in dataclasses.fields(af2ig.AF2IGInput)]
    for fn in (af2ig.read_af2ig_input, af2ig.features, af2ig.fold):
        params += list(inspect.signature(fn).parameters)
    params += [o for p in cli.commands["predict"].params for o in p.opts]
    assert not [p for p in params if "guess" in p.lower()]


def test_the_first_pass_starts_from_the_design(tmp_path):
    """The behaviour the name promises, read off what the trunk is actually handed."""
    spec = af2ig.read_af2ig_input(EXAMPLE)
    feats = af2ig.features(spec)
    seen = []

    class _Recording(_StubTrunk):
        def __call__(self, f, prev):
            seen.append(prev["prev_pos"].clone())
            return super().__call__(f, prev)

    af2ig.fold(_Recording(len(feats["residue_index"])), spec, recycles=0)
    design = torch.from_numpy(feats["batch/all_atom_positions"].astype(np.float32))
    assert design.abs().sum() > 0 and torch.equal(seen[0], design)
