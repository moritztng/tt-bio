"""RF3's YAML door drops what its chain reader does not return, so it says so.

`_predict_rf3_one` is the only RF3 spec builder and it constructs every component from
`_read_bio_chains`, which returns (chain_id, sequence, msa_spec, mol_type). A
`constraints:`, `modifications:` or `cyclic:` block therefore never reaches the
featurizer. It used to be accepted, dropped and never mentioned — and a dropped
covalent bond changes the answer rather than omitting an output, so these refuse
rather than warn.

RF3 the model does carry all three; they reach it through its own JSON/CIF spec, which
`featurize(src)` reads straight off disk. That path is not what tt-bio's YAML builds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tt_bio.worker import _validate_cyclic_unsupported, _validate_rf3_yaml_unsupported

SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
PLAIN = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {SEQ}\n"
CONSTRAINTS = (f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {SEQ}\n"
               "  - ligand:\n      id: B\n      ccd: SAH\n"
               "constraints:\n  - bond:\n      atom1: [A, 5, SG]\n      atom2: [B, 1, C]\n")
MODIFICATIONS = (f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {SEQ}\n"
                 "      modifications:\n        - position: 5\n          ccd: TPO\n")
CYCLIC = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {SEQ}\n      cyclic: true\n"


def _write(tmp_path, text, name="in.yaml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _check(path):
    _validate_rf3_yaml_unsupported(path)
    _validate_cyclic_unsupported(path, "rf3")


@pytest.mark.parametrize("content,needle", [
    (CONSTRAINTS, "constraints"),
    (MODIFICATIONS, "modifications"),
    (CYCLIC, "cyclic"),
])
def test_a_block_rf3_cannot_read_is_refused(tmp_path, content, needle):
    with pytest.raises(RuntimeError) as e:
        _check(_write(tmp_path, content))
    msg = str(e.value)
    assert "rf3" in msg and needle in msg
    assert "boltz2" in msg or "JSON spec" in msg      # somewhere else to go


def test_a_plain_input_is_untouched(tmp_path):
    """The control: without this the refusals above would pass on a check that
    refuses everything."""
    _check(_write(tmp_path, PLAIN))


def test_the_spec_builder_actually_runs_them(tmp_path):
    """A validator nothing calls is not a guard."""
    import inspect

    from tt_bio.worker import _WorkerState

    src = inspect.getsource(_WorkerState._predict_rf3_one)
    assert "_validate_rf3_yaml_unsupported(path)" in src
    assert '_validate_cyclic_unsupported(path, "rf3")' in src
