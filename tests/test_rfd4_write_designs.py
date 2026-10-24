"""`write_designs` writes a batch on worker processes and the files match a serial `write_design`;
`writer_count` sizes the pool by the batch, a cap, the CPUs and the memory available.

Card-free. The file test needs the PD-L1 benchmark spec (`RFD4_SPEC`, default ~/rfd4-inputs/pdl1-128.yaml) and stands in an engine
output: residue types legal per token, the decoder's atom37 table, input coordinates plus noise.
"""
import os
from pathlib import Path

import pytest
import torch

SPEC = Path(os.environ.get("RFD4_SPEC", Path.home() / "rfd4-inputs" / "pdl1-128.yaml"))

needs_spec = pytest.mark.skipif(not SPEC.exists(), reason=f"no RFD4 spec at {SPEC}")


def _stand_in(features, n_designs):
    from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_ae_masks
    from tt_bio.rfd4 import ae_fold as AF, network as N
    from tt_bio.rfd4.engine import DesignOutput

    tab = AF.tables()
    b = N._broadcast_batch(dict(features), n_designs)
    _, emit = compute_ae_masks(b)
    g = torch.Generator().manual_seed(0)
    seq = b[AF.C_SEQ].long()
    rt = torch.where((seq >= 1) & (seq <= 20), seq, torch.randint(1, 21, seq.shape, generator=g))
    atoms = tab["atom_mask"][rt] & emit[..., None]
    atoms[..., 1] |= b["padding_mask"].bool()
    xyz = (b["coords"].float() / 10 + 0.05 * torch.randn(*b["coords"].shape, generator=g)) * atoms[..., None]
    n = features["padding_mask"].shape[1]
    return DesignOutput(coords_nm=xyz, residue_type=torch.where(emit, rt, torch.full_like(rt, -1)),
                        atom_mask=atoms, insertion_logits=torch.randn(n_designs, n, n, generator=g))


def _body(path):
    return [line for line in Path(path).read_text().splitlines() if "time" not in line.split(" ")[0]]


@needs_spec
def test_a_pooled_batch_writes_the_same_files_as_a_serial_one(tmp_path, monkeypatch):
    from tt_bio import rfd4
    from tt_bio.rfd4 import io

    monkeypatch.setattr(io, "_available_mib", lambda: 1 << 20)  # the pool, whatever this host has free

    ex = rfd4.prepare_examples(str(SPEC), str(tmp_path / "inputs"))[0]
    features, atom_array = rfd4.featurize(ex.cif, seed=0)
    output = _stand_in(features, 3)
    serial = [rfd4.write_design(output, features, atom_array, tmp_path / "a" / f"d{k}.cif", row=k) for k in range(3)]
    pooled = rfd4.write_designs(output, features, atom_array, [tmp_path / "b" / f"d{k}.cif" for k in range(3)])
    assert [p.name for p in pooled] == ["d0.cif", "d1.cif", "d2.cif"]
    assert [_body(p) for p in pooled] == [_body(p) for p in serial]
    # A writer forks from a single-threaded server and stays single-threaded itself.
    assert len(io.writer_pool(3).submit(os.listdir, "/proc/self/task").result()) == 1


@pytest.mark.parametrize("n_designs, cpus, available_mib, expected", [
    (1, 12, 1 << 20, 1),      # one design is written here, no pool
    (3, 12, 1 << 20, 3),      # never more writers than designs
    (16, 12, 1 << 20, 8),     # the fixed cap
    (16, 4, 1 << 20, 4),      # one per CPU
    (16, 12, None, 2),        # what fits in half the memory, beside the server
    (16, 12, 0, 1),           # nothing free: write here
])
def test_writer_count(monkeypatch, n_designs, cpus, available_mib, expected):
    from tt_bio.rfd4 import io

    if available_mib is None:
        available_mib = 2 * (io.SERVER_MIB + 2 * io.WRITER_MIB) + 1
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    assert io.writer_count(n_designs, available_mib) == expected
