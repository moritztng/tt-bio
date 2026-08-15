#!/usr/bin/env python3
"""Can the per-residue conformers be computed in PARALLEL and stay bit-exact?

The memo (`REF_MOL_MEMO`) is worth 2.6 s and is not bit-exact: it serves 512 residues from
~20 cached embeddings, and every residue currently gets its own randomly seeded ETKDG
conformer. Parallelising instead of skipping keeps all 512 embeddings and every value, IF
each residue can be given exactly the seed the sequential run would have drawn for it.

The seed is drawn inside `_compute_conformer` as `random.randint(0, 10**9)`, once per call,
in residue order. So: pre-draw the seeds in order, hand residue i its own, and run the
residues on a thread pool. Any residue whose conformer generation retries would draw a
SECOND seed, and from there the sequential stream and the pre-drawn one diverge -- that is
detected here, not assumed.

Patches nothing on disk: it swaps the conformer module's `random` for a thread-aware shim
inside this process only. No device.
"""
import random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402

from tt_bio._vendor.openfold3.core.data.primitives.structure import conformer as C  # noqa: E402
from tt_bio._vendor.openfold3.core.data.primitives.structure.query import (  # noqa: E402
    atom_array_from_ccd_code, processed_reference_molecule_from_atom_array)
from tt_bio._vendor.openfold3.core.data.resources.residues import (  # noqa: E402
    MOLECULE_TYPE_TO_LEAVING_ATOMS, PROTEIN_RESTYPE_1TO3)
from tt_bio._vendor.openfold3.core.data.primitives.structure.labels import MoleculeType  # noqa: E402

POLY = MoleculeType.PROTEIN
LEAVING = MOLECULE_TYPE_TO_LEAVING_ATOMS[POLY]
SEED = 0
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def sequence_from_fixture(p: Path) -> str:
    """The page fixture's protein sequence, straight out of the yaml."""
    seq, want = [], False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("sequence:"):
            seq.append(s.split(":", 1)[1].strip().strip("'\""))
        want = want
    return "".join(seq)


seq = sequence_from_fixture(ROOT / "perf" / "size512" / "fixtures" / "cdk2x2_512.yaml")
codes = [PROTEIN_RESTYPE_1TO3.get(c, "UNK") for c in seq]
print(f"{len(codes)} residues, {len(set(codes))} distinct types", flush=True)
arrays = [atom_array_from_ccd_code(c, chain_id="A", res_id=i + 1, molecule_type=POLY)
          for i, c in enumerate(codes)]
N = len(arrays)


def coords(pr):
    return np.array(pr.mol.GetConformer().GetPositions(), dtype=np.float64)


def run_sequential():
    random.seed(SEED)
    t0 = time.perf_counter()
    out = [processed_reference_molecule_from_atom_array(a, atoms_to_mask=LEAVING) for a in arrays]
    return out, time.perf_counter() - t0


class SeedShim:
    """Stands in for the conformer module's `random`, one pre-drawn seed per residue.

    A second draw inside the same residue means the generation retried, so the pre-drawn
    stream no longer matches what the sequential run would have produced. That is recorded
    and the caller falls back rather than shipping a silently different conformer.
    """

    def __init__(self):
        self.tl = threading.local()
        self.diverged = []

    def randint(self, lo, hi):
        s = getattr(self.tl, "seed", None)
        if s is None:
            self.diverged.append(getattr(self.tl, "idx", -1))
            return random.randint(lo, hi)
        self.tl.seed = None
        return s


def run_pooled(workers):
    random.seed(SEED)
    seeds = [random.randint(0, 10 ** 9) for _ in range(N)]     # the sequential draw order
    shim = SeedShim()
    real_random, C.random = C.random, shim

    def one(i):
        shim.tl.seed, shim.tl.idx = seeds[i], i
        return processed_reference_molecule_from_atom_array(arrays[i], atoms_to_mask=LEAVING)

    try:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(one, range(N)))
        dt = time.perf_counter() - t0
    finally:
        C.random = real_random
    return out, dt, shim.diverged


seq_out, seq_s = run_sequential()
print(f"sequential      {seq_s*1e3:8.1f} ms", flush=True)

for w in (WORKERS,):
    par_out, par_s, diverged = run_pooled(w)
    same = sum(1 for a, b in zip(seq_out, par_out)
               if np.array_equal(coords(a), coords(b)))
    smiles_same = sum(1 for a, b in zip(seq_out, par_out)
                      if Chem.MolToSmiles(a.mol) == Chem.MolToSmiles(b.mol))
    print(f"{w:2d} threads      {par_s*1e3:8.1f} ms   {seq_s/par_s:5.2f}x", flush=True)
    print(f"  conformers bit-identical to sequential: {same}/{N}", flush=True)
    print(f"  molecules identical (SMILES):           {smiles_same}/{N}", flush=True)
    print(f"  residues that redrew (divergence):      {len(diverged)}"
          f"{'' if not diverged else ' at ' + str(diverged[:8])}", flush=True)
