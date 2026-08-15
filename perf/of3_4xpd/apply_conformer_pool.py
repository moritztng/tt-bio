#!/usr/bin/env python3
"""Apply the parallel per-residue conformer route to a tt-bio tree.

Kept as a patch script rather than an edit because the parity gate was folding out of the
worktree when this was written, and every `tt_bio.main predict` it spawns imports that
checkout: editing `tt_bio/` mid-gate would silently change what the gate is testing. Run
this against a scratch copy to measure, and against the worktree once the gate is done.

    python3 perf/of3_4xpd/apply_conformer_pool.py <tree-root>
"""
import pathlib, sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def sub(rel, old, new, count=1):
    p = ROOT / rel
    t = p.read_text()
    if t.count(old) != count:
        sys.exit("PATCH MISS in %s: found %d, expected %d\n---\n%s" % (rel, t.count(old), count, old[:300]))
    p.write_text(t.replace(old, new, count))
    print("patched", rel)


C = "tt_bio/_vendor/openfold3/core/data/primitives/structure/conformer.py"
Q = "tt_bio/_vendor/openfold3/core/data/primitives/structure/query.py"

# ---- conformer.py: let a caller hand this residue its ETKDG seed ----
sub(C,
'''def _compute_conformer(''',
'''# The ETKDG seed is drawn per call from the global `random`, in residue order, so a caller
# that wants to run the residues concurrently has to hand each one the seed the sequential
# path would have given it. `pool_seed` does that for the calling thread. A second draw
# inside the same residue means the generation retried, which in the sequential path would
# have consumed the NEXT residue's seed -- that is recorded, not hidden, and the caller
# recomputes sequentially.
_POOL = threading.local()
_POOL_DIVERGED = []


def pool_seed(seed: int) -> None:
    _POOL.seed = seed
    _POOL.active = True


def pool_reset() -> None:
    _POOL.seed = None
    _POOL.active = False
    _POOL_DIVERGED.clear()


def pool_diverged() -> bool:
    return bool(_POOL_DIVERGED)


def _etkdg_seed() -> int:
    seed = getattr(_POOL, "seed", None)
    if seed is not None:
        _POOL.seed = None
        return seed
    if getattr(_POOL, "active", False):
        _POOL_DIVERGED.append(1)
    return random.randint(0, 10**9)


def _compute_conformer(''')

sub(C,
'''    strategy.randomSeed = random.randint(0, 10**9)''',
'''    strategy.randomSeed = _etkdg_seed()''')

sub(C,
'''import random''',
'''import random
import threading''')

# ---- query.py: compute the residues' reference molecules on a pool ----
sub(Q,
'''# One reference conformer per residue TYPE instead of one per residue.''',
'''# Residues whose reference conformer is computed concurrently. Every conformer is still
# computed -- this changes WHERE the work runs, not what it produces, so it is bit-exact
# whenever no residue redraws its seed (0 of 512 on the page fixture, and on any
# all-canonical protein). 0 keeps the sequential path. 24 is where the scaling stops on a
# 32-core host; a smaller box gets its own core count.
CONFORMER_THREADS = min(24, os.cpu_count() or 1)


def _pooled_reference_molecules(jobs: list) -> list:
    """Every job's reference molecule, computed on a thread pool.

    The seeds are pre-drawn in residue order and handed out one per residue, so residue i
    embeds with exactly the seed the sequential loop would have drawn for it. If any
    residue retried it consumed a seed the sequential path would have given to the next
    one, and from there the two streams differ, so the whole sequence is recomputed
    sequentially -- a valid conformer set, just not the same draw.
    """
    seeds = [random.randint(0, 10**9) for _ in jobs]
    conformer_mod.pool_reset()

    def one(i):
        conformer_mod.pool_seed(seeds[i])
        arr, leaving = jobs[i]
        return processed_reference_molecule_from_atom_array(arr, atoms_to_mask=leaving)

    with ThreadPoolExecutor(max_workers=CONFORMER_THREADS) as ex:
        out = list(ex.map(one, range(len(jobs))))
    if conformer_mod.pool_diverged():
        logger.warning("conformer pool: a residue redrew its seed, recomputing sequentially")
        conformer_mod.pool_reset()
        out = [processed_reference_molecule_from_atom_array(a, atoms_to_mask=l)
               for a, l in jobs]
    return out


# One reference conformer per residue TYPE instead of one per residue.''')

sub(Q,
'''import dataclasses
import logging''',
'''import dataclasses
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor''')

sub(Q,
'''from tt_bio._vendor.openfold3.core.data.primitives.structure.conformer import (''',
'''from tt_bio._vendor.openfold3.core.data.primitives.structure import conformer as conformer_mod
from tt_bio._vendor.openfold3.core.data.primitives.structure.conformer import (''')

sub(Q,
'''    atom_array = None
    processed_reference_mols = []''',
'''    atom_array = None
    processed_reference_mols = []
    pool_jobs = [] if CONFORMER_THREADS > 0 else None''')

sub(Q,
'''        # Parse into RDKit mol and compute conformer
        if REF_MOL_MEMO:
            processed_ref_mol = _memo_processed_reference_molecule(
                res_array, leaving_atoms,
                (resname_3, tuple(leaving_atoms), poly_type))
        else:
            processed_ref_mol = processed_reference_molecule_from_atom_array(
                res_array, atoms_to_mask=leaving_atoms
            )
        processed_reference_mols.append(processed_ref_mol)''',
'''        # Parse into RDKit mol and compute conformer
        if pool_jobs is not None:
            pool_jobs.append((res_array, leaving_atoms))
        else:
            if REF_MOL_MEMO:
                processed_ref_mol = _memo_processed_reference_molecule(
                    res_array, leaving_atoms,
                    (resname_3, tuple(leaving_atoms), poly_type))
            else:
                processed_ref_mol = processed_reference_molecule_from_atom_array(
                    res_array, atoms_to_mask=leaving_atoms
                )
            processed_reference_mols.append(processed_ref_mol)''')

sub(Q,
'''    return StructureWithReferenceMolecules(
        atom_array=atom_array,
        processed_reference_mols=processed_reference_mols,
    )''',
'''    if pool_jobs is not None:
        processed_reference_mols = _pooled_reference_molecules(pool_jobs)
    return StructureWithReferenceMolecules(
        atom_array=atom_array,
        processed_reference_mols=processed_reference_mols,
    )''')
print("CONFORMER POOL APPLIED")
