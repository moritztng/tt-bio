#!/usr/bin/env python3
"""RF3 host-featurizer parity gate (card-free, CPU-only).

Scores :func:`tt_bio.rf3.featurize.featurize` against the committed reference
captures in ``parity_artifacts/``. Those were captured from the real upstream
``RF3InferenceEngine`` on Python 3.12, so this gate checks the vendored AtomWorks
pipeline on tt-bio's own runtime against what upstream inference actually sees.
No foundry install, no checkpoint and no device are needed at gate time.

The bar is bit-exact on every comparable key, on every fixture: the upstream
featurizer is reproducible process to process once ``random``, ``numpy`` and
``torch`` are all seeded, so anything less is a real difference.

That only holds against a matching RDKit. ``f["ref_pos"]`` is an RDKit-generated
conformer, and on ligand inputs RDKit's chiral-centre perception feeds
``chiral_centers`` / ``chiral_feats`` / ``chiral_center_dihedral_angles`` as well,
both of which move between RDKit releases. The captures record the environment
that produced them and this gate reports a mismatch rather than letting one look
like a port defect.

Two keys are compared as shape-only zero stubs rather than by value:
``feats/atom_level_embedding`` and ``feats/mean_atom_level_embedding``. That is the
MLFF/MACE track, whose cache is an IPD-internal path that is not distributed, so
it is exactly zero on any public run and the fixtures cannot exercise it. The
contract checked here is all-zeros of the recorded shape.

Returns a report dict for ``scripts/full_parity_gate.py`` (mode ==
"rf3_featurizer").
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO / "scripts" / "rf3_port" / "parity_artifacts"

#: Per-fixture capture flags. These have to match how each reference was captured;
#: `template` and `cyclic` are the only fixtures that pass any.
FIXTURES: dict[str, dict] = {
    "glke": {},
    "two_protein_chains": {},
    "protein_dna": {},
    "rna": {},
    "ligands": {},
    "covalent_glycan": {},
    "ncaa_small": {},
    "monomer_msa": {},
    "cyclic": {"cyclic_chains": ["A"]},
    "template": {
        "template_selection": ["A"],
        "ground_truth_conformer_selection": ["C"],
    },
}


#: Non-tensor entries that carry provenance rather than model input. `example_id`
#: is derived from the input filename, and the fixtures rename theirs to `input.*`
#: so each directory is self-contained, so it cannot match by construction.
PLAIN_PROVENANCE_KEYS = frozenset({"example_id"})


def _flatten(obj, prefix=""):
    """Flatten the pipeline output the same way ``capture_ref_f.py`` does."""
    import torch

    tensors, plain = {}, {}

    def walk(o, p):
        key = p.rstrip("/")
        if isinstance(o, torch.Tensor):
            tensors[key] = o
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}{k}/")
        elif isinstance(o, (list, tuple)) and any(
            isinstance(v, (torch.Tensor, dict)) for v in o
        ):
            for i, v in enumerate(o):
                walk(v, f"{p}{i}/")
        else:
            try:
                json.dumps(o)
                plain[key] = o
            except (TypeError, ValueError):
                pass

    walk(obj, prefix)
    return tensors, plain


@contextlib.contextmanager
def _in_dir(path: Path):
    """Run with ``path`` as cwd.

    Fixture inputs reference their ligand and MSA files by paths relative to the
    fixture directory, so that each fixture is self-contained and movable. Those
    resolve against the process cwd, so the gate has to be there.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _input_path(fixture_dir: Path) -> Path:
    for name in ("input.json", "input.cif", "input.pdb"):
        if (fixture_dir / name).exists():
            return fixture_dir / name
    raise FileNotFoundError(f"no input file in {fixture_dir}")


def score_fixture(name: str, flags: dict, pipeline=None) -> dict:
    import torch

    from tt_bio.rf3.featurize import featurize

    d = ARTIFACTS / name
    ref_pt, ref_meta = d / "ref_f.pt", d / "ref_f.meta.json"
    if not ref_pt.exists():
        return {"fixture": name, "verdict": "ERROR", "error": f"missing {ref_pt}"}

    ref = torch.load(ref_pt, weights_only=False)
    meta = json.loads(ref_meta.read_text())
    stubs = meta.get("__zero_stub_keys__", {})
    ref_plain = meta.get("__non_tensor__", {})

    with _in_dir(d):
        out = featurize(_input_path(d).name, **flags)
    got, got_plain = _flatten(out[0])

    total = exact = 0
    mismatches = []

    for key in sorted(set(ref) | set(got) | set(stubs)):
        total += 1
        if key in stubs:
            # zero-stub: the port must emit all-zeros of the recorded shape
            spec = stubs[key]
            if key not in got:
                mismatches.append({"key": key, "reason": "MISSING_STUB"})
                continue
            t = got[key]
            if list(t.shape) != spec["shape"]:
                mismatches.append(
                    {"key": key, "reason": "STUB_SHAPE",
                     "ported": list(t.shape), "ref": spec["shape"]}
                )
            elif int(torch.count_nonzero(t)) != 0:
                mismatches.append({"key": key, "reason": "STUB_NONZERO",
                                   "nonzero": int(torch.count_nonzero(t))})
            else:
                exact += 1
            continue
        if key not in ref or key not in got:
            mismatches.append({"key": key, "reason": "MISSING",
                               "in_ref": key in ref, "in_ported": key in got})
            continue
        a, b = ref[key], got[key]
        if a.shape != b.shape:
            mismatches.append({"key": key, "reason": "SHAPE",
                               "ref": list(a.shape), "ported": list(b.shape)})
        elif torch.equal(a, b):
            exact += 1
        else:
            diff = (a != b)
            mismatches.append({
                "key": key, "reason": "VALUE",
                "n_differing": int(diff.sum()), "n_total": int(a.numel()),
                "maxabs": (float((a.float() - b.float()).abs().max())
                           if a.is_floating_point() or not a.dtype == torch.bool else None),
            })

    # non-tensor features (cyclic_asym_ids and friends)
    plain_mismatches = []
    for key, want in ref_plain.items():
        if isinstance(want, dict) and "__repr__" in want:
            continue  # unserialisable object (the AtomArray), not a feature
        if key in PLAIN_PROVENANCE_KEYS:
            continue
        if got_plain.get(key) != want:
            plain_mismatches.append({"key": key, "ref": want,
                                     "ported": got_plain.get(key)})

    verdict = "PASS" if not mismatches and not plain_mismatches else "GAP"
    return {
        "fixture": name,
        "verdict": verdict,
        "keys_total": total,
        "keys_bitexact": exact,
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "plain_mismatches": plain_mismatches,
        "tokens": int(ref["feats/restype"].shape[0]),
        "atoms": int(ref["feats/ref_pos"].shape[0]),
    }


#: Keys derived from RDKit's conformer embedding and chiral-centre perception.
#: These are only comparable against a capture made with the same RDKit.
RDKIT_DERIVED_KEYS = frozenset({
    "feats/ref_pos",
    "feats/ref_pos_ground_truth",
    "feats/chiral_centers",
    "feats/chiral_feats",
    "feats/chiral_center_dihedral_angles",
    "ground_truth/coord_atom_lvl",
    "ground_truth/coord_token_lvl",
    "symmetry_resolution/coord_atom_lvl",
})

#: Keys carrying a random rigid augmentation, which `get_random_rots` builds from
#: `torch.linalg.qr`. The RNG draws are reproducible across torch builds -- `t` and
#: `noise` stay bit-exact -- but QR's sign convention is the LAPACK backend's, so a
#: different torch gives a different rotation for the same random matrix and the
#: augmented coordinates move rigidly while every deterministic key still matches.
#: Measured 2026-08-23 on torch 2.8.0+cpu against captures made on 2.11.0+cpu: 500 of
#: 510 keys bit-exact, `coord_atom_lvl_to_be_noised` the only one off, on all ten
#: fixtures. torch is unpinned in pyproject.toml, so this is not a host quirk to wait
#: out. Only `--partial_t` reads the key (tt_bio/worker.py), and there its orientation
#: is arbitrary by construction.
TORCH_QR_DERIVED_KEYS = frozenset({"coord_atom_lvl_to_be_noised"})


def _env_mismatch() -> dict | None:
    """Every recorded capture-environment entry that the running one does not match.

    Checking rdkit alone let a torch difference read as a port defect: the QR-derived
    key above is off on any torch but the captures', and one un-excusable key is enough
    to keep the whole leg at GAP.
    """
    import numpy
    import torch

    import rdkit

    meta = json.loads((ARTIFACTS / "glke" / "ref_f.meta.json").read_text())
    want = meta.get("__env__", {})
    running = {"rdkit": rdkit.__version__, "torch": torch.__version__,
               "numpy": numpy.__version__}
    off = {k: {"capture": v, "running": running[k]}
           for k, v in want.items() if k in running and v != running[k]}
    return off or None


def excusable_keys(env_mismatch: dict | None) -> set:
    """Key sets a differing dependency owns. A key set is only excusable when ITS
    dependency is the one that differs: a torch difference does not excuse a
    conformer, and an RDKit difference does not excuse a rotation."""
    if not env_mismatch:
        return set()
    out = set()
    if "rdkit" in env_mismatch:
        out |= RDKIT_DERIVED_KEYS
    if "torch" in env_mismatch:
        out |= TORCH_QR_DERIVED_KEYS
    return out


def featurizer_parity() -> dict:
    sys.path.insert(0, str(REPO))

    env_mismatch = _env_mismatch()
    results = [score_fixture(n, f) for n, f in FIXTURES.items()]
    passed = [r for r in results if r["verdict"] == "PASS"]
    verdict = "PASS" if len(passed) == len(results) else "GAP"
    if verdict == "GAP" and env_mismatch:
        # Every surviving mismatch belongs to a key set the differing dependency owns,
        # so this is an environment difference, not a port defect. Say which, and do
        # not call it a PASS. A key set is only excusable when ITS dependency is the
        # one that differs: a torch difference does not excuse a conformer, and an
        # RDKit difference does not excuse a rotation.
        excusable = excusable_keys(env_mismatch)
        off_keys = {
            m["key"]
            for r in results
            for m in r.get("mismatches", [])
        }
        if off_keys and off_keys <= excusable:
            verdict = "GAP_ENV"
    return {
        "mode": "rf3_featurizer",
        "verdict": verdict,
        "env_mismatch": env_mismatch,
        "fixtures_total": len(results),
        "fixtures_pass": len(passed),
        "keys_total": sum(r.get("keys_total", 0) for r in results),
        "keys_bitexact": sum(r.get("keys_bitexact", 0) for r in results),
        "results": results,
    }


def main() -> int:
    rep = featurizer_parity()
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
