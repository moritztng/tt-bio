"""Score tt-bio OpenFold3 predictions with upstream openfold-3's own metric.

Run under the upstream CPU venv. Imports openfold3.core.metrics.alignment; the only
substitution against the upstream test is the model that produced the cifs.
"""
import json, statistics, sys
from pathlib import Path
import openfold3
from openfold3.core.metrics.alignment import Structure, best_ca_rmsd

MMCIFS = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"


def read_prediction(path: Path) -> Structure:
    """Parse a tt-bio prediction into upstream's ``Structure``.

    Upstream's own ``Structure.from_cif`` goes through ``parse_mmcif``, which requires a
    ``chem_comp`` category the tt-bio writer does not emit; it raises KeyError on our
    files. The reader is adapted here, the metric is not: biotite produces the same
    ``AtomArray`` that ``Structure`` holds, and ``best_ca_rmsd`` runs untouched on it.
    References are still read by upstream's own parser.
    """
    import biotite.structure.io.pdbx as pdbx

    block = pdbx.CIFFile.read(str(path))
    array = pdbx.get_structure(block, model=1, use_author_fields=False)
    return Structure(path=Path(path), atom_array=array)

def sample_cifs(d: Path):
    d = Path(d)
    stems = sorted(p for p in d.glob("*.cif") if "_model_" not in p.name)
    assert len(stems) == 1, f"expected one rank-0 cif in {d}, got {stems}"
    rest = sorted(d.glob("*_model_*.cif"),
                  key=lambda p: int(p.stem.rsplit("_model_", 1)[1]))
    return stems + rest

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--ref", required=True, help="pdb id (looked up in upstream mmcifs) or a path")
    ap.add_argument("--ref-chains", default="A")
    ap.add_argument("--expected-samples", type=int, required=True)
    ap.add_argument("--ceiling", type=float)
    ap.add_argument("--floor", type=float)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    ref_path = Path(a.ref) if Path(a.ref).exists() else MMCIFS / f"{a.ref}.cif"
    cifs = sample_cifs(a.pred_dir)
    assert len(cifs) == a.expected_samples, (
        f"Expected {a.expected_samples} predicted samples, found {len(cifs)}: "
        f"{[c.name for c in cifs]}")
    ref = Structure.from_cif(ref_path)
    chains = tuple(a.ref_chains.split(","))
    ms = [best_ca_rmsd(pred=read_prediction(c), ref=ref, ref_chains=chains) for c in cifs]
    vals = [m.rmsd for m in ms]
    gdt = [m.gdt_ts for m in ms]
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    verdict = "MEASURED"
    if a.ceiling is not None:
        verdict = "PASS" if mean < a.ceiling else "FAIL"
    if a.floor is not None:
        v = "PASS" if mean > a.floor else "FAIL"
        verdict = v if a.ceiling is None else (verdict if verdict == "FAIL" else v)
    rec = {"label": a.label, "pred_dir": str(a.pred_dir), "ref_cif": ref_path.name,
           "ref_chains": chains, "n": len(vals), "values": vals, "gdt_ts": gdt,
           "mean": mean, "sd": sd, "ceiling": a.ceiling, "floor": a.floor,
           "verdict": verdict,
           "metric": "openfold3.core.metrics.alignment.best_ca_rmsd",
           "openfold3_commit": "72fc3a9534d37291b1ca7f02f11a8a0b12cd80c9",
           "cifs": [c.name for c in cifs]}
    print(json.dumps(rec, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")

main()
