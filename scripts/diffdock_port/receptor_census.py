"""Residue count per PoseBusters receptor, through the port's own parser.

Upstream refuses a receptor over 3000 residues (process_mols.py:169, on len(coords)
after parsing); tt_bio/diffdock/receptor.py:205 mirrors it. This counts the set so
the exclusion list in EVAL.md is measured, not assumed.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.diffdock.structure import parse_receptor  # noqa: E402
from tt_bio.diffdock.receptor import MAX_RECEPTOR_RESIDUES  # noqa: E402

PB = Path(os.environ.get("DFD_PB", "/home/moritz/scratch/dfd/pb/posebusters_benchmark_set"))
out = {}
for d in sorted(PB.iterdir()):
    if not d.is_dir():
        continue
    pdb = d / f"{d.name}_protein.pdb"
    try:
        out[d.name] = int(parse_receptor(pdb).n_residues)
    except Exception as e:  # a parse failure is a result too, recorded not dropped
        out[d.name] = f"PARSE_ERROR: {type(e).__name__}: {e}"

over = {k: v for k, v in out.items() if isinstance(v, int) and v > MAX_RECEPTOR_RESIDUES}
errs = {k: v for k, v in out.items() if not isinstance(v, int)}
sizes = sorted(v for v in out.values() if isinstance(v, int))
res = {
    "cap": MAX_RECEPTOR_RESIDUES,
    "n_complexes": len(out),
    "over_cap": over,
    "parse_errors": errs,
    "max_residues": sizes[-1] if sizes else None,
    "median_residues": sizes[len(sizes) // 2] if sizes else None,
    "per_complex": out,
}
p = Path(sys.argv[1] if len(sys.argv) > 1 else "residue_counts.json")
p.write_text(json.dumps(res, indent=1))
print("n", len(out), "over cap", len(over), over)
print("parse errors", len(errs), list(errs)[:5])
print("max", res["max_residues"], "median", res["median_residues"])
