"""Score every run under out/ against its crystal and print the before/after table.

  python collect.py [out_dir]

Per model x target x arm: DockQ, fnat, iRMS, LRMS for seeds 0 and 1, the seed floor (CA-RMSD
between the two seeds of the same arm), and whether the two arms' structures are bit-identical
(ATOM record md5). The monomer control (im9) reports CA-RMSD and the md5 check only.
"""
import hashlib
import sys
from pathlib import Path

from score import _fit, score

HERE = Path(__file__).resolve().parent
CRYSTAL = {"1emv": ("1EMV", "A,B"), "1brs": ("1BRS", "A,D"), "1hvr": ("1HVR", "A,B"),
           "im9": ("1EMV", "A")}


def _pred(out, tag, target):
    hits = [p for p in (out / tag).rglob(f"{target}.cif")]
    return hits[0] if hits else None


def _atoms_md5(path):
    lines = [l for l in path.read_text().splitlines() if l.startswith(("ATOM", "HETATM"))]
    return hashlib.md5("\n".join(lines).encode()).hexdigest()[:12]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out"
    models = sorted({d.name.split("-before-")[0] for d in out.glob("*-before-s*")})
    for m in models:
        for t, (pdb, chains) in CRYSTAL.items():
            cells = {}
            for arm in ("before", "after"):
                for s in (0, 1):
                    p = _pred(out, f"{m}-{arm}-s{s}", t)
                    if p is not None:
                        cells[arm, s] = (p, score(str(HERE / "crystal" / f"{pdb}.cif.gz"), str(p),
                                                  chains.split(",")))
            if not cells:
                continue
            for arm in ("before", "after"):
                row = [c for (a, _s), c in sorted(cells.items()) if a == arm]
                if not row:
                    print(f"{m:12s} {t:5s} {arm:6s} missing")
                    continue
                if "dockq" in row[0][1]:
                    vals = "  ".join(f"DockQ {r['dockq']:.3f} fnat {r['fnat']:.2f} iRMS {r['irms']:.2f} "
                                     f"LRMS {r['lrms']:.2f}" for _p, r in row)
                else:
                    vals = "  ".join(f"CA-RMSD {r['rmsd']:.2f}" for _p, r in row)
                floor = (f" | seed floor {_fit(row[0][1]['ca'], row[1][1]['ca'])[0]:.2f} A"
                         if len(row) == 2 else "")
                print(f"{m:12s} {t:5s} {arm:6s} {vals}{floor}")
            same = [s for s in (0, 1) if ("before", s) in cells and ("after", s) in cells
                    and _atoms_md5(cells["before", s][0]) == _atoms_md5(cells["after", s][0])]
            both = [s for s in (0, 1) if ("before", s) in cells and ("after", s) in cells]
            print(f"{m:12s} {t:5s} bit-identical before/after on seeds {same} of {both}")


if __name__ == "__main__":
    main()
