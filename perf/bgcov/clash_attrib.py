"""Where the clashes in a BoltzGen design are: the designed chain, or the fixture it was given.

`check_structure.py` reports one clash count for the whole complex, and at these sizes that
number FAILS the rule. A count cannot say whether the engine packed the binder badly or whether
the target crop arrived with those contacts already in it, and those are different results: one
is the model, the other is the fixture. So this splits the same pairs by chain pair and runs the
identical count over the INPUT crop as a control.

    python3 perf/bgcov/clash_attrib.py <complex.cif> [--control <input.cif>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wh-correctness"))
from check_structure import CLASH_DIST, DISULFIDE_MAX, VIRTUAL_ATOM  # noqa: E402


def pairs(path: Path) -> tuple[Counter, int, float]:
    """The same pairs check_structure counts, keyed by the chain pair they fall between."""
    st = gemmi.read_structure(str(path))
    st.remove_alternative_conformations()
    st.setup_entities()
    model = st[0]
    ns = gemmi.NeighborSearch(st, 5.0).populate()
    seen, heavy, worst = set(), 0, math.inf
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.element == gemmi.Element("H") or VIRTUAL_ATOM.match(atom.name):
                    continue
                heavy += 1
                for m in ns.find_atoms(atom.pos, "\0", radius=CLASH_DIST):
                    cra = m.to_cra(model)
                    if (cra.atom.element == gemmi.Element("H")
                            or VIRTUAL_ATOM.match(cra.atom.name)):
                        continue
                    if (cra.chain.name == chain.name
                            and abs(cra.residue.seqid.num - res.seqid.num) < 2):
                        continue
                    d = cra.atom.pos.dist(atom.pos)
                    if (d < DISULFIDE_MAX and atom.name == "SG" and cra.atom.name == "SG"
                            and res.name == "CYS" and cra.residue.name == "CYS"):
                        continue
                    if d < CLASH_DIST and cra.atom.serial != atom.serial:
                        seen.add((min(atom.serial, cra.atom.serial),
                                  max(atom.serial, cra.atom.serial),
                                  "-".join(sorted((chain.name, cra.chain.name)))))
                        worst = min(worst, d)
    return (Counter(k[2] for k in seen), heavy,
            0.0 if worst is math.inf else round(worst, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("struct", type=Path)
    ap.add_argument("--control", type=Path,
                    help="the input crop the design was conditioned on")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out = {}
    for label, p in (("design", a.struct), ("control", a.control)):
        if p is None:
            continue
        by_pair, heavy, worst = pairs(p)
        out[label] = {"file": p.name, "heavy_atoms": heavy, "n": sum(by_pair.values()),
                      "worst_dist": worst, "by_chain_pair": dict(sorted(by_pair.items()))}
        print(f"{label:8s} {p.name}: {sum(by_pair.values())} pairs / {heavy} heavy atoms "
              f"({sum(by_pair.values()) / max(heavy, 1):.2%}), worst {worst} A, "
              f"{dict(sorted(by_pair.items()))}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
