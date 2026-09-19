"""Where are the clashes that fail the 1536 rung: inside a chain, or between the two?

check_structure reports one clash count, and that number cannot distinguish "the model
folded a chain into itself" from "the fixture asked for an interface that does not exist".
The 1536 rung is two CDK2 tandem chains where B is an exact prefix of A, so there is no real
A-B interface to find and contacts across it are the fixture rather than the size or the
board. Splitting the count makes that measurable instead of argued.

Both numbers come from the check_structure functions themselves, so they reconcile with the
verdict it printed rather than being a second opinion with a second definition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wh-correctness"))
from check_structure import clashes, interface_clashes  # noqa: E402


def main(path: str, interface_chain: str = "B") -> int:
    st = gemmi.read_structure(path)
    st.remove_hydrogens()
    st.setup_entities()
    total, heavy, worst = clashes(st)
    names = [ch.name for ch in st[0]]
    print(f"{Path(path).name}: chains {names}, {heavy} heavy atoms")
    print(f"  total clashes < 2.0 A : {total}  (worst {worst} A)")
    if interface_chain in names and len(names) > 1:
        n_if, heavy_if = interface_clashes(st, interface_chain)
        print(f"  across the {interface_chain}-to-rest interface : {n_if}")
        print(f"  inside a single chain : {total - n_if}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
