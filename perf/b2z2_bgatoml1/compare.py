"""Read the three arm records and say what the comparison actually supports.

The verdict has two independent parts and both have to hold:

  * the A/A control (base vs base2) says a digest compare is meaningful here at all;
  * the census says the L1 arm reached the gate. `l1 == 0` with `shape > 0` is a lever that
    never fired, and a bit-exact result from it proves nothing about the lever.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load(root: Path, arm: str) -> dict:
    """An arm's record, or None if that arm never wrote one -- a crashed arm is a result too."""
    f = root / arm / "arm.json"
    return json.loads(f.read_text()) if f.is_file() else None


def _coords(root: Path, arm: str):
    """(labels, Nx3 array) for the one design an arm wrote, or None if gemmi is unavailable."""
    try:
        import gemmi
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    cif = sorted((root / arm / "intermediate_designs").glob("*.cif"))[0]
    st = gemmi.read_structure(str(cif))
    labels, xyz = [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    labels.append(f"{chain.name}/{res.seqid.num}/{res.name}/{atom.name}")
                    xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
        break
    return labels, np.asarray(xyz)


def _deviation(root: Path, a: str, b: str) -> str:
    """Max per-atom displacement in Angstrom, so a failed digest still has a size.

    Only meaningful when both arms placed the same atoms; a sequence that differs between arms
    is a bigger difference than any distance, and says so instead of averaging over a mismatch.
    """
    ca, cb = _coords(root, a), _coords(root, b)
    if ca is None or cb is None:
        return "unavailable (no gemmi)"
    if ca[0] != cb[0]:
        common = sum(1 for x, y in zip(ca[0], cb[0]) if x == y)
        return (f"atom sets differ: {len(ca[0])} vs {len(cb[0])} atoms, "
                f"{common} labels agree in order -- not a distance")
    import numpy as np

    d = np.linalg.norm(ca[1] - cb[1], axis=1)
    return f"max {d.max():.4f} A, mean {d.mean():.4f} A over {len(d)} atoms"


def main() -> int:
    root = Path(sys.argv[1])
    arms = {a: _load(root, a) for a in ("base", "l1", "base2")}
    missing = [a for a, r in arms.items() if r is None]
    if missing:
        print(f"arms that produced no record: {', '.join(missing)}")
        print("\nVERDICT: NO-GO -- an arm did not finish. A lever that crashes the pipeline is a "
              "NO-GO whatever the surviving arms agree on; read that arm's log for the throw.")
        return 1

    print(f"{'arm':6} {'wall_s':>9}  {'ATOM_L1_STATS':38} file_sha256")
    for name, rec in arms.items():
        for cif, dig in sorted(rec["designs"].items()):
            print(f"{name:6} {rec['wall_s']:>9.1f}  {str(rec['atom_l1_stats']):38} "
                  f"{dig['file_sha256'][:16]}  {cif}")

    def digests(rec, key):
        return {c: d[key] for c, d in rec["designs"].items()}

    aa_file = digests(arms["base"], "file_sha256") == digests(arms["base2"], "file_sha256")
    aa_coord = digests(arms["base"], "coord_sha256") == digests(arms["base2"], "coord_sha256")
    ab_file = digests(arms["base"], "file_sha256") == digests(arms["l1"], "file_sha256")
    ab_coord = digests(arms["base"], "coord_sha256") == digests(arms["l1"], "coord_sha256")

    st = arms["l1"]["atom_l1_stats"]
    fired = st.get("l1", 0) > 0 and st.get("dram", 0) == 0 and st.get("shape", 0) == 0
    base_st = arms["base"]["atom_l1_stats"]
    base_clean = base_st.get("l1", 0) == 0 and base_st.get("off", 0) > 0

    print()
    print(f"A/A  base vs base2 : file {aa_file}  coord {aa_coord}  "
          f"[{_deviation(root, 'base', 'base2')}]")
    print(f"A/B  base vs l1    : file {ab_file}  coord {ab_coord}  "
          f"[{_deviation(root, 'base', 'l1')}]")
    print(f"gate l1 arm fired  : {fired}  ({st})")
    print(f"gate base arm off  : {base_clean}  ({base_st})")

    if not aa_file:
        print("\nVERDICT: INCONCLUSIVE -- the seeded base arm does not reproduce itself, so a "
              "digest compare between arms cannot attribute a difference to the lever.")
        return 2
    if not fired:
        print("\nVERDICT: INCONCLUSIVE -- the L1 arm never took the L1 path; bit-exactness here "
              "is the absence of the lever, not evidence about it.")
        return 3
    if ab_file and ab_coord:
        print("\nVERDICT: GO -- the lever fires on every atom-layer call and BoltzGen writes a "
              "byte-identical design.")
        return 0
    print("\nVERDICT: NO-GO -- the lever fires and the design moves.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
