#!/usr/bin/env python3
"""Pairwise CA-RMSD of 7aqx_1024 folds, whole and with the nanobodies' HA+His6 tag left out.

    python3 perf/mgx/acc/tagfree.py ref0.cif ref1.cif tt_s0.cif tt_s1.cif ...

The nanobody chains end in AAAYPYDVPDYGSHHHHHH (residues 124-142). The crystal does not resolve
it and every fold, upstream included, gives it pLDDT ~40 against ~84 for the core, so its
placement is not a structural claim. This prints every pair both ways, with the mean pLDDT of
core and tag per structure, so a whole-complex excess can be attributed to the tag or not.
"""

import itertools
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
from ca_rmsd import best_rmsd, ca_chains  # noqa: E402

TAG_START = 124


def nb(seq: str) -> bool:
    return len(seq) < 200


def strip(d: dict) -> dict:
    return {n: (s, {k: v for k, v in ca.items() if not nb(s) or k < TAG_START})
            for n, (s, ca) in d.items()}


def plddt(path: str) -> str:
    core, tag = [], []
    for ch in gemmi.read_structure(path)[0]:
        b = [r.find_atom("CA", "*").b_iso for r in ch if r.find_atom("CA", "*")]
        if len(b) < 200:
            core += b[:TAG_START - 1]
            tag += b[TAG_START - 1:]
    return f"nanobody pLDDT core {sum(core) / len(core):.1f} tag {sum(tag) / len(tag):.1f}"


def main() -> None:
    paths = sys.argv[1:]
    chains = {p: ca_chains(p) for p in paths}
    for p in paths:
        print(f"{Path(p).name}: {plddt(p)}")
    for a, b in itertools.combinations(paths, 2):
        w, t = best_rmsd(chains[a], chains[b]), best_rmsd(strip(chains[a]), strip(chains[b]))
        print(f"{Path(a).name} {Path(b).name}: whole {w[0]:.2f} A (n {w[1]}), tag-free {t[0]:.2f} A (n {t[1]})")


if __name__ == "__main__":
    main()
