"""OpenBind-0 capacity rungs: the OpenFold3 ceiling ladder, with and without a ligand.

Two arms, because OB0 has a token axis the residue count does not describe:

  apo   the polymer alone, so tokens == residues and the rung lines up with the openfold3
        ladder in perf/ceiling_of3 (same sequences, same alignment, same depth).
  lig   the same polymer plus one ligand. Every ligand heavy atom is its own token, so a
        1024-residue target folds a 1059-token trunk -- and 1059 buckets to 1088, one rung
        ABOVE the apo 1024. A residue-denominated ceiling cannot see that.

Reads the openfold3 rung yamls rather than re-deriving their sequences: the alignment cache is
keyed on the protein sequence, so an OB0 rung built this way hits the same 14190-row alignment
its openfold3 twin folded, and the two arms differ only by the checkpoint and the ligand.

    python perf/ceiling_openbind/make_rungs.py <of3-rung-dir> <out-dir>
"""

import pathlib
import sys

# Staurosporine, C28H26N4O3, 35 heavy atoms. The medium ligand of perf/openbind/make_inputs.py, so
# the ligand cost here is priced on a molecule the perf page already carries.
LIGAND_CCD = "STU"
LIGAND_HEAVY_ATOMS = 35


def residues(yaml_text: str) -> int:
    return sum(len(line.split("sequence:", 1)[1].strip())
               for line in yaml_text.splitlines() if "sequence:" in line)


def bucketed(n_token: int, multiple: int = 32) -> int:
    return -(-n_token // multiple) * multiple


def main() -> None:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(src.glob("*.yaml")):
        text = path.read_text()
        n_res = residues(text)
        (out / f"ob_apo_{path.stem}.yaml").write_text(text)
        lig = text.rstrip("\n") + f"\n  - ligand:\n      id: L\n      ccd: {LIGAND_CCD}\n"
        (out / f"ob_lig_{path.stem}.yaml").write_text(lig)
        rows.append((path.stem, n_res, bucketed(n_res),
                     n_res + LIGAND_HEAVY_ATOMS, bucketed(n_res + LIGAND_HEAVY_ATOMS)))
    print(f"{'rung':<12}{'residues':>9}{'apo tokens':>12}{'lig tokens':>12}{'lig bucketed':>14}")
    for stem, n_res, apo, lig_tok, lig_b in rows:
        print(f"{stem:<12}{n_res:>9}{apo:>12}{lig_tok:>12}{lig_b:>14}")


if __name__ == "__main__":
    main()
