#!/usr/bin/env python3
"""Write the alignment-representatives FASTA and a single-sequence MSA per representative.

Upstream's training path always has MSAs, and the representatives FASTA is how a chain finds its
alignment directory. A single-row MSA is a legitimate degenerate OF3 input: it keeps the MSA stack
live without needing a real search. It does NOT exercise pairing or subsampling, which is
`of3t-data`'s job, and the frozen bundle should be rebuilt on real MSAs before anything downstream
is called final.
"""
import json
import shutil
import sys
from pathlib import Path

MOLTYPE = {"PROTEIN": "protein", "RNA": "rna"}


def main() -> int:
    root = Path(sys.argv[1])
    aln = root / "alignments"
    shutil.rmtree(aln, ignore_errors=True)
    aln.mkdir()
    meta = json.loads((root / "metadata.json").read_text())["structure_data"]

    out = []
    for fa in sorted(root.glob("structure_files/*/*.fasta")):
        pdb = fa.parent.name
        recs, hdr, seq = [], None, []
        for line in fa.read_text().splitlines():
            if line.startswith(">"):
                if hdr:
                    recs.append((hdr, "".join(seq)))
                hdr, seq = line[1:].strip(), []
            elif line.strip():
                seq.append(line.strip())
        if hdr:
            recs.append((hdr, "".join(seq)))

        chains = meta.get(pdb, {}).get("chains", {})
        for chain, sequence in recs:
            moltype = MOLTYPE.get(chains.get(chain, {}).get("molecule_type"))
            if moltype is None:
                continue
            msa_id = f"{pdb}_{chain}"
            out.append(f">{msa_id}|{moltype}\n{sequence}")
            d = aln / msa_id
            d.mkdir(exist_ok=True)
            (d / "uniref90_hits.a3m").write_text(f">{msa_id}\n{sequence}\n")

    (root / "alignment_representatives.fasta").write_text("\n".join(out) + "\n")
    print(f"{len(out)} alignment representatives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
