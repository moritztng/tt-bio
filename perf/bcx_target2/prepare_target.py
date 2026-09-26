#!/usr/bin/env python3
"""Turn an RCSB PDB entry into a BindCraft 2 target structure.

This is the whole of the 28 Sep motion for a target BC2 does not ship: download the entry,
keep one chain, drop waters, ligands, hydrogens and alternate locations, write it next to the
settings JSON. BindCraft 2 reads `target_path` relative to the settings file it came from
(`bindcraft/settings.py:711`), so the pair travels together.

    python3 prepare_target.py 193L A hewl_193L_A.pdb

The shipped targets under `settings/target/structures/` are ATOM-only single-chain files with
their original residue numbering, and this writes the same shape, because `hotspots` in the
settings JSON are read in the structure own numbering.
"""
import sys
import urllib.request

def main():
    entry, chain, out = sys.argv[1], sys.argv[2], sys.argv[3]
    url = f"https://files.rcsb.org/download/{entry}.pdb"
    src = urllib.request.urlopen(url, timeout=60).read().decode().splitlines()
    kept, residues = [], {}
    for line in src:
        if not line.startswith("ATOM") or line[21] != chain:
            continue
        if line[16] not in (" ", "A"):          # one alternate location, the first
            continue
        if line[76:78].strip() == "H":
            continue
        kept.append(line[:16] + " " + line[17:])
        residues[int(line[22:26])] = line[17:20]
    if not kept:
        raise SystemExit(f"{entry} chain {chain}: no ATOM records")
    lo, hi = min(residues), max(residues)
    gaps = [i for i in range(lo, hi) if i not in residues]
    with open(out, "w") as fh:
        fh.write("\n".join(kept) + "\nTER\nEND\n")
    print(f"{out}: {len(residues)} residues {lo}-{hi}, {len(kept)} atoms, "
          f"{len(gaps)} numbering gaps: {gaps}")

if __name__ == "__main__":
    main()
