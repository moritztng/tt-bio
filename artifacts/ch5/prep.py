"""Build the three Challenge 5 target systems from PDB 7OW6.

7OW6 is an HLA-A*11:01 / beta-2m / KRAS-G12D-peptide complex bound by a TCR. The competition
reference is "chains A-C, TCR removed", so this strips chains D and E and writes:

  target_ABC.cif  the TCR-free complex, used as the co-fold template and to pick hotspots
  systems.json    the three co-fold inputs that differ only in the peptide:
                    MUT  VVVGADGVGK  the challenge target
                    WT   VVVGAGGVGK  counter-screen 1
                    APO  no peptide  counter-screen 2 (peptide-free MHC + b2m)

Nothing here mutates coordinates: the WT system is the same receptor with a different peptide
sequence, and letting the co-folder place it is more honest than editing a side chain by hand.
"""
import json, sys, collections, pathlib

KEEP = ("A", "B", "C")
PEPTIDE_CHAIN = "C"
MUT = "VVVGADGVGK"
WT = "VVVGAGGVGK"

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

src, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
lines = src.read_text().splitlines()

# locate the _atom_site loop
start = next(i for i, l in enumerate(lines) if l.startswith("_atom_site."))
i = start
cols = []
while lines[i].startswith("_atom_site."):
    cols.append(lines[i].strip().split(".", 1)[1]); i += 1
body_start = i
while i < len(lines) and not lines[i].startswith("#"):
    i += 1
body_end = i
ci = {c: n for n, c in enumerate(cols)}

kept, seqs, seen = [], collections.OrderedDict(), set()
for raw in lines[body_start:body_end]:
    f = raw.split()
    if not f or f[ci["auth_asym_id"]] not in KEEP:
        continue
    kept.append(raw)
    if f[ci["group_PDB"]] != "ATOM":
        continue
    ch, num = f[ci["auth_asym_id"]], f[ci["auth_seq_id"]]
    if (ch, num) in seen:
        continue
    seen.add((ch, num))
    seqs.setdefault(ch, []).append(AA3.get(f[ci["label_comp_id"]], "X"))
seqs = {ch: "".join(s) for ch, s in seqs.items()}

if seqs[PEPTIDE_CHAIN] != MUT:
    raise SystemExit("chain {} is {!r}, expected the G12D peptide {!r}".format(
        PEPTIDE_CHAIN, seqs[PEPTIDE_CHAIN], MUT))

(outdir / "target_ABC.cif").write_text(
    "data_7OW6_ABC\n#\nloop_\n"
    + "".join("_atom_site.{}\n".format(c) for c in cols)
    + "\n".join(kept) + "\n#\n")

heavy, b2m = seqs["A"], seqs["B"]
systems = {
    "source": "PDB 7OW6 chains A-C, TCR chains D/E removed",
    "receptor": {"heavy_chain_HLA_A1101": heavy, "beta2m": b2m},
    "systems": {
        "MUT": {"chains": [heavy, b2m, MUT], "role": "challenge target, KRAS G12D"},
        "WT":  {"chains": [heavy, b2m, WT],  "role": "counter-screen, wild-type KRAS"},
        "APO": {"chains": [heavy, b2m],      "role": "counter-screen, peptide-free MHC"},
    },
    "discriminating_residue": {"chain": PEPTIDE_CHAIN, "peptide_position": 6, "mut": "D", "wt": "G"},
}
(outdir / "systems.json").write_text(json.dumps(systems, indent=2) + "\n")

print("kept {} atom_site rows over chains {}".format(len(kept), "".join(KEEP)))
for ch in KEEP:
    print("  chain {}: {} residues".format(ch, len(seqs[ch])))
print("MUT/WT differ at peptide position 6 only: {} vs {}".format(MUT, WT))
