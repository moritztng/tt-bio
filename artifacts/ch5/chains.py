"""List per-chain one-letter sequences from an mmCIF, for target prep."""
import sys, collections

AA3 = {"ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
       "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
       "THR":"T","TRP":"W","TYR":"Y","VAL":"V"}

def atom_site(path):
    lines = open(path).read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("_atom_site."):
            cols = []
            while lines[i].startswith("_atom_site."):
                cols.append(lines[i].strip().split(".", 1)[1]); i += 1
            rows = []
            while i < len(lines) and not lines[i].startswith("#"):
                rows.append(lines[i].split()); i += 1
            return cols, rows
        i += 1
    raise SystemExit("no _atom_site loop")

cols, rows = atom_site(sys.argv[1])
ci = {c: n for n, c in enumerate(cols)}
seqs, seen = collections.OrderedDict(), set()
for r in rows:
    if r[ci["group_PDB"]] != "ATOM":
        continue
    ch, num, comp = r[ci["auth_asym_id"]], r[ci["auth_seq_id"]], r[ci["label_comp_id"]]
    if (ch, num) in seen:
        continue
    seen.add((ch, num))
    seqs.setdefault(ch, []).append(AA3.get(comp, "X"))
for ch, s in seqs.items():
    s = "".join(s)
    print("chain {}: len {}  {}".format(ch, len(s), s))
