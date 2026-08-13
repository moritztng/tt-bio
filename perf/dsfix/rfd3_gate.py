"""Accuracy / geometry gate for the RFD3 finalist, run on a full 200-timestep production design.

Per plan sec.1.5 the perf runs get only a cheap structural check because a 2-step diffusion output is
noise by construction. The chosen fixture gets the real gate, once, at shipped settings:

  1. the designed chain's sequence is a REAL sequence, not a poly-alanine fallback (sec.1.5 names
     this as the first thing to confirm on the RFD3 leg);
  2. consecutive CA-CA distance over the designed chain is 3.80 +/- 0.30 A;
  3. no non-bonded heavy-atom pair closer than 1.5 A.

Reads the CIF written by `tt_bio.main design --num_timesteps 200`.
"""
import collections, json, math, pathlib, sys

CIF = sys.argv[1]
DESIGNED_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# Column order is header-defined and differs between writers: tt_bio's RFD3 CIF puts Cartn_x at
# index 15, BoltzGen's at index 10. Read _atom_site.* in order and index by name, never by a
# hardcoded position.
cols, atoms = [], []
for line in pathlib.Path(CIF).read_text().splitlines():
    t = line.strip()
    if t.startswith("_atom_site."):
        cols.append(t.split(".", 1)[1].split()[0])
        continue
    if line.startswith(("ATOM", "HETATM")):
        f = line.split()
        g = dict(zip(cols, f))
        atoms.append({"name": g["label_atom_id"], "res": g["label_comp_id"],
                      "chain": g["label_asym_id"], "seq": g["label_seq_id"],
                      "x": float(g["Cartn_x"]), "y": float(g["Cartn_y"]),
                      "z": float(g["Cartn_z"])})
if not atoms:
    print("GATE FAIL: parsed 0 atoms from %s (cols=%s)" % (CIF, cols)); sys.exit(1)

chains = collections.OrderedDict()
for a in atoms:
    chains.setdefault(a["chain"], collections.OrderedDict())[a["seq"]] = a["res"]

print("chains:", {c: len(r) for c, r in chains.items()})

# RFD3 writes the contig as ONE chain: `A1-585,100` emits 585 fixed target residues followed by
# 100 designed ones, all under label_asym_id A. BoltzGen instead writes two chains. Handle both:
# prefer a chain whose residue count is exactly the binder length, else take the tail of the
# largest chain.
designed = [c for c, r in chains.items() if len(r) == DESIGNED_LEN]
if designed:
    dc = designed[0]
    keys = list(chains[dc])
    tgt_keys = []
else:
    dc = max(chains, key=lambda c: len(chains[c]))
    allk = list(chains[dc])
    if len(allk) <= DESIGNED_LEN:
        print("GATE FAIL: chain %s has only %d residues" % (dc, len(allk)))
        sys.exit(1)
    keys, tgt_keys = allk[-DESIGNED_LEN:], allk[:-DESIGNED_LEN]
    print("single-chain contig layout: chain %s, %d target + %d designed"
          % (dc, len(tgt_keys), len(keys)))

seq = [chains[dc][k] for k in keys]
DESIGNED_KEYS = set(keys)

# The fixed motif must come through untouched: compare the target-half sequence against the
# converted target PDB the fixture actually reads.
if tgt_keys and len(sys.argv) > 4:
    tgt_pdb = pathlib.Path(sys.argv[4])
    seen, ref = set(), []
    for line in tgt_pdb.read_text().splitlines():
        if line.startswith("ATOM"):
            rid = line[22:27]
            if rid not in seen:
                seen.add(rid); ref.append(line[17:20].strip())
    got = [chains[dc][k] for k in tgt_keys]
    same = sum(1 for a, b in zip(ref, got) if a == b)
    print("[0] fixed motif vs %s: %d/%d residues identical: %s"
          % (tgt_pdb.name, same, len(got), "PASS" if same == len(got) else "FAIL"))
    MOTIF_OK = (same == len(got) and len(ref) == len(got))
else:
    MOTIF_OK = True
comp = collections.Counter(seq)
top, topn = comp.most_common(1)[0]
frac_top = topn / len(seq)
n_distinct = len(comp)

print("designed chain %s: %d residues, %d distinct aa, most common %s at %.1f%%"
      % (dc, len(seq), n_distinct, top, 100 * frac_top))
print("sequence:", "".join({
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}.get(r, "X")
    for r in seq))

# 1. real sequence, not a poly-A fallback
seq_ok = n_distinct >= 10 and frac_top <= 0.30
print("[1] real sequence (>=10 distinct aa and no residue >30%%): %s" % ("PASS" if seq_ok else "FAIL"))

# 2. consecutive CA-CA over the designed chain
ca = [a for a in atoms if a["chain"] == dc and a["name"] == "CA" and a["seq"] in DESIGNED_KEYS]
ca.sort(key=lambda a: int(a["seq"]))
d = [math.dist((ca[i]["x"], ca[i]["y"], ca[i]["z"]),
               (ca[i + 1]["x"], ca[i + 1]["y"], ca[i + 1]["z"])) for i in range(len(ca) - 1)]
d.sort()
med = d[len(d) // 2]
out = [v for v in d if abs(v - 3.80) > 0.30]
raw = [math.dist((ca[i]["x"], ca[i]["y"], ca[i]["z"]),
                 (ca[i + 1]["x"], ca[i + 1]["y"], ca[i + 1]["z"])) for i in range(len(ca) - 1)]
bad_at = [(i, ca[i]["seq"], round(v, 3)) for i, v in enumerate(raw) if abs(v - 3.80) > 0.30]
print("    CA-CA outliers (index, seq, dist):", bad_at)
ca_ok = not out
print("[2] CA-CA over designed chain: n=%d median %.3f A, range %.3f-%.3f, %d outside 3.80+/-0.30: %s"
      % (len(d), med, d[0], d[-1], len(out), "PASS" if ca_ok else "FAIL"))

# 3. non-bonded heavy-atom clashes, whole complex, grid-bucketed.
# RFD3 pads every DESIGNED token to MAX_ATOMS_PER_TOKEN = 14: 5 real atoms (N, CA, C, O, CB) and
# 9 placeholders named V0-V8. Placeholders are not atoms and must not be scored as clashes.
atoms = [a for a in atoms if not (len(a["name"]) == 2 and a["name"][0] == "V"
                                  and a["name"][1].isdigit())]
print("    (excluded V0-V8 padding placeholders; %d real atoms scored)" % len(atoms))
CELL = 4.0
grid = collections.defaultdict(list)
for i, a in enumerate(atoms):
    grid[(int(a["x"] // CELL), int(a["y"] // CELL), int(a["z"] // CELL))].append(i)
clashes = []
for i, a in enumerate(atoms):
    gx, gy, gz = int(a["x"] // CELL), int(a["y"] // CELL), int(a["z"] // CELL)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for j in grid.get((gx + dx, gy + dy, gz + dz), ()):
                    if j <= i:
                        continue
                    b = atoms[j]
                    same = a["chain"] == b["chain"] and a["seq"] == b["seq"]
                    adj = (a["chain"] == b["chain"]
                           and abs(int(a["seq"]) - int(b["seq"])) == 1)
                    if same or adj:      # intra-residue and peptide-bonded neighbours are bonded
                        continue
                    r = math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"]))
                    if r < 1.5:
                        clashes.append((round(r, 3), a["chain"] + a["seq"] + a["name"],
                                        b["chain"] + b["seq"] + b["name"]))
cl_ok = not clashes
print("[3] non-bonded heavy-atom pairs < 1.5 A: %d: %s" % (len(clashes), "PASS" if cl_ok else "FAIL"))
for c in clashes[:5]:
    print("     ", c)

allok = seq_ok and ca_ok and cl_ok and MOTIF_OK
print("\nGATE %s" % ("PASS" if allok else "FAIL"))
json.dump({"cif": CIF, "designed_chain": dc, "n_designed": len(seq), "motif_ok": MOTIF_OK,
           "n_distinct_aa": n_distinct, "most_common_aa": top, "most_common_frac": round(frac_top, 4),
           "seq_ok": seq_ok, "ca_n": len(d), "ca_median": round(med, 4),
           "ca_min": round(d[0], 4), "ca_max": round(d[-1], 4), "ca_outside": len(out),
           "ca_ok": ca_ok, "n_clashes": len(clashes), "clash_ok": cl_ok, "gate_pass": allok},
          open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout, indent=1)
