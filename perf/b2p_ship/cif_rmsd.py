"""CA-RMSD between two mmCIF outputs of the same fold, no superposition and with one.

Same sequence and same residue numbering by construction (the same input file), so the atoms are
matched on (chain, residue number) rather than by order.
"""
import math
import sys


def ca(path):
    out = {}
    cols = None
    for line in open(path):
        if line.startswith("_atom_site."):
            cols = cols or []
            cols.append(line.strip().split(".")[1])
            continue
        if cols and (line.startswith("ATOM") or line.startswith("HETATM")):
            f = line.split()
            if len(f) < len(cols):
                continue
            r = dict(zip(cols, f))
            if r.get("label_atom_id") == "CA":
                key = (r.get("label_asym_id"), r.get("label_seq_id"))
                out[key] = (float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"]))
        elif cols and line.startswith("#") and out:
            break
    return out


a, b = ca(sys.argv[1]), ca(sys.argv[2])
keys = sorted(set(a) & set(b))
if not keys:
    print("NO_COMMON_CA", len(a), len(b))
    raise SystemExit(1)
sq = sum(sum((a[k][i] - b[k][i]) ** 2 for i in range(3)) for k in keys)
print(f"CA_RMSD_NOSUPERPOSE {math.sqrt(sq / len(keys)):.4f} A over {len(keys)} CA "
      f"(of {len(a)} and {len(b)})")
mx = max(math.sqrt(sum((a[k][i] - b[k][i]) ** 2 for i in range(3))) for k in keys)
print(f"CA_MAX_DEV {mx:.4f} A")
