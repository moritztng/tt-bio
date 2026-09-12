"""Max absolute coordinate delta between two mmCIF structures, in Angstrom.

An identical sha256 says "the bytes match". It does not say how much numeric room there was:
mmCIF coordinates are written to three decimals, so anything under 0.0005 A writes the same bytes
and the digest cannot see it. This turns a control from a yes/no into a number.

Column positions come from the file's own _atom_site loop header rather than a fixed index, so a
writer that adds a field does not silently shift the comparison onto the wrong column.
"""
import sys


def coords(path):
    cols, out, in_loop = [], [], False
    for line in open(path):
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1])
            in_loop = True
            continue
        if in_loop and s.startswith(("ATOM", "HETATM")):
            f = s.split()
            ix = [cols.index(c) for c in ("Cartn_x", "Cartn_y", "Cartn_z")]
            out.append(tuple(float(f[i]) for i in ix))
        elif in_loop and out and not s.startswith(("ATOM", "HETATM")):
            in_loop = False
    return out


def main(a, b):
    ca, cb = coords(a), coords(b)
    if not ca or len(ca) != len(cb):
        print(f"ATOM-COUNT {len(ca)} vs {len(cb)} -- not comparable")
        return 1
    worst = max(abs(va - vb) for xa, xb in zip(ca, cb) for va, vb in zip(xa, xb))
    print(f"atoms {len(ca)}  MAX-ABS-DELTA {worst:.4f} A")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
