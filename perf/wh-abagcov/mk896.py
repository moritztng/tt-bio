import sys
D = "/home/cust-team/mthuening/abagcov"
NCOL = 896
def read_a3m(p):
    hs, ss, h, buf = [], [], None, []
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if h is not None: hs.append(h); ss.append("".join(buf))
            h, buf = line, []
        else: buf.append(line)
    if h is not None: hs.append(h); ss.append("".join(buf))
    return hs, ss
def trim(seq, ncol):
    out, c = [], 0
    for ch in seq:
        if ch.islower():
            if c: out.append(ch)
            continue
        if c >= ncol: break
        out.append(ch); c += 1
    return "".join(out)
hs, ss = read_a3m(f"{D}/rungs/cdk2x2_1024_d8192.a3m")
q = trim(ss[0], NCOL)
assert len(q) == NCOL and "-" not in q
rows, seen, uniq = [trim(s, NCOL) for s in ss], set(), []
for h, r in zip(hs, rows):
    if r in seen: continue
    seen.add(r); uniq.append((h, r))
with open(f"{D}/rungs/cdk2x2_896_d8192.a3m", "w") as f:
    for h, r in uniq: f.write(h + "\n" + r + "\n")
open(f"{D}/rungs/cdk2x2_896_d8192.yaml", "w").write(
    "version: 1\n"
    "# CDK2 (PDB 1HCL) tiled to 896 residues: the published `measured_wall` rung, re-cut from\n"
    "# the 1024 rung's own alignment so the two differ in length and in nothing else.\n"
    "sequences:\n  - protein:\n      id: A\n"
    f"      sequence: {q}\n      msa: {D}/rungs/cdk2x2_896_d8192.a3m\n")
print("896 rung: unique rows", len(uniq), "query len", len(q))
