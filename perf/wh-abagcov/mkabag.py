import sys, yaml, hashlib
D = "/home/cust-team/mthuening/abagcov"
SRC_A3M = f"{D}/rungs/cdk2x2_1024_d8192.a3m"
NCOL = 596

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
    """Keep the first `ncol` MATCH columns of an a3m row. Lowercase = insertion, not a column."""
    out, c = [], 0
    for ch in seq:
        if ch.islower(): 
            if c: out.append(ch)          # insertions belong to the preceding match column
            continue
        if c >= ncol: break
        out.append(ch); c += 1
    return "".join(out)

hs, ss = read_a3m(SRC_A3M)
query = trim(ss[0], NCOL)
assert len(query.replace("-","")) == NCOL and "-" not in query, (len(query), query[:80])
rows = [trim(s, NCOL) for s in ss]
uniq, seen = [], set()
for hdr, r in zip(hs, rows):
    if r in seen: continue
    seen.add(r); uniq.append((hdr, r))
with open(f"{D}/rungs/abag1024_A.a3m", "w") as f:
    for hdr, r in uniq: f.write(hdr + "\n" + r + "\n")
print("antigen a3m rows written:", len(uniq), "of", len(rows), "(dedup collapses the rest)")

d = yaml.safe_load(open(f"{D}/eng-main/examples/1ahw_abag.yaml"))
chains = {s["protein"]["id"]: s["protein"]["sequence"] for s in d["sequences"]}
for cid in ("H", "L"):
    with open(f"{D}/rungs/abag1024_{cid}.a3m", "w") as f:
        f.write(f">{cid}_query\n{chains[cid]}\n")

body = [f"""  - protein:
      id: A
      sequence: {query}
      msa: {D}/rungs/abag1024_A.a3m"""]
for cid in ("H", "L"):
    body.append(f"""  - protein:
      id: {cid}
      sequence: {chains[cid]}
      msa: {D}/rungs/abag1024_{cid}.a3m""")
hdr = f"""version: 1
# OpenDDE-abag capacity rung, 1024 tokens, antibody-antigen SHAPED (3 chains) rather than the
# single-chain ladder rung. Antigen A is CDK2 (PDB 1HCL) tiled to {NCOL} residues so the
# 8192-row alignment of the single-chain rung can be trimmed to it column-for-column; H and L
# are the 1ahw Fab 5G9 heavy and light chains, run single-sequence as antibody chains are.
# {NCOL} + {len(chains['H'])} + {len(chains['L'])} = {NCOL+len(chains['H'])+len(chains['L'])} tokens.
sequences:
"""
open(f"{D}/rungs/abag1024x.yaml", "w").write(hdr + "\n".join(body) + "\n")
tot = NCOL + len(chains["H"]) + len(chains["L"])
print("tokens:", tot, "= 32 *", tot/32)
