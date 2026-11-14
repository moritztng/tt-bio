import sys, urllib.request, torch
from tt_bio.protenix_data import residue_atoms
seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGGSGSGSNLYIQWLKDGGPSSGRPPPS"
d = torch.load(sys.argv[1], weights_only=False)
folds = d["coords"] if isinstance(d["coords"], dict) else {"-": d["coords"]}
three = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU","G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}
off, ca = 0, []
for i, c in enumerate(seq):
    names = residue_atoms(three[c])
    ca.append(off + list(names).index("CA")); off += len(names)
pdb = urllib.request.urlopen("https://files.rcsb.org/download/1UBQ.pdb", timeout=30).read().decode()
R = torch.tensor([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in pdb.splitlines()
                  if l.startswith("ATOM") and l[12:16].strip() == "CA"])[:76]
B = R.double() - R.double().mean(0)
for seed, X in folds.items():
    P = X.reshape(-1, 3).float()[ca]
    A = P[:76].double(); A = A - A.mean(0)
    U, S, Vt = torch.linalg.svd(A.T @ B); dd = torch.sign(torch.det(U @ Vt))
    Rm = U @ torch.diag(torch.tensor([1, 1, dd], dtype=torch.float64)) @ Vt
    print("seed %s: CA-CA %.2f A, CA RMSD vs 1UBQ (76 res) %.2f A" % (
        seed, (P[1:] - P[:-1]).norm(dim=-1).mean(), (A @ Rm - B).norm(dim=-1).pow(2).mean().sqrt()))
