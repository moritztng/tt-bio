"""Key-by-key diff of two featurize_dry.py outputs; rigid-motion check per residue for ref_pos."""
import sys, numpy as np, torch
a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
ma, mb = a.pop("_meta"), b.pop("_meta")
print("atoms equal:", all(ma[k] == mb[k] for k in ("chain_id", "res_name", "res_id", "atom_name")))
ch = [i for i, (x, y) in enumerate(zip(ma["charge"], mb["charge"])) if x != y]
print("charge diffs:", [(ma["chain_id"][i], ma["res_name"][i], ma["res_id"][i], ma["atom_name"][i], ma["charge"][i], mb["charge"][i]) for i in ch])
same = 0
for k in sorted(set(a) | set(b)):
    x, y = a.get(k), b.get(k)
    if x is None or y is None or x.shape != y.shape:
        print("SHAPE/MISSING", k); continue
    if torch.equal(x, y): same += 1; continue
    print(f"DIFF {k}: {int((x != y).sum())}/{x.numel()} maxabs {float((x.double()-y.double()).abs().max()):.3g}")
print("identical keys", same, "of", len(set(a) | set(b)))
def kab(P, Q):
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q); d = np.sign(np.linalg.det(U @ Vt))
    return float(np.sqrt((((P @ (U @ np.diag([1, 1, d]) @ Vt)) - Q) ** 2).sum(1).mean()))
for key in ("feats/ref_pos", "feats/ref_pos_ground_truth"):
    x, y = a[key].numpy(), b[key].numpy(); n = len(ma["res_id"]); x, y = x[:n], y[:n]
    res = np.array([f"{c}{r}" for c, r in zip(ma["chain_id"], ma["res_id"])])
    worst, nd = 0.0, 0
    for r in dict.fromkeys(res):
        s = res == r
        if not np.array_equal(x[s], y[s]):
            nd += 1; worst = max(worst, kab(x[s], y[s]))
    print(f"{key}: residues differing {nd}/{len(set(res))}, worst post-superposition RMSD {worst:.2e} A, nonzero rows {int(np.any(y != 0, 1).sum())}")
