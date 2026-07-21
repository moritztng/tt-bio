import sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/opendde-covalent-bond-support")
import torch
from tt_bio.protenix_data import build_complex_features

chains = [("AC", None, "protein"), ("AC", None, "protein")]
chain_ids = ["A", "B"]
bonds = [(("A", 2, "SG"), ("B", 2, "SG"))]

feats = build_complex_features(chains, chain_ids=chain_ids, bonds=bonds)
tb = feats["token_bonds"]
asym = feats["asym_id"]
print("n_tokens:", tb.shape[0], "asym_id:", asym.tolist())
nz = (tb != 0).nonzero().tolist()
print("nonzero token_bonds pairs:", nz)
print("residue_index:", feats["residue_index"].tolist())
a_tok = [i for i in range(len(asym)) if asym[i] == 0 and feats["residue_index"][i] == 2]
b_tok = [i for i in range(len(asym)) if asym[i] == 1 and feats["residue_index"][i] == 2]
print("A-res2 token idx:", a_tok, "B-res2 token idx:", b_tok)
for i in a_tok:
    for j in b_tok:
        print("  tb[%d,%d] = %s  sym tb[%d,%d] = %s" % (i, j, tb[i,j].item(), j, i, tb[j,i].item()))
feats_nob = build_complex_features(chains, chain_ids=chain_ids, bonds=None)
print("no-bond token_bonds sum:", feats_nob["token_bonds"].sum().item())
print("with-bond token_bonds sum:", tb.sum().item())
