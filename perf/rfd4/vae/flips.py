"""Which residue-head tokens flip on the device for one golden case, with token type, reference top-2 and TV.

    TT_VISIBLE_DEVICES=<chip> python perf/rfd4/vae/flips.py <case.pt> <ae.ckpt>
"""
import torch, sys
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio.rfd4 import ae_fold as AF
from tt_bio.rfd4.autoencoder import RFD4AutoEncoder
sd = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["state_dict"]
ae = RFD4AutoEncoder(sd, parts=("decoder",))
g = torch.load(sys.argv[1], map_location="cpu", weights_only=False); b = g["batch"]
d = ae.decode_raw(b); ref = g["decoder"]; em = ref["emit_mask"][0]
tt = b["token_type"][0]
legal = ae.tab["allowed"][b["token_type"].long()][0]
neg = torch.finfo(torch.float64).min
pr = torch.softmax(ref["seq_logits"][0].double().masked_fill(~legal, neg), -1)
pt = torch.softmax(d["seq_logits"][0].double().masked_fill(~legal, neg), -1)
tv = 0.5 * (pr - pt).abs().sum(-1)
print("token types emitted:", {int(t): int(((tt == t) & em).sum()) for t in tt.unique()})
print("tv by token type:", {int(t): round(float(tv[(tt == t) & em].mean()), 4) for t in tt.unique() if ((tt == t) & em).any()})
print("legal classes per type:", {int(t): int(legal[(tt == t)].sum(-1).max()) for t in tt.unique()})
fl = ((d["aatype_max"][0] != ref["aatype_max"][0]) & em).nonzero().flatten()
for i in fl.tolist():
    print("flip", i, "type", int(tt[i]), "ref", int(ref["aatype_max"][0, i]), "tt", int(d["aatype_max"][0, i]), "p_ref top2", [round(x, 3) for x in pr[i].topk(2).values.tolist()], "tv", round(float(tv[i]), 3))
top = tv[em].topk(5)
print("top tv tokens", [(int(em.nonzero().flatten()[j]), round(float(v), 3)) for v, j in zip(top.values, top.indices)])
if fl.numel():
    a, c = int(ref["aatype_max"][0, fl[0]]), int(d["aatype_max"][0, fl[0]])
    m = em & (tt == tt[fl[0]])
    dl = ((d["seq_logits"][0, :, a] - d["seq_logits"][0, :, c]).double() - (ref["seq_logits"][0, :, a] - ref["seq_logits"][0, :, c]).double())[m]
    print(f"logit{a} - logit{c} shift, card - ref, over {int(m.sum())} tokens: mean {float(dl.mean()):.4f} std {float(dl.std()):.4f} "
          f"ref gap median {float((ref['seq_logits'][0, :, a] - ref['seq_logits'][0, :, c]).double()[m].abs().median()):.4f}")
