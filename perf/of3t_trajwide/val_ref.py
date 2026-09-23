import sys, os, importlib.util, torch
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location(
    "tw", "/home/ttuser/.coworker/wt/of3t-trajwide/perf/of3t_trajwide/trajwide.py")
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)
tw.refpath.install()
# stub is applied AFTER build_theirs in the real path
m, own, inc, moved = tw.build_theirs(torch.float64)
print("reference tree resolved:", tw.REF_TREE)
print("load is TOTAL: %d missing, %d unexpected" % (len(inc.missing_keys), len(inc.unexpected_keys)))
print("layer_norm_z realigned:", moved)
np_ = dict(m.named_parameters())
print("reference parameters now:", len(np_), "(was 738)")
k = "diffusion_transformer.blocks.0.attention_pair_bias.layer_norm_z.weight"
print("per-block key present  :", k in np_)
if k in np_:
    v = np_[k].detach().double()
    ck = own[k].double()
    print("   matches checkpoint  :", bool(torch.equal(v, ck)))
    print("   first 8             :", [round(float(x), 5) for x in v[:8]])
print("shared key still a param:", "diffusion_transformer.layer_norm_z.weight" in np_)
print("atom enc shared kept    :", "atom_attn_enc.atom_transformer.layer_norm_z.weight" in np_)
