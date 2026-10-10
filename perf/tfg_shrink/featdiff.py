"""Compare tt-bio's guidance features for an example against upstream's (GPU dump feats)."""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, ".")
from tt_bio.main import _read_bio_bonds, _read_bio_chains
from tt_bio.protenix_data import build_complex_features
from tt_bio.worker import _tfg_guidance
path = Path(sys.argv[1]); up = torch.load(sys.argv[2], weights_only=False, map_location="cpu")["feats"]
chains = _read_bio_chains(path); bonds = _read_bio_bonds(path, chains)
feats = build_complex_features([(s, None, mt) for _c, s, _sp, mt, _m in chains], chain_ids=[c for c, *_r in chains],
                               bonds=bonds, modifications=[m for *_x, m in chains])
g = _tfg_guidance(path, {"use_tfg_guidance": True}, feats, chains, bonds)
ours = g.feats
for k in sorted(up):
    if k.startswith(("msa", "template", "has_del", "deletion", "profile")): continue
    if k not in ours: print("MISSING in ours", k, tuple(up[k].shape)); continue
    a, b = ours[k], up[k]
    a = torch.as_tensor(a)
    if a.shape != b.shape: print("SHAPE", k, tuple(a.shape), tuple(b.shape)); continue
    eq = torch.equal(a.to(b.dtype), b)
    print("EQ  " if eq else "DIFF", k, tuple(b.shape), "" if eq else f"maxdiff {(a.double()-b.double()).abs().max().item():.3g} n={(a.to(b.dtype)!=b).sum().item()}")
print("extra in ours:", sorted(set(ours) - set(up))[:40])
