"""Consistency of OpenDDE's atom -> structural-token map against the residue-token map.

Host only, no device. The invariant the count assert in build_structural_token_features cannot
see: for every atom, parent[a2s[atom]] must be the residue token a2t[atom] already says the atom
belongs to. If the two featurizers ever walk atoms in a different order, the counts still match
and every atom silently lands on a neighbouring residue's structural token.
"""
import sys
import torch

from tt_bio.protenix_data import build_complex_features
from tt_bio.opendde_data import build_structural_token_features

CDK2 = ("MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQ"
        "DLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPE"
        "ILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDG"
        "RSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL")


def check(seq, tag):
    feats = build_complex_features([(seq, None, "protein")])
    ifd = build_structural_token_features(feats)
    a2t = feats["atom_to_token_idx"].long()
    a2s = ifd["atom_to_structural_token_idx"].long()
    parent = ifd["parent_residue_idx"].long()
    role = ifd["subtoken_role_id"].long()
    tokatom = ifd["atom_to_structural_tokatom_idx"].long()
    Ns, N = parent.shape[0], a2t.shape[0]

    bad_parent = (parent.index_select(0, a2s) != a2t).nonzero().flatten()
    oob = ((a2s < 0) | (a2s >= Ns)).nonzero().flatten()

    # every structural token must own a contiguous 0..k-1 tokatom block, and own >=1 atom
    counts = torch.bincount(a2s, minlength=Ns)
    empty = (counts == 0).nonzero().flatten()
    bad_tokatom = []
    for s in range(Ns):
        m = (a2s == s).nonzero().flatten()
        if m.numel() == 0:
            continue
        got = sorted(tokatom.index_select(0, m).tolist())
        if got != list(range(len(got))):
            bad_tokatom.append((s, got))

    print("%-10s len=%4d  N_atom=%5d  Ns=%4d  bad_parent=%d  oob=%d  empty_tokens=%d  bad_tokatom=%d"
          % (tag, len(seq), N, Ns, bad_parent.numel(), oob.numel(), empty.numel(), len(bad_tokatom)))
    if bad_parent.numel():
        print("   first bad atoms:", bad_parent[:12].tolist())
        print("   their a2t :", a2t.index_select(0, bad_parent[:12]).tolist())
        print("   their a2s->parent:", parent.index_select(0, a2s.index_select(0, bad_parent[:12])).tolist())
    if empty.numel():
        print("   empty structural tokens:", empty[:12].tolist(),
              "roles", role.index_select(0, empty[:12]).tolist())
    if bad_tokatom:
        print("   bad tokatom blocks:", bad_tokatom[:4])
    return bad_parent.numel() + oob.numel() + empty.numel() + len(bad_tokatom)


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [64, 110, 128, 256, 320, 384, 512, 544, 608]
    total = 0
    for n in sizes:
        total += check(CDK2[:n], "cdk2(%d)" % n)
    print("\nTOTAL violations:", total)
