"""Where on the structural-token axis does OpenDDE's conditioning go bad?

Runs a real fold and, at the seam, reports per-token statistics of every tensor the diffusion
is conditioned on -- before the refiner, after the refiner, and after the pair conditioning --
then folds and maps the ejected residues back to the structural tokens that produced them.

The symptom being chased: all atoms of one structural token collapse onto each other 15-20 A off
the chain (residue 121: N-CA 0.79, CA-C 0.62, C-O 0.20 A). Every atom of a token shares the same
broadcast token representation, so a token whose representation blows up drags its whole atom set
to one point. This finds those tokens if they exist.

Usage: python perf/wh-correctness/struct_cond_probe.py <fasta> [n_step]
"""
import sys
import torch


def outliers(name, t, k=10):
    """Per-token L2 norm profile of a (Ns, C) or (Ns, Ns, C) tensor."""
    x = t.float()
    if x.dim() == 3:
        n = x.flatten(1).norm(dim=1)
    else:
        n = x.norm(dim=1)
    med = n.median()
    ratio = n / med.clamp_min(1e-9)
    top = ratio.topk(min(k, n.numel()))
    print("%-16s  median=%10.3f  max=%10.3f (%.1fx median)  nonfinite=%d"
          % (name, med.item(), n.max().item(), ratio.max().item(), int((~torch.isfinite(x)).sum())))
    print("     worst tokens:", [(int(i), round(float(r), 2)) for i, r in zip(top.indices, top.values)])
    return ratio


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.opendde import OpenDDE

    fasta = sys.argv[1]
    n_step = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seq = "".join(l.strip() for l in open(fasta) if not l.startswith(">"))
    print("seq len %d, n_step %d" % (len(seq), n_step))
    feats = build_complex_features([(seq, None, "protein")])

    get_device()
    model = OpenDDE.load_from_checkpoint()

    captured = {}
    from tt_bio.opendde import StructuralTokenExpander
    exp_call = StructuralTokenExpander.__call__
    ear = model.expand_and_refine

    def wrapped_expander(self, ifd, s_inputs_res, s_res, z_res):
        out = exp_call(self, ifd, s_inputs_res, s_res, z_res)
        Ns = ifd["subtoken_role_id"].shape[0]
        captured["ifd"] = ifd
        captured["s_exp"] = ttnn.to_torch(out[1])[:Ns, :model.expander.c_s].clone()
        captured["si_exp"] = ttnn.to_torch(out[0])[:Ns, :model.expander.c_s_inputs].clone()
        captured["ab"] = ttnn.to_torch(out[3])[:Ns, :Ns].clone()
        return out

    def wrapped_ear(ifd, si, s, z, **kw):
        out = ear(ifd, si, s, z, **kw)
        Ns = ifd["subtoken_role_id"].shape[0]
        captured["s_ref"] = ttnn.to_torch(out[1])[:Ns, :model.expander.c_s].clone()
        return out

    StructuralTokenExpander.__call__ = wrapped_expander
    model.expand_and_refine = wrapped_ear

    traj = {}
    coords = model.fold(feats, n_step=n_step, n_cycles=2, seed=0, n_sample=1,
                        dump_fn=lambda s, st, x: traj.setdefault(st, x.clone()))
    coords = coords[0]

    ifd = captured["ifd"]
    parent = ifd["parent_residue_idx"].long()
    role = ifd["subtoken_role_id"].long()
    a2s = ifd["atom_to_structural_token_idx"].long()
    Ns = parent.shape[0]
    print("\nNs=%d  N_atom=%d" % (Ns, coords.shape[0]))

    print("\n--- conditioning, per structural token ---")
    r_si = outliers("s_inputs_struct", captured["si_exp"])
    r_se = outliers("s_expander", captured["s_exp"])
    r_sr = outliers("s_refined", captured["s_ref"])
    ab = captured["ab"].float()
    print("attn_bias         min=%.4f max=%.4f nonfinite=%d" % (ab.min(), ab.max(), int((~torch.isfinite(ab)).sum())))

    # --- which tokens actually collapsed? ---
    print("\n--- collapsed tokens in the output structure ---")
    bad = []
    for s in range(Ns):
        m = (a2s == s).nonzero().flatten()
        if m.numel() < 2:
            continue
        p = coords.index_select(0, m)
        span = (p[:, None, :] - p[None, :, :]).norm(dim=-1).max()
        if span < 1.0:                      # a real 4-atom backbone spans ~2.4 A
            bad.append((s, float(span), int(parent[s]), int(role[s]), int(m.numel())))
    print("collapsed tokens (max intra-token atom span < 1.0 A): %d / %d" % (len(bad), Ns))
    for s, span, p, ro, na in bad[:25]:
        print("   tok %4d  parent res %4d  role %d  natoms %d  span %.3f A   |s_exp|=%.2fx |s_ref|=%.2fx"
              % (s, p, ro, na, span, r_se[s], r_sr[s]))

    if bad:
        idx = torch.tensor([b[0] for b in bad])
        print("\n   collapsed-token norm ratios: s_inputs %.2f  s_exp %.2f  s_ref %.2f (median over collapsed)"
              % (r_si[idx].median(), r_se[idx].median(), r_sr[idx].median()))
        print("   attn_bias row mean over collapsed: %.4f  vs all: %.4f"
              % (ab[idx].mean(), ab.mean()))

    # --- when in the trajectory does it happen? ---
    if traj:
        print("\n--- trajectory: intra-token span of the collapsed tokens by step ---")
        steps = sorted(traj)
        for st in steps[:: max(1, len(steps) // 8)] + [steps[-1]]:
            x = traj[st].reshape(-1, 3)
            sp = []
            for s, *_ in bad[:8]:
                m = (a2s == s).nonzero().flatten()
                p = x.index_select(0, m)
                sp.append(float((p[:, None, :] - p[None, :, :]).norm(dim=-1).max()))
            print("   step %3d: %s" % (st, " ".join("%6.2f" % v for v in sp)))


if __name__ == "__main__":
    main()
