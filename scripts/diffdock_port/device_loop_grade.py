"""Grade each `tt_bio.diffdock.device_loop` block on a card against its float64 torch twin.

    TT_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/diffdock_port/device_loop_grade.py --out g.json

Edge vectors span 0-40 A in every direction, plus exact zeros (pad edges). Inputs are drawn once
on the host, so card and reference see the same fp32 numbers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))


def stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.detach().double().flatten(), ref.detach().double().flatten()
    return {"max_abs": float((g - r).abs().max()),
            "rel_frobenius": float((g - r).norm() / r.norm().clamp_min(1e-30)),
            "pcc": float(torch.corrcoef(torch.stack([g, r]))[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import device_loop as dl
    from tt_bio.diffdock.features import gaussian_smearing_reference
    from tt_bio.diffdock.harmonics import spherical_harmonics
    from tt_bio.diffdock.score_model import load_released
    from tt_bio.tenstorrent import get_device

    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    model = load_released(sd, cfg).double()
    g = torch.Generator().manual_seed(0)
    vec = (torch.rand(args.edges, 3, generator=g) * 2 - 1) * 40
    vec[:32] = 0
    vec = vec.float()
    dev = get_device()
    xyz = [dl.upload(dev, vec[:, k]) for k in range(3)]
    back = lambda t, w: ttnn.to_torch(t)[:args.edges, :w]
    out = {"edges": args.edges}

    def flush_json():
        """Write what is measured so far, so a later leg's failure costs only that leg."""
        Path(args.out).write_text(json.dumps(out, indent=1))

    def record(name, got, ref):
        """Measure one leg and print it at once: a later failure keeps the earlier numbers."""
        out[name] = stats(got, ref)
        print(name, out[name], flush=True)
        flush_json()
        return out[name]

    dist = vec.double().norm(dim=-1)
    record("norm3", back(dl.norm3(*xyz), 1)[:, 0], dist)
    for ls in ((0, 1, 2), (2,)):
        ref = spherical_harmonics(vec.double(), ls)
        record(f"harmonics_{''.join(map(str, ls))}", back(dl.harmonics(*xyz, ls), ref.shape[1]), ref)
    d_dev = dl.upload(dev, dist.float())
    for name in ("lig", "rec", "cross"):
        mod = getattr(model, f"{name}_distance_expansion")
        ref = gaussian_smearing_reference(dist.float(), mod)
        record(f"smear_{name}", back(dl.Smearing(dev, mod)(d_dev), ref.shape[1]), ref)
    for name in ("rec_edge_embedding", "cross_edge_embedding", "lig_edge_embedding",
                 "rec_sigma_embedding"):
        seq = getattr(model, name)
        x = torch.randn(args.edges, seq[0].in_features, generator=g).float()
        ref = seq(x.double())
        record(name, back(dl.MLP(dev, seq.float())(dl.upload(dev, x)), ref.shape[1]), ref)
        seq.double()
    # --- the pose update ------------------------------------------------------------
    from tt_bio.diffdock.sampling import axis_angle_to_matrix, guard_scores

    B = 320
    aa = torch.randn(B, 3, generator=g).float() * 1.5
    aa[:32] = 0                       # the small-angle series, and the exact-zero axis
    aa[32:64] *= 1e-7
    cols = [dl.upload(dev, aa[:, k]) for k in range(3)]
    ref_m = axis_angle_to_matrix(aa.double())
    got_m = dl.axis_angle_matrix(*cols)
    record("axis_angle_matrix", 
        torch.cat([ttnn.to_torch(c)[:B, :1] for c in got_m], 1), ref_m.reshape(B, 9))

    pts = torch.randn(B, 3, generator=g).float() * 12
    pcols = [dl.upload(dev, pts[:, k]) for k in range(3)]
    ref_r = torch.bmm(pts.double()[:, None, :], ref_m.transpose(1, 2))[:, 0]
    record("rotate", 
        torch.cat([ttnn.to_torch(c)[:B, :1] for c in dl.rotate(pcols, got_m)], 1), ref_r)

    score = torch.randn(B, 3, generator=g).float()
    score[3, 1], score[7, 0], score[11, 2] = float("nan"), float("inf"), float("-inf")
    ref_g = guard_scores(score.double())[0]
    got_g = dl.guard([dl.upload(dev, score[:, k]) for k in range(3)])
    record("guard", torch.cat([ttnn.to_torch(c)[:B, :1] for c in got_g], 1), ref_g)

    # --- gathered edge geometry -----------------------------------------------------
    n_nodes, E = 512, args.edges
    pos = (torch.rand(n_nodes, 3, generator=g) * 2 - 1).float() * 30
    src = torch.randint(0, n_nodes, (E,), generator=g)
    dst = torch.randint(0, n_nodes, (E,), generator=g)
    dst[:32] = src[:32]                       # pad edges: a zero vector, zero harmonics
    geo = dl.EdgeGeometry(dev, src, dst, E)
    pos_dev = dl.upload(dev, pos)
    vcols = geo.vectors(pos_dev)
    ref_v = pos.double()[dst] - pos.double()[src]
    record("gathered_edge_vectors", 
        torch.cat([ttnn.to_torch(c)[:E, :1] for c in vcols], 1), ref_v)
    ref_sh = spherical_harmonics(ref_v, (0, 1, 2))
    record("gathered_harmonics", back(dl.harmonics(*vcols), ref_sh.shape[1]), ref_sh)
    d_gath = dl.norm3(*vcols)
    ref_m = (ref_v.norm(dim=-1) <= 5.0).double()
    got_m = ttnn.to_torch(geo.within(d_gath, 5.0))[:E, 0]
    out["within"] = {"mismatches": int((got_m.double() - ref_m).abs().sum()),
                     "kept": int(ref_m.sum())}
    print("within", out["within"], flush=True)
    flush_json()

    # --- the step's masks ------------------------------------------------------------
    from tt_bio.diffdock.graph import StaticGraphs

    n_lig, n_rec, n_poses = 24, 48, 4
    bonds = torch.stack([torch.arange(0, 10), torch.arange(1, 11)])
    sg = StaticGraphs.build(n_lig, n_rec, bonds, n_poses=n_poses)
    lp = (torch.rand(sg.n_lig_padded, 3, generator=g) * 2 - 1).float() * 6
    rp = (torch.rand(sg.n_rec_padded, 3, generator=g) * 2 - 1).float() * 18
    nl, nc = sg.lig_edge_index.shape[1], sg.cross_edge_index.shape[1]
    masks = dl.StepMasks(dev, sg, nl, nc)
    lp_dev, rp_dev = dl.upload(dev, lp), dl.upload(dev, rp)

    # ligand: bond rows always on, the complete block by radius
    lgeo = dl.EdgeGeometry(dev, sg.lig_edge_index[0], sg.lig_edge_index[1], nl)
    ldist = dl.norm3(*lgeo.vectors(lp_dev))
    ref_lm = sg.lig_mask(lp.double(), cfg.lig_max_radius)
    got_lm = ttnn.to_torch(masks.lig(ldist, cfg.lig_max_radius))[:nl, 0]
    out["mask_lig"] = {"mismatches": int((got_lm.double() - ref_lm).abs().round().sum()),
                       "on": int(ref_lm.sum()), "edges": nl}
    print("mask_lig", out["mask_lig"], flush=True)
    flush_json()

    # cross: the host divides the vector by the cutoff and compares the norm with 1
    cgeo = dl.EdgeGeometry(dev, sg.cross_edge_index[0], sg.cross_edge_index[1], nc)
    cvec = cgeo.vectors(lp_dev, rp_dev)
    for label, cutoff in (("scalar", 20.0), ("per_pose", None)):
        if cutoff is None:
            per_pose = torch.tensor([14.0, 20.0, 26.0, 32.0])[:n_poses]
            ref_cm = sg.cross_mask(lp.double(), rp.double(), per_pose.double())
            col = per_pose.repeat_interleave(n_lig * n_rec).float()
            col = torch.nn.functional.pad(col, (0, nc - col.shape[0]), value=1.0)
            arg = dl.upload(dev, col)
        else:
            ref_cm = sg.cross_mask(lp.double(), rp.double(), float(cutoff))
            arg = float(cutoff)
        got_cm = ttnn.to_torch(masks.cross(*cvec, arg))[:nc, 0]
        key = f"mask_cross_{label}"
        out[key] = {"mismatches": int((got_cm.double() - ref_cm).abs().round().sum()),
                    "on": int(ref_cm.sum()), "edges": nc}
        print(key, out[key], flush=True)
        flush_json()

    # receptor edges: an edge survives iff both ends do
    rei = sg.rec_edge_index(torch.stack([torch.arange(0, 30), torch.arange(1, 31)]))
    ne = rei.shape[1]
    keep = (torch.rand(sg.n_rec_padded, generator=g) > 0.35).float()
    keep[sg.n_rec * n_poses:] = 0
    ref_rm = StaticGraphs.rec_edge_mask(keep.double(), rei)
    rsrc = dl.EdgeGeometry(dev, rei[0], rei[1], ne)
    got_rm = ttnn.to_torch(dl.StepMasks.rec_edges(
        dl.upload(dev, keep), rsrc.src, rsrc.dst, ne))[:ne, 0]
    out["mask_rec_edges"] = {"mismatches": int((got_rm.double() - ref_rm).abs().round().sum()),
                             "on": int(ref_rm.sum()), "edges": ne}
    print("mask_rec_edges", out["mask_rec_edges"], flush=True)
    flush_json()

    # --- the three towers' edge and node features ------------------------------------
    from tt_bio.diffdock.features import timestep_embedding

    def cut(x, n, w):
        """back() clips at --edges; the cross tower has more edges than that."""
        return ttnn.to_torch(x)[:n, :w]

    t_tr = 0.73
    sigma_emb = timestep_embedding(torch.tensor([t_tr]).double(), cfg.sigma_embed_dim,
                                   cfg.embedding_scale)
    sigma_row = dl.upload(dev, sigma_emb.float())
    tf = dl.TowerFeatures(dev, model.float(), sg, nl, nc)
    model.double()

    # receptor: static over the rollout
    rvec_cols = [dl.upload(dev, x) for x in
                 (rp[rei[1]] - rp[rei[0]]).float().unbind(1)]
    ref_rv = (rp.double()[rei[1]] - rp.double()[rei[0]])
    ref_re = model.rec_edge_embedding(model.rec_distance_expansion(ref_rv.norm(dim=-1)))
    got_re, got_rsh = tf.receptor_static(rvec_cols)
    record("tower_rec_eattr", cut(got_re, ne, ref_re.shape[1]), ref_re)
    ref_rsh = spherical_harmonics(ref_rv, tuple(model.sh_ls))
    record("tower_rec_sh", cut(got_rsh, ne, ref_rsh.shape[1]), ref_rsh)

    # ligand: bond block + broadcast sigma + distance expansion
    lsig = sigma_emb.expand(sg.n_lig_padded, -1)
    lvec = (lp.double()[sg.lig_edge_index[1]] - lp.double()[sg.lig_edge_index[0]])
    bond_w = (model.lig_edge_embedding[0].in_features - sigma_emb.shape[1]
              - model.lig_distance_expansion.offset.numel())
    bond = torch.zeros(nl, bond_w, dtype=torch.float64)
    bond[:sg.lig_n_bond_edges] = torch.rand(sg.lig_n_bond_edges, bond_w, generator=g).double()
    ref_le = model.lig_edge_embedding(torch.cat(
        [bond, lsig[sg.lig_edge_index[0]], model.lig_distance_expansion(lvec.norm(dim=-1))], 1))
    got_le, got_lsh = tf.ligand(lgeo.vectors(lp_dev), dl.upload(dev, bond.float()), sigma_row)
    record("tower_lig_eattr", cut(got_le, nl, ref_le.shape[1]), ref_le)
    ref_lsh = spherical_harmonics(lvec, tuple(model.sh_ls))
    record("tower_lig_sh", cut(got_lsh, nl, ref_lsh.shape[1]), ref_lsh)

    # cross: edge attributes, harmonics and the reverse harmonics
    cvec_h = (rp.double()[sg.cross_edge_index[1]] - lp.double()[sg.cross_edge_index[0]])
    ref_ce = model.cross_edge_embedding(torch.cat(
        [lsig[sg.cross_edge_index[0]], model.cross_distance_expansion(cvec_h.norm(dim=-1))], 1))
    got_ce, got_csh, got_rev = tf.cross(cvec, sigma_row)
    record("tower_cross_eattr", cut(got_ce, nc, ref_ce.shape[1]), ref_ce)
    ref_csh = spherical_harmonics(cvec_h, tuple(model.sh_ls))
    record("tower_cross_sh", cut(got_csh, nc, ref_csh.shape[1]), ref_csh)
    ref_rev = spherical_harmonics(-cvec_h, tuple(model.sh_ls))
    record("tower_cross_rev_sh", cut(got_rev, nc, ref_rev.shape[1]), ref_rev)

    # --- the centre head: geometry, attributes and the final scaling -----------------
    head = model.center_head
    P, ns = sg.n_poses, model.ns
    valid = torch.zeros(sg.n_lig_padded, 1, dtype=torch.float64)
    valid[:sg.n_lig * P] = 1.0
    ch = dl.CenterHead(dev, head.float(), sg, cfg.sigma_embed_dim)
    head.double()

    ref_center = lp.double()[:sg.n_lig * P].reshape(P, sg.n_lig, 3).mean(1)
    pose_of = (torch.arange(sg.n_lig_padded).div(sg.n_lig, rounding_mode="floor")
               .clamp(max=P - 1))
    ref_hv = (lp.double() - ref_center[pose_of]) * valid
    got_hv = ch.vectors(lp_dev, dl.upload(dev, valid.float()))
    record("center_head_vec",
           torch.cat([ttnn.to_torch(c)[:sg.n_lig_padded, :1] for c in got_hv], 1), ref_hv)

    lig_sig_pad = sigma_emb.expand(sg.n_lig_padded, -1)
    lig_h = torch.randn(sg.n_lig_padded, ns, generator=g).double()
    ref_cattr = head.center_edge_embedding(torch.cat(
        [head.center_distance_expansion(ref_hv.norm(dim=-1)), lig_sig_pad], 1))
    src = torch.arange(sg.n_lig_padded) if head.fixed_center_conv else pose_of
    ref_cattr = torch.cat([ref_cattr, lig_h[src, :ns]], -1)
    src_dev = ttnn.from_torch(src.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    got_cattr = ch.attributes(got_hv, dl.upload(dev, lig_h.float()), ns, sigma_row, src_dev)
    record("center_cattr", cut(got_cattr, sg.n_lig_padded, ref_cattr.shape[1]), ref_cattr)

    # the norm-and-rescale, against the host's own expression
    from tt_bio.diffdock.schedules import so3_score_norm

    conv_out = torch.randn(P, 6 if head.odd_parity else 12, generator=g).double()
    tr_sigma, rot_sigma = 1.7, 0.9
    if head.odd_parity:
        rtr, rrot = conv_out[:, :3], conv_out[:, 3:6]
    else:
        rtr = conv_out[:, :3] + conv_out[:, 6:9]
        rrot = conv_out[:, 3:6] + conv_out[:, 9:]
    gg = sigma_emb.expand(P, -1)
    trn = torch.linalg.vector_norm(rtr, dim=1, keepdim=True)
    rtr = rtr / trn * head.tr_final_layer(torch.cat([trn, gg], 1))
    rotn = torch.linalg.vector_norm(rrot, dim=1, keepdim=True)
    rrot = rrot / rotn * head.rot_final_layer(torch.cat([rotn, gg], 1))
    so3n = float(so3_score_norm(torch.as_tensor(rot_sigma).reshape(1)).item())
    if head.scale_by_sigma:
        rtr, rrot = rtr / tr_sigma, rrot * so3n
    g_tr, g_rot = ch.scale(dl.upload(dev, conv_out.float()), sigma_row, tr_sigma, so3n, P)
    record("center_tr", ttnn.to_torch(g_tr)[:P, :3], rtr)
    record("center_rot", ttnn.to_torch(g_rot)[:P, :3], rrot)

    # --- the receptor crop, as a reduction over the complete cross graph --------------
    # Last leg on purpose: it is the newest block and the json is flushed after each leg,
    # so if the ttnn reduction raises, every measurement above it is already on disk.
    cross_dev = masks.cross(*cvec, 20.0)
    ref_near = sg.cross_mask(lp.double(), rp.double(), 20.0)
    ref_keep = sg.rec_keep_from_cross(ref_near)
    got_keep = ttnn.to_torch(dl.rec_keep(cross_dev, n_poses, n_lig, n_rec,
                                         sg.n_rec_padded))[:sg.n_rec_padded, 0]
    out["rec_keep"] = {"mismatches": int((got_keep.double() - ref_keep).abs().round().sum()),
                       "kept": int(ref_keep.sum()), "nodes": int(sg.n_rec_padded)}
    print("rec_keep", out["rec_keep"], flush=True)
    flush_json()

    # --- the torsion head around its two tensor products -----------------------------
    # Newest block, so last: the json is flushed after each leg.
    th = model.torsion_head
    if th is not None:
        from tt_bio.diffdock.graph import pad_edge_index, tile_disjoint
        from tt_bio.diffdock.schedules import torus_score_norm

        rot = torch.tensor([[2, 5, 8], [3, 6, 9]])
        sgt = StaticGraphs.build(n_lig, n_rec, bonds, n_rot_bonds=3, n_poses=n_poses)
        rb = pad_edge_index(tile_disjoint(rot, n_poses, n_lig, n_lig), sgt.n_bond_padded,
                            sgt.lig_pad_node)
        tdev = dl.TorsionHead(dev, th.float(), sgt, rb)
        th.double()
        ns, ei, ne, nb = th.ns, sgt.tor_edge_index, sgt.tor_edge_index.shape[1], rb.shape[1]
        lh = torch.randn(sgt.n_lig_padded, 2 * ns, generator=g).double()
        p64 = lp.double()[:sgt.n_lig_padded]
        bpos = (p64[rb[0]] + p64[rb[1]]) / 2
        ev = p64[ei[1]] - bpos[ei[0]]
        ref_attr = torch.cat([th.final_edge_embedding(th.lig_distance_expansion(ev.norm(dim=-1))),
                              lh[ei[1], :ns], (lh[rb[0]] + lh[rb[1]])[ei[0], :ns]], 1)
        ref_bsh = spherical_harmonics(p64[rb[1]] - p64[rb[0]], (2,))[ei[0]]
        ref_m = sgt.tor_mask(p64, bpos, th.lig_max_radius)
        g_attr, g_sh, g_bsh, g_m = tdev(dl.upload(dev, p64.float()), dl.upload(dev, lh.float()))
        record("tor_bond_pos", cut(tdev.bonds(dl.upload(dev, p64.float()))[0], nb, 3), bpos)
        record("tor_edge_attr", cut(g_attr, ne, 3 * ns), ref_attr)
        record("tor_edge_sh", cut(g_sh, ne, sum(2 * l + 1 for l in th.sh_ls)),
               spherical_harmonics(ev, tuple(th.sh_ls)))
        record("tor_bond_sh", cut(g_bsh, ne, 5), ref_bsh)
        out["tor_mask"] = {"mismatches": int((cut(g_m, ne, 1)[:, 0].double() - ref_m).abs()
                                             .round().sum()), "on": int(ref_m.sum()), "edges": ne}
        print("tor_mask", out["tor_mask"], flush=True)
        width = th.tor_final_layer[0].in_features
        conv = torch.randn(nb, width, generator=g).double()
        sigma = 0.37
        ref_tor = th.tor_final_layer(conv)
        if th.scale_by_sigma:
            ref_tor = ref_tor * float(torus_score_norm(torch.tensor([sigma]).double())[0]) ** 0.5
        record("tor_score", cut(tdev.score(dl.upload(dev, conv.float()), sigma), nb, 1), ref_tor)
    flush_json()

    print(json.dumps({k: v for k, v in out.items() if isinstance(v, (int, float))}), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
