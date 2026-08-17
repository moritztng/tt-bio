"""Device StructuralTokenExpander vs a torch fp32 reference, on real ifd + synthetic trunk.

The expander is a pure function of (ifd, s_inputs_res, s_res, z_res) and the routed weights, so
it can be graded without paying for a trunk run. Everything host-side (_pair_features_rows) is
shared with the reference on purpose: what is under test is the device path.

Usage:  python -m perf.wh-correctness.expander_ref <fasta>   (or run the file directly)
"""
import sys
import torch


def build_ifd(seq):
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.opendde_data import build_structural_token_features
    feats = build_complex_features([(seq, None, "protein")])
    return feats, build_structural_token_features(feats)


def host_reference(exp, ifd, s_inputs_res, s_res, z_res):
    """Everything the device expander computes, in torch fp32."""
    w = exp._w
    parent = ifd["parent_residue_idx"].long()
    role = ifd["subtoken_role_id"].long()
    Ns = role.shape[0]
    C = exp.c_z

    s_inputs_struct = s_inputs_res.index_select(0, parent) + w["single_input_role_embedding.weight"].index_select(0, role)
    s_parent = s_res.index_select(0, parent)
    mlp = torch.nn.functional.layer_norm(
        s_parent, (s_parent.shape[-1],), w["single_split_mlp.0.weight"], w["single_split_mlp.0.bias"], 1e-5)
    mlp = mlp @ w["single_split_mlp.1.weight"].t()
    mlp = torch.nn.functional.silu(mlp)
    mlp = mlp @ w["single_split_mlp.3.weight"].t()
    s_struct = s_parent + mlp + w["single_role_embedding.weight"].index_select(0, role)

    z_struct = torch.empty(Ns, Ns, C)
    attn_bias = torch.empty(Ns, Ns)
    bias_only = torch.empty(Ns, Ns, C)
    proj_only = torch.empty(Ns, Ns, C)
    chunk = min(exp.pair_chunk_size or Ns, Ns)
    for start in range(0, Ns, chunk):
        end = min(start + chunk, Ns)
        ri = torch.arange(start, end)
        pf = exp._pair_features_rows(ifd, role, parent, ri)
        rp = parent.index_select(0, ri)
        zc = z_res[rp][:, parent]                       # (clen, Ns, C)

        # 49 role-pair projections
        row_role = role.index_select(0, ri)
        pidx = (row_role[:, None] * exp.n_roles + role[None, :])
        delta = torch.zeros_like(zc)
        for g in pidx.unique().tolist():
            m = pidx == g
            delta[m] = zc[m] @ w["pair_block_proj.%d.weight" % g].t()

        # five additive pair-init embeddings
        b = torch.zeros(end - start, Ns, C)
        for tkey, ikey in (("same_parent_embedding.weight", "same_parent_residue"),
                           ("same_residue_twin_embedding.weight", "same_residue_twin"),
                           ("prev_bb_chain_embedding.weight", "prev_bb_chain"),
                           ("next_bb_chain_embedding.weight", "next_bb_chain"),
                           ("role_pair_type_embedding.weight", "role_pair_type")):
            idx = pf[ikey].long()
            b = b + w[tkey].index_select(0, idx.reshape(-1)).reshape(end - start, Ns, C)

        z_struct[start:end] = zc + delta + b
        bias_only[start:end] = b
        proj_only[start:end] = delta

        rpb = w["attn_bias_role_pair_type"].index_select(0, pf["role_pair_type"].reshape(-1)).reshape(pf["role_pair_type"].shape)
        attn_bias[start:end] = (
            pf["same_parent_residue"].float() * float(w["attn_bias_same_parent"])
            + pf["same_residue_twin"].float() * float(w["attn_bias_same_residue_twin"])
            + pf["prev_bb_chain"].float() * float(w["attn_bias_prev_bb_chain"])
            + pf["next_bb_chain"].float() * float(w["attn_bias_next_bb_chain"])
            + rpb)
    return s_inputs_struct, s_struct, z_struct, attn_bias, bias_only, proj_only


def report(name, dev, ref):
    d = (dev.float() - ref.float()).abs()
    scale = ref.abs().mean().clamp_min(1e-9)
    print("%-18s max|err|=%9.4f  mean|err|=%9.6f  mean|ref|=%8.4f  rel_max=%8.4f"
          % (name, d.max().item(), d.mean().item(), scale.item(), (d.max() / scale).item()))
    return d


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.opendde import StructuralTokenExpander, OPENDDE_CONFIG, load_opendde_checkpoint, route_opendde_weights

    fasta = sys.argv[1] if len(sys.argv) > 1 else "state/cdk2_128.fasta"
    seq = "".join(l.strip() for l in open(fasta) if not l.startswith(">"))
    print("seq len %d" % len(seq))
    feats, ifd = build_ifd(seq)
    Ns = ifd["subtoken_role_id"].shape[0]
    NT = feats["restype"].shape[0]
    print("NT=%d Ns=%d  Ns%%32=%d  chunk=128 last_chunk=%d" % (NT, Ns, Ns % 32, Ns % 128 or 128))

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    routed = route_opendde_weights(load_opendde_checkpoint())
    C = OPENDDE_CONFIG
    exp = StructuralTokenExpander(routed["expander"], ckc, c_s=C["c_s"], c_z=C["c_z"],
                                  c_s_inputs=C["c_s_inputs"], n_roles=C["n_roles"],
                                  pair_chunk_size=C["pair_chunk_size"])

    torch.manual_seed(0)
    # Trunk-scale inputs: LayerNorm'd trunk outputs sit near unit variance.
    s_inputs_res = torch.randn(NT, C["c_s_inputs"])
    s_res = torch.randn(NT, C["c_s"])
    z_res = torch.randn(NT, NT, C["c_z"]) * 0.5

    si_d, s_d, z_d, ab_d = exp(ifd, s_inputs_res, s_res, z_res)
    si_h = ttnn.to_torch(si_d)[:Ns, :C["c_s_inputs"]]
    s_h = ttnn.to_torch(s_d)[:Ns, :C["c_s"]]
    z_h = ttnn.to_torch(z_d).reshape(Ns, Ns, C["c_z"])
    ab_h = ttnn.to_torch(ab_d)[:Ns, :Ns]

    si_r, s_r, z_r, ab_r, bias_r, proj_r = host_reference(exp, ifd, s_inputs_res, s_res, z_res)

    print("\n--- device vs host fp32 reference ---")
    report("s_inputs_struct", si_h, si_r)
    report("s_struct", s_h, s_r)
    dz = report("z_struct", z_h, z_r)
    report("attn_bias", ab_h, ab_r)

    # Where in z_struct is the error? Per-row-block profile.
    per_row = dz.amax(dim=(1, 2))
    print("\nz_struct max|err| by row, worst 12 rows:")
    for i in per_row.topk(min(12, Ns)).indices.tolist():
        print("   row %4d (chunk %d, off %3d, role %d)  max|err|=%9.4f" %
              (i, i // 128, i % 128, int(ifd["subtoken_role_id"][i]), per_row[i].item()))
    print("median row max|err| = %.4f" % per_row.median().item())

    # Isolate the two device sub-terms on chunk 0.
    role = ifd["subtoken_role_id"].long()
    parent = ifd["parent_residue_idx"].long()
    ri = torch.arange(0, min(128, Ns))
    pf = exp._pair_features_rows(ifd, role, parent, ri)
    b_dev = ttnn.to_torch(exp._pair_init_bias(pf)).reshape(ri.numel(), Ns, C["c_z"])
    print()
    report("pair_init_bias c0", b_dev, bias_r[: ri.numel()])
    pb = (b_dev.float() - bias_r[: ri.numel()].float()).abs().amax(dim=(1, 2))
    print("   rows with max|err| > 0.05: %d / %d" % (int((pb > 0.05).sum()), pb.numel()))
    if int((pb > 0.05).sum()):
        print("   first such rows:", pb.gt(0.05).nonzero().flatten()[:16].tolist())

    z_flat = ttnn.from_torch(z_res.reshape(NT * NT, C["c_z"]), layout=ttnn.ROW_MAJOR_LAYOUT,
                             device=dev, dtype=ttnn.bfloat16)
    rp = parent.index_select(0, ri)
    gidx = (rp[:, None] * NT + parent[None, :]).reshape(1, -1).to(torch.int32)
    z_dev = ttnn.embedding(ttnn.from_torch(gidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32),
                           z_flat, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    z_dev = ttnn.reshape(z_dev, (ri.numel() * Ns, C["c_z"]))
    p_dev = ttnn.to_torch(exp._pair_project_full(z_dev, role, ri)).reshape(ri.numel(), Ns, C["c_z"])
    report("pair_project c0", p_dev, proj_r[: ri.numel()])
    pp = (p_dev.float() - proj_r[: ri.numel()].float()).abs().amax(dim=(1, 2))
    print("   rows with max|err| > 0.5: %d / %d" % (int((pp > 0.5).sum()), pp.numel()))


if __name__ == "__main__":
    main()
