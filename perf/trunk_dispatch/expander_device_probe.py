"""Bit-exactness probe for the on-device StructuralTokenExpander pair path.

Compares, on the real opendde_v1 expander weights and a synthetic 298-aa-like
structural-token layout (all 7 roles present, all 49 role-pair groups
nonempty, ragged last chunk):

  OLD (branch HEAD~ code: host fp32 gather/sum + per-chunk from_torch uploads)
  vs
  NEW (device embedding gathers; fp32 table sum + one fp32->bf16 cast for the
  bias; device permutation gather + group slices for the projection)

Compares are torch.equal on to_torch() bit patterns, for (1) _pair_init_bias
per chunk, (2) _pair_project_full per chunk, (3) the full pair loop outputs
z_struct / attn_bias. The bias is additionally checked against the naive fp32
reference (host fp32 sum cast to bf16 by torch). Also reports old vs new
pair-loop wall time (synced, warm).

Run from the repo root:
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:<slug> PYTHONPATH=$PWD \
      python3 perf/trunk_dispatch/expander_device_probe.py
"""
import time

import torch
import ttnn

from tt_bio.opendde import (OPENDDE_CONFIG, StructuralTokenExpander,
                            load_opendde_checkpoint, route_opendde_weights)
from tt_bio.tenstorrent import get_device

C = OPENDDE_CONFIG


def synthetic_inputs(Nr=298, seed=0):
    """298-aa-like layout: 1 backbone + 1 sidechain token per residue, plus 35
    atom/dna/rna tokens so all 7 roles (hence all 49 role pairs) appear."""
    g = torch.Generator().manual_seed(seed)
    roles, parent = [], []
    for i in range(Nr):
        roles += [1, 2]
        parent += [i, i]
    extra = [0, 3, 4, 5, 6] * 7
    roles += extra
    parent += torch.randint(0, Nr, (len(extra),), generator=g).tolist()
    role = torch.tensor(roles, dtype=torch.long)
    parent = torch.tensor(parent, dtype=torch.long)
    Ns = role.numel()
    ifd = {
        "parent_residue_idx": parent,
        "subtoken_role_id": role,
        "asym_id": torch.ones(Nr, dtype=torch.long),
        "prev_parent_residue_idx": torch.where(
            role == 1, (parent - 1).clamp(min=0), torch.full((Ns,), -1)),
        "next_parent_residue_idx": torch.where(
            role == 1, (parent + 1).clamp(max=Nr - 1), torch.full((Ns,), -1)),
    }
    s_inputs_res = torch.randn(Nr, C["c_s_inputs"], generator=g)
    s_res = torch.randn(Nr, C["c_s"], generator=g)
    z_res = torch.randn(Nr, Nr, C["c_z"], generator=g)
    return ifd, s_inputs_res, s_res, z_res


# --- OLD implementations: verbatim copies of the pre-change code ---

def old_pair_init_bias_h(exp, pf):
    """Host fp32 gather/sum (returns the host tensor, as the old code did)."""
    b = exp._emb("same_parent_embedding.weight", pf["same_parent_residue"].long())
    b = b + exp._emb("same_residue_twin_embedding.weight", pf["same_residue_twin"].long())
    b = b + exp._emb("prev_bb_chain_embedding.weight", pf["prev_bb_chain"].long())
    b = b + exp._emb("next_bb_chain_embedding.weight", pf["next_bb_chain"].long())
    b = b + exp._emb("role_pair_type_embedding.weight", pf["role_pair_type"])
    return b


def old_pair_project_full(exp, z_chunk_h, role, row_index):
    clen = row_index.numel()
    Ns = role.shape[0]
    Cz = exp.c_z
    flat = z_chunk_h.reshape(clen * Ns, Cz)
    row_role = role.index_select(0, row_index)
    role_i = row_role[:, None].expand(clen, Ns).reshape(-1)
    role_j = role[None, :].expand(clen, Ns).reshape(-1)
    pidx = role_i * exp.n_roles + role_j
    perm = torch.argsort(pidx, stable=True)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    flat_sorted = flat.index_select(0, perm).contiguous()
    uniq, counts = torch.unique_consecutive(pidx.index_select(0, perm), return_counts=True)
    pieces = []
    off = 0
    for g_, c in zip(uniq.tolist(), counts.tolist()):
        seg = exp._up(flat_sorted[off:off + c].contiguous())
        out = exp._lin(seg, "pair_block_proj.%d.weight" % g_)
        pieces.append(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))
        off += c
    sorted_delta = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)
    inv_idx = ttnn.from_torch(inv.reshape(1, -1).to(torch.int32),
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=get_device(),
                              dtype=ttnn.uint32)
    flat_delta = ttnn.embedding(inv_idx, sorted_delta, layout=ttnn.ROW_MAJOR_LAYOUT,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_layout(ttnn.reshape(flat_delta, (clen, Ns, Cz)), ttnn.TILE_LAYOUT), len(uniq)


def old_pair_loop(exp, ifd, z_res, role, parent):
    """The pre-change __call__ pair section (device z gather from 4859524e,
    host z_chunk_h for the projection, host bias sum + upload)."""
    Ns = role.shape[0]
    chunk = min(exp.pair_chunk_size or Ns, Ns)
    Nr = z_res.shape[0]
    z_flat = ttnn.from_torch(
        z_res.reshape(Nr * Nr, exp.c_z), layout=ttnn.ROW_MAJOR_LAYOUT,
        device=get_device(), dtype=ttnn.bfloat16)
    z_chunks, ab_chunks = [], []
    for start in range(0, Ns, chunk):
        end = min(start + chunk, Ns)
        row_index = torch.arange(start, end)
        pf = exp._pair_features_rows(ifd, role, parent, row_index)
        row_parent = parent.index_select(0, row_index)
        z_chunk_h = z_res.index_select(0, row_parent).index_select(1, parent).contiguous()
        gidx = (row_parent[:, None] * Nr + parent[None, :]).reshape(1, -1).to(torch.int32)
        z_dev = ttnn.embedding(
            ttnn.from_torch(gidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=get_device(),
                            dtype=ttnn.uint32),
            z_flat, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        z_dev = ttnn.to_layout(
            ttnn.reshape(z_dev, (end - start, Ns, exp.c_z)), ttnn.TILE_LAYOUT)
        proj, _ = old_pair_project_full(exp, z_chunk_h, role, row_index)
        z_dev = ttnn.add(z_dev, proj)
        z_dev = ttnn.add(z_dev, exp._up(old_pair_init_bias_h(exp, pf)))
        z_chunks.append(z_dev)
        ab_chunks.append(exp._attn_bias(pf))
    ttnn.deallocate(z_flat)
    z_struct = z_chunks[0] if len(z_chunks) == 1 else ttnn.concat(z_chunks, dim=-3)
    attn_bias = ab_chunks[0] if len(ab_chunks) == 1 else ttnn.concat(ab_chunks, dim=0)
    return z_struct, attn_bias


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = route_opendde_weights(load_opendde_checkpoint())["expander"]
    exp = StructuralTokenExpander(sd, ckc)

    ifd, s_inputs_res, s_res, z_res = synthetic_inputs()
    role = ifd["subtoken_role_id"]
    parent = ifd["parent_residue_idx"]
    Ns = role.shape[0]
    Nr = z_res.shape[0]
    chunk = exp.pair_chunk_size
    print(f"synthetic layout: Nr={Nr} Ns={Ns} chunks="
          f"{list(range(0, Ns, chunk))}")

    # --- (1)+(2): per-chunk unit compares, first and last chunk ---
    z_flat = ttnn.from_torch(
        z_res.reshape(Nr * Nr, exp.c_z), layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev, dtype=ttnn.bfloat16)
    for start in sorted({0, max(0, Ns - chunk)}):
        end = min(start + chunk, Ns)
        row_index = torch.arange(start, end)
        pf = exp._pair_features_rows(ifd, role, parent, row_index)
        row_parent = parent.index_select(0, row_index)

        bias_old = exp._up(old_pair_init_bias_h(exp, pf))
        bias_new = exp._pair_init_bias(pf)
        bias_ref = old_pair_init_bias_h(exp, pf).to(torch.bfloat16)
        assert torch.equal(ttnn.to_torch(bias_old), ttnn.to_torch(bias_new)), \
            f"bias old!=new on chunk {start}"
        assert torch.equal(ttnn.to_torch(bias_new), bias_ref), \
            f"bias new!=fp32-ref on chunk {start}"
        print(f"chunk {start}: _pair_init_bias bit-exact (old==new==fp32 ref)")

        z_chunk_h = z_res.index_select(0, row_parent).index_select(1, parent).contiguous()
        gidx = (row_parent[:, None] * Nr + parent[None, :]).reshape(1, -1).to(torch.int32)
        z_dev = ttnn.embedding(
            ttnn.from_torch(gidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            dtype=ttnn.uint32),
            z_flat, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        z_dev = ttnn.reshape(z_dev, ((end - start) * Ns, exp.c_z))
        proj_old, ngroups = old_pair_project_full(exp, z_chunk_h, role, row_index)
        proj_new = exp._pair_project_full(z_dev, role, row_index)
        assert torch.equal(ttnn.to_torch(proj_old), ttnn.to_torch(proj_new)), \
            f"projection old!=new on chunk {start}"
        print(f"chunk {start}: _pair_project_full bit-exact ({ngroups} role-pair groups)")
    ttnn.deallocate(z_flat)

    # --- (3): full pair loop, old vs new __call__ pair outputs ---
    z_old, ab_old = old_pair_loop(exp, ifd, z_res, role, parent)
    _, _, z_new, ab_new = exp(ifd, s_inputs_res, s_res, z_res)
    z_old_t, z_new_t = ttnn.to_torch(z_old), ttnn.to_torch(z_new)
    assert torch.equal(z_old_t, z_new_t), "pair loop z_struct old!=new"
    assert torch.equal(ttnn.to_torch(ab_old), ttnn.to_torch(ab_new)), \
        "pair loop attn_bias old!=new"
    print(f"full pair loop bit-exact: z_struct {tuple(z_old_t.shape)}, "
          f"attn_bias {tuple(ttnn.to_torch(ab_old).shape)}")

    # --- timing: old vs new full expander call, synced, warm ---
    def old_full():
        s_inputs_struct = ttnn.add(
            exp._up(s_inputs_res.index_select(0, parent).contiguous()),
            exp._up(exp._emb("single_input_role_embedding.weight", role)))
        s_parent = exp._up(s_res.index_select(0, parent).contiguous())
        mlp = exp._ln(s_parent, "single_split_mlp.0.weight", "single_split_mlp.0.bias")
        mlp = exp._lin(mlp, "single_split_mlp.1.weight")
        mlp = ttnn.silu(mlp)
        mlp = exp._lin(mlp, "single_split_mlp.3.weight")
        ttnn.add(ttnn.add(s_parent, mlp),
                 exp._up(exp._emb("single_role_embedding.weight", role)))
        return old_pair_loop(exp, ifd, z_res, role, parent)

    def timed(fn):
        fn()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        return time.perf_counter() - t0

    t_old = timed(old_full)
    t_new = timed(lambda: exp(ifd, s_inputs_res, s_res, z_res))
    print(f"expander wall (warm, synced): old {t_old:.3f}s  new {t_new:.3f}s  "
          f"(-{t_old - t_new:.3f}s)")
    print("PROBE PASS")


if __name__ == "__main__":
    main()
