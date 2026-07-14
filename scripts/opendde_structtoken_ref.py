# SPDX-License-Identifier: Apache-2.0
"""Random-weight torch reference for OpenDDE's StructuralTokenExpander (the one
novel compute block; the rest of OpenDDE is Protenix-v2's already-ported stack).

Builds the module at the real opendde_v1 config (pair_projection_mode="full",
pair_chunk_size=128, init_mode="scratch"), runs a deterministic forward on
synthetic residue-trunk inputs, and saves inputs+golden outputs for the ttnn
PCC gate. No checkpoints, no data pipeline -- per-module parity methodology.

Usage: OPENDDE_SRC=/tmp/opendde-src python3 scripts/opendde_structtoken_ref.py
"""
import os, sys, types

SRC = os.environ.get("OPENDDE_SRC", "/tmp/opendde-src")
sys.path.insert(0, SRC)

import torch
import torch.nn as nn
torch.set_grad_enabled(False)

# --- stub the FoldCP (context-parallel) imports pulled in at module import
# time. Only the single-device local forward() path is exercised here, which
# never calls these; the stubs just satisfy the top-level imports. ---
def _stub(name, attrs):
    m = types.ModuleType(name)
    for a in attrs:
        setattr(m, a, type(a, (), {}))
    sys.modules[name] = m

sys.modules.setdefault("optree", types.ModuleType("optree"))

_stub("opendde.distributed.foldcp.atom_window",
      ["gather_pair_embedding_in_dense_trunk_from_foldcp_local"])
_stub("opendde.distributed.foldcp.mesh", ["FoldCPProcessMesh"])
_stub("opendde.distributed.foldcp.pair_sharding",
      ["FoldCPPairShardSpec", "make_pair_shard_spec"])

from opendde.model.modules.structural_tokens import StructuralTokenExpander
from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES

# --- opendde_v1 config for this block (opendde/config/model_base.py) ---
C_S, C_Z, C_S_INPUTS = 384, 384, 449
cfg = dict(
    c_s=C_S, c_z=C_Z, c_s_inputs=C_S_INPUTS, n_roles=7,
    init_mode="scratch", role_init_std=0.02, pair_feature_init_std=0.02,
    attention_bias_init=0.1, pair_projection_mode="full", pair_chunk_size=128,
)

g = torch.Generator().manual_seed(0)
mod = StructuralTokenExpander(**cfg).eval()
# scratch init already randomizes role/pair conditioning; also randomize the
# split-MLP + full pair projections (zero-init by default) so the reference
# is a non-trivial function, not a near-identity residual.
for p in mod.parameters():
    if p.dim() >= 1:
        p.copy_(torch.empty_like(p).normal_(0.0, 0.05, generator=g))

# --- synthetic residue-trunk inputs + structural-token index map ---
N_RES, N_STRUCT = 32, 64
sg = torch.Generator().manual_seed(1)
parent = torch.randint(0, N_RES, (N_STRUCT,), generator=sg)
# keep parent sorted so prev/next-parent adjacency is meaningful
parent, _ = torch.sort(parent)
role = torch.randint(0, 7, (N_STRUCT,), generator=sg)
prev_parent = torch.clamp(parent - 1, min=-1)
next_parent = torch.clamp(parent + 1, max=N_RES - 1)

ifd = {
    "parent_residue_idx": parent,
    "subtoken_role_id": role,
    "residue_index": torch.arange(N_RES),
    "asym_id": torch.zeros(N_RES, dtype=torch.long),
    "prev_parent_residue_idx": prev_parent,
    "next_parent_residue_idx": next_parent,
}
s_inputs_res = torch.empty(N_RES, C_S_INPUTS).normal_(0, 1, generator=sg)
s_res = torch.empty(N_RES, C_S).normal_(0, 1, generator=sg)
z_res = torch.empty(N_RES, N_RES, C_Z).normal_(0, 1, generator=sg)

s_inputs_st, s_st, z_st, pair_feats = mod(ifd, s_inputs_res, s_res, z_res)

out = os.path.join(SRC, "..", "opendde_structtoken_golden.pt")
out = os.path.abspath(out)
torch.save({
    "cfg": cfg,
    "state_dict": mod.state_dict(),
    "inputs": {"ifd": ifd, "s_inputs_res": s_inputs_res, "s_res": s_res, "z_res": z_res},
    "outputs": {
        "s_inputs_struct": s_inputs_st, "s_struct": s_st, "z_struct": z_st,
        "structural_pair_attn_bias": pair_feats.get("structural_pair_attn_bias"),
    },
}, out)

def stat(t):
    return f"shape={tuple(t.shape)} mean={t.float().mean():.5f} std={t.float().std():.5f}"
print("StructuralTokenExpander opendde_v1 forward OK")
print("  s_inputs_struct", stat(s_inputs_st))
print("  s_struct       ", stat(s_st))
print("  z_struct       ", stat(z_st))
ab = pair_feats.get("structural_pair_attn_bias")
if ab is not None:
    print("  attn_bias      ", stat(ab))
print("  pair_feat keys :", sorted(pair_feats.keys()))
print("  saved golden ->", out)
