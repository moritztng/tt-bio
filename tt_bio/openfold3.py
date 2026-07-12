"""OpenFold3 device port for tt-bio.

Assembled from the shared ``tenstorrent`` primitives (the same Pairformer / MSA /
Triangle / DiT / AdaLN / AttentionPairBias building blocks Protenix-v2 and Boltz-2
reuse), driven by the OF3->tt-bio weight remap in ``tt_bio.openfold3_weights``.

OF3 is *not* a pure weight-remap onto ``protenix.Trunk``: the trunk hyperparameters
differ (OF3 c_z=128, no_heads_pair=4 vs Protenix-v2 c_z=256, no_heads_pair=8), so the
trunk is assembled from the shared primitives at OF3 dims rather than instantiated from
``protenix.py``. The DiffusionModule's core DiT/AtomTransformer dims (24 DiT blocks,
16 heads, head_dim=48, c_atom=128, c_atom_pair=16, NQ=32/NK=128/PAD_LEFT=48) match
Protenix-v2 and reuse its ``DiffusionModule`` via key remap.

Status (P6): the InputEmbedder *glue* leg (s_input -> s, z via the 5 weight-only
linears + outer-sum z) is PCC-gated on device (``tests/test_openfold3_input_embedder.py``).
The atom-encoder -> s_input leg, trunk assembly, template embedder, DiffusionModule and
confidence heads are landed incrementally in subsequent ticks, each PCC-gated vs the
golden in ``~/of3_ref_out.pkl`` exactly as the Pairformer / MSA legs were in P3-P5.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.tenstorrent import Module


class InputEmbedderGlue(Module):
    """OF3 ``InputEmbedderAllAtom`` glue: ``s_input -> (s, z)``.

    Implements the five weight-only linears (``linear_s``, ``linear_z_i``,
    ``linear_z_j``, ``linear_relpos``, ``linear_token_bonds``) plus the outer-sum pair
    representation

        z[i,j] = linear_z_i(s_input)[i] + linear_z_j(s_input)[j]
                 + linear_relpos(relpos)[i,j] + linear_token_bonds(token_bonds)[i,j]

    The atom-encoder leg (atom features -> ``s_input``) is gated separately; this module
    isolates the glue linears so the device linear precision is PCC-gated independently of
    the atom-transformer attention. All five linears are bias-free in the OF3 checkpoint.

    Args:
        state_dict: the ``input_embedder`` sub-dict (keys ``linear_*.weight``).
        compute_kernel_config: device compute kernel config (HiFi4 + fp32 dest acc).

    Inputs (device bf16):
        s_input:     [1, N_token, c_s_input=449]
        relpos:      [1, N_token, N_token, 139]  (OF3 ``relpos_complex`` feature)
        token_bonds: [1, N_token, N_token, 1]

    Outputs (device bf16):
        s: [1, N_token, c_s=384]
        z: [1, N_token, N_token, c_z=128]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_s = self.torch_to_tt("linear_s.weight")
        self.w_zi = self.torch_to_tt("linear_z_i.weight")
        self.w_zj = self.torch_to_tt("linear_z_j.weight")
        self.w_relpos = self.torch_to_tt("linear_relpos.weight")
        self.w_tb = self.torch_to_tt("linear_token_bonds.weight")

    def __call__(self, s_input, relpos, token_bonds):
        lin = self._lin
        s = lin(s_input, self.w_s)
        zi = lin(s_input, self.w_zi)
        zj = lin(s_input, self.w_zj)
        relpos_emb = lin(relpos, self.w_relpos)
        tb_emb = lin(token_bonds, self.w_tb)

        # Outer sum z[i,j] = zi[i] + zj[j]. ttnn add broadcasts a [1,N,1,c] operand over
        # dim -2 and a [1,1,N,c] operand over dim -3; seed with a zero [1,N,N,c] so both
        # single-dim broadcasts follow the same path as the pair-bias add in protenix.py.
        n = s_input.shape[-2]
        c_z = self.w_zj.shape[-1]
        z = ttnn.zeros((1, n, n, c_z), device=self.device, dtype=ttnn.bfloat16)
        z = ttnn.add(z, ttnn.unsqueeze(zi, -2))
        z = ttnn.add(z, ttnn.unsqueeze(zj, -3))
        z = ttnn.add(z, relpos_emb)
        z = ttnn.add(z, tb_emb)
        ttnn.deallocate(relpos_emb)
        ttnn.deallocate(tb_emb)
        return s, z
