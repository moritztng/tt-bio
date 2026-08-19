"""RF3 feature initializer: the host-computable half.

`RelativePositionEncoding` and `process_token_bonds` are pure functions of integer
features, so they are built here on host in fp32 and only their single learned linear
runs on device -- the same split that worked for `template_features()`. The one-hot
construction is integer gather/compare work that ttnn is poor at, and the learned part
is one `[I, I, 139] @ [139, c_z]` matmul.

Kept deliberately close to `RelativePositionEncoding.forward` in
`_vendor/rf3/model/layers/pairformer_layers.py`, including the cyclic-chain branch and
the Protenix `same_entity` bugfix, because the concatenation ORDER is load-bearing:
the linear's 139 input columns are [relpos 66 | reltoken 66 | same_entity 1 | relchain 6].
"""

from __future__ import annotations

import torch


def relpos_features(
    f: dict,
    r_max: int = 32,
    s_max: int = 2,
) -> torch.Tensor:
    """Build the [I, I, 139] one-hot block that feeds `relative_position_encoding.linear`.

    Mirrors the reference exactly. Returns float32 on host.
    """
    asym_id = f["asym_id"]
    residue_index = f["residue_index"]
    entity_id = f["entity_id"]
    token_index = f["token_index"]
    sym_id = f["sym_id"]

    b_samechain = asym_id.unsqueeze(-1) == asym_id.unsqueeze(-2)
    b_sameresidue = residue_index.unsqueeze(-1) == residue_index.unsqueeze(-2)
    b_same_entity = entity_id.unsqueeze(-1) == entity_id.unsqueeze(-2)

    cyclic_asym_ids = f.get("cyclic_asym_ids", []) or []
    if len(cyclic_asym_ids) > 0:
        # A cyclic chain's residue offsets wrap, so the offset used is whichever of
        # (d, d + len, d - len) is smallest in absolute value.
        offset = residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2)
        for cyclic_asym_id in cyclic_asym_ids:
            len_cyclic_chain = (
                residue_index[asym_id == cyclic_asym_id].unique().shape[0]
            )
            cyclic_chain_mask = (asym_id.unsqueeze(-1) == cyclic_asym_id) & (
                asym_id.unsqueeze(-2) == cyclic_asym_id
            )
            if len_cyclic_chain > 0:
                offset_plus = offset + len_cyclic_chain
                offset_minus = offset - len_cyclic_chain
                abs_offset = offset.abs()
                abs_offset_plus = offset_plus.abs()
                abs_offset_minus = offset_minus.abs()
                choice_plus_or_minus = torch.where(
                    abs_offset_plus <= abs_offset_minus, offset_plus, offset_minus
                )
                c_offset = torch.where(
                    (abs_offset <= abs_offset_plus) & (abs_offset <= abs_offset_minus),
                    offset,
                    choice_plus_or_minus,
                )
                offset = torch.where(cyclic_chain_mask, c_offset, offset)
        offset = (offset + r_max).clamp(0, 2 * r_max)
        d_residue = torch.where(
            b_samechain, offset, (2 * r_max + 1) * torch.ones_like(offset)
        )
    else:
        d_residue = torch.where(
            b_samechain,
            torch.clip(
                residue_index.unsqueeze(-1) - residue_index.unsqueeze(-2) + r_max,
                0,
                2 * r_max,
            ),
            2 * r_max + 1,
        )

    a_relpos = torch.nn.functional.one_hot(d_residue.long(), 2 * r_max + 2)

    d_token = torch.where(
        b_samechain * b_sameresidue,
        torch.clip(
            token_index.unsqueeze(-1) - token_index.unsqueeze(-2) + r_max,
            0,
            2 * r_max,
        ),
        2 * r_max + 1,
    )
    a_reltoken = torch.nn.functional.one_hot(d_token.long(), 2 * r_max + 2)

    # NOTE: `same_entity`, not `not same_chain` -- the Protenix technical-report fix
    # that RF3 adopts over the AF-3 pseudocode.
    d_chain = torch.where(
        b_same_entity,
        torch.clip(sym_id.unsqueeze(-1) - sym_id.unsqueeze(-2) + s_max, 0, 2 * s_max),
        2 * s_max + 1,
    )
    a_relchain = torch.nn.functional.one_hot(d_chain.long(), 2 * s_max + 2)

    return torch.cat(
        [a_relpos, a_reltoken, b_same_entity.unsqueeze(-1), a_relchain], dim=-1
    ).to(torch.float32)


def token_bond_features(f: dict) -> torch.Tensor:
    """The [I, I, 1] input to `process_token_bonds`."""
    return f["token_bonds"].unsqueeze(-1).to(torch.float32)


def mlff_constant(process_atom_level_embedding, n_conformers: int = 8,
                  embedding_dim: int = 384) -> torch.Tensor:
    """The constant this checkpoint's MLFF track adds to every atom's C_L.

    `use_atom_level_embedding` is True, but at public inference the MACE cache is
    absent and `atom_level_embedding` is all-zero. That does NOT make the track a
    no-op: the downcast MLP's Linears carry biases and its final Linear is followed
    by a LayerNorm, whose output on an all-zero input is its own bias. Because the
    input is all-zero and the MLP is position-independent, the result is one
    constant [c_atom] vector, identical for every atom -- so it is precomputed here
    instead of running the MLP on device.

    Callers MUST check the embeddings really are all-zero (see `assert_mlff_inputs_zero`);
    with real embeddings this constant is wrong.
    """
    with torch.no_grad():
        out = process_atom_level_embedding(
            torch.zeros(n_conformers, 1, embedding_dim)
        )
    return out[0].clone()


def assert_mlff_inputs_zero(f: dict) -> None:
    """Fail loudly rather than silently folding in a constant that no longer applies."""
    ale = f.get("atom_level_embedding")
    if ale is not None and int(torch.count_nonzero(ale)) != 0:
        raise ValueError(
            "atom_level_embedding is non-zero: the MLFF contribution is only a "
            "constant while the embeddings are all-zero (public inference, no MACE "
            "cache). Port the ConformerEmbeddingWeightedAverage MLP before running this."
        )
