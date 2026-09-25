"""`batch["offset"]` reaches the relative encoding, for both AF2 variants.

BindCraft 2 fills `offset` on every call (`bindcraft/af2.py:128`, from
`cyclic_sequence_offsets`). For an ordinary chain that function returns the residue-index
difference, so recomputing it looked harmless for as long as nobody designed a cyclic binder.
For a chain whose residues carry the CYCLIC flag it does not, and the port recomputed it
unconditionally until `05eea9df8`: a cyclic binder folded with the wrong relative encoding and
nothing raised.

The float64 red/green lives in that commit and needs the 373 MB parameter file
(`scripts/af2_port/multimer_ref_jax.py --cyclic`). What is testable card-free and
checkpoint-free is the property the fix turns on, and it is the one that regresses: the
encoding reads `offset` and nothing else about position. The first two assertions fail with
the lookup taken out.

`Linear` initialises to zeros, so a model straight out of the constructor maps every relative
feature to the same zero pair and any comparison of two encodings passes without measuring
anything. `_model` fills the one embedding under test with a seeded `randn` for that reason,
and `test_the_fixture_is_not_vacuous` holds it to it.
"""
from __future__ import annotations

import torch

from tt_bio import af2_reference as ref

N = 8


def _model(multimer: bool) -> ref.AF2Model:
    model = ref.AF2Model(template=False, structure=False, multimer=multimer,
                         num_evoformer_blocks=0, num_extra_msa_blocks=0)
    embed = model.embed["position_activations" if multimer else "pair_activations"]
    with torch.no_grad():
        embed.weight.copy_(torch.randn(embed.weight.shape,
                                       generator=torch.Generator().manual_seed(0)))
    return model


def _feats(index: torch.Tensor) -> dict:
    return {"residue_index": index,
            "asym_id": torch.zeros(N, dtype=torch.long),
            "entity_id": torch.zeros(N, dtype=torch.long),
            "sym_id": torch.zeros(N, dtype=torch.long)}


def _cyclic_offset(index: torch.Tensor) -> torch.Tensor:
    """What `cyclic_sequence_offsets` returns for a chain flagged cyclic end to end: the
    difference wrapped into the shorter way round the ring."""
    plain = index[:, None] - index[None, :]
    return (plain + N // 2) % N - N // 2


@torch.no_grad()
def test_the_fixture_is_not_vacuous():
    """Two different relative features must reach two different encodings, or the tests below
    would pass on a model that encodes nothing."""
    index = torch.arange(N)
    for multimer in (False, True):
        model = _model(multimer)
        a = model.relative_encoding(_feats(index), torch.float32)
        b = model.relative_encoding(_feats(index * 7), torch.float32)
        assert not torch.allclose(a, b), f"multimer={multimer}"


@torch.no_grad()
def test_the_relative_encoding_reads_offset_and_not_the_residue_index():
    """Two different residue indices, one supplied offset: the encoding may not move.

    Position enters the monomer encoding only through the offset, and the multimer one adds
    chain features that are not derived from `residue_index` either. So holding `offset` fixed
    pins the output, and it stops doing so the moment the offset is recomputed from the index.
    """
    offset = _cyclic_offset(torch.arange(N))
    for multimer in (False, True):
        model = _model(multimer)
        a = model.relative_encoding({**_feats(torch.arange(N)), "offset": offset}, torch.float32)
        b = model.relative_encoding({**_feats(torch.arange(N) * 7), "offset": offset},
                                    torch.float32)
        assert torch.equal(a, b), f"multimer={multimer}"


@torch.no_grad()
def test_a_cyclic_offset_encodes_differently_from_the_index_difference():
    """The fix is not cosmetic: on a cyclic chain the two offsets disagree, and so must the two
    encodings. 16 of the 64 pairs move here, which is the shape of the 3.6e-3 1-pcc the float64
    reference read at the first tap."""
    index = torch.arange(N)
    cyclic = _cyclic_offset(index)
    plain = index[:, None] - index[None, :]
    assert int((cyclic != plain).sum()) == 16
    for multimer in (False, True):
        model = _model(multimer)
        honoured = model.relative_encoding({**_feats(index), "offset": cyclic}, torch.float32)
        fallback = model.relative_encoding(_feats(index), torch.float32)
        assert not torch.allclose(honoured, fallback), f"multimer={multimer}"


@torch.no_grad()
def test_an_ordinary_chain_is_untouched_by_the_fallback():
    """Every shipped number stays where it is: for a non-cyclic chain the supplied offset IS the
    index difference, so honouring it and recomputing it agree exactly."""
    index = torch.arange(N)
    plain = index[:, None] - index[None, :]
    for multimer in (False, True):
        model = _model(multimer)
        supplied = model.relative_encoding({**_feats(index), "offset": plain}, torch.float32)
        recomputed = model.relative_encoding(_feats(index), torch.float32)
        assert torch.equal(supplied, recomputed), f"multimer={multimer}"
