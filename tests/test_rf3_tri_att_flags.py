"""The two triangle-attention flags move together, at every RF3 site, or the ceiling comes back.

RF3 has four sites that build a pairformer block: the trunk stack, the confidence head, the
template embedder and the MSA module. Two of them were moved to the fused SDPA and two were not,
and the two left behind were the 627 aa Wormhole ceiling -- at 656 tokens the materialised route
asks for one 2 369 912 832 B buffer. `fp32_softmax` and `tri_att_sdpa_ragged_pad` are also not
independent: these blocks see the RAW token axis, so an unmasked ragged tail makes the fused
kernel 71-76x wrong and its output allocation-history dependent.

So the pair is a single decision, `remap.tri_att_fused_flags`, and this file keeps it that way.
"""
import re
from pathlib import Path

import pytest

RF3 = Path(__file__).resolve().parent.parent / "tt_bio" / "rf3"
KWARG = re.compile(r"\b(fp32_softmax|tri_att_sdpa_ragged_pad)\s*=")
#: Only files that BUILD a pairformer block. `fp32_softmax` is also a flag on RF3's diffusion
#: attention -- atom_encoder.py, diffusion_atom_decoder.py and token_dit.py all set it -- and
#: those are a different block on a different axis. Their score tensor is
#: [1, heads, N, N] rather than triangle attention's [tokens, heads, S, S], one whole factor of
#: N smaller, so they are not the ceiling mechanism and this pair does not apply to them.
BUILDER = re.compile(r"\bPairformer(Layer)?\s*\(")


def test_the_pair_moves_together():
    from tt_bio.rf3.remap import tri_att_fused_flags
    assert tri_att_fused_flags(True) == {
        "fp32_softmax": False, "tri_att_sdpa_ragged_pad": True}
    assert tri_att_fused_flags(False) == {
        "fp32_softmax": True, "tri_att_sdpa_ragged_pad": False}


def test_pairformer_flags_carry_the_fused_pair():
    """The trunk's flag dict is built from the same helper, not from its own copy."""
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS, tri_att_fused_flags
    for k, v in tri_att_fused_flags(True).items():
        assert PAIRFORMER_FLAGS[k] == v, f"trunk disagrees with the helper on {k}"


def _offenders(root=RF3):
    out = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text()
        if path.name == "remap.py" or not BUILDER.search(text):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if KWARG.search(line):
                out.append(f"{path.name}:{n}: {line.strip()}")
    return out


def test_no_site_sets_either_flag_by_hand():
    assert not _offenders(), (
        "these lines pass fp32_softmax or tri_att_sdpa_ragged_pad directly instead of "
        "**tri_att_fused_flags(...), which is how one site gets left on the materialised route "
        "and the size ceiling comes back:\n  " + "\n  ".join(_offenders()))


def test_the_guard_above_actually_fires(tmp_path):
    """Negative control: a guard that cannot fail is not a guard."""
    fake = tmp_path / "rf3"
    fake.mkdir()
    (fake / "sneaky.py").write_text("layer = PairformerLayer(fp32_softmax=True)\n")
    assert _offenders(fake) == ["sneaky.py:1: layer = PairformerLayer(fp32_softmax=True)"]
    # and it is the pairformer restriction doing the narrowing, not a dead regex: the same
    # kwarg in a file that builds no pairformer block is correctly ignored.
    (fake / "diffusion_ish.py").write_text("blk = TokenDiT(fp32_softmax=True)\n")
    assert len(_offenders(fake)) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
