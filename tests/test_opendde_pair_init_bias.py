"""Host check: OpenDDE's joint pair-init table gives every pair the bf16 rounding of the fp32 sum of its
five embedding rows, in table order, i.e. exactly what the where-chain path computes per pair."""
import pytest
import torch

pytest.importorskip("ttnn")
from tt_bio.opendde import StructuralTokenExpander as STE  # noqa: E402


@pytest.mark.parametrize("c_z", [128, 384])
def test_joint_table_matches_per_pair_sum(c_z):
    g = torch.Generator().manual_seed(0)
    # tables carry more rows than the indices use (`[:n]`), as the checkpoint's padded vocabularies do
    weights = [torch.randn(n + 3, c_z, generator=g).to(torch.bfloat16) for _, _, n in STE._PIB_TABLES]
    pf = {ikey: torch.randint(0, n, (7, 33), generator=g) for _, ikey, n in STE._PIB_TABLES}
    got = STE._pib_joint(weights)[STE._pib_index(pf)]
    ref = None
    for w, (_, ikey, _) in zip(weights, STE._PIB_TABLES):
        rows = w.float()[pf[ikey]]
        ref = rows if ref is None else ref + rows
    assert got.dtype == torch.bfloat16 and got.shape == (7, 33, c_z)
    assert torch.equal(got, ref.to(torch.bfloat16))
