"""sdpa_generic's `kv_window`: the reader takes K/V as sliding windows of a frame.

Opens no device. Checks that the generated reader is the running wheel's reader with only the
two K/V placement lines twinned under `KV_WINDOW_NB`, and that the twin's tile arithmetic picks
the same frame tiles `AtomTransformer._attention_superset` copies with its slice + concat.
"""
import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from tt_bio import sdpa_generic as SG                                       # noqa: E402


def test_the_generated_reader_only_twins_the_two_placement_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = SG.kv_window_kernel_dir()
    gen = (d / "dataflow/reader_interleaved.cpp").read_text()
    stock = (SG._kdir() / "dataflow/reader_interleaved.cpp").read_text()
    a = gen.index("#ifdef KV_WINDOW_NB")
    b = gen.index("#else\n", a)
    assert gen[:a] + gen[b + len("#else\n"):].replace("\n#endif", "", 1) == stock
    assert (d / "dataflow/dataflow_common.hpp").exists()
    assert SG.kv_window_kernel_dir() == d                                   # cached, same hash


@pytest.mark.parametrize("M,H,nb,S", [(1, 4, 3, 5), (2, 4, 185, 5), (5, 2, 7, 3)])
def test_the_window_tile_ids_are_the_superset_windows(M, H, nb, S):
    nbk = nb + S - 1
    frame = torch.arange(M * H * nbk).reshape(M, H, nbk)                    # one tile id per frame tile row
    # `_attention_superset`'s win(): window i is frame rows i .. i + S - 1, heads-major (h, i)
    ref = torch.stack([frame[:, :, j:j + nb] for j in range(S)], dim=-1).reshape(M, H * nb, S)
    got = torch.empty_like(ref)
    for b in range(M):
        for head in range(H * nb):
            for r in range(S):                                              # kv_row_start_tile 0, one chunk
                got[b, head, r] = (b * H + head // nb) * nbk + head % nb + r
    assert torch.equal(got, ref)
