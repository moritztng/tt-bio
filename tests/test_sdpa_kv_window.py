"""sdpa_generic's `kv_window`: the reader takes K/V as sliding windows of a frame.

Opens no device. Checks that the generated reader is the running wheel's reader with only the
K/V and Q placement lines twinned under `KV_WINDOW_NB`, and that the twin's tile arithmetic picks
the same frame tiles `AtomTransformer._attention_superset` copies with its slice + concat. With a lead
(`kv_window=(nb, W, lead)`) the frame is virtual and every tile it reads is the zero-framed one's or sits
under the mask.
"""
import re

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from tt_bio import sdpa_generic as SG                                       # noqa: E402


def test_the_generated_reader_only_twins_the_two_placement_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    d = SG.kv_window_kernel_dir()
    gen = (d / "dataflow/reader_interleaved.cpp").read_text()
    stock = (SG._kdir() / "dataflow/reader_interleaved.cpp").read_text()
    assert gen.count("#ifdef KV_WINDOW_") == 5
    gen = gen.replace("\n" + SG._KV_LEAD_READ, "", 1)
    gen = re.sub(r"#ifdef KV_WINDOW_\w+\n.*?#else\n", "", gen, flags=re.S)
    for line in SG._KV_WINDOW_ORIG[1:] + (SG._Q_WINDOW_ORIG,) + SG._KV_READERS_ORIG:
        gen = gen.replace(line + "\n#endif", line, 1)
    assert gen == stock
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


@pytest.mark.parametrize("H,nbr,lead,S", [(4, 16, 2, 5), (8, 3, 1, 3), (2, 1, 2, 5)])
def test_a_virtual_frame_reads_the_real_rows_or_a_masked_slot(H, nbr, lead, S):
    """`_KV_LEAD_READ`'s remap, in torch: K is [H, nbr] tile rows, the window of query block i is virtual
    frame rows i .. i + S - 1 with `lead` zero rows in front. Wherever the zero frame has a real row the
    virtual frame reads that row; elsewhere it reads some real row of the same head, which the bias masks."""
    ft = nbr + lead + S                                                     # KV_WINDOW_FRAME_T
    for head in range(H * nbr):
        for r in range(S):
            vid = (head // nbr) * ft + head % nbr + r                       # the twin's start id, DHt 1
            h, fr = divmod(vid, ft)
            real = h * nbr + min(max(fr - lead, 0), nbr - 1)
            framed = head % nbr + r - lead                                  # its row in the zero frame
            assert real // nbr == head // nbr
            if 0 <= framed < nbr:
                assert real == (head // nbr) * nbr + framed
