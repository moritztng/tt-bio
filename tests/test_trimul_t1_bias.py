"""T1's bias tiles (trimul_tail.t1_bias_tiles) in the order and replication reader.cpp and compute.cpp read them."""
import torch

from tt_bio import trimul_tail as TT


def test_bias_tiles_layout():
    ct2, ntg = 8, 4
    p_b, g_b, go_b = torch.randn(ct2 * 32), torch.randn(ct2 * 32), torch.randn(ntg * 32)
    t = TT.t1_bias_tiles(p_b, g_b, go_b)
    assert t.shape == (32, (2 * ct2 + ntg) * 32)
    tile = lambda j: t[:, j * 32:(j + 1) * 32]  # noqa: E731  (a [32, W] TILE tensor's page j)
    r = torch.arange(32)
    for q in range(ct2):
        # [a | b] tiles are channel-major: tile row = channel, every column the same.
        assert torch.equal(tile(q), p_b[q * 32 + r][:, None].expand(32, 32))
        assert torch.equal(tile(ct2 + q), g_b[q * 32 + r][:, None].expand(32, 32))
    for n in range(ntg):
        # g_out is position-major: tile column = channel, every row the same.
        assert torch.equal(tile(2 * ct2 + n), go_b[n * 32 + r][None, :].expand(32, 32))


def test_bias_tiles_reproduce_the_biased_projection():
    """Seeding dest with the tile and adding W^T xn^T is xn @ W + b, channel-major."""
    torch.manual_seed(0)
    K, ct2, ntg = 128, 8, 4
    xn = torch.randn(32, K, dtype=torch.float64)
    wp, wg, wo = (torch.randn(K, ct2 * 32, dtype=torch.float64), torch.randn(K, ct2 * 32, dtype=torch.float64),
                  torch.randn(K, ntg * 32, dtype=torch.float64))
    p_b, g_b, go_b = (torch.randn(ct2 * 32, dtype=torch.float64), torch.randn(ct2 * 32, dtype=torch.float64),
                      torch.randn(ntg * 32, dtype=torch.float64))
    t = TT.t1_bias_tiles(p_b, g_b, go_b)
    tile = lambda j: t[:, j * 32:(j + 1) * 32]  # noqa: E731
    for q in range(ct2):
        cm = tile(q) + wp[:, q * 32:(q + 1) * 32].T @ xn.T                 # [channels, positions]
        assert torch.allclose(cm, (xn @ wp + p_b)[:, q * 32:(q + 1) * 32].T)
    for n in range(ntg):
        pm = tile(2 * ct2 + n) + xn @ wo[:, n * 32:(n + 1) * 32]            # [positions, channels]
        assert torch.allclose(pm, (xn @ wo + go_b)[:, n * 32:(n + 1) * 32])
