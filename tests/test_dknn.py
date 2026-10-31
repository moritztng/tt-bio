"""tt_bio.dknn's contract, host only: its fixed point and packed mask, with the kernel's per-row rule run in numpy,
reproduce _neighbours_kdtree (the float64 k-d tree the model uses today) row for row."""
import numpy as np
import pytest
import torch

dknn = pytest.importorskip("tt_bio.dknn")
M = pytest.importorskip("tt_bio.rfd3.model")


def _row(mw, r, c, L, K, WO):
    """dknn.cpp's output row: kept atoms c merged with the row's first K - n_fill mask bits, padded with L."""
    j = np.arange(L)
    bits = np.flatnonzero((mw[r, 1 + (j >> 5)] >> (j & 31).astype(np.uint32)) & 1)[: K - mw[r, 0]]
    row = np.full(WO, L, np.int32)
    row[: len(bits) + len(c)] = np.sort(np.concatenate([bits, c]))
    return row


def _kernel_rows(q, mw, L, K, WO):
    """dknn.cpp per row: k-th uint64 key (quickselect on the atoms under a 1-in-8 sample's bound, or on every atom if
    fewer than K are under it), outside-mask atoms within it, the n_fill nearest (ties to the lower index), merged
    with the sequence entries."""
    out = np.full((L, WO), L, np.int32)
    j = np.arange(L)
    ts, sampled = 2 * K // 8 - 1, L >= 4 * K
    for r in range(L):
        d = ((q[:, :3].astype(np.int64) - q[r, :3]) ** 2).sum(1).astype(np.uint64)
        under = d <= np.sort(d[::8])[ts] if sampled else np.ones(L, bool)
        if under.sum() < K:
            under[:] = True
        dk = np.sort(d[under])[K - 1]
        assert dk == np.sort(d)[K - 1]
        bit = (mw[r, 1 + (j >> 5)] >> (j & 31).astype(np.uint32)) & 1
        c = np.where((d <= dk) & (bit == 0))[0]
        if len(c) > mw[r, 0]:
            c = c[np.lexsort((c, d[c]))][: mw[r, 0]]
        out[r] = _row(mw, r, c, L, K, WO)
    return out


@pytest.mark.parametrize("sigma", [0.0, 5.0, 160.0])
def test_dknn_rule_equals_kdtree(sigma):
    L, K, NT = 700, 64, 80
    rng = np.random.default_rng(3)
    tok = np.repeat(np.arange(NT), np.diff(np.linspace(0, L, NT + 1).astype(int)))
    x = np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)[tok] + rng.normal(size=(L, 3)) * (1.0 + sigma)
    x = torch.from_numpy(x.astype(np.float32))
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    n_fill = (K - (seq < L).sum(1)).numpy()
    assert (n_fill == K - np.minimum(mask.numpy().sum(1), K)).all()   # how dknn.neighbours derives it
    mw = dknn.mask_words(mask.numpy(), n_fill).view(np.uint32)
    assert (mw[:, 0] == n_fill).all()
    q = dknn.fixed_point(x.numpy())
    assert np.abs(q).max() <= 2 ** 30
    out = _kernel_rows(q, mw, L, K, dknn._align(K))
    got = torch.from_numpy(out[:, :K]).long().unsqueeze(0)
    ref = M._neighbours_kdtree(mask, seq, x.unsqueeze(0), K)
    assert torch.equal(got, ref)



def test_fits_l1_bound():
    """The card index only takes designs whose two RISC-Vs' buffers fit one core; bigger ones keep the k-d tree."""
    assert dknn.fits(6051, 128)
    assert dknn.fits(16000, 128)
    assert not dknn.fits(20000, 128)
    big = max(L for L in range(1000, 30000, 8) if dknn.fits(L, 128))
    MW, WO = dknn._align(1 + (big + 31) // 32), dknn._align(128)
    assert 2 * sum(dknn._cb_sizes(big, MW, WO)) + dknn.PROGRAM_RESERVE <= dknn.L1_PER_CORE


def _cand_rows(q, mw, L, K, WO, table, m0, m1, full):
    """dknn.cpp's candidate mode: per table row, keys over its static candidates and the movers; the full row when
    the k-th of them is not under the row's bound. Returns the output and how many rows the bound proved."""
    out = full.copy()
    j = np.arange(L)
    proved = 0
    for w in table.view(np.uint32):
        r, bound = int(w[0]), int(w[1]) | (int(w[2]) << 32)
        c = np.concatenate([w[4:][w[4:] < L].astype(np.int64), np.arange(m0, m1)])
        d = ((q[c, :3].astype(np.int64) - q[r, :3]) ** 2).sum(1).astype(np.uint64)
        if len(c) < K or not int(np.sort(d)[K - 1]) < bound:
            continue
        proved += 1
        dk = np.sort(d)[K - 1]
        bit = (mw[r, 1 + (c >> 5)] >> (c & 31).astype(np.uint32)) & 1
        keep = (d <= dk) & (bit == 0)
        cc, dd = c[keep], d[keep]
        if len(cc) > mw[r, 0]:
            cc = cc[np.lexsort((cc, dd))][: mw[r, 0]]
        out[r] = _row(mw, r, cc, L, K, WO)
    return out, proved


@pytest.mark.parametrize("sigma", [0.3, 8.0, 3000.0])
def test_static_candidates_equal_full(monkeypatch, sigma):
    """Static's table + bound give each static row the full rule's result, at a coarse and a fine fixed point, with
    a binder pressed onto the target and duplicate coordinates (ties)."""
    monkeypatch.setattr(dknn, "_upload", lambda a, device: a)
    L, K, NT, NM = 900, 64, 100, 200
    rng = np.random.default_rng(11)
    tok = np.repeat(np.arange(NT), L // NT)
    base = np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)[tok] + rng.normal(size=(L, 3))
    base[L - NM:] = base[:NM] + rng.normal(size=(NM, 3)) * 0.5   # the binder lies on the target's first atoms
    base[300:310] = base[290:300]                                  # ties among static atoms
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    n_fill = (K - (seq < L).sum(1)).numpy()
    mw = dknn.mask_words(mask.numpy(), n_fill).view(np.uint32)
    WO = dknn._align(K)
    st = dknn.Static()
    for step in range(3):
        x = base.copy()
        x[L - NM:] += rng.normal(size=(NM, 3)) * sigma / (1 + step)
        x = x.astype(np.float32)
        plan = st.plan(None, x, K)
        assert (plan is None) == (step == 0)
        if plan is None:
            continue
        table, m0, m1 = plan
        assert (m0, m1) == (L - NM, L)
        q = dknn.fixed_point(x)
        full = _kernel_rows(q, mw, L, K, WO)
        got, proved = _cand_rows(q, mw, L, K, WO, table, m0, m1, full)
        assert np.array_equal(got, full)
        assert proved >= 0.5 * (L - NM), proved


def test_static_candidates_fall_back_on_ties(monkeypatch):
    """On a lattice the k-th and next static distances tie, so many bounds fail; those rows run in full."""
    monkeypatch.setattr(dknn, "_upload", lambda a, device: a)
    monkeypatch.setattr(dknn, "SLACK", 0)
    g = np.stack(np.meshgrid(*[np.arange(9)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    L, K, NM = len(g), 32, 100
    rng = np.random.default_rng(2)
    tok = np.arange(L) // 4
    mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 1)
    seq = torch.where(mask, torch.arange(L).expand(L, L), L).topk(K, dim=1, largest=False, sorted=True).values
    n_fill = (K - (seq < L).sum(1)).numpy()
    mw = dknn.mask_words(mask.numpy(), n_fill).view(np.uint32)
    WO = dknn._align(K)
    st = dknn.Static()
    st.plan(None, g.astype(np.float32), K)
    x = g.copy()
    x[L - NM:] += rng.normal(size=(NM, 3)) * 0.7
    x = x.astype(np.float32)
    table, m0, m1 = st.plan(None, x, K)
    q = dknn.fixed_point(x)
    full = _kernel_rows(q, mw, L, K, WO)
    got, proved = _cand_rows(q, mw, L, K, WO, table, m0, m1, full)
    assert np.array_equal(got, full)
    assert 0 < proved < L - NM, proved
