"""r4t-perf L4 bar (a)/(d): per-site rel vs float64, served path (`autograd._mmvjp_dx/_dw`, TT_BIO_MMVJP_MIN=1) vs stock
(`autograd._matmul`), at every PairUpdate crop-256 linear-VJP shape. (a): served rel <= 1.05x stock rel at every site
and every site served. (d) with TT_BIO_MMVJP_BROKEN=1: the square dX sites take the un-transposed W and must FAIL (a)."""
import argparse, json, os, subprocess, time
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch, ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    assert ag.MMVJP_MIN, "run with TT_BIO_MMVJP_MIN=1"
    dev = T.get_device(); cfg = ag.precise_config(); K = 65536
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    gen = torch.Generator().manual_seed(1); rows = []
    rel = lambda o, r: float((ttnn.to_torch(o).double().reshape(r.shape) - r).norm() / r.norm())
    for i_, o_ in ((384, 1024), (1024, 384), (384, 384), (768, 384), (384, 768), (256, 384), (384, 256)):  # W [in, out]
        X = torch.randn(K, i_, generator=gen).bfloat16(); G = torch.randn(K, o_, generator=gen).bfloat16()
        W = (torch.randn(i_, o_, generator=gen) / i_ ** 0.5).bfloat16()
        dx_ref = G.double() @ W.double().T; dw_ref = X.double().T @ G.double()
        tx, tg, tw = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev) for t in (X, G, W))
        s_dx = ag._matmul(tg, tw, transpose_b=True, compute_kernel_config=cfg)
        s_dw = ag._matmul(tx, tg, transpose_a=True, compute_kernel_config=cfg, dtype=ttnn.float32)
        m_dx = ag._mmvjp_dx(tg, tw, cfg); m_dw = ag._mmvjp_dw(tx, tg, cfg)
        for kind, s_, m_, r_ in (("dX", s_dx, m_dx, dx_ref), ("dW", s_dw, m_dw, dw_ref)):
            row = dict(kind=kind, w=[i_, o_], served=m_ is not None, stock_rel=rel(s_, r_))
            row["served_rel"] = rel(m_, r_) if m_ is not None else None
            row["ok"] = bool(m_ is not None and row["served_rel"] <= 1.05 * row["stock_rel"])
            rows.append(row); print(json.dumps(row), flush=True)
        for t in (tx, tg, tw, s_dx, s_dw) + tuple(t for t in (m_dx, m_dw) if t is not None):
            ttnn.deallocate(t)
    res = dict(host="qb2", chip=int(os.environ["TT_VISIBLE_DEVICES"]), board="p300c", sha=sha, broken=ag.MMVJP_BROKEN,
               rows=rows, stats=dict(ag.MMVJP_STATS), verdict="PASS" if all(r["ok"] for r in rows) else "FAIL")
    json.dump(res, open(a.out, "w"), indent=1); print(res["verdict"], res["stats"])
if __name__ == "__main__":
    main()
