#!/usr/bin/env python3
"""gap_dpb: the DiT's stacked SDPA masks, shipped ttnn chain against the one-program `tt_bio.dit_pair_bias`.
  shipped  _tuned_linear(z, W) -> permute (0, 3, 1, 2) -> _sdpa_dense_mask (multiply, add, pad rows to n)
  dpb      dit_pair_bias(z, W^T, mask padded once, sqrt(head_dim))
  dpb_split  the same with n_out=18 (one [1, 16, n, n] tensor per block); must equal dpb bit for bit
Both against a float64 reference of the same bf16 inputs (pad rows must be the pad value exactly), and against
each other (bit-equal fraction, max abs). fid_<F>[_raw]: dpb at each math fidelity, with and without scale + mask. Sizes: the 685-token page and a 418-token one (partial last row tile).
    python scripts/rfd3_port/gap_dpb.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.tiles import pad_axis                                   # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402
from tt_bio import dit_pair_bias as D                                    # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_dpb.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
C, H, HD = 128, 16 * 18, 48


def timeit(fn, dev):
    o = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N):
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), min(ts), max(ts)


def case(dev, ckc, I):
    n = (I + 31) // 32 * 32
    g = torch.Generator().manual_seed(I)
    z_t = torch.randn(1, I, I, C, generator=g).bfloat16()   # logical I x I, tile-padded to n like the DiT's z
    w_t = (torch.randn(C, H, generator=g) / C ** 0.5).bfloat16()
    m_t = torch.where(torch.rand(1, 1, I, I, generator=g) < 0.3, -1e4, 0.0).bfloat16()
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z, w, m = up(z_t), up(w_t), up(m_t)
    wt = up(w_t.t().contiguous().reshape(1, 1, H, C))
    mp = pad_axis(pad_axis(m, n, 3, -1e4), n, 2, -1e4)
    pad = float(ttnn.to_torch(mp)[0, 0, -1, 0])
    ref = torch.full((1, H, n, n), pad, dtype=torch.float64)
    ref[:, :, :I, :I] = torch.einsum("bijc,ch->bhij", z_t.double(), w_t.double()) * HD ** 0.5 + m_t.double()

    def shipped():
        pb = M._tuned_linear(z, w, ckc=ckc, dtype=ttnn.bfloat16, core_grid=M.CORE_GRID_MAIN)
        pb_t = ttnn.permute(pb, (0, 3, 1, 2))
        ttnn.deallocate(pb)
        out = M._sdpa_dense_mask(pb_t, m, HD, n)
        ttnn.deallocate(pb_t)
        return out

    def dpb():
        return D.dit_pair_bias(z, wt, mp, HD ** 0.5, ckc.math_fidelity)

    def dpb_raw():
        return D.dit_pair_bias(z, wt, mp, HD ** 0.5, ckc.math_fidelity, raw=True)

    raw_ref = torch.zeros(1, H, n, n, dtype=torch.float64)
    raw_ref[:, :, :I, :I] = torch.einsum("bijc,ch->bhij", z_t.double(), w_t.double())
    rec, outs = {"I": I, "n": n, "eligible": D.eligible(z, wt, mp)}, {}
    try:
        o = D.dit_pair_bias(z, wt, mp, HD ** 0.5, ckc.math_fidelity, raw=True, k1=False)
        d = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
        e = (d - raw_ref).abs()
        idx = (e > 0.25).nonzero()
        r = {"rel_err_f64": float((d - raw_ref).norm() / raw_ref.norm()), "max_abs_f64": float(e.max()),
             "n_bad": int(idx.shape[0]), "bad_first": [[int(x) for x in row] + [float(d[tuple(row)]),
                                                         float(raw_ref[tuple(row)])] for row in idx[:12]]}
        for k, nm in ((1, "h"), (2, "i"), (3, "j")):
            r[f"bad_{nm}_mod32"] = sorted(set(int(x) for x in (idx[:, k] % 32).tolist()))[:40]
            r[f"bad_{nm}_tile"] = sorted(set(int(x) for x in (idx[:, k] // 32).tolist()))[:40]
    except Exception as ex:                                              # noqa: BLE001
        r = {"error": repr(ex)[:400]}
    rec["dpb_raw"] = r
    print(I, "dpb_raw", r, flush=True)
    def dpb_fpu():
        return D.dit_pair_bias(z, wt, mp, HD ** 0.5, ckc.math_fidelity, k1=False)

    for name, fn in (("shipped", shipped), ("dpb", dpb), ("dpb_fpu", dpb_fpu)):
        try:
            o, med, lo, hi = timeit(fn, dev)
            d = ttnn.to_torch(o).double()
            ttnn.deallocate(o)
            assert tuple(d.shape) == tuple(ref.shape), (tuple(d.shape), tuple(ref.shape))
            v = ref[:, :, :I, :I] > -5e3
            e = (d - ref)[:, :, :I, :I][v]
            r = {"ms_median": med, "ms_min": lo, "ms_max": hi,
                 "rel_err_f64": float(e.norm() / ref[:, :, :I, :I][v].norm()), "max_abs_f64": float(e.abs().max()),
                 "pad_exact": bool((d[:, :, I:] == ref[:, :, I:]).all() and (d[..., I:] == ref[..., I:]).all()),
                 "masked_max_abs": float((d - ref)[:, :, :I, :I][~v].abs().max())}
            bad = ((d - ref).abs() > 1.0)
            bad[:, :, :I, :I] &= v
            bad[:, :, I:] = False
            bad[..., I:] = False
            idx = bad.nonzero()
            r["n_bad"] = int(idx.shape[0])
            if idx.shape[0]:
                r["bad_first"] = [[int(x) for x in row] + [float(d[tuple(row)]), float(ref[tuple(row)])]
                                  for row in idx[:12]]
                for k, nm in ((1, "h"), (2, "i"), (3, "j")):
                    r[f"bad_{nm}_mod32"] = sorted(set(int(x) for x in (idx[:, k] % 32).tolist()))[:40]
                    r[f"bad_{nm}_tile"] = sorted(set(int(x) for x in (idx[:, k] // 32).tolist()))[:40]
            outs[name] = d
        except Exception as ex:                                          # noqa: BLE001
            r = {"error": repr(ex)[:400]}
        rec[name] = r
        print(I, name, r, flush=True)
    # fidelity sweep: is the program compute-bound? The same dpb at each fidelity (and raw, no scale/mask read).
    for fid in ("HiFi4", "HiFi3", "HiFi2", "LoFi"):
        f = getattr(ttnn.MathFidelity, fid)
        for raw in (False, True):
            try:
                o, med, lo, _ = timeit(lambda: D.dit_pair_bias(z, wt, mp, HD ** 0.5, f, raw=raw), dev)
                d = ttnn.to_torch(o).double()
                ttnn.deallocate(o)
                if raw:
                    r = {"ms_median": med, "ms_min": lo}
                else:
                    v = ref[:, :, :I, :I] > -5e3
                    e = (d - ref)[:, :, :I, :I][v]
                    r = {"ms_median": med, "ms_min": lo, "rel_err_f64": float(e.norm() / ref[:, :, :I, :I][v].norm()),
                         "max_abs_f64": float(e.abs().max()), "mean_abs_f64": float(e.abs().mean())}
            except Exception as ex:                                      # noqa: BLE001
                r = {"error": repr(ex)[:400]}
            rec[f"fid_{fid}{'_raw' if raw else ''}"] = r
            print(I, fid, "raw" if raw else "", r, flush=True)
    try:   # dpb_split: the same program writing each block's 16 heads to its own tensor; must equal dpb exactly
        split = lambda: D.dit_pair_bias(z, wt, mp, HD ** 0.5, ckc.math_fidelity, n_out=H // 16)  # noqa: E731
        os_ = split()
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(N):
            for o in os_:
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            os_ = split()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        d = torch.cat([ttnn.to_torch(o).double() for o in os_], dim=1)
        r = {"ms_median": statistics.median(ts), "ms_min": min(ts), "ms_max": max(ts)}
        if "dpb" in outs:
            r["equal_dpb"] = bool(torch.equal(d, outs["dpb"]))
    except Exception as ex:                                              # noqa: BLE001
        r = {"error": repr(ex)[:400]}
    rec["dpb_split"] = r
    print(I, "dpb_split", r, flush=True)
    if "shipped" in outs and "dpb" in outs:
        a, b = outs["shipped"], outs["dpb"]
        rec["dpb_vs_shipped"] = {"bit_equal_frac": float((a == b).double().mean()), "max_abs": float((a - b).abs().max())}
        print(I, "dpb_vs_shipped", rec["dpb_vs_shipped"], flush=True)
    return rec


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    rec = {"reps": N, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
           "arch": str(dev.arch()), "cases": [case(dev, ckc, I) for I in (685, 418)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
