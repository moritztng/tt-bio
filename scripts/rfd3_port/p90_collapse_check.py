#!/usr/bin/env python3
"""p90 -- the shipped process_z block against DiffusionTokenEncoder._process_z_collapsed.

p89 measured the identity on hand-built arms. This runs the code that actually landed: the
run_device branch's two sides, at the page fixture's [1,685,685,*], for both recycles, plus the
three things the collapsed path can get wrong that a numeric diff would not catch --

  * the invariant cache must HIT across steps (Ainv is O(I^2); rebuilding it per step would cost
    more than the lever saves) and must be RELEASED when Z_init_II changes, or eight designs leak
    ~2 GB of DRAM;
  * the table must be indexed the way _combined_onehot_dev lays its columns out, not the way this
    file assumes;
  * the first recycle (bins_self None, n_ones=1) and the second (n_ones=2) are different tables
    and different scales.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p90/collapse_check.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
I = int(os.environ.get("P90_I", "685"))
NB, CZ, W258, EPS = 65, 128, 258, 1e-6
STEPS, CALLS_PER_STEP = 200, 2


def timeit(fn, dev, n=N, warm=2):
    for _ in range(warm):
        o = fn()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    return statistics.median(out), min(out), max(out)


def main():
    dev = get_device()
    torch.manual_seed(7)
    ckc = M._default_compute_kernel_config()
    dt, B = ttnn.bfloat16, 1
    e = object.__new__(M.DiffusionTokenEncoder)
    e.device, e.compute_kernel_config, e.dtype = dev, ckc, dt
    e._const, e._zinv = {}, None
    e.process_z_n = ttnn.from_torch(torch.randn(W258) * 0.3 + 1.0, dtype=dt,
                                   layout=ttnn.TILE_LAYOUT, device=dev)
    e.process_z_w = ttnn.from_torch(torch.randn(W258, CZ) * 0.05, dtype=dt,
                                    layout=ttnn.TILE_LAYOUT, device=dev)

    Z = (torch.randn(B, I, I, CZ) * 0.4)
    bins = torch.randint(0, NB, (B, I, I))
    bins_self = torch.randint(0, NB, (B, I, I))
    print("[p90] I=%d  collapse flag default=%s" % (I, M._PROCESS_Z_COLLAPSE), flush=True)

    def shipped(bs):
        z = e._batched(M._tt_cached(Z, dev, dt), B)
        dself = e._combined_onehot_dev(bins, bs, B, I)
        wide = ttnn.concat([z, dself], dim=-1)
        ttnn.deallocate(dself)
        zcat = ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, W258])
        ttnn.deallocate(wide)
        o = ttnn.rms_norm(zcat, weight=e.process_z_n, epsilon=EPS, compute_kernel_config=ckc)
        o = ttnn.linear(o, e.process_z_w, compute_kernel_config=ckc, dtype=dt,
                        core_grid=M.CORE_GRID_MAIN)
        ttnn.deallocate(zcat)
        return o

    res, tot = {}, {"shipped": 0.0, "collapse": 0.0}
    for rec, bs in (("r0_self_none", None), ("r1_self", bins_self)):
        ref = ttnn.to_torch(shipped(bs)).float()
        got = ttnn.to_torch(e._process_z_collapsed(Z, bins, bs, B, I)).float()
        d = (got - ref).abs()
        refmax = ref.abs().max().item()
        relmed = (d / ref.abs().clamp_min(1e-3)).median().item()
        ms_s = timeit(lambda: shipped(bs), dev)
        ms_c = timeit(lambda: e._process_z_collapsed(Z, bins, bs, B, I), dev)
        tot["shipped"] += ms_s[0]
        tot["collapse"] += ms_c[0]
        print("\n  %-12s shipped %8.3f ms  collapse %8.3f ms  (%.2fx)"
              % (rec, ms_s[0], ms_c[0], ms_s[0] / ms_c[0]), flush=True)
        print("               maxabs %.3e  rel_max %.3e  rel_med %.3e  refmax %.4f"
              % (d.max().item(), d.max().item() / refmax, relmed, refmax), flush=True)
        res[rec] = dict(shipped_ms=ms_s, collapse_ms=ms_c, maxabs=d.max().item(),
                        rel_max=d.max().item() / refmax, rel_med=relmed)

    # --- cache behaviour ------------------------------------------------------------------
    keys = sorted(e._zinv[1].keys())
    held_key = e._zinv[0]
    e._process_z_collapsed(Z, bins, bins_self, B, I)
    hit = e._zinv[0] == held_key and sorted(e._zinv[1].keys()) == keys
    old = [t for pair in e._zinv[1].values() for t in pair]
    Z2 = (torch.randn(B, I, I, CZ) * 0.4)
    e._process_z_collapsed(Z2, bins, bins_self, B, I)
    released = all(not t.is_allocated() for t in old)
    newkeys = sorted(e._zinv[1].keys())
    print("\n  invariant cache: same Z_init reuses = %s ; new Z_init releases the old = %s ;"
          " n_ones keys %s -> %s" % (hit, released, keys, newkeys), flush=True)
    assert hit, "invariant cache MISSED on an unchanged Z_init -- Ainv would rebuild per step"
    assert released, "old design's invariants NOT released -- O(I^2) leak per design"

    # --- the table's column mapping, derived from the shipped one-hot -----------------------
    wn = ttnn.to_torch(e.process_z_n).float().reshape(-1)
    ww = ttnn.to_torch(e.process_z_w).float()
    # _combined_onehot_dev keys its table on `bins_self is None`, so the no-self table is True
    comb = ttnn.to_torch(e._const[("comb", True, dt)]).float()
    tab = ttnn.to_torch(e._process_z_table(False)).float()
    bd = 17
    want = (comb[bd][:130] * wn[CZ:]) @ ww[CZ:]
    err = (tab[bd] - want).abs().max().item()
    print("  table row vs shipped one-hot contribution, bd=%d: maxabs %.3e" % (bd, err), flush=True)
    assert err < 5e-3, err
    comb2 = ttnn.to_torch(e._const[("comb", False, dt)]).float()
    tab2 = ttnn.to_torch(e._process_z_table(True)).float()
    bs_ = 41
    want2 = (comb2[bd * NB + bs_][:130] * wn[CZ:]) @ ww[CZ:]
    err2 = (tab2[bd * NB + bs_] - want2).abs().max().item()
    print("  table row vs shipped one-hot contribution, (bd,bs)=(%d,%d): maxabs %.3e"
          % (bd, bs_, err2), flush=True)
    assert err2 < 5e-3, err2

    print("\n=== %d steps x %d recycles ===" % (STEPS, CALLS_PER_STEP), flush=True)
    print("  shipped  %8.3f ms/step -> %6.3f s/design" % (tot["shipped"], tot["shipped"] * STEPS / 1e3),
          flush=True)
    print("  collapse %8.3f ms/step -> %6.3f s/design   saves %6.3f s/design (ISOLATED, a ceiling)"
          % (tot["collapse"], tot["collapse"] * STEPS / 1e3,
             (tot["shipped"] - tot["collapse"]) * STEPS / 1e3), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(I=I, N=N, recycles=res, ms_step=tot,
                                   isolated_saving_s_design=(tot["shipped"] - tot["collapse"]) * STEPS / 1e3,
                                   cache_hit=hit, cache_released=released,
                                   table_err=[err, err2]), indent=2))
    print("wrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
