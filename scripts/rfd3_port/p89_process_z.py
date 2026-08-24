#!/usr/bin/env python3
"""p89 -- the token encoder's process_z chain: what it costs and what is left in it.

The census (perf/p84/census_p3.txt) puts the token encoder at 206.0 ms/step, 42 % of the step and
the largest region in the model. E4.2 measured its eight pair Transitions at 141.01 ms/step, so
~65 ms/step is everything else, and the board has carried that as one never-screened "encoder
one-hot + rms_norm + process_z in one pass ~4 s/design" item since p2.

N_RECYCLE=2, so DiffusionTokenEncoder.run_device runs TWICE per step and this chain with it. Per
call at the production [1,685,685,*] the shipped route is

    embedding -> to_layout(TILE) -> concat(z,dself) 288 -> slice 258 -> rms_norm -> linear

which moves ~2.4 GB, i.e. ~4.8 GB/step -- exactly the census's unaccounted remainder in this
region (33.58 total, 25.68 in the pair Transitions, 3.10 in pf attn).

Three arms, and the split between them is the whole point because a non-bit-exact lever in this
lineage costs ~90 h of card time to license (E3.5):

  shipped   the code in model.py:2517-2530, run as-is
  tile_emb  ttnn.embedding straight to TILE_LAYOUT, dropping the to_layout           BIT-EXACT?
  no_slice  concat whose last operand is 130 wide, so the 288->258 slice goes away   BIT-EXACT?
  collapse  the algebraic identity: rms scale and the z half of the linear are both
            step-invariant, and the one-hot half of the linear is an embedding lookup NOT bit-exact

The identity: zcat = [z(128) | e_bd(65) | e_bs(65)], so sum(zcat^2) = sum(z^2) + n_ones with
n_ones constant, hence the rms scale depends only on z -- which is Z_init_II, invariant across all
200 steps. And linear(x*inv*w_n) = inv * ((z*w_n_z) @ W_z + w_n[128+bd]*W[128+bd] + w_n[193+bs]*W[193+bs]),
whose last two terms are a lookup in a [65*65, 128] table built on host.
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p89/process_z.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
I = int(os.environ.get("P89_I", "685"))
NB, CZ, W258 = 65, 128, 258
EPS = 1e-6
STEPS, CALLS_PER_STEP = 200, 2      # N_RECYCLE=2


def timeit(fn, dev, n=N, warm=2):
    for _ in range(warm):
        o = fn()
        if o is not None and hasattr(o, "deallocate"):
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3)
        if o is not None and hasattr(o, "deallocate"):
            ttnn.deallocate(o)
    return statistics.median(out), min(out), max(out)


def mk_enc(dev, ckc, w_n_h, w_w_h):
    e = object.__new__(M.DiffusionTokenEncoder)
    e.device, e.compute_kernel_config, e.dtype = dev, ckc, ttnn.bfloat16
    e._const = {}
    e.process_z_n = ttnn.from_torch(w_n_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    e.process_z_w = ttnn.from_torch(w_w_h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return e


def shipped(e, z, bins, bins_self, B, dev, ckc, dt):
    """model.py:2517-2530 verbatim, _CONCAT_ALIGNED path."""
    dself = e._combined_onehot_dev(bins, bins_self, B, I)
    wide = ttnn.concat([z, dself], dim=-1)
    ttnn.deallocate(dself)
    zcat = ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, W258])
    ttnn.deallocate(wide)
    out = ttnn.rms_norm(zcat, weight=e.process_z_n, epsilon=EPS, compute_kernel_config=ckc)
    out = ttnn.linear(out, e.process_z_w, compute_kernel_config=ckc, dtype=dt,
                      core_grid=M.CORE_GRID_MAIN)
    ttnn.deallocate(zcat)
    return out


def onehot_tile(e, bins, bins_self, B, dev):
    """_combined_onehot_dev with the embedding writing TILE_LAYOUT directly."""
    dt, n, w = e.dtype, NB, e.COMBINED_ONEHOT_W
    tab = e._const[("comb", bins_self is None, dt)]
    idx = M._tt_idx(bins if bins_self is None else bins * n + bins_self, dev)
    oh = ttnn.embedding(idx, tab, layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(oh, (B, I, I, w))


def main():
    dev = get_device()
    torch.manual_seed(7)
    ckc = M._default_compute_kernel_config()
    dt, B = ttnn.bfloat16, 1
    w_n_h = (torch.randn(W258) * 0.3 + 1.0)
    w_w_h = torch.randn(W258, CZ) * 0.05                 # already .t()-ed, [in, out]
    e = mk_enc(dev, ckc, w_n_h, w_w_h)

    z_h = torch.randn(B, I, I, CZ) * 0.4
    z = ttnn.from_torch(z_h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    bins = torch.randint(0, NB, (B, I, I))
    bins_self = torch.randint(0, NB, (B, I, I))

    print("[p89] I=%d  z %s padded %s  ttnn %s" % (I, tuple(z.shape), tuple(z.padded_shape),
                                                   ttnn.__version__ if hasattr(ttnn, "__version__") else "?"),
          flush=True)

    res, arms = {}, {}
    for rec, bs in (("r0_self_none", None), ("r1_self", bins_self)):
        print("\n=== recycle arm %s  (n_ones=%d) ===" % (rec, 1 if bs is None else 2), flush=True)
        # prime the constant table exactly as the shipped path does
        e._combined_onehot_dev(bins, bs, B, I)

        # ---- per-op split of the shipped chain -------------------------------------------
        dself = e._combined_onehot_dev(bins, bs, B, I)
        split = {}
        split["onehot"] = timeit(lambda: e._combined_onehot_dev(bins, bs, B, I), dev)[0]
        wide = ttnn.concat([z, dself], dim=-1)
        split["concat"] = timeit(lambda: ttnn.concat([z, dself], dim=-1), dev)[0]
        zcat = ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, W258])
        split["slice"] = timeit(lambda: ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, W258]), dev)[0]
        rn = ttnn.rms_norm(zcat, weight=e.process_z_n, epsilon=EPS, compute_kernel_config=ckc)
        split["rms_norm"] = timeit(lambda: ttnn.rms_norm(zcat, weight=e.process_z_n, epsilon=EPS,
                                                         compute_kernel_config=ckc), dev)[0]
        split["linear"] = timeit(lambda: ttnn.linear(rn, e.process_z_w, compute_kernel_config=ckc,
                                                     dtype=dt, core_grid=M.CORE_GRID_MAIN), dev)[0]
        for k, v in split.items():
            print("    %-9s %8.3f ms/call" % (k, v), flush=True)
        print("    %-9s %8.3f ms  (sum of parts)" % ("SUM", sum(split.values())), flush=True)
        for t in (dself, wide, zcat, rn):
            ttnn.deallocate(t)

        # ---- arm: shipped ----------------------------------------------------------------
        med, lo, hi = timeit(lambda: shipped(e, z, bins, bs, B, dev, ckc, dt), dev)
        ref = ttnn.to_torch(shipped(e, z, bins, bs, B, dev, ckc, dt)).float()
        arms.setdefault(rec, {})["shipped"] = dict(ms=med, lo=lo, hi=hi, maxabs=0.0, rel=0.0)
        print("  shipped   %8.3f ms/call  [%.3f, %.3f]" % (med, lo, hi), flush=True)
        refmax = ref.abs().max().item()

        # ---- arm: tile_emb ---------------------------------------------------------------
        try:
            def a_tile():
                dself = onehot_tile(e, bins, bs, B, dev)
                wide = ttnn.concat([z, dself], dim=-1)
                ttnn.deallocate(dself)
                zc = ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, W258])
                ttnn.deallocate(wide)
                o = ttnn.rms_norm(zc, weight=e.process_z_n, epsilon=EPS, compute_kernel_config=ckc)
                o = ttnn.linear(o, e.process_z_w, compute_kernel_config=ckc, dtype=dt,
                                core_grid=M.CORE_GRID_MAIN)
                ttnn.deallocate(zc)
                return o
            got = ttnn.to_torch(a_tile()).float()
            d = (got - ref).abs().max().item()
            med, lo, hi = timeit(a_tile, dev)
            arms[rec]["tile_emb"] = dict(ms=med, lo=lo, hi=hi, maxabs=d, rel=d / refmax)
            print("  tile_emb  %8.3f ms/call  [%.3f, %.3f]  maxabs %.3e  %s"
                  % (med, lo, hi, d, "BIT-EXACT" if d == 0 else "DIFFERS"), flush=True)
        except Exception as ex:
            arms[rec]["tile_emb"] = dict(error=str(ex)[:300])
            print("  tile_emb  FAILED: %s" % str(ex)[:200], flush=True)

        # ---- arm: no_slice ---------------------------------------------------------------
        try:
            n = NB
            ar = torch.arange(n)
            if bs is None:
                t130 = torch.zeros(n, 130); t130[ar, ar] = 1.0
            else:
                t130 = torch.zeros(n * n, 130)
                row = ar.repeat_interleave(n) * n + ar.repeat(n)
                t130[row, ar.repeat_interleave(n)] = 1.0
                t130[row, n + ar.repeat(n)] = 1.0
            tab130 = ttnn.from_torch(t130, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

            def a_nos():
                idx = M._tt_idx(bins if bs is None else bins * n + bs, dev)
                oh = ttnn.embedding(idx, tab130, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                oh = ttnn.reshape(oh, (B, I, I, 130))
                oh = ttnn.to_layout(oh, ttnn.TILE_LAYOUT)
                zc = ttnn.concat([z, oh], dim=-1)
                ttnn.deallocate(oh)
                o = ttnn.rms_norm(zc, weight=e.process_z_n, epsilon=EPS, compute_kernel_config=ckc)
                o = ttnn.linear(o, e.process_z_w, compute_kernel_config=ckc, dtype=dt,
                                core_grid=M.CORE_GRID_MAIN)
                ttnn.deallocate(zc)
                return o
            o = a_nos()
            print("     no_slice concat width -> %s padded %s" % (tuple(o.shape), tuple(o.padded_shape)),
                  flush=True)
            got = ttnn.to_torch(o).float()
            d = (got - ref).abs().max().item()
            med, lo, hi = timeit(a_nos, dev)
            arms[rec]["no_slice"] = dict(ms=med, lo=lo, hi=hi, maxabs=d, rel=d / refmax)
            print("  no_slice  %8.3f ms/call  [%.3f, %.3f]  maxabs %.3e  %s"
                  % (med, lo, hi, d, "BIT-EXACT" if d == 0 else "DIFFERS"), flush=True)
        except Exception as ex:
            arms[rec]["no_slice"] = dict(error=str(ex)[:300])
            print("  no_slice  FAILED: %s" % str(ex)[:200], flush=True)

        # ---- arm: collapse ---------------------------------------------------------------
        try:
            n_ones = 1.0 if bs is None else 2.0
            zb = z_h.to(torch.bfloat16).float()
            ss = (zb * zb).sum(-1) + n_ones
            inv_h = torch.rsqrt(ss / W258 + EPS).unsqueeze(-1)          # [B,I,I,1]
            wn = w_n_h.to(torch.bfloat16).float()
            ww = w_w_h.to(torch.bfloat16).float()
            # invariant half: (z * w_n_z) @ W_z, then scaled by inv
            wnz = ttnn.from_torch((wn[:CZ]).reshape(1, 1, 1, CZ), dtype=dt,
                                  layout=ttnn.TILE_LAYOUT, device=dev)
            Wz = ttnn.from_torch(ww[:CZ], dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            zs = ttnn.multiply(z, wnz)
            A = ttnn.linear(zs, Wz, compute_kernel_config=ckc, dtype=dt, core_grid=M.CORE_GRID_MAIN)
            ttnn.deallocate(zs)
            inv = ttnn.from_torch(inv_h, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            Ainv = ttnn.multiply(A, inv)
            ttnn.deallocate(A)
            # the one-hot half as a lookup table
            Td = wn[CZ:CZ + NB].unsqueeze(-1) * ww[CZ:CZ + NB]           # [65,128]
            if bs is None:
                Tc = Td
                idx_h = bins
            else:
                Ts = wn[CZ + NB:].unsqueeze(-1) * ww[CZ + NB:]           # [65,128]
                Tc = (Td.unsqueeze(1) + Ts.unsqueeze(0)).reshape(NB * NB, CZ)
                idx_h = bins * NB + bs
            Tt = ttnn.from_torch(Tc, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

            def mk_col(tile):
                def a_col():
                    idx = M._tt_idx(idx_h, dev)
                    if tile:
                        em = ttnn.embedding(idx, Tt, layout=ttnn.TILE_LAYOUT,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        em = ttnn.reshape(em, (B, I, I, CZ))
                    else:
                        em = ttnn.embedding(idx, Tt, layout=ttnn.ROW_MAJOR_LAYOUT,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        em = ttnn.reshape(em, (B, I, I, CZ))
                        em = ttnn.to_layout(em, ttnn.TILE_LAYOUT)
                    o = ttnn.multiply(em, inv)
                    ttnn.deallocate(em)
                    o2 = ttnn.add(o, Ainv)
                    ttnn.deallocate(o)
                    return o2
                return a_col
            for name, tile in (("collapse", False), ("collapse_tile", True)):
                try:
                    fn = mk_col(tile)
                    got = ttnn.to_torch(fn()).float()
                    d = (got - ref).abs().max().item()
                    rel = d / refmax
                    den = ref.abs().clamp_min(1e-3)
                    relel = ((got - ref).abs() / den).median().item()
                    med, lo, hi = timeit(fn, dev)
                    arms[rec][name] = dict(ms=med, lo=lo, hi=hi, maxabs=d, rel=rel, rel_median=relel)
                    print("  %-9s %8.3f ms/call  [%.3f, %.3f]  maxabs %.3e  rel_max %.3e  rel_med %.3e"
                          % (name, med, lo, hi, d, rel, relel), flush=True)
                except Exception as ex:
                    arms[rec][name] = dict(error=str(ex)[:300])
                    print("  %-9s FAILED: %s" % (name, str(ex)[:200]), flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            arms[rec]["collapse"] = dict(error=str(ex)[:300])
            print("  collapse  FAILED: %s" % str(ex)[:200], flush=True)

        res[rec] = dict(split={k: round(v, 4) for k, v in split.items()},
                        arms={k: v for k, v in arms[rec].items()})

    print("\n=== s/design, %d steps x %d recycles ===" % (STEPS, CALLS_PER_STEP), flush=True)
    base = sum(res[r]["arms"]["shipped"]["ms"] for r in res)          # one r0 + one r1 per step
    print("  shipped chain, both recycles      %8.3f ms/step -> %6.3f s/design"
          % (base, base * STEPS / 1e3), flush=True)
    for arm in ("tile_emb", "no_slice", "collapse", "collapse_tile"):
        if all("ms" in res[r]["arms"].get(arm, {}) for r in res):
            tot = sum(res[r]["arms"][arm]["ms"] for r in res)
            ex = all(res[r]["arms"][arm]["maxabs"] == 0 for r in res)
            print("  %-33s %8.3f ms/step -> %6.3f s/design   saves %6.3f s/design  %s"
                  % (arm, tot, tot * STEPS / 1e3, (base - tot) * STEPS / 1e3,
                     "BIT-EXACT" if ex else "not bit-exact"), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(I=I, N=N, steps=STEPS, calls_per_step=CALLS_PER_STEP,
                                   shipped_ms_step=base, recycles=res), indent=2))
    print("\nwrote %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
