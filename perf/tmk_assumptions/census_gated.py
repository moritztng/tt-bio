#!/usr/bin/env python3
"""tmk-assumptions pass 2: what channel move does a 512 aa production fold actually call?

Pass 1 priced 'the shipped channel move' as ttnn.permute(zc, (0,3,1,2)) and read 63.3 GB/s. The
production path at 512 aa does not call that op: tt_bio/tenstorrent.py:6492-6495 calls
reblock_permute_gated, our own generic_op kernel, behind eligible_gated. This script folds
cdk2x2_512 once through the production predict path and COUNTS, at run time, which move every
trimul call takes and at which shape -- firing is counted, never read off a gate
(eligibility-firing-condition-is-not-a-code-fact).

No timing here: a fold with per-call syncs is not a fold. The shapes and counts this writes are the
input to granule2.py, which times them standalone under a pinned clock.
"""
from __future__ import annotations

import argparse, collections, importlib.util, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)
FIX = REPO / "perf" / "size512" / "fixtures"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--out", default=str(REPO / "perf/tmk_assumptions/census_gated.json"))
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio import reblock_permute as RP
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *x, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), _TB.__file__

    C = dict(gated=collections.Counter(), gated_elig=collections.Counter(),
             fwd=collections.Counter(), fwd_elig=collections.Counter(),
             back=collections.Counter(), permute=collections.Counter(),
             transpose=collections.Counter())

    def shp(t):
        return "x".join(str(int(d)) for d in t.shape)

    o_gated, o_gelig = RP.reblock_permute_gated, RP.eligible_gated
    o_fwd, o_felig = RP.reblock_permute, RP.eligible
    o_back = RP.reblock_permute_back
    o_perm, o_tr = ttnn.permute, ttnn.transpose

    def gated(xw, p_slice, g_slice, slice_c, memory_config=None, device=None, out=None, row_off=0):
        bt = str((memory_config or (out.memory_config() if out is not None
                                    else xw.memory_config())).buffer_type).split(".")[-1]
        C["gated"][f"in={shp(xw)} slice_c={slice_c} out={shp(out) if out is not None else '-'} "
                   f"rowblk={out is not None} bt={bt} dt={str(xw.dtype).split('.')[-1]}"] += 1
        return o_gated(xw, p_slice, g_slice, slice_c, memory_config=memory_config,
                       device=device, out=out, row_off=row_off)

    def gelig(xw, slice_c, memory_config):
        r = o_gelig(xw, slice_c, memory_config)
        C["gated_elig"][f"in={shp(xw)} slice_c={slice_c} -> {r}"] += 1
        return r

    def fwd(x, memory_config=None, device=None):
        C["fwd"][f"in={shp(x)}"] += 1
        return o_fwd(x, memory_config=memory_config, device=device)

    def felig(x, memory_config):
        r = o_felig(x, memory_config)
        C["fwd_elig"][f"in={shp(x)} -> {r}"] += 1
        return r

    def back(x, memory_config=None, device=None):
        C["back"][f"in={shp(x)}"] += 1
        return o_back(x, memory_config=memory_config, device=device)

    def perm(t, dims, *rest, **kw):
        C["permute"][f"in={shp(t)} dims={tuple(int(d) for d in dims)}"] += 1
        return o_perm(t, dims, *rest, **kw)

    def tr(t, d0, d1, *rest, **kw):
        C["transpose"][f"in={shp(t)} dims=({d0},{d1})"] += 1
        return o_tr(t, d0, d1, *rest, **kw)

    RP.reblock_permute_gated = gated; RP.eligible_gated = gelig
    RP.reblock_permute = fwd; RP.eligible = felig
    RP.reblock_permute_back = back
    ttnn.permute = perm; ttnn.transpose = tr

    dev = get_device()
    work = Path(tempfile.mkdtemp(prefix="tmk-census-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{a.fixture}.yaml", (FIX / f"{a.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("tmk-assumptions-census", cfg)

    clk = Path("/sys/class/tenstorrent/tenstorrent!%s/tt_aiclk"
               % os.environ.get("TT_BIO_LEASE_CARDS", "3"))
    t0 = time.perf_counter()
    metrics, _b, _f = state.predict_one(FIX / f"{a.fixture}.yaml", cfg)
    ttnn.synchronize_device(dev)
    wall = time.perf_counter() - t0

    res = {"host": socket.gethostname(), "fixture": a.fixture, "fold_s": round(wall, 3),
           "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
           "tt_visible": os.environ.get("TT_VISIBLE_DEVICES"),
           "aiclk_after": clk.read_text().strip() if clk.exists() else None,
           "stats_gated_counter": int(RP.STATS_GATED[0]),
           "counts": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:40]) for k, v in C.items()},
           "totals": {k: sum(v.values()) for k, v in C.items()}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("fold_s", "totals", "aiclk_after")}, indent=1))
    for k in ("gated", "gated_elig", "fwd", "fwd_elig", "back"):
        for kk, vv in res["counts"][k].items():
            print(f"{k:11s} {vv:6d}  {kk}")
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
