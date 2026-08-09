#!/usr/bin/env python3
"""Trace every L1-residency guard decision during a real run, and prove it does not OOM.

The merged whole-tensor guard (cd4b71e67) and the row block on this branch were both
tuned on standalone modules. This runs the real thing -- a 298 aa fold, a ligand
co-fold, a BoltzGen design -- and records, per call site, the shapes the guard saw and
what it decided. A guard that silently never fires at the priority size is as much a
bug as one that fires and blows L1, so both directions are reported.

    TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=... python3 perf/qkv_rootcause/guard_trace.py \
        --model protenix-v2 --target examples/prot300.yaml --out results/guard_prot300.json
"""
import argparse, collections, json, os, sys, tempfile, time, traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--target", required=True)
ap.add_argument("--a3m", default=None, help="seed the MSA cache instead of hitting the server")
ap.add_argument("--recycling", type=int, default=10)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--single-sequence", action="store_true")
ap.add_argument("--design", action="store_true", help="BoltzGen design instead of a fold")
ap.add_argument("--design-steps", type=int, default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()


def _site():
    """Nearest tt_bio/tenstorrent.py frame above the guard itself."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        if fr.filename.endswith("tenstorrent.py") and not fr.name.startswith("_l1_resident"):
            return f"{fr.name}:{fr.lineno}"
    return "?"


def main():
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name
    import tt_bio.tenstorrent as T
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    noop = lambda *a, **k: None
    _E.set_progress(noop)
    get_device()

    # ---- instrument both guards ---------------------------------------------------
    # Trace the OUTER entry points, not just the inner budget check: the shape/dtype
    # preconditions in _l1_resident_linear_config return before the budget is ever
    # consulted, so tracing only the inner function reports "never called" for a path
    # that is in fact being rejected on dtype.
    mm = collections.Counter()      # (site, m,k,n, dtype, full_k, admitted) -> count
    rb = collections.Counter()      # (site, rows, cols, k, n, chosen) -> count
    real_cfg = T._l1_resident_linear_config
    real_rb = T._tri_att_row_block

    def _shape(x, w):
        try:
            xs, ws = list(x.shape), list(w.shape)
            m = 1
            for d in xs[:-1]:
                m *= int(d)
            return m, int(xs[-1]), int(ws[-1])
        except Exception:
            return -1, -1, -1

    def cfg_trace(x, w, dtype, full_k=True):
        c = real_cfg(x, w, dtype, full_k=full_k)
        m, k, n = _shape(x, w)
        mm[(_site(), m, k, n, str(dtype), full_k, c is not None)] += 1
        return c

    def rb_trace(x, w, dtype):
        r = real_rb(x, w, dtype)
        m, k, n = _shape(x, w)
        rows = int(x.shape[0]) if len(x.shape) > 1 else -1
        rb[(_site(), rows, m, k, n, str(dtype), r)] += 1
        return r

    T._l1_resident_linear_config = cfg_trace
    T._tri_att_row_block = rb_trace

    target = Path(args.target)
    work = Path(tempfile.mkdtemp(prefix="guardtrace-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = work / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        model=args.model, fast=False, output_format="cif",
        recycling_steps=args.recycling, sampling_steps=args.steps,
        diffusion_samples=args.samples, seed=0, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=args.single_sequence, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None, msa_server_password=None,
        api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    if args.design:
        cfg["design_samples"] = 1
        if args.design_steps is not None:
            cfg["design_steps"] = args.design_steps
    _ensure_local_artifacts(cfg)

    if args.a3m:
        import hashlib
        from tt_bio.main import _read_bio_chains
        for ch in _read_bio_chains(target):
            seq = ch[1]
            text = Path(args.a3m).read_text()
            if text.split("\n")[1] != seq:
                continue
            (msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(text)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("guardtrace", cfg)
    state.pfn = noop

    t = time.perf_counter()
    err = None
    metrics = None
    try:
        metrics, _best, _feats = state.predict_one(target, dict(cfg))
    except Exception as e:  # an L1 OOM shows up here, and that is the point of the run
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    wall = time.perf_counter() - t

    def dump(counter, keys):
        rows = [dict(zip(keys, k), calls=v) for k, v in sorted(counter.items())]
        return rows

    mm_rows = dump(mm, ("site", "m", "k", "n", "dtype", "full_k", "admitted"))
    rb_rows = dump(rb, ("site", "rows", "m", "k", "n", "dtype", "block"))

    print(f"\n=== {args.model} {target.name}  wall={wall:.2f}s  err={err} ===", flush=True)
    print("--- whole-tensor guard (_l1_resident_linear_config) ---", flush=True)
    for r in mm_rows:
        print(f"  {r['site']:30s} m={r['m']:7d} k={r['k']:4d} n={r['n']:5d} {r['dtype']:22s}"
              f"full_k={str(r['full_k']):5s} -> {'L1' if r['admitted'] else 'fallback':8s} "
              f"x{r['calls']}", flush=True)
    print("--- row block (_tri_att_row_block) ---", flush=True)
    for r in rb_rows:
        print(f"  {r['site']:30s} rows={r['rows']:5d} m={r['m']:7d} k={r['k']:4d} "
              f"n={r['n']:5d} {r['dtype']:22s}-> block={r['block']} x{r['calls']}", flush=True)

    n_l1 = sum(r["calls"] for r in mm_rows if r["admitted"])
    n_fb = sum(r["calls"] for r in mm_rows if not r["admitted"])
    n_blk = sum(r["calls"] for r in rb_rows if r["block"])
    print(f"summary: whole-tensor L1 {n_l1} / fallback {n_fb}; row-blocked {n_blk}", flush=True)

    out = {"model": args.model, "target": target.name, "wall_s": round(wall, 3),
           "error": err, "metrics": metrics, "arch": arch_name(),
           "grid": list(T.COMPUTE_GRID_MAIN),
           "matmul_guard": mm_rows, "row_block": rb_rows,
           "counts": {"l1": n_l1, "fallback": n_fb, "row_blocked": n_blk}}
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(p, "w"), indent=2)
        print("wrote", p, flush=True)
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
