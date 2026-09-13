#!/usr/bin/env python3
"""`TT_BIO_SDPA_GRID_Q_CHUNK` on Blackhole, isolated: does the lever have one sign, or one per site?

Three phases in ONE process on ONE device, each written to --out as it finishes.

  census   one fold per arm, discarded for timing, with `SDPA_GRID_Q_CHUNK_PICKS` snapshotted
           after each. That counter sits outside the rule's lru_cache, so it counts CALLS, and
           it records the shipped chunk beside the chosen one: a site where they are equal is a
           site the rule DECLINED to move, which is the reading that has been mistaken for a win.
           The previous row could not read this counter at all -- it censused the CLI parent while
           the fold ran in spawned device workers. This harness drives `predict_one` in-process.

  ops      an isolated A/B at every shape the census saw, shipped chunk against the rule's chunk,
           interleaved, `torch.equal` on the outputs. This is also the arm-switching CONTROL: if
           the two chunks do not produce measurably different programs here, the in-process fold
           arms below are sharing a compiled program and the fold A/B would be meaningless.

  fold     paired interleaved A/B, `base ship ship base` per rep so the order reverses inside the
           rep, cold folds discarded, loadavg per fold, CIF sha256 per fold. The A/A floor comes
           from THIS session's same-arm adjacent pairs, never from another row's draws.

The arms are set in-process by assigning the module constant the rule reads per call; nothing is
exported, so neither arm can inherit the other's environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- the cell's fixtures, cfg and MSA seeding, unmodified

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def census_rows(T):
    return [{"site": k[0], "q_len": k[1], "k_len": k[2], "work": k[3], "d": k[4],
             "calls": v[0], "shipped_chunk": v[1], "chunk": v[2], "units": v[3],
             "moved": v[2] != v[1]}
            for k, v in sorted(T.SDPA_GRID_Q_CHUNK_PICKS.items())]


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--op-reps", type=int, default=30)
    ap.add_argument("--size", default="512")
    ap.add_argument("--skip-fold", action="store_true")
    args = ap.parse_args()
    OUT_PATH = args.out

    assert "TT_BIO_SDPA_GRID_Q_CHUNK" not in os.environ, \
        "the lever may not be pinned in the environment; the arms are set in-process"
    assert os.environ.get("TT_BIO_ATOM_L1") in (None, "0"), \
        "this row measures main's atom path: TT_BIO_ATOM_L1 must be off"

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    assert not T._SDPA_GRID_Q_CHUNK, "the flag's default must be off"

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
    OUT.update({"doc": __doc__, "env": {
        "host": socket.gethostname(), "arch": str(dev.arch()),
        "storage_grid": [g.x, g.y], "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
        "cores": cores, "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "torch": torch.__version__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": os.getloadavg(),
        "protocol": {"recycling_steps": AB.RECYCLING_STEPS, "steps": AB.SAMPLING_STEPS,
                     "seed": AB.SEED, "size_aa": args.size, "atom_l1": False},
    }, "census": {}, "ops": [], "folds": []})
    dump()

    def arm(name):
        T._SDPA_GRID_Q_CHUNK = (name == "ship")
        T._grid_q_chunk.cache_clear()

    # ---------------------------------------------------------------- the model and the fixture
    work = Path(tempfile.mkdtemp(prefix="b2z2-qchunk-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    name = f"cdk2x2_{args.size}"
    AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-qchunk-isolated-bh", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()
    target = AB.FIX / f"{name}.yaml"

    def fold(a, keep, tag):
        arm(a)
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        la = os.getloadavg()[0]
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        r = {"arm": a, "tag": tag, "fold_s": round(wall, 3),
             "sha256": hashlib.sha256(body).hexdigest(),
             "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6),
             "loadavg_1m": round(la, 2)}
        OUT["folds"].append(r); dump()
        print(f"  {tag:12s} {a:5s} {r['fold_s']:7.3f}s load {la:5.2f} sha={r['sha256'][:12]}",
              flush=True)
        return r

    # ------------------------------------------------------------------------------ 1. census
    for a in ("base", "ship"):
        T.SDPA_GRID_Q_CHUNK_PICKS.clear()
        fold(a, args.cifdir / f"cold_{a}", "cold")
        OUT["census"][a] = census_rows(T)
        dump()
    for row in OUT["census"]["ship"]:
        print(f"   census {row['site']:16s} q{row['q_len']:5d} k{row['k_len']:5d} "
              f"work {row['work']:5d} calls {row['calls']:6d} {row['shipped_chunk']:4d}"
              f" -> {row['chunk']:4d} units {row['units']:5d}/{cores} "
              f"{'MOVED' if row['moved'] else 'declined'}", flush=True)
    OUT["census_summary"] = {
        a: {"calls": sum(r["calls"] for r in rows),
            "calls_moved": sum(r["calls"] for r in rows if r["moved"]),
            "sites_moved": sorted({r["site"] for r in rows if r["moved"]}),
            "sites_declined": sorted({r["site"] for r in rows if not r["moved"]})}
        for a, rows in OUT["census"].items()}
    dump()

    # --------------------------------------------------------- 2. the op A/B at every shape seen
    def timed(fn, reps):
        fn(); ttnn.synchronize_device(dev)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); o = fn(); ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t); ttnn.deallocate(o)
        return ts

    for row in OUT["census"]["ship"]:
        q_len, k_len, w = row["q_len"], row["k_len"], row["work"]
        heads = w if w else 1
        try:
            tq = torch.randn(1, heads, q_len, row["d"], dtype=torch.bfloat16)
            tk = torch.randn(1, heads, k_len, row["d"], dtype=torch.bfloat16)
            tv = torch.randn(1, heads, k_len, row["d"], dtype=torch.bfloat16)
            tb = torch.randn(1, heads, q_len, k_len, dtype=torch.bfloat16)
            mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q, k, v, b = mk(tq), mk(tk), mk(tv), mk(tb)
            outs, times = {}, {}
            for a, chunk in (("base", row["shipped_chunk"]), ("ship", row["chunk"])):
                cfgp = T._sdpa_program_config(q_chunk_size=chunk,
                                              k_chunk_size=T._capped_sdpa_chunk_size(k_len))
                f = lambda: ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=b, is_causal=False, scale=row["d"] ** -0.5, program_config=cfgp)
                times[a] = timed(f, args.op_reps)
                o = f(); ttnn.synchronize_device(dev)
                outs[a] = ttnn.to_torch(o); ttnn.deallocate(o)
            eq = bool(torch.equal(outs["base"], outs["ship"]))
            mx = float((outs["base"].float() - outs["ship"].float()).abs().max())
            med = {a: st.median(times[a]) for a in times}
            rec = {"site": row["site"], "q_len": q_len, "k_len": k_len, "work": w,
                   "heads": heads, "calls_in_fold": row["calls"],
                   "shipped_chunk": row["shipped_chunk"], "chunk": row["chunk"],
                   "units": row["units"], "cores": cores, "moved": row["moved"],
                   "base_us": round(med["base"] * 1e6, 2), "ship_us": round(med["ship"] * 1e6, 2),
                   "ratio": round(med["base"] / med["ship"], 5),
                   "base_spread_pct": round(100 * (max(times["base"]) - min(times["base"]))
                                            / med["base"], 2),
                   "ship_spread_pct": round(100 * (max(times["ship"]) - min(times["ship"]))
                                            / med["ship"], 2),
                   "torch_equal": eq, "max_abs": mx,
                   "fold_s_delta": round(row["calls"] * (med["base"] - med["ship"]), 4)}
            OUT["ops"].append(rec); dump()
            print(f"   op {rec['site']:16s} q{q_len:5d} w{w:4d} {rec['shipped_chunk']:4d}->"
                  f"{rec['chunk']:4d} {rec['base_us']:8.1f} -> {rec['ship_us']:8.1f} us "
                  f"{rec['ratio']:7.4f}x equal={eq} fold_delta {rec['fold_s_delta']:+.3f}s",
                  flush=True)
            for t in (q, k, v, b):
                ttnn.deallocate(t)
        except Exception as exc:  # noqa: BLE001 -- a shape we cannot rebuild is reported, not fatal
            OUT["ops"].append({"site": row["site"], "q_len": q_len, "work": w,
                               "error": f"{type(exc).__name__}: {exc}"[:400]})
            dump()
            print(f"   op {row['site']} q{q_len} w{w} FAILED: {exc}"[:300], flush=True)

    if args.skip_fold:
        return 0

    # ------------------------------------------------------------------- 3. the paired fold A/B
    for i in range(args.reps):
        for j, a in enumerate(("base", "ship", "ship", "base")):
            fold(a, args.cifdir / f"r{i}_{j}_{a}", f"rep{i}.{j}")
    OUT["env"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT["env"]["loadavg_end"] = os.getloadavg()
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
