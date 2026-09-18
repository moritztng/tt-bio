#!/usr/bin/env python3
"""Which default-off levers actually FIRE in a Boltz-2 fold, at 298 aa and at 512 aa.

A firing census, not a timing read. It answers the question that decides whether a
measured-but-unshipped lever deserves a clean board pair: does the fold ever execute the site
the flag switches, at the shape the flag needs? Several op-level wins in the campaign record
were measured at shapes a Boltz-2 fold on this part never runs, and
`eligibility-firing-condition-is-not-a-code-fact` says the firing condition has to be executed,
not read off the source.

Admissible on a CONTENDED chip. The shared resource on a p300c board pair is the board power
budget, which moves the AICLK and therefore the seconds. It does not move a route pick or a
call count, and DRAM is per chip. Nothing here is a wall-clock claim.

Arms flip IN PROCESS with the affected lru_caches cleared, one device open for the whole run.
`tenstorrent.py:7059` documents that pattern for `_TRIATT_FUSED_HIFI`; the chunk-pick flags need
their caches dropped as well because the picks are memoised per shape.

TT_BIO_APB_CONCAT_HEADS is NOT flippable in process: it re-lanes proj_g/proj_o at construction
(tenstorrent.py:7756), so a post-load flip would run padded lanes against unpadded weights and
produce a wrong answer rather than a slow one. Run it as its own process with the env var set and
--arms none; its "base" row is then the APB-on arm.

Usage (contended chip is fine):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    python3 perf/c14_land/firing_census.py --sizes 298,512 --out perf/c14_land/firing.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1, default=str))


def _load_helpers():
    """Reuse b2x-flag-levers cfg builder rather than re-deriving the 40-line Boltz-2 config."""
    p = REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py"
    spec = importlib.util.spec_from_file_location("_b2x_flaglev", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def snapshot(TT, trimul_tail, triatt_sdpa, block_pairwise_calls=None) -> dict:
    """Every firing counter the candidates touch. Copied, so the next arm cannot mutate it."""
    def g(mod, name):
        v = getattr(mod, name, None)
        if v is None:
            return None
        if isinstance(v, dict):
            return {str(k): (list(x) if isinstance(x, list) else x) for k, x in v.items()}
        return list(v) if isinstance(v, (list, tuple)) else v
    return {
        "sdpa_chunk_picks": g(TT, "SDPA_CHUNK_PICKS"),
        "sdpa_route_counts": g(TT, "SDPA_ROUTE_COUNTS"),
        "apb_concat_heads": g(TT, "APB_CONCAT_HEADS_STATS"),
        "sdpa_k_chunk": g(TT, "SDPA_K_CHUNK_STATS"),
        "sdpa_fused_large_s": g(TT, "SDPA_FUSED_LARGE_S_STATS"),
        "sdpa_ragged_pad": g(TT, "SDPA_RAGGED_PAD_STATS"),
        "sdpa_ragged_sites": g(TT, "SDPA_RAGGED_SITES"),
        "triatt_fused_hifi": g(TT, "TRIATT_FUSED_HIFI_STATS"),
        "trimul_mask_l1": g(TT, "TRIMUL_MASK_L1_STATS"),
        "trimul_gout": g(TT, "TRIMUL_GOUT_STATS"),
        "opm_small_depth": g(TT, "OPM_SMALL_DEPTH_STATS"),
        "atom_axis_bucket": g(TT, "ATOM_AXIS_BUCKET_STATS"),
        "token_dit_sdpa": g(TT, "B2_TOKEN_DIT_SDPA_STATS"),
        "trimul_tail": g(trimul_tail, "STATS"),
        "trimul_tail_out_l1": g(trimul_tail, "OUT_L1_STATS"),
        "trimul_tail_rejects": g(trimul_tail, "REJECTS"),
        "triatt_gate": g(triatt_sdpa, "GATE_STATS"),
        "block_pairwise_calls": list(block_pairwise_calls)
        if block_pairwise_calls is not None else None,
    }


def reset_counters(TT, trimul_tail, triatt_sdpa):
    """Zero every counter so an arm reports its OWN firing, not the run-to-date total."""
    for mod, name in (
        (TT, "APB_CONCAT_HEADS_STATS"), (TT, "SDPA_K_CHUNK_STATS"),
        (TT, "SDPA_FUSED_LARGE_S_STATS"), (TT, "SDPA_RAGGED_PAD_STATS"),
        (TT, "TRIMUL_MASK_L1_STATS"), (TT, "TRIMUL_GOUT_STATS"),
        (TT, "OPM_SMALL_DEPTH_STATS"), (TT, "ATOM_AXIS_BUCKET_STATS"),
        (TT, "B2_TOKEN_DIT_SDPA_STATS"), (trimul_tail, "STATS"),
        (trimul_tail, "OUT_L1_STATS"), (triatt_sdpa, "GATE_STATS"),
    ):
        v = getattr(mod, name, None)
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = 0
    for mod, name in ((TT, "SDPA_CHUNK_PICKS"), (TT, "SDPA_ROUTE_COUNTS"),
                      (TT, "SDPA_RAGGED_SITES"), (TT, "TRIATT_FUSED_HIFI_STATS"),
                      (trimul_tail, "REJECTS")):
        v = getattr(mod, name, None)
        if isinstance(v, dict):
            if name in ("SDPA_ROUTE_COUNTS", "TRIATT_FUSED_HIFI_STATS"):
                for k in list(v):
                    v[k] = 0            # keep the keys: a route that stops firing must read 0
            else:
                v.clear()


def clear_pick_caches(TT):
    """Chunk picks are memoised per shape, so a flag flip is invisible until they are dropped."""
    for name in ("_sdpa_chunks_shipped", "_dividing_sdpa_chunk_size", "_capped_sdpa_chunk_size",
                 "_tri_att_q_chunks", "_tri_att_k_chunks", "_tri_att_sdpa_program_config",
                 "_padded_sdpa_len"):
        fn = getattr(TT, name, None)
        cc = getattr(fn, "cache_clear", None)
        if cc:
            cc()


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="298,512")
    ap.add_argument("--steps", type=int, default=25,
                    help="sampling steps. FIRING is binary, so the census does not need the "
                         "production 200; route COUNTS here are therefore not production counts "
                         "and are read for zero-vs-nonzero only.")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--arms", default="band_div_k,narrow_q,hifi_min_s,fused_hifi,bias_b8,"
                                     "trimul_cz128,host_block_pairwise")
    ap.add_argument("--label", default="base")
    args = ap.parse_args()
    OUT_PATH = args.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio import trimul_tail, triatt_sdpa
    from tt_bio import boltz2 as _B2
    # TT_BIO_HOST_BLOCK_PAIRWISE has no counter in production, so without this the census
    # would report it as "does not fire" on an instrument blind to it. Wrap the predicate:
    # BLOCK_PAIRWISE_CALLS = [taken, not taken] at boltz2.py:1569.
    BLOCK_PAIRWISE_CALLS = [0, 0]
    _orig_block_pairwise = _B2._block_pairwise

    def _counted_block_pairwise():
        r = _orig_block_pairwise()
        BLOCK_PAIRWISE_CALLS[0 if r else 1] += 1
        return r

    _B2._block_pairwise = _counted_block_pairwise
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree "
        "(memory parity-gate-scores-installed-package-not-checkout)")

    H = _load_helpers()
    dev = get_device()
    try:
        g = dev.compute_with_storage_grid_size()
        grid = [g.x, g.y]
    except Exception:
        grid = None

    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "grid": grid, "torch": torch.__version__,
        "tt_bio_file": _TB.__file__,
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "steps": args.steps, "recycles": args.recycles,
        "note": "FIRING census. Counters and route picks only -- no wall-clock claim, so a "
                "contended board-pair sibling does not invalidate it.",
        "flag_defaults_as_imported": {
            "APB_CONCAT_HEADS": getattr(TT, "_APB_CONCAT_HEADS", None),
            "SDPA_BAND_DIV_K": getattr(TT, "_SDPA_BAND_DIV_K", None),
            "SDPA_NARROW_Q_FALLBACK": getattr(TT, "_SDPA_NARROW_Q_FALLBACK", None),
            "TRIATT_HIFI_MIN_S_PADDED": getattr(TT, "_TRIATT_HIFI_MIN_S_PADDED", None),
            "TRIATT_FUSED_HIFI": getattr(TT, "_TRIATT_FUSED_HIFI", None),
            "TRIATT_BIAS_B8": getattr(TT, "_TRIATT_BIAS_B8", None),
            "SDPA_WIDE_Q": getattr(TT, "_SDPA_WIDE_Q", None),
            "SDPA_DIV_K": getattr(TT, "_SDPA_DIV_K", None),
            "_env_APB_CONCAT_HEADS": os.environ.get("TT_BIO_APB_CONCAT_HEADS"),
        },
    }
    dump()

    # ---- arm setters. Each returns a restore callable. ----------------------
    def _tt_flag(attr):
        def setter():
            prev = getattr(TT, attr)
            setattr(TT, attr, True)
            clear_pick_caches(TT)
            def restore():
                setattr(TT, attr, prev)
                clear_pick_caches(TT)
            return restore
        return setter

    def _cz128():
        prev = trimul_tail.set_f1_cz128(True)
        return lambda: trimul_tail.set_f1_cz128(prev)

    def _host_block():
        prev = os.environ.get("TT_BIO_HOST_BLOCK_PAIRWISE")
        os.environ["TT_BIO_HOST_BLOCK_PAIRWISE"] = "1"      # read per call, boltz2.py:1168
        def restore():
            if prev is None:
                os.environ.pop("TT_BIO_HOST_BLOCK_PAIRWISE", None)
            else:
                os.environ["TT_BIO_HOST_BLOCK_PAIRWISE"] = prev
        return restore

    ARMS = {
        "band_div_k": _tt_flag("_SDPA_BAND_DIV_K"),
        "narrow_q": _tt_flag("_SDPA_NARROW_Q_FALLBACK"),
        "hifi_min_s": _tt_flag("_TRIATT_HIFI_MIN_S_PADDED"),
        "fused_hifi": _tt_flag("_TRIATT_FUSED_HIFI"),
        "bias_b8": _tt_flag("_TRIATT_BIAS_B8"),
        "trimul_cz128": _cz128,
        "host_block_pairwise": _host_block,
    }

    work = Path(tempfile.mkdtemp(prefix="c14-firing-", dir=str(REPO / "perf" / "c14_land")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    for s in sizes:
        H._seed_msa(FIX / f"cdk2x2_{s}.yaml", (FIX / f"cdk2x2_{s}.a3m").read_text(), msa_dir)

    cfg = H.build_cfg(msa_dir, struct_dir)
    cfg["recycling_steps"] = args.recycles
    cfg["sampling_steps"] = args.steps
    cfg["conf_kwargs"]["predict_args"]["recycling_steps"] = args.recycles
    cfg["conf_kwargs"]["predict_args"]["sampling_steps"] = args.steps
    _ensure_local_artifacts(cfg)

    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c14-land-tail-firing", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    def clk():
        try:
            return TT.aiclk() if hasattr(TT, "aiclk") else None
        except Exception:
            return None

    def fold(size: str, arm: str) -> dict:
        target = FIX / f"cdk2x2_{size}.yaml"
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        reset_counters(TT, trimul_tail, triatt_sdpa)
        BLOCK_PAIRWISE_CALLS[0] = BLOCK_PAIRWISE_CALLS[1] = 0
        state.pfn = None
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        err = None
        metrics = {}
        try:
            metrics, _b, _f = state.predict_one(target, cfg)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        ttnn.synchronize_device(dev)
        wall = round(time.perf_counter() - t0, 3)
        cifs = sorted(struct_dir.glob("*.cif"))
        row = {
            "size": size, "arm": arm, "error": err,
            # Kept for a sanity read only. This chip shares a board power budget with a busy
            # sibling, so these seconds are NOT a measurement and must not be quoted as one.
            "wall_s_NOT_A_MEASUREMENT": wall,
            "plddt": (metrics or {}).get("complex_plddt", (metrics or {}).get("plddt")),
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest() if cifs else None,
            "counters": snapshot(TT, trimul_tail, triatt_sdpa, BLOCK_PAIRWISE_CALLS),
        }
        return row

    rows = []
    arms = ["base"] + [a.strip() for a in args.arms.split(",") if a.strip() and a.strip() != "none"]
    for size in sizes:
        # one discarded warm fold per size: program cache, and it also proves the size runs at all
        w = fold(size, "warmup")
        w["warmup"] = True
        rows.append(w); OUT["rows"] = rows; dump()
        print(f"[{size}] warmup err={w['error']} cif={str(w['cif_sha256'])[:12]}", flush=True)
        for arm in arms:
            restore = None
            if arm != "base":
                restore = ARMS[arm]()
            try:
                r = fold(size, arm)
            finally:
                if restore:
                    restore()
            r["warmup"] = False
            rows.append(r); OUT["rows"] = rows; dump()
            picks = r["counters"].get("sdpa_chunk_picks") or {}
            print(f"[{size}] {arm:20s} err={r['error']} plddt={r['plddt']} "
                  f"picks={len(picks)} cif={str(r['cif_sha256'])[:12]}", flush=True)

    # ---- firing verdict: an arm fires iff some counter or pick differs from base -------------
    verdict = {}
    for size in sizes:
        base = next((r for r in rows if r["size"] == size and r["arm"] == "base"), None)
        if not base:
            continue
        for r in rows:
            if r["size"] != size or r["arm"] in ("base", "warmup"):
                continue
            diffs = {}
            for k, v in r["counters"].items():
                bv = base["counters"].get(k)
                if v != bv:
                    diffs[k] = {"base": bv, "arm": v}
            verdict[f"{size}/{r['arm']}"] = {
                "fires": bool(diffs),
                "error": r["error"],
                "cif_changed": r["cif_sha256"] != base["cif_sha256"],
                "plddt_base": base["plddt"], "plddt_arm": r["plddt"],
                "diffs": diffs,
            }
    OUT["firing_verdict"] = verdict
    OUT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    shutil.rmtree(work, ignore_errors=True)
    for k, v in verdict.items():
        print(f"VERDICT {k:34s} fires={v['fires']!s:5s} cif_changed={v['cif_changed']!s:5s} "
              f"err={v['error']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
