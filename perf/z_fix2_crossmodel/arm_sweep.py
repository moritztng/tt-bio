#!/usr/bin/env python3
"""Four-arm cross-model test of FIX-2 (`5a207fee`) and FIX-D (`142e0109`) at 512 aa.

Cut from `perf/z_flip_land/census_sweep.py` (308 lines) so the p300 single-chip mesh-descriptor
fix, the boltz2 `conf_kwargs` injection, `seed_msa_cache` and the per-target folds off one model
load all come across unchanged rather than being re-derived. Seven edits, listed in
`state/protenix-trunk--z-fix2-crossmodel.md` §5.1:

  1  --arm {A,B,C,D}, recorded in the header and every row
  2  arm fingerprint read off `inspect.getsource`, asserted against --arm
  3  full untruncated traceback per failing fold
  4  predicate fields parsed out of the throw: core range, both addresses, held/bank, over_by
  5  `_BMM_CFG_REFUSED` tally, cleared per fold -- this leg's positive control
  6  a `ttnn.matmul` wrapper that records every raise, because in arm B FIX-2 swallows the throw
     and the wrapper is then the only way to read `held` and the CB end from a PASSING arm
  7  full sha256 (not 16 hex), and `on=False` only -- the reblock_permute kernel stays at its
     default, which is OFF

    TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$WT \
      python3 perf/z_fix2_crossmodel/arm_sweep.py --arm A --model openfold3 \
        --targets perf/size512/fixtures/cdk2x2_298.yaml,perf/size512/fixtures/cdk2x2_512.yaml \
        --out perf/z_fix2_crossmodel/armA_openfold3.json
"""
from __future__ import annotations

import argparse, hashlib, inspect, json, os, re, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np
import torch
import ttnn

OUT = Path(__file__).resolve().parent

# arm -> (fix2 present, fixd present)
ARMS = {"A": (False, False), "B": (True, False), "C": (True, True), "D": (False, True)}

THROW_RE = re.compile(r"core range \[(.*?)\].*?allocated at (\d+).*?ends at (\d+)", re.S)


def load():
    return [round(v, 2) for v in os.getloadavg()]


def a3m_for(target: Path) -> Path | None:
    sib = target.with_suffix(".a3m")
    if sib.exists():
        return sib
    for name in ("prot117.a3m", "prot300.a3m"):
        cand = REPO / "scripts/gpu_vs_tt/fixtures" / name
        if cand.exists():
            try:
                from tt_bio.main import _read_bio_chains
                if cand.read_text().split("\n")[1] == _read_bio_chains(target)[0][1]:
                    return cand
            except Exception:                                          # noqa: BLE001
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True, help="comma-separated yaml paths, repo-relative")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    out_path = Path(a.out) if a.out else OUT / f"arm{a.arm}_{a.model}.json"
    targets = [REPO / t.strip() for t in a.targets.split(",") if t.strip()]
    for t in targets:
        assert t.exists(), f"missing target {t}"

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    assert Path(T.__file__).resolve().is_relative_to(REPO), (
        f"tt_bio loaded from {T.__file__}, not from {REPO} -- set PYTHONPATH")

    # ---- edit 2: arm fingerprint, read off the source of the two patched functions --------------
    fix2_present = "_BMM_CFG_REFUSED" in inspect.getsource(T.batched_matmul)
    fixd_present = ("The normed pair tensor is dead"
                    in inspect.getsource(T.AttentionPairBias.__call__))
    want = ARMS[a.arm]
    print(f"arm {a.arm}: fix2={fix2_present} fixd={fixd_present} "
          f"(want fix2={want[0]} fixd={want[1]})", flush=True)
    if (fix2_present, fixd_present) != want:
        print(f"FINGERPRINT MISMATCH for arm {a.arm} -- working tree does not match the label",
              file=sys.stderr, flush=True)
        return 3

    msa_dir = Path.home() / "ypx_msa"

    # ---- instrumentation, installed before the model is built -----------------------------------
    SEEN: dict = {}
    _orig_elig = RP.eligible

    def _elig(x, mc):
        v = _orig_elig(x, mc)
        key = (int(x.shape[1]), int(x.shape[3]),
               str(x.memory_config().buffer_type).rsplit(".", 1)[-1],
               str(mc.buffer_type).rsplit(".", 1)[-1],
               str(x.dtype).rsplit(".", 1)[-1], str(x.layout).rsplit(".", 1)[-1], bool(v))
        SEEN[key] = SEEN.get(key, 0) + 1
        return v

    RP.eligible = _elig

    norm = {"l1": 0, "dram": 0}
    _orig_l1ln = T._l1_layer_norm

    def _counting_l1ln(x, headroom, **kw):
        t, in_l1 = _orig_l1ln(x, headroom, **kw)
        norm["l1" if in_l1 else "dram"] += 1
        return t, in_l1

    T._l1_layer_norm = _counting_l1ln

    TM_CALLS: dict = {}
    throws: list = []
    _orig_tm = T.TriangleMultiplication.__call__

    def _wrapped_tm(self, x, mask=None):
        s = tuple(int(d) for d in x.shape)
        TM_CALLS[s] = TM_CALLS.get(s, 0) + 1
        try:
            return _orig_tm(self, x, mask)
        except Exception as e:                                         # noqa: BLE001
            throws.append(repr(e)[:400])
            raise

    T.TriangleMultiplication.__call__ = _wrapped_tm

    # ---- edit 6: the ttnn.matmul wrapper -------------------------------------------------------
    # In arm B the clash is caught inside FIX-2 and never reaches the fold's own `except`, so this
    # is the ONLY way to read the throwing core range and the two L1 addresses from an arm that
    # passes. It re-raises unchanged, so it is behaviour-neutral; arm C's plDDT against arm A's at
    # 298 aa is the check that it is.
    MM_THROWS: list = []
    _orig_mm = ttnn.matmul

    def _mm(*args, **kw):
        try:
            return _orig_mm(*args, **kw)
        except Exception as e:                                         # noqa: BLE001
            try:
                sa = [int(d) for d in args[0].shape]
                sb = [int(d) for d in args[1].shape]
                dt = str(args[0].dtype)
            except Exception:                                          # noqa: BLE001
                sa, sb, dt = None, None, None
            MM_THROWS.append({"a": sa, "b": sb, "dtype": dt,
                              "had_program_config": kw.get("program_config") is not None,
                              "err": str(e)})
            raise

    ttnn.matmul = _mm

    if a.model == "boltz2":
        from tt_bio import worker as _W
        from tt_baseline import RECYCLING_STEPS, SAMPLING_STEPS, DIFFUSION_SAMPLES

        _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
                      "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
                      "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
                      "alignment_reverse_diff": True, "synchronize_sigmas": True}
        _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
        _msa = {"subsample_msa": True, "num_subsampled_msa": 1024, "use_paired_feature": True,
                "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
                "pairwise_head_width": 32, "pairwise_num_heads": 4,
                "activation_checkpointing": True}
        _steering = {"fk_steering": False, "physical_guidance_update": False,
                     "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                     "fk_resampling_interval": 3, "num_gd_steps": 20}
        _conf_kwargs = dict(
            predict_args={"recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                          "diffusion_samples": DIFFUSION_SAMPLES, "max_parallel_samples": None},
            diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
            steering_args=_steering, use_kernels=True, use_tenstorrent=True, trace=False,
            diffusion_trace=False)
        _aff_kwargs = dict(
            predict_args={"recycling_steps": 5, "sampling_steps": 200, "diffusion_samples": 5,
                          "max_parallel_samples": 1},
            diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
            steering_args=dict(_steering, contact_guidance_update=False),
            affinity_mw_correction=False, use_tenstorrent=True, trace=False, diffusion_trace=False)
        _orig_load_model = _W._WorkerState.load_model

        def _load_model(self, cfg):
            cfg.setdefault("conf_kwargs", _conf_kwargs)
            cfg.setdefault("aff_kwargs", _aff_kwargs)
            cfg.setdefault("use_potentials", False)
            return _orig_load_model(self, cfg)

        _W._WorkerState.load_model = _load_model

    from tt_baseline import build_fold, seed_msa_cache

    t_load = time.perf_counter()
    one_fold, meta, state = build_fold(a.model, msa_dir, targets[0], a3m_for(targets[0])
                                       or REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    job_cfg = dict(meta["job_cfg"])
    struct_dir = Path(meta["struct_dir"])

    import importlib.metadata as md
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    try:
        bank = int(ttnn.get_max_worker_l1_unreserved_size())
    except Exception as e:                                             # noqa: BLE001
        bank = None
        print("bank probe failed:", e, flush=True)
    R = {"wheel": md.version("ttnn"), "host": os.uname().nodename,
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model, "arm": a.arm,
         "fix2_present": fix2_present, "fixd_present": fixd_present,
         "bank_bytes": bank,
         "reblock_permute_constant": getattr(RP, "REBLOCK_PERMUTE", None),
         "enabled_at_import": RP._ENABLED,
         "grid": [g.x, g.y], "cores": g.x * g.y,
         "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
         "l1_max_seq": T._trimul_l1_max_seq(),
         "meta": {k: meta.get(k) for k in ("hardware", "card_type", "aiclk_mhz", "load_s")},
         "rows": []}
    print("head:", json.dumps({k: R[k] for k in (
        "wheel", "host", "card", "model", "arm", "fix2_present", "fixd_present", "bank_bytes",
        "grid", "enabled_at_import", "l1_max_seq")}), flush=True)

    def cif_sha():
        # edit 7: the FULL digest. 16 hex is a truncation and the brief asks for sha256.
        return sorted((p.name, hashlib.sha256(p.read_bytes()).hexdigest())
                      for p in struct_dir.glob("*") if p.is_file())

    def bmm_refused():
        s = getattr(T, "_BMM_CFG_REFUSED", None)
        if s is None:
            return None, []
        return len(s), sorted(str(k) for k in s)

    def predicate(text):
        """edit 4: read the predicate straight off the throw. `held_per_bank = bank - address`,
        because the reported address is measured from the bottom of the bank and `held` is its
        complement -- adding the two, as the brief does, counts the same region twice."""
        m = THROW_RE.search(text or "")
        if not m:
            return None
        core_range, addr, cb_end = m.group(1), int(m.group(2)), int(m.group(3))
        d = {"core_range": core_range, "l1_buffer_addr": addr, "cb_region_end": cb_end,
             "overlap_bytes": cb_end - addr}
        if bank:
            d["held_per_bank"] = bank - addr
            d["sum_vs_bank"] = (bank - addr) + cb_end
            d["over_by"] = d["sum_vs_bank"] - bank
            d["bank"] = bank
        return d

    def fold(target, a3m, on=False):
        SEEN.clear(); TM_CALLS.clear(); throws.clear(); RP.REJECTS.clear()
        MM_THROWS.clear()
        RP.STATS[0] = RP.STATS[1] = 0
        norm["l1"] = norm["dram"] = 0
        T._L1_OUT_REFUSED.clear()
        s = getattr(T, "_BMM_CFG_REFUSED", None)
        if s is not None:
            s.clear()
        RP.set_enabled(on)
        if a3m is not None:
            seed_msa_cache(target, a3m, msa_dir)
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        err = err_tb = None
        metrics = {}
        try:
            metrics, _b, _f = state.predict_one(target, job_cfg)
        except Exception as e:                                         # noqa: BLE001
            err = repr(e)                                             # edit 3: untruncated
            err_tb = traceback.format_exc()
            traceback.print_exc()
        wall = time.perf_counter() - t0
        RP.set_enabled(False)
        pair_n = max((k[1] for k in TM_CALLS), default=0)
        pair_c = max((k[3] for k in TM_CALLS if k[1] == pair_n), default=0)
        n_ref, keys = bmm_refused()
        clash_mm = [m for m in MM_THROWS if "clash with L1 buffers" in (m["err"] or "")]
        return {
            "arm": a.arm, "target": str(target.relative_to(REPO)), "kernel_on": on,
            "rc_ok": err is None, "error": err, "error_tb": err_tb, "wall_s": round(wall, 2),
            "n_tokens": metrics.get("n_tokens"), "n_residues": metrics.get("n_residues"),
            "plddt": metrics.get("plddt"),
            "cif_sha": cif_sha(),
            # edit 5: the positive control. A fold that passes with this at 0 has not tested FIX-2.
            "bmm_cfg_refused_n": n_ref, "bmm_cfg_refused": keys,
            # edit 6: throws seen inside ttnn.matmul, including ones FIX-2 swallowed
            "mm_throws_n": len(MM_THROWS),
            "mm_clash_n": len(clash_mm),
            "mm_clash": [{k: m[k] for k in ("a", "b", "dtype", "had_program_config", "err")}
                         for m in clash_mm[:4]],
            "predicate_from_mm": (predicate(clash_mm[0]["err"]) if clash_mm else None),
            "predicate_from_err": predicate(err),
            "channel_move_calls": sum(SEEN.values()),
            "calls_served": RP.STATS[0], "calls_refused": RP.STATS[1],
            "trimul_invocations_total": sum(TM_CALLS.values()),
            "trimul_throws": list(throws),
            "l1_out_refused_n": len(T._L1_OUT_REFUSED),
            "l1_out_refused": sorted(str(k)[:160] for k in T._L1_OUT_REFUSED)[:6],
            "l1_layer_norm": dict(norm),
            "pair_shape": [pair_n, pair_c],
            "load": load(),
        }

    for target in targets:
        a3m = a3m_for(target)
        row = fold(target, a3m, on=False)
        R["rows"].append(row)
        print("ROW:", json.dumps({k: row[k] for k in (
            "arm", "target", "rc_ok", "wall_s", "n_tokens", "plddt", "pair_shape",
            "bmm_cfg_refused_n", "mm_clash_n", "l1_out_refused_n", "predicate_from_mm",
            "predicate_from_err", "error")}), flush=True)
        print("     sha:", json.dumps(row["cif_sha"]), flush=True)
        print("     bmm_keys:", json.dumps(row["bmm_cfg_refused"]), flush=True)
        out_path.write_text(json.dumps(R, indent=1))

    RP.eligible = _orig_elig
    ttnn.matmul = _orig_mm
    out_path.write_text(json.dumps(R, indent=1))
    print("wrote", out_path, flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
