#!/usr/bin/env python3
"""Per-model, per-size census for the `reblock_permute` default flip, plus the three interaction
checks the landing leg has to see with the kernel ON.

Cut down from `perf/y_permute_crossmodel/crossmodel_ab.py`: the A/A rounds, the A/B rounds and the
block wall are all gone. The flip's value is settled at 209-251 ms/fold on the trimul block wall
(three sessions, byte-identical code) and this host's fold wall cannot resolve it -- the qb1 A/A
floor alone runs to 1480 ms. What is NOT settled is whether the kernel's circular buffers coexist
with main's L1 consumers at every size a model actually folds at, so this script folds and counts.

One model load serves every `--targets` entry: `build_fold` binds the model, not the target, so
later sizes go straight through `state.predict_one`. That saves a model load per size and, more
importantly, avoids the device open/close churn that has wedged cards on this fleet.

Per (model, target) one JSON row: pair shapes seen with counts, calls eligible, calls SERVED,
rejects by reason, trimul invocations, plDDT, the CIF sha, and the four acceptance checks --
`_L1_OUT_REFUSED`, the `_l1_layer_norm` L1/DRAM tally, `_transpose_memory_config`'s branch at the
fold's own pair shape, and any exception raised inside `TriangleMultiplication.__call__`.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=$WT \
      python3 perf/z_flip_land/census_sweep.py --model openfold3 \
        --targets examples/prot.yaml,perf/size512/fixtures/cdk2x2_298.yaml \
        --out perf/z_flip_land/sweep_openfold3_qb2.json
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys, time, traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

import numpy as np
import torch
import ttnn

OUT = Path(__file__).resolve().parent


def load():
    return [round(v, 2) for v in os.getloadavg()]


def a3m_for(target: Path) -> Path | None:
    """The sibling alignment, or the two committed ones by residue count, or None (the msa_dir
    already carries it and seeding would be a no-op)."""
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", required=True, help="comma-separated yaml paths, repo-relative")
    ap.add_argument("--out", default="")
    ap.add_argument("--control", action="store_true",
                    help="fold every target a second time with the kernel OFF and compare the CIF "
                         "sha; the flip is bit-exact, so anything but identical is a finding")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    out_path = Path(a.out) if a.out else OUT / f"sweep_{a.model}.json"
    targets = [REPO / t.strip() for t in a.targets.split(",") if t.strip()]
    for t in targets:
        assert t.exists(), f"missing target {t}"

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T

    # qb2 is two dual-chip p300 boards, so a bare single-chip open fails with "has 1 chips, but
    # expected 2 chips for board type p300". `release_gate.py` does exactly this; a hand-written
    # harness has to as well.
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    assert Path(T.__file__).resolve().is_relative_to(REPO), (
        f"tt_bio loaded from {T.__file__}, not from {REPO} -- set PYTHONPATH")

    msa_dir = Path.home() / "ypx_msa"

    # ---- instrumentation, installed before the model is built ------------------------------------
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

    # `build_fold`'s cfg dict was written for protenix-v2 and carries no Boltz-2 hyperparameters, so
    # `_WorkerState.load_model` raises KeyError('conf_kwargs') for boltz2. Inject exactly what
    # `tt_bio.main` builds; the patch lives in this process and production is untouched.
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

    if a.fast:
        from tt_bio import worker as _W2
        _prev_bind, _prev_lm = _W2._WorkerState.bind_run, _W2._WorkerState.load_model
        _W2._WorkerState.bind_run = lambda s, r, c: (c.__setitem__("fast", True),
                                                     _prev_bind(s, r, c))[1]
        _W2._WorkerState.load_model = lambda s, c: (c.__setitem__("fast", True),
                                                    _prev_lm(s, c))[1]

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
    R = {"wheel": md.version("ttnn"), "host": os.uname().nodename,
         "card": os.environ.get("TT_VISIBLE_DEVICES"), "model": a.model,
         "reblock_permute_constant": RP.REBLOCK_PERMUTE, "enabled_at_import": RP._ENABLED,
         "grid": [g.x, g.y], "cores": g.x * g.y,
         "compute_grid_main": list(T.COMPUTE_GRID_MAIN),
         "l1_max_seq": T._trimul_l1_max_seq(),
         "meta": {k: meta.get(k) for k in ("hardware", "card_type", "aiclk_mhz", "load_s")},
         "rows": []}
    print("head:", json.dumps({k: R[k] for k in ("wheel", "host", "card", "model", "grid",
                                                 "reblock_permute_constant", "enabled_at_import",
                                                 "l1_max_seq")}), flush=True)

    def cif_sha():
        return sorted((p.name, hashlib.sha256(p.read_bytes()).hexdigest()[:16])
                      for p in struct_dir.glob("*") if p.is_file())

    def transpose_branch(n, c):
        """`_transpose_memory_config`'s branch at this fold's own pair shape. Above
        `_trimul_l1_max_seq()` L1 is not the expected answer, so the caller records n/a."""
        t = ttnn.from_torch(torch.zeros(n, n, c, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b = str(T._transpose_memory_config(t).buffer_type)
        ttnn.deallocate(t)
        return b

    def fold(target, a3m, on):
        SEEN.clear(); TM_CALLS.clear(); throws.clear(); RP.REJECTS.clear()
        RP.STATS[0] = RP.STATS[1] = 0
        norm["l1"] = norm["dram"] = 0
        T._L1_OUT_REFUSED.clear()
        RP.set_enabled(on)
        if a3m is not None:
            seed_msa_cache(target, a3m, msa_dir)
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        err = None
        metrics = {}
        try:
            metrics, _b, _f = state.predict_one(target, job_cfg)
        except Exception as e:                                         # noqa: BLE001
            err = repr(e)[:600]
            traceback.print_exc()
        wall = time.perf_counter() - t0
        RP.set_enabled(False)
        # TM_CALLS keys are the full [1, N, N, C], so N is index 1 -- index 0 is the batch dim and
        # probing the transpose at N=1 would answer L1 for every fold, which is not a check.
        pair_n = max((k[1] for k in TM_CALLS), default=0)
        pair_c = max((k[3] for k in TM_CALLS if k[1] == pair_n), default=0)
        return {
            "target": str(target.relative_to(REPO)), "kernel_on": on, "rc_ok": err is None,
            "error": err, "wall_s": round(wall, 2),
            "n_tokens": metrics.get("n_tokens"), "n_residues": metrics.get("n_residues"),
            "plddt": metrics.get("plddt"),
            "cif_sha": cif_sha(),
            "channel_move_calls": sum(SEEN.values()),
            "calls_served": RP.STATS[0], "calls_refused": RP.STATS[1],
            "by_shape": [{"N": k[0], "C": k[1], "in": k[2], "out": k[3], "dtype": k[4],
                          "layout": k[5], "eligible": k[6], "calls": v}
                         for k, v in sorted(SEEN.items())],
            "reject_reasons": {f"{k[0]}:{list(k[1])}": v for k, v in RP.REJECTS.items()},
            "trimul_invocations": {"x".join(str(d) for d in k): v for k, v in sorted(TM_CALLS.items())},
            "trimul_invocations_total": sum(TM_CALLS.values()),
            "trimul_throws": list(throws),
            "l1_out_refused_n": len(T._L1_OUT_REFUSED),
            "l1_out_refused": sorted(str(k)[:160] for k in T._L1_OUT_REFUSED)[:6],
            "l1_layer_norm": dict(norm),
            "pair_shape": [pair_n, pair_c],
            "transpose_memory_config": (transpose_branch(pair_n, pair_c) if pair_n else None),
            "transpose_l1_expected": bool(pair_n and pair_n <= T._trimul_l1_max_seq()),
            "load": load(),
        }

    for target in targets:
        a3m = a3m_for(target)
        row = fold(target, a3m, True)
        R["rows"].append(row)
        print("ON :", json.dumps({k: row[k] for k in (
            "target", "rc_ok", "wall_s", "n_tokens", "plddt", "channel_move_calls", "calls_served",
            "calls_refused", "pair_shape", "l1_out_refused_n", "l1_layer_norm",
            "transpose_memory_config", "trimul_throws", "error")}), flush=True)
        print("     shapes:", json.dumps(row["by_shape"]), flush=True)
        out_path.write_text(json.dumps(R, indent=1))
        if a.control or not row["rc_ok"]:
            ctl = fold(target, a3m, False)
            ctl["control_for"] = row["target"]
            ctl["cif_sha_identical_to_on"] = (ctl["cif_sha"] == row["cif_sha"])
            # The program serial in a TT_THROW is a per-session counter and differs between arms
            # whenever the program cache does, so comparing it makes two identical failures look
            # different. What identifies the failure is the throwing core range and the two L1
            # addresses, all of which survive this substitution.
            def _sig(e):
                return re.sub(r"program \d+", "program N", (e or ""))[:300]

            ctl["same_failure_signature"] = _sig(ctl["error"]) == _sig(row["error"])
            R["rows"].append(ctl)
            print("OFF:", json.dumps({k: ctl[k] for k in (
                "target", "rc_ok", "wall_s", "plddt", "calls_served", "cif_sha_identical_to_on",
                "same_failure_signature", "error")}), flush=True)
            out_path.write_text(json.dumps(R, indent=1))

    RP.eligible = _orig_elig
    out_path.write_text(json.dumps(R, indent=1))
    print("wrote", out_path, flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
