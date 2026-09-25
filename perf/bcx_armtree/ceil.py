"""At which drawn binder length does a BindCraft 2 gradient step stop fitting in one card?

`perf/bcx_dram/memtrace.py` asked whether DRAM RISES across steps at one draw. This asks the
other question: hold the number of steps small and vary the DRAW, which is what BindCraft 2
itself varies, 60 to 180 against a 115-residue target.

The draw is pinned through `binder_lengths`, BindCraft 2s own setting, so the campaign still
samples it. Nothing here shortens a step, changes a recycle or skips a block. Only the first
gradient stage of one trajectory runs, and only `--steps` of it: an allocation failure is
visible in the first few steps, or it is an accumulation, and the two are told apart by whether
the resident line rises.

What the allocator sees is NOT the drawn length. `bindcraft/af2.py` pads the design chain up to
`length_bucket_size` (`pad_design_chains`), adds the target, and `_predict_complex` pads that
total again; then `splice._pad_inputs` pads the token axis up to a multiple of 32 for tt-bio.
The last of those is the number the DRAM buffers are shaped by, so it is read off the real call
here rather than recomputed.
"""
import argparse, json, os, pathlib, resource, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_predictor")):
    sys.path.insert(0, p)

import ttnn                                                             # noqa: E402
import jax                                                              # noqa: E402
import afgrad as A, stack as S                                          # noqa: E402
import bc2_state as B                                                   # noqa: E402
import splice as SP                                                     # noqa: E402
import ttbio_predictor as T                                             # noqa: E402
from tt_bio import autograd as ag                                       # noqa: E402

from bindcraft.af2 import MONOMER_POOL, campaign_length_bucket, padded_prediction_length  # noqa: E402
from bindcraft.design_workers import campaign_subbatch_size             # noqa: E402
from bindcraft.protein_preparation import (design_residue_count,        # noqa: E402
                                           initialize_design_trajectory,
                                           sampled_trajectory_values)
from bindcraft.settings import (build_design_settings,                   # noqa: E402
                                resolve_cyclic_offset_mode,
                                select_design_and_validation_models)
from bindcraft.target_schedule import build_design_schedule, build_target_schedule  # noqa: E402
from bindcraft.trajectory import (build_stage_plan, induced_fit_reference_arguments,  # noqa: E402
                                  run_gradient_design_stage, run_stage_operations,
                                  TrajectoryState)

GB = float(2 ** 30)


def dram_used_free(dev):
    mv = ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)
    banks = int(mv.num_banks)
    return (int(mv.total_bytes_allocated_per_bank) * banks,
            int(mv.total_bytes_free_per_bank) * banks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binder-length", type=int, required=True,
                    help="the drawn binder length to pin. BindCraft 2 draws 60..180.")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--trajectory", type=int, default=1)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "3")))
    ap.add_argument("--node-peak", action="store_true",
                    help="sample DRAM at every tape node and after every backward node, which "
                         "sees the in-step transient the two seam reads miss. Each sample drains "
                         "the pipeline, so a node-peak run wall is NOT a timing")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ov = [f"campaign_seed={args.seed}", "max_trajectories=1", "validation_model=monomer",
          'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]',
          f"binder_lengths=[{args.binder_length}]",
          "autotune=false", "compile_next_length=false"]
    settings = B.campaign_settings(overrides=ov)
    design_settings = build_design_settings(settings)
    bucket = campaign_length_bucket(settings)

    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), args.trajectory)
    drawn, targets = sampled_trajectory_values(design_settings, key)
    binder_key, _mut = jax.random.split(key)
    conformation_key = jax.random.split(binder_key)[1]
    protein_states, multi_chain_binders, losses = initialize_design_trajectory(
        design_settings, binder_key, targets)

    selected = select_design_and_validation_models(settings, MONOMER_POOL, MONOMER_POOL)
    model = T.TTBioAlphaFoldDesignModel(
        presets=selected.design_models, data_dir="/home/ttuser/bcx_e2e/af2_params",
        max_cache_size=16, num_recycle=settings.get("design_recycles", 1),
        models=selected.design_models, cyclic_offset_mode=resolve_cyclic_offset_mode(settings),
        subbatch_size=campaign_subbatch_size(settings, design_residue_count(settings)),
        attention_backend=settings.get("attention_backend", "auto"),
        use_cueq=bool(settings.get("use_cueq", False)), length_bucket_size=bucket,
        multi_chain_binders=multi_chain_binders, target_pad_length=0, trunk="device")
    model.key = key

    binder_len = int(drawn["binder_length"])
    n_complex = sum(len(p) for c in protein_states.values() for p in c.values())
    n_binder_bucketed = padded_prediction_length(binder_len, bucket) + (n_complex - binder_len)

    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device())
    lv.arm("stack")
    evo = SP.EvoformerOnDevice(dev, k_evo=48)

    phase = {"now": "before first step", "step": 0}
    peak = {"bytes": 0, "at": None, "samples": 0, "free_gb": None}

    def node_sample(where):
        peak["samples"] += 1
        used, free = dram_used_free(dev)
        if used > peak["bytes"]:
            peak.update(bytes=used, at=f"{phase['now']}:{where}:step{phase['step']}",
                        free_gb=round(free / GB, 4))

    if args.node_peak:
        # Borrowed from perf/bcx_large/ladder.py --node-peak. Both bindings: taped_ttnn imports
        # its own reference to _tape, so rebinding only autograd misses every op that routes
        # through it.
        from tt_bio import taped_ttnn as _TT
        _orig_tape = ag._tape

        def tape(out_value, parents, make_fn):
            def make(*a, **k):
                fn = make_fn(*a, **k)

                def bw(g):
                    r = fn(g)
                    node_sample("node_bw")
                    return r
                return bw
            node = _orig_tape(out_value, parents, make)
            node_sample("node_fw")
            return node
        ag._tape = _TT._tape = tape

    # The token axis the DEVICE is handed, read off the real call. `_pad_inputs` is the last
    # thing between a BindCraft 2 array and a ttnn buffer, so its n32 is what the allocator
    # shapes every activation by. Every predicted state passes here and they are not the same
    # size, so keep the set and the max rather than the last one seen.
    seen = {"n32": 0, "n": 0, "n32_seen": set(), "n_seen": set()}
    _pad_inputs = SP._pad_inputs

    def pad_inputs(m, z, mask, pair_mask):
        out = _pad_inputs(m, z, mask, pair_mask)
        seen["n_seen"].add(int(out[4])); seen["n32_seen"].add(int(out[5]))
        seen["n"] = max(seen["n"], int(out[4]))
        seen["n32"] = max(seen["n32"], int(out[5]))
        seen["msa_rows"] = int(out[0].shape[0])
        return out
    SP._pad_inputs = pad_inputs

    _taped, _backward = SP.EvoformerOnDevice._taped, SP.EvoformerOnDevice._backward
    seam = {"fwd": 0, "bwd": 0}

    def taped(self, *a, **kw):
        phase["now"] = "forward"
        out = _taped(self, *a, **kw)
        seam["fwd"] = max(seam["fwd"], dram_used_free(dev)[0])
        return out

    def backward(self, *a, **kw):
        phase["now"] = "backward"
        out = _backward(self, *a, **kw)
        seam["bwd"] = max(seam["bwd"], dram_used_free(dev)[0])
        return out
    SP.EvoformerOnDevice._taped, SP.EvoformerOnDevice._backward = taped, backward

    rows = []
    real_sg = T.TTBioAlphaFoldDesignModel.sequence_gradients
    clock = S.Clock()
    tag = args.tag or f"L{args.binder_length}"
    dest = pathlib.Path(args.out) if args.out else HERE / f"ceil_{tag}.json"
    jsonl = dest.with_suffix(".jsonl")
    jsonl.unlink(missing_ok=True)

    def traced(self, states, active_losses, *a, **kw):
        step = len(rows) + 1
        phase["step"] = step
        seam["fwd"] = seam["bwd"] = 0
        before = dram_used_free(dev)[0]
        t0 = time.time()
        preds, grads, loss = real_sg(self, states, active_losses, *a, **kw)
        t1 = time.time()
        row = {"step": step, "seconds": round(t1 - t0, 3), "span": (t0, t1),
               "design_loss": float(loss),
               "dram_before_gb": round(before / GB, 4),
               "dram_peak_fwd_gb": round(seam["fwd"] / GB, 4),
               "dram_peak_bwd_gb": round(seam["bwd"] / GB, 4),
               "dram_after_gb": round(dram_used_free(dev)[0] / GB, 4),
               "dram_peak_node_gb": round(peak["bytes"] / GB, 4) if args.node_peak else None,
               "peak_at": peak["at"], "free_at_peak_gb": peak["free_gb"],
               "node_samples": peak["samples"],
               "live_tapes": SP.EvoformerOnDevice.live_tapes(),
               "ckpt_pins": len(ag._CKPT_PINS),
               "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
               "loadavg1": round(os.getloadavg()[0], 2)}
        peak.update(bytes=0, at=None, samples=0, free_gb=None)
        rows.append(row)
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if step >= args.steps:
            raise SystemExit(0)
        return preds, grads, loss
    T.TTBioAlphaFoldDesignModel.sequence_gradients = traced

    out = {"binder_length": binder_len, "n_complex": n_complex,
           "n_binder_bucketed": n_binder_bucketed, "bucket": bucket,
           "steps_requested": args.steps, "card": args.card, "node_peak": bool(args.node_peak),
           "weakrefs_in_autograd": (ROOT / "tt_bio" / "autograd.py").read_text().count("weakref.ref"),
           "head": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
           "dirty_autograd": bool(os.popen(
               f"git -C {ROOT} status --porcelain tt_bio/autograd.py").read().strip())}
    print(json.dumps(out), flush=True)
    try:
        with SP.evoformer_on_device(evo):
            stage_plan = build_stage_plan(design_settings, losses, multi_chain_binders,
                                          model, protein_states)
            stage = stage_plan[0]
            target_schedule = build_target_schedule(
                design_settings, protein_states, stage.sequence_optimizer.iterations, stage.name)
            trajectory = TrajectoryState(protein_states, {}, protein_states, losses)
            trajectory = run_stage_operations(stage.prepare, trajectory)
            model.dropout = stage.dropout and settings.get("design_dropout", False)
            schedule = build_design_schedule(
                design_settings, trajectory.design_target_states, trajectory.losses,
                stage.sequence_optimizer.iterations, conformation_key, False,
                target_schedule, stage.name)
            out["stage"] = stage.name
            run_gradient_design_stage(
                trajectory.protein_states, model, sequence_optimizer=stage.sequence_optimizer,
                design_schedule=schedule, select_best_round=True,
                **induced_fit_reference_arguments(trajectory.losses,
                                                  trajectory.binder_alone_reference,
                                                  model, design_settings))
    except SystemExit:
        out["stopped"] = "step budget reached"
    except Exception as exc:                                             # noqa: BLE001
        # WHERE it died is half the answer. An OOM in the forward is a different finding from
        # one in the backward, and a step that never reached the backward has not been tested.
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["failed_step"] = phase["step"]
        out["failed_phase"] = phase["now"]
        out["steps_completed"] = len(rows)
        used, free = dram_used_free(dev)
        out["dram_at_failure_gb"] = round(used / GB, 4)
        out["dram_free_at_failure_gb"] = round(free / GB, 4)
        print(f"FAILED step {phase['step']} in the {phase['now']}: {out['error']}", flush=True)
    finally:
        clock.stop()
        out["rows"] = rows
        out["device_n"] = seen["n"]
        out["device_n32"] = seen["n32"]
        out["device_n32_all"] = sorted(seen["n32_seen"])
        out["device_n_all"] = sorted(seen["n_seen"])
        out["msa_rows"] = seen.get("msa_rows")
        out["device_calls"] = dict(evo.calls)
        out["aiclk"] = clock.window([r["span"] for r in rows]) if rows else {"n": 0}
        out["stamp"] = A.stamp(args.card)
        if rows:
            out["resident_first_gb"] = rows[0]["dram_after_gb"]
            out["resident_last_gb"] = rows[-1]["dram_after_gb"]
            out["resident_rise_gb"] = round(rows[-1]["dram_after_gb"] - rows[0]["dram_after_gb"], 4)
            out["peak_max_gb"] = round(max(max(r["dram_peak_fwd_gb"], r["dram_peak_bwd_gb"])
                                           for r in rows), 4)
            if args.node_peak:
                out["peak_node_max_gb"] = round(
                    max(r["dram_peak_node_gb"] or 0 for r in rows), 4)
        dest.write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
