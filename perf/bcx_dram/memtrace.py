"""Per design step: DRAM high-water, DRAM resident after the step, and what is still live.

BindCraft 2s own `run_gradient_design_stage` drives this, on the same predictor
`campaign.py` constructs and the same seam `bcx-predictor` ships, so the tape traffic is the
loops, not a harness. Only the first gradient stage of one trajectory runs: the question is
whether DRAM resident RISES across steps, and that is visible inside one stage.

Resident is read after the step returns and after `release_pins`, so it is what the next step
starts from; the high-water is sampled at the two seam points that already synchronise, the end
of the taped forward and the end of the backward. A flat resident line with a high peak is a
per-step cost; a rising one is a leak, and they need different fixes.
"""
import argparse, gc, json, os, pathlib, resource, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_predictor")):
    sys.path.insert(0, p)

import numpy as np                                                      # noqa: E402
import jax                                                              # noqa: E402
import ttnn                                                             # noqa: E402
import afgrad as A, stack as S                                          # noqa: E402
import bc2_state as B                                                   # noqa: E402
import splice as SP                                                     # noqa: E402
import ttbio_predictor as T                                             # noqa: E402
from tt_bio import autograd as ag                                       # noqa: E402

import bindcraft.campaign as campaign                                   # noqa: E402
from bindcraft.af2 import MONOMER_POOL, campaign_length_bucket, padded_prediction_length  # noqa: E402
from bindcraft.design_workers import campaign_subbatch_size                # noqa: E402
from bindcraft.protein_preparation import (design_residue_count,           # noqa: E402
                                           initialize_design_trajectory,
                                           sampled_trajectory_values)
from bindcraft.settings import (build_design_settings,                     # noqa: E402
                                resolve_cyclic_offset_mode,
                                select_design_and_validation_models)
from bindcraft.target_schedule import build_design_schedule, build_target_schedule  # noqa: E402
from bindcraft.trajectory import (build_stage_plan, induced_fit_reference_arguments,  # noqa: E402
                                  run_gradient_design_stage, run_stage_operations,
                                  TrajectoryState)


def dram_bytes(dev):
    mv = ttnn.get_memory_view(dev.device, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


def live_counts():
    """Objects the tape can still reach. `gc.get_objects` is the only honest count here: a
    tape node graph is cyclic, so a Python refcount says nothing about whether it is garbage."""
    n_tensor = n_node = n_ttnn = 0
    for o in gc.get_objects():
        t = type(o)
        if t is ag.Tensor:
            n_tensor += 1
        elif t is ttnn.Tensor:
            n_ttnn += 1
        elif t.__name__ == "Node" and t.__module__.endswith("autograd"):
            n_node += 1
    return {"ag_tensor": n_tensor, "ag_node": n_node, "ttnn_tensor": n_ttnn}


def cycle_probe(dev):
    """What a collect finds, and how much DRAM it releases.

    `perf/bcx_stack/stack.py gcdiag` asks this of ONE recomputed block; the question here is
    the design step, so the analysis is borrowed and the subject is the loop. DEBUG_SAVEALL
    keeps the unreachable set instead of freeing it, so the types and the back-references are
    readable before the release is priced.
    """
    import stack as _S
    before = dram_bytes(dev)
    gc.set_debug(gc.DEBUG_SAVEALL)
    t0 = time.time()
    n = gc.collect()
    dt = time.time() - t0
    gc.set_debug(0)
    garbage = list(gc.garbage)
    kinds = {}
    for o in garbage:
        k = type(o).__name__
        kinds[k] = kinds.get(k, 0) + 1
    tens = [o for o in garbage if type(o) is ag.Tensor]
    out = {"seconds": round(dt, 3), "unreachable": n,
           "types": dict(sorted(kinds.items(), key=lambda kv: -kv[1])[:12]),
           "tensors": len(tens),
           "tensors_with_node": sum(t.node is not None for t in tens),
           "tensors_with_shares": sum(t.shares is not None for t in tens),
           "cycles": _S._cycles(garbage) if garbage else {},
           "dram_before_gb": round(before / 2**30, 4)}
    gc.garbage.clear()
    gc.collect()
    out["dram_after_gb"] = round(dram_bytes(dev) / 2**30, 4)
    out["released_gb"] = round(out["dram_before_gb"] - out["dram_after_gb"], 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=200, help="campaign seed; 200 draws binder 176 -> n=307")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--trajectory", type=int, default=1,
                    help="trajectory number the campaign would be on; seed 200 traj 1 is the "
                         "binder-176 draw device_summary.json records at n=307")
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "2")))
    ap.add_argument("--gc-probe", action="store_true",
                    help="gc.collect() after the resident read, as a CONTROL: it reports what a "
                         "collect would have freed without changing what the next step sees")
    ap.add_argument("--cycle-probe", type=int, default=0, metavar="K",
                    help="at step K, collect with DEBUG_SAVEALL and report what was "
                         "unreachable, by type and by the reference that closed the cycle, "
                         "with the DRAM the collect released")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ov = [f"campaign_seed={args.seed}", "max_trajectories=1", "validation_model=monomer",
          "design_models=[\"model_1_ptm\"]", "validation_models=[\"model_2_ptm\"]",
          "autotune=false", "compile_next_length=false"]
    settings = B.campaign_settings(overrides=ov)
    design_settings = build_design_settings(settings)
    bucket = campaign_length_bucket(settings)

    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), args.trajectory)
    drawn, targets = sampled_trajectory_values(design_settings, key)
    binder_key, _mutation_key = jax.random.split(key)
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

    binder_len = drawn["binder_length"]
    n_total = sum(len(p) for c in protein_states.values() for p in c.values())
    n_padded = padded_prediction_length(drawn["binder_length"], bucket) + (
        n_total - drawn["binder_length"])
    print(f"seed {args.seed}: binder {binder_len}, complex {n_total}, "
          f"bucket {bucket}, predictor sees n={n_padded}", flush=True)

    lv = S.Levers(); dm, _ = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
    evo = SP.EvoformerOnDevice(dev, k_evo=48)

    rows, seam = [], {"fwd": 0, "bwd": 0}
    jsonl = HERE / f"memtrace_seed{args.seed}t{args.trajectory}.jsonl"
    base = dram_bytes(dev)

    _taped, _backward = SP.EvoformerOnDevice._taped, SP.EvoformerOnDevice._backward

    def taped(self, *a, **kw):
        out = _taped(self, *a, **kw)
        seam["fwd"] = max(seam["fwd"], dram_bytes(dev))
        return out

    def backward(self, *a, **kw):
        out = _backward(self, *a, **kw)
        seam["bwd"] = max(seam["bwd"], dram_bytes(dev))
        return out

    SP.EvoformerOnDevice._taped, SP.EvoformerOnDevice._backward = taped, backward

    real_sg = T.TTBioAlphaFoldDesignModel.sequence_gradients
    clock = S.Clock()

    def traced(self, states, active_losses, *a, **kw):
        step = len(rows) + 1
        seam["fwd"] = seam["bwd"] = 0
        before = dram_bytes(dev)
        t0 = time.time()
        preds, grads, loss = real_sg(self, states, active_losses, *a, **kw)
        t1 = time.time()
        after = dram_bytes(dev)
        row = {"step": step, "seconds": round(t1 - t0, 3), "span": (t0, t1),
               "design_loss": float(loss),
               "dram_before_gb": round(before / 2**30, 4),
               "dram_peak_fwd_gb": round(seam["fwd"] / 2**30, 4),
               "dram_peak_bwd_gb": round(seam["bwd"] / 2**30, 4),
               "dram_after_gb": round(after / 2**30, 4),
               "live_tapes": SP.EvoformerOnDevice.live_tapes(),
               "ckpt_pins": len(ag._CKPT_PINS),
               "calls": dict(self_calls) if (self_calls := evo.calls) else {},
               "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)}
        row.update(live_counts())
        if step == args.cycle_probe:
            row["cycle"] = cycle_probe(dev)
            print(json.dumps({"step": step, "cycle": row["cycle"]}), flush=True)
        if args.gc_probe:
            gc.collect()
            row["dram_after_gc_gb"] = round(dram_bytes(dev) / 2**30, 4)
        rows.append(row)
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if step >= args.steps:
            raise SystemExit(0)
        return preds, grads, loss

    T.TTBioAlphaFoldDesignModel.sequence_gradients = traced

    out = {"seed": args.seed, "binder": drawn["binder_length"], "n_complex": n_total,
           "n_padded": n_padded, "bucket": bucket, "steps_requested": args.steps,
           "base_dram_bytes": base, "gc_probe": bool(args.gc_probe)}
    try:
        with SP.evoformer_on_device(evo):
            stage_plan = build_stage_plan(design_settings, losses, multi_chain_binders,
                                          model, protein_states)
            stage = stage_plan[0]
            target_schedule = build_target_schedule(design_settings, protein_states,
                                                    stage.sequence_optimizer.iterations, stage.name)
            trajectory = TrajectoryState(protein_states, {}, protein_states, losses)
            trajectory = run_stage_operations(stage.prepare, trajectory)
            model.dropout = stage.dropout and settings.get("design_dropout", False)
            schedule = build_design_schedule(design_settings, trajectory.design_target_states,
                                             trajectory.losses, stage.sequence_optimizer.iterations,
                                             conformation_key, False, target_schedule, stage.name)
            out["stage"] = stage.name
            out["stage_iterations"] = stage.sequence_optimizer.iterations
            run_gradient_design_stage(
                trajectory.protein_states, model, sequence_optimizer=stage.sequence_optimizer,
                design_schedule=schedule, select_best_round=True,
                **induced_fit_reference_arguments(trajectory.losses,
                                                  trajectory.binder_alone_reference,
                                                  model, design_settings))
    except SystemExit:
        out["stopped"] = "step budget reached"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        print(out["error"], flush=True)
    finally:
        clock.stop()
        out["rows"] = rows
        out["device_calls"] = dict(evo.calls)
        out["aiclk"] = clock.window([r["span"] for r in rows]) if rows else {"n": 0}
        out["stamp"] = A.stamp(args.card)
        if len(rows) >= 2:
            out["resident_first_gb"] = rows[0]["dram_after_gb"]
            out["resident_last_gb"] = rows[-1]["dram_after_gb"]
            out["resident_rise_gb"] = round(rows[-1]["dram_after_gb"] - rows[0]["dram_after_gb"], 4)
            out["peak_max_gb"] = round(max(max(r["dram_peak_fwd_gb"], r["dram_peak_bwd_gb"])
                                           for r in rows), 4)
        dest = (pathlib.Path(args.out) if args.out
                else HERE / f"memtrace_seed{args.seed}t{args.trajectory}.json")
        dest.write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))


if __name__ == "__main__":
    main()
