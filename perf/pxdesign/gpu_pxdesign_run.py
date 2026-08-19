"""One PXDesign pipeline invocation with every stage timed and every kernel path counted.

    python perf/pxdesign/gpu_pxdesign_run.py --yaml task.yaml --out-dir /work/out/x \
        --preset extended --n-sample 8 --report /work/results/run_x.json

PXDesign is a four-model pipeline, not a model, so a total is useless: the number that decides
whether a tt-bio port is mostly new work or mostly wiring is the per-stage split. This wraps
upstream at its own stage boundaries instead of forking it, so the pipeline that runs is exactly
`pxdesign pipeline`:

    prep_host      input YAML -> json, bioassembly dict, checkpoint cache check
    model_init     DesignPipeline construction + pxdesign-d checkpoint load
    gen_feat       PXDesign-d host featurisation (the dataloader)
    gen_device     PXDesign-d diffusion sampling                      <- the new model
    gen_write      CIF write for the generated backbones
    tgt_template   Protenix fold of the bare target (extended only)   <- protenix
    mpnn           ProteinMPNN                                       (subprocess)
    af2_complex    AF2-IG complex                                    (subprocess)
    af2_monomer    AF2-IG monomer                                    (subprocess)
    ptx_mini/ptx   Protenix filter                                    <- protenix
    metrics_host   secondary structure, diversity, success rates
    rank_host      ranking + top-design collection

AF2-IG and ProteinMPNN are spawned as fresh subprocesses by pxdbench, one per stage per call, so
each pays its own model load and JAX compile. That is a property of the pipeline, not of this
harness, and it is why every stage records wall time and an epoch window: the sweep driver
integrates nvidia-smi over each window to get a stage's device utilisation, which is the only way
to see device vs host inside a subprocess we do not own.

Kernel paths are COUNTED, never inferred from a flag. `--use_deepspeed_evo_attention` only sets
DEEPSPEED_EVO, and whether the kernel is reached also depends on the tensor shapes and on
deepspeed importing at all; `--use_fast_ln` only sets LAYERNORM_TYPE and the fused kernel silently
falls back if the extension is missing. So we count DS4Sci_EvoformerAttention calls and census
which branch every LayerNorm construction took.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

STAGES = {}
COUNTS = {}
WINDOWS = []


def _bump(name, n=1):
    COUNTS[name] = COUNTS.get(name, 0) + n


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class stage:
    """Time a stage, cuda-synced on both edges so a stage owns its own device time."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _sync()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        _sync()
        t1 = time.time()
        s = STAGES.setdefault(self.name, {"s": 0.0, "calls": 0})
        s["s"] += t1 - self.t0
        s["calls"] += 1
        WINDOWS.append({"stage": self.name, "t0": self.t0, "t1": t1})
        return False


def wrap(obj, attr, name, counter=None):
    """Wrap a callable attribute in a stage timer. Returns True if the target existed."""
    fn = getattr(obj, attr, None)
    if fn is None:
        return False

    def inner(*a, **kw):
        if counter:
            _bump(counter)
        with stage(name):
            return fn(*a, **kw)

    inner.__name__ = getattr(fn, "__name__", attr)
    setattr(obj, attr, inner)
    return True


def count_only(obj, attr, counter):
    fn = getattr(obj, attr, None)
    if fn is None:
        return False

    def inner(*a, **kw):
        _bump(counter)
        return fn(*a, **kw)

    setattr(obj, attr, inner)
    return True


# ---------------------------------------------------------------------------------------------
# kernel-path counters. These are installed BEFORE the model is built but AFTER protenix is
# imported, which is safe because `_deepspeed_evo_attn` and the `LayerNorm` factory look their
# targets up in the primitives module globals at call time, not at import time.
# ---------------------------------------------------------------------------------------------
def install_kernel_counters(report):
    import torch
    import torch.nn.functional as F

    got = {}
    try:
        import protenix.openfold_local.model.primitives as prim
    except Exception as e:
        report["counter_install_error"] = f"{type(e).__name__}: {e}"
        return got

    got["ds4sci_present"] = hasattr(prim, "DS4Sci_EvoformerAttention")
    got["fused_ln_present"] = hasattr(prim, "FusedLayerNorm")
    got["deepspeed_evo_env"] = os.environ.get("DEEPSPEED_EVO")
    got["layernorm_type_env"] = os.environ.get("LAYERNORM_TYPE")

    count_only(prim, "DS4Sci_EvoformerAttention", "ds4sci_evo_attention")
    count_only(prim, "_deepspeed_evo_attn", "deepspeed_evo_attn_fn")

    # The LayerNorm branch is censused after the fact (layernorm_census) rather than by wrapping
    # the classes: OpenFoldLayerNorm.__init__ calls the explicit super(OpenFoldLayerNorm, self),
    # which resolves the name from module globals at call time, so replacing the class with a
    # factory makes that super() call fail.

    orig_sdpa = F.scaled_dot_product_attention

    def sdpa(*a, **kw):
        _bump("torch_sdpa")
        return orig_sdpa(*a, **kw)

    F.scaled_dot_product_attention = sdpa
    torch.nn.functional.scaled_dot_product_attention = sdpa
    return got


def install_stage_timers():
    """Patch the pipeline's own stage boundaries. Records which targets were found."""
    import pxdbench.tasks.base as PB_BASE
    import pxdbench.tasks.binder as PB_BINDER
    import pxdbench.tools.base as PB_TOOLS
    import pxdesign.model.generator as PXD_GEN
    import pxdesign.runner.dumper as PXD_DUMP
    import pxdesign.runner.inference as PXD_INF
    import pxdesign.runner.pipeline as PXD_PIPE

    found = {}
    found["gen_device"] = wrap(PXD_INF.InferenceRunner, "predict", "gen_device", "pxd_predict")
    found["gen_write"] = wrap(PXD_DUMP.DataDumper, "dump", "gen_write")
    found["gen_total"] = wrap(PXD_INF.InferenceRunner, "_inference", "gen_total")
    found["sample_diffusion"] = count_only(PXD_GEN, "sample_diffusion", "sample_diffusion")
    found["prep_input"] = wrap(PXD_PIPE, "process_input_file", "prep_host")
    found["prep_cache"] = wrap(PXD_PIPE, "download_inference_cache", "prep_host")
    found["prep_bioassembly"] = wrap(PXD_PIPE, "convert_to_bioassembly_dict", "prep_host")
    found["prep_toolweights"] = wrap(PXD_PIPE, "check_tool_weights", "prep_host")
    found["model_init"] = wrap(PXD_PIPE.DesignPipeline, "__init__", "model_init")
    found["tgt_template"] = wrap(PXD_PIPE, "use_target_template_or_not", "tgt_template",
                                 "target_template_probe")
    found["rank_host"] = wrap(PXD_PIPE, "save_top_designs", "rank_host")
    found["mpnn"] = wrap(PB_BINDER.BinderTask, "design_sequence", "mpnn")
    found["af2_complex"] = wrap(PB_BASE.BaseTask, "af2_complex_predict", "af2_complex",
                               "af2_complex_calls")
    found["af2_monomer"] = wrap(PB_BASE.BaseTask, "af2_monomer_predict", "af2_monomer",
                                "af2_monomer_calls")
    found["metrics_secondary"] = wrap(PB_BASE.BaseTask, "cal_secondary", "metrics_host")
    found["metrics_diversity"] = wrap(PB_BASE.BaseTask, "cal_diversity", "metrics_host")
    found["eval_total"] = wrap(PB_BINDER.BinderTask, "run", "eval_total")

    # protenix_predict carries is_large, so it needs its own wrapper to land in the right bucket.
    ptx_fn = PB_BASE.BaseTask.protenix_predict

    def ptx(self, data_list, orig_seqs=None, is_large=False):
        _bump("protenix_filter_calls_large" if is_large else "protenix_filter_calls_mini")
        with stage("ptx" if is_large else "ptx_mini"):
            return ptx_fn(self, data_list, orig_seqs=orig_seqs, is_large=is_large)

    PB_BASE.BaseTask.protenix_predict = ptx
    found["protenix_predict"] = True

    # every pxdbench subprocess: count them and record the argv, because "which model actually
    # ran" is a property of the spawned script, not of the config.
    sub_fn = PB_TOOLS.BasePredictor.run
    subprocs = []

    def subrun(self, input_data):
        t0 = time.time()
        try:
            return sub_fn(self, input_data)
        finally:
            subprocs.append({"script": os.path.basename(getattr(self, "script_path", "?")),
                             "wall_s": round(time.time() - t0, 3)})
            _bump("pxdbench_subprocesses")

    PB_TOOLS.BasePredictor.run = subrun
    return found, subprocs


# ---------------------------------------------------------------------------------------------
def layernorm_census():
    """Which LayerNorm implementation is actually INSTANTIATED, counted over every live module.
    --use_fast_ln only sets LAYERNORM_TYPE; the fused path falls back silently if the compiled
    extension is missing, so the flag is not evidence."""
    import gc

    import torch.nn as nn
    tally = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, nn.Module):
                n = type(o).__name__
                if "LayerNorm" in n or "Attention" in n:
                    tally[n] = tally.get(n, 0) + 1
        except ReferenceError:
            pass
    return tally


def gpu_state():
    """Every compute app on the card. A vast.ai 'single GPU' box can have a co-tenant on the same
    physical card, which silently inflates every number (root-caused on rf3-gpu-reference-vast)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception as e:
        return ["error: %s" % e]


def env_stack():
    from importlib.metadata import PackageNotFoundError, version

    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None

    import torch
    e = {p: v(p) for p in ("pxdesign", "protenix", "pxdbench", "torch", "jax", "jaxlib",
                           "deepspeed", "numpy", "transformers", "dm-haiku", "optax",
                           "biotite", "rdkit", "colabdesign")}
    e["torch_version"] = torch.__version__
    e["torch_cuda"] = torch.version.cuda
    e["cudnn"] = torch.backends.cudnn.version()
    e["gpu"] = torch.cuda.get_device_name(0)
    e["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    try:
        import jax
        e["jax_platform"] = jax.devices()[0].platform
        e["jax_devices"] = str(jax.devices())
    except Exception as ex:
        e["jax_platform"] = "IMPORT_FAILED: %s" % ex
    for k in ("DEEPSPEED_EVO", "LAYERNORM_TYPE", "CUTLASS_PATH", "TOOL_WEIGHTS_ROOT",
              "PROTENIX_DATA_ROOT_DIR", "CUDA_VISIBLE_DEVICES"):
        e["env_" + k] = os.environ.get(k)
    try:
        e["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return e


def validate_outputs(out_dir, task_name, expect_designs):
    """A design model that emits garbage fast is not a reference. Read the pipeline's OWN reported
    metrics and check the designs are real binders before any second is recorded."""
    import math

    import pandas as pd

    v = {"ok": True, "why": []}
    root = pathlib.Path(out_dir)
    design_dir = root / "design_outputs" / task_name
    v["design_dir"] = str(design_dir)

    samples = sorted(root.glob("global_run_*/*/seed_*/predictions/*_sample_*.cif"))
    v["n_generated_cif"] = len(samples)
    if len(samples) != expect_designs:
        v["ok"] = False
        v["why"].append("expected %d generated cif, found %d" % (expect_designs, len(samples)))

    # coordinates must be finite and the binder must actually have atoms
    bad = 0
    n_atoms = []
    for p in samples:
        na = 0
        for line in p.read_text().splitlines():
            if line.startswith("ATOM"):
                na += 1
                for tok in line.split()[10:13]:
                    try:
                        if not math.isfinite(float(tok)):
                            bad += 1
                    except ValueError:
                        pass
        n_atoms.append(na)
    v["n_atoms_per_design"] = n_atoms[:5]
    v["nonfinite_coord_tokens"] = bad
    if bad:
        v["ok"] = False
        v["why"].append("%d non-finite coordinate tokens" % bad)
    if n_atoms and min(n_atoms) < 100:
        v["ok"] = False
        v["why"].append("a design has only %d atoms" % min(n_atoms))

    csvs = sorted(root.glob("global_run_*/*/seed_*/predictions/sample_level_output.csv"))
    v["sample_csvs"] = [str(c) for c in csvs]
    if not csvs:
        v["ok"] = False
        v["why"].append("no sample_level_output.csv written")
        return v

    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    v["n_scored_rows"] = int(len(df))
    v["columns"] = list(df.columns)
    metric_cols = [c for c in ("pLDDT", "i_pTM", "i_pAE", "unscaled_i_pAE",
                               "bound_unbound_RMSD", "af2_binder_pred_design_rmsd",
                               "ptx_iptm_binder", "ptx_ptm_binder", "ptx_pred_design_rmsd",
                               "ptx_mini_iptm_binder", "ptx_mini_ptm_binder",
                               "alpha", "beta", "loop", "Rg") if c in df.columns]
    v["metrics"] = {}
    for c in metric_cols:
        col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(col):
            v["metrics"][c] = {"min": float(col.min()), "median": float(col.median()),
                               "max": float(col.max()), "n": int(len(col))}
    for c in [c for c in df.columns if c.endswith("_success")]:
        try:
            v["metrics"][c + ".count"] = int(df[c].fillna(0).astype(bool).sum())
        except Exception:
            pass
    v["sequences"] = [str(s) for s in df.get("sequence", pd.Series(dtype=str)).head(3)]

    # the confidence metrics must exist and be in range, or the timing measured nothing real
    if "pLDDT" not in v["metrics"]:
        v["ok"] = False
        v["why"].append("no AF2 pLDDT column - eval did not run")
    else:
        p = v["metrics"]["pLDDT"]
        if not (0.0 <= p["min"] <= p["max"] <= 1.0):
            v["ok"] = False
            v["why"].append("pLDDT out of [0,1]: %s" % p)

    summ = design_dir / "summary.csv"
    v["summary_csv_exists"] = summ.exists()
    if summ.exists():
        sdf = pd.read_csv(summ)
        v["summary_rows"] = int(len(sdf))
    ti = design_dir / "task_info.json"
    if ti.exists():
        v["task_info"] = json.loads(ti.read_text())
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--preset", default="extended", choices=["extended", "preview", "custom"])
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--n-step", type=int, default=400)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--fast-ln", default="True")
    ap.add_argument("--ds-evo", default="True")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="run")
    ap.add_argument("--extra", default="", help="extra argv passed through to pxdesign pipeline")
    a = ap.parse_args()

    report = {"label": a.label, "preset": a.preset, "n_sample": a.n_sample, "n_step": a.n_step,
              "dtype": a.dtype, "seed": a.seed, "yaml": a.yaml,
              "yaml_sha256": None, "ok": False, "why": []}
    import hashlib
    report["yaml_sha256"] = hashlib.sha256(pathlib.Path(a.yaml).read_bytes()).hexdigest()

    report["compute_apps_before"] = gpu_state()

    import torch
    torch.cuda.reset_peak_memory_stats()

    # importing pxdesign sets PROTENIX_DATA_ROOT_DIR / TOOL_WEIGHTS_ROOT defaults; the env vars
    # that pick the kernels are set by pxdesign itself from --use_fast_ln / --use_deepspeed_*.
    import pxdesign  # noqa: F401
    import pxdesign.runner.pipeline as PXD_PIPE

    counter_info = install_kernel_counters(report)
    found, subprocs = install_stage_timers()
    report["patch_targets_found"] = found
    report["counter_info"] = counter_info

    # pipeline.main() is the argparse layer BELOW click, so it wants the long names click's
    # build_argv emits (--dump_dir / --input_json_path), not click's -o / -i.
    argv = ["--preset", a.preset, "--dump_dir", a.out_dir, "--input_json_path", a.yaml,
            "--N_sample", str(a.n_sample), "--N_step", str(a.n_step),
            "--dtype", a.dtype, "--use_fast_ln", a.fast_ln,
            "--use_deepspeed_evo_attention", a.ds_evo,
            "--eta_type", "const", "--eta_min", "2.5", "--eta_max", "2.5",
            "--seeds", str(a.seed), "--N_max_runs", "1"]
    if a.preset == "preview":
        argv += ["--eval.binder.eval_complex", "true", "--eval.binder.eval_binder_monomer", "true",
                 "--eval.binder.eval_protenix", "false",
                 "--eval.binder.eval_protenix_mini", "false"]
    elif a.preset == "extended":
        argv += ["--eval.binder.eval_complex", "true", "--eval.binder.eval_binder_monomer", "true",
                 "--eval.binder.eval_protenix", "true",
                 "--eval.binder.eval_protenix_mini", "false"]
    argv += ["--min_total_return", str(a.n_sample), "--max_success_return", str(a.n_sample)]
    if a.extra:
        argv += a.extra.split()
    report["argv"] = argv

    t0 = time.time()
    try:
        PXD_PIPE.main(argv)
        report["pipeline_rc"] = 0
    except SystemExit as e:
        report["pipeline_rc"] = int(e.code or 0)
    except Exception as e:
        report["pipeline_rc"] = 1
        report["why"].append("pipeline raised %s: %s" % (type(e).__name__, e))
        report["traceback"] = traceback.format_exc()
    _sync()
    report["total_s"] = round(time.time() - t0, 4)
    report["t_start"] = t0
    report["t_end"] = time.time()

    report["stages"] = {k: {"s": round(v["s"], 4), "calls": v["calls"]}
                        for k, v in STAGES.items()}
    report["windows"] = WINDOWS
    report["counts"] = COUNTS
    report["module_census"] = layernorm_census()
    report["subprocesses"] = subprocs
    report["peak_vram_alloc_B"] = int(torch.cuda.max_memory_allocated())
    report["peak_vram_reserved_B"] = int(torch.cuda.max_memory_reserved())
    report["compute_apps_after"] = gpu_state()
    report["gpu_exclusive"] = (len(report["compute_apps_before"]) <= 1
                              and len(report["compute_apps_after"]) <= 1)
    report["env"] = env_stack()

    # LEAF stages partition the run; gen_total and eval_total are umbrellas over leaves and are
    # reported but never summed, or the accounting double-counts.
    leaves = ("prep_host", "model_init", "gen_feat", "gen_device", "gen_write", "tgt_template",
              "mpnn", "af2_complex", "af2_monomer", "ptx_mini", "ptx", "metrics_host",
              "rank_host")
    gen_total = report["stages"].get("gen_total", {}).get("s", 0.0)
    inner = sum(report["stages"].get(k, {}).get("s", 0.0) for k in ("gen_device", "gen_write"))
    report["stages"].setdefault("gen_feat", {"s": 0.0, "calls": 0})
    report["stages"]["gen_feat"]["s"] = round(max(0.0, gen_total - inner), 4)
    accounted = sum(report["stages"].get(k, {}).get("s", 0.0) for k in leaves)
    report["unattributed_s"] = round(report["total_s"] - accounted, 4)

    report["split"] = {
        "pxdesign_d_s": report["stages"].get("gen_device", {}).get("s", 0.0),
        "protenix_s": round(sum(report["stages"].get(k, {}).get("s", 0.0)
                                for k in ("tgt_template", "ptx_mini", "ptx")), 4),
        "af2ig_s": round(sum(report["stages"].get(k, {}).get("s", 0.0)
                             for k in ("af2_complex", "af2_monomer")), 4),
        "proteinmpnn_s": report["stages"].get("mpnn", {}).get("s", 0.0),
        "host_data_s": round(sum(report["stages"].get(k, {}).get("s", 0.0)
                                 for k in ("prep_host", "model_init", "gen_feat", "gen_write",
                                           "metrics_host", "rank_host"))
                             + report["unattributed_s"], 4),
    }
    tot = report["total_s"] or 1.0
    report["split_pct"] = {k.replace("_s", "_pct"): round(100.0 * v / tot, 2)
                           for k, v in report["split"].items()}
    report["s_per_design"] = round(report["total_s"] / max(1, a.n_sample), 4)

    task_name = pathlib.Path(a.yaml).stem
    try:
        report["validation"] = validate_outputs(a.out_dir, task_name, a.n_sample)
    except Exception as e:
        report["validation"] = {"ok": False, "why": ["validator raised %s: %s"
                                                     % (type(e).__name__, e)],
                                "traceback": traceback.format_exc()}

    report["ok"] = (report["pipeline_rc"] == 0 and report["gpu_exclusive"]
                    and report["validation"].get("ok", False))
    if not report["gpu_exclusive"]:
        report["why"].append("card was shared: before=%s after=%s"
                             % (report["compute_apps_before"], report["compute_apps_after"]))
    report["why"] += report["validation"].get("why", [])

    pathlib.Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.report).write_text(json.dumps(report, indent=2, default=str))
    print("[pxd] %s total=%.2fs s/design=%.2f | pxd-d=%.1f ptx=%.1f af2=%.1f mpnn=%.1f host=%.1f "
          "| ok=%s %s" % (a.label, report["total_s"], report["s_per_design"],
                          report["split"]["pxdesign_d_s"], report["split"]["protenix_s"],
                          report["split"]["af2ig_s"], report["split"]["proteinmpnn_s"],
                          report["split"]["host_data_s"], report["ok"], report["why"]), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
