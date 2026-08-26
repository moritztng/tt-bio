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

Three modes, one file. `--preset extended|preview` runs the pipeline as upstream ships it.
`--preset gen` switches every eval stage off and times the generator alone, which is what the perf
page's cell publishes; its gate is the written backbone, not AF2-IG's metrics, because AF2-IG did
not run. `--rounds N` invokes the pipeline N times in ONE process with seeds [0,1,2,3,0], drops
round 0 as cold and digests every design, which is the protocol
`perf/newmodelcells/pxd_pagecell.py` uses on the Tenstorrent side -- the last round repeats the
first round's seed and its coordinate digest has to match.

Kernel paths are COUNTED, never inferred from a flag. `--use_deepspeed_evo_attention` only sets
DEEPSPEED_EVO, and whether the kernel is reached also depends on the tensor shapes and on
deepspeed importing at all; `--use_fast_ln` only sets LAYERNORM_TYPE and the fused kernel silently
falls back if the extension is missing. So we count DS4Sci_EvoformerAttention calls and census
which branch every LayerNorm construction took.

Counts are recorded PER STAGE, not only per run. "DeepSpeed was reached somewhere in the pipeline"
and "DeepSpeed was reached inside the three steps the cell publishes" are different claims, and only
the second one decides whether the torch pin is load-bearing for the measurement. The jax counters
carry a self-test for the same reason in reverse: AF2-IG runs in a subprocess, so an in-process jax
counter reads zero whether or not jax ran, and a zero with no positive control checks nothing.
"""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
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
    """Time a stage, cuda-synced on both edges so a stage owns its own device time.

    Also records which COUNTS moved INSIDE the stage. The run total says a kernel was reached
    somewhere in the pipeline; only the per-stage delta says whether it was reached inside the
    three steps the perf-page cell publishes. Nested stages (gen_total over gen_device and
    gen_write) each get their own delta, so an umbrella double-counts its leaves on purpose.
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _sync()
        self._c0 = dict(COUNTS)
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        _sync()
        t1 = time.time()
        s = STAGES.setdefault(self.name, {"s": 0.0, "calls": 0, "counts": {}})
        s["s"] += t1 - self.t0
        s["calls"] += 1
        inside = s.setdefault("counts", {})
        for k, v in COUNTS.items():
            d = v - self._c0.get(k, 0)
            if d:
                inside[k] = inside.get(k, 0) + d
        WINDOWS.append({"stage": self.name, "t0": self.t0, "t1": t1})
        return False


CENSUS = {}


def census_once(tag):
    if tag not in CENSUS:
        CENSUS[tag] = layernorm_census()


def wrap(obj, attr, name, counter=None):
    """Wrap a callable attribute in a stage timer. Returns True if the target existed."""
    fn = getattr(obj, attr, None)
    if fn is None:
        return False

    def inner(*a, **kw):
        if counter:
            _bump(counter)
        with stage(name):
            r = fn(*a, **kw)
        if name in ("gen_device", "tgt_template"):
            census_once(name)
        return r

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


def install_jax_counters(report):
    """Count jax calls in THIS process, and prove the counter can fire.

    AF2-IG is the only JAX model in the pipeline and pxdbench spawns it as a subprocess, so an
    in-process jax counter reads zero in every run including the ones where AF2-IG ran. A zero with
    no positive control checks nothing, so `jax_selftest` calls one wrapped entrypoint after the
    pipeline is done and asserts the counter moved.
    """
    info = {"present": False, "wrapped": []}
    try:
        import jax
        import jax.numpy as jnp
    except Exception as e:
        info["why"] = "%s: %s" % (type(e).__name__, e)
        report["jax_counters"] = info
        return
    info["present"] = True
    info["version"] = getattr(jax, "__version__", None)
    for obj, attr, counter in ((jax, "jit", "jax_jit"),
                               (jax, "device_put", "jax_device_put"),
                               (jnp, "add", "jnp_add")):
        if count_only(obj, attr, counter):
            info["wrapped"].append(counter)
    report["jax_counters"] = info


def jax_selftest(report):
    """Fire one wrapped jax entrypoint and check the counter saw it. Runs after every stage, so it
    cannot pollute a stage's own delta -- only the run total, which is why it is reported."""
    if not report.get("jax_counters", {}).get("present"):
        report["jax_counter_selftest"] = "absent"
        return
    before = COUNTS.get("jnp_add", 0)
    try:
        import jax.numpy as jnp
        jnp.add(1.0, 2.0)
    except Exception as e:
        report["jax_counter_selftest"] = "RAISED %s: %s" % (type(e).__name__, e)
        return
    moved = COUNTS.get("jnp_add", 0) - before
    report["jax_counter_selftest"] = "ok" if moved else "DEAD"
    report["jax_selftest_calls"] = moved


def ds_selftest(report):
    """Fire the wrapped DeepSpeed Evoformer symbol and check the counter saw it.

    The DeepSpeed half of H1 needs the same shape of proof the jax half gets. `counts_in_gen` being
    empty in a generator-only run is a zero, and a zero only means something if the counter that
    produced it can be shown to fire. The positive control was meant to come from an `extended` run,
    where the Protenix filter calls the kernel for real -- but `extended` against a yaml with no
    `msa` key sends the target-template stage into a search the reference cells skipped (measured:
    20 min of CPU with the GPU idle, against 44.2 s for the reference cell that had an MSA). So the
    control is taken here instead, in the exact process shape the published cell runs in.

    This calls the wrapped attribute, not the kernel: what has to be established is that a call
    through `prim.DS4Sci_EvoformerAttention` increments `ds4sci_evo_attention`. Whether the CUDA
    kernel underneath would then succeed is a different question and not the one H1 asks. The
    underlying call is expected to raise on the dummy tensors, and a raise still proves the counter
    moved, which is the whole point.
    """
    try:
        import protenix.openfold_local.model.primitives as prim
    except Exception as e:
        report["ds_counter_selftest"] = "absent: %s" % type(e).__name__
        return
    if not hasattr(prim, "DS4Sci_EvoformerAttention"):
        report["ds_counter_selftest"] = "absent"
        return
    before = COUNTS.get("ds4sci_evo_attention", 0)
    try:
        prim.DS4Sci_EvoformerAttention(None, None, None, None)
    except Exception:
        pass
    moved = COUNTS.get("ds4sci_evo_attention", 0) - before
    report["ds_counter_selftest"] = "ok" if moved else "DEAD"
    report["ds_selftest_calls"] = moved


def subprocess_overlap(report, stages=("gen_feat", "gen_device", "gen_write")):
    """Did any pxdbench subprocess run while a published stage's clock was running?

    The second, independent instrument for the JAX question: AF2-IG's device work happens in a
    process we do not instrument, so what has to be shown is that its wall window is disjoint from
    the windows the cell publishes.
    """
    out = []
    for sp in report.get("subprocesses", []):
        if sp.get("t0") is None:
            continue
        for w in report.get("windows", []):
            if w["stage"] in stages and sp["t0"] < w["t1"] and w["t0"] < sp["t1"]:
                out.append({"script": sp["script"], "stage": w["stage"]})
    return out


def install_stage_timers():
    """Patch the pipeline's own stage boundaries. Records which targets were found."""
    import importlib

    import pxdesign.model.generator as PXD_GEN
    import pxdesign.runner.dumper as PXD_DUMP
    import pxdesign.runner.inference as PXD_INF
    import pxdesign.runner.pipeline as PXD_PIPE

    found = {}

    # The eval half is timed when it is there and skipped when it is not. `--preset gen` runs the
    # generator on a stack that need not carry AF2-IG's JAX at all, and an ImportError here would
    # take the generator down with it.
    def opt(name):
        try:
            return importlib.import_module(name)
        except Exception as e:
            found["import_" + name] = "%s: %s" % (type(e).__name__, e)
            return None

    PB_BASE = opt("pxdbench.tasks.base")
    PB_BINDER = opt("pxdbench.tasks.binder")
    PB_TOOLS = opt("pxdbench.tools.base")
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
    if PB_BINDER is not None:
        found["mpnn"] = wrap(PB_BINDER.BinderTask, "design_sequence", "mpnn")
        found["eval_total"] = wrap(PB_BINDER.BinderTask, "run", "eval_total")
    if PB_BASE is not None:
        found["af2_complex"] = wrap(PB_BASE.BaseTask, "af2_complex_predict", "af2_complex",
                                    "af2_complex_calls")
        found["af2_monomer"] = wrap(PB_BASE.BaseTask, "af2_monomer_predict", "af2_monomer",
                                    "af2_monomer_calls")
        found["metrics_secondary"] = wrap(PB_BASE.BaseTask, "cal_secondary", "metrics_host")
        found["metrics_diversity"] = wrap(PB_BASE.BaseTask, "cal_diversity", "metrics_host")

    subprocs = []
    if PB_BASE is None or PB_TOOLS is None:
        return found, subprocs

    # protenix_predict carries is_large, so it needs its own wrapper to land in the right bucket.
    ptx_fn = PB_BASE.BaseTask.protenix_predict

    def ptx(self, data_list, orig_seqs=None, is_large=False):
        _bump("protenix_filter_calls_large" if is_large else "protenix_filter_calls_mini")
        with stage("ptx" if is_large else "ptx_mini"):
            r = ptx_fn(self, data_list, orig_seqs=orig_seqs, is_large=is_large)
        census_once("ptx")
        return r

    PB_BASE.BaseTask.protenix_predict = ptx
    found["protenix_predict"] = True

    # every pxdbench subprocess: count them and record the argv, because "which model actually
    # ran" is a property of the spawned script, not of the config.
    sub_fn = PB_TOOLS.BasePredictor.run

    def subrun(self, input_data):
        t0 = time.time()
        try:
            return sub_fn(self, input_data)
        finally:
            t1 = time.time()
            subprocs.append({"script": os.path.basename(getattr(self, "script_path", "?")),
                             "wall_s": round(t1 - t0, 3), "t0": t0, "t1": t1})
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
    metrics and check the designs are real binders before any second is recorded.

    The authoritative table is design_outputs/<task>/summary.csv (one row per returned design,
    scalar columns). sample_level_output.csv stores the same AF2 metrics as one-element list
    STRINGS ("[0.91]"), because AF2 can run several models per design, so it needs unwrapping."""
    import ast
    import math

    import pandas as pd

    def unwrap(x):
        if isinstance(x, str) and x.startswith("["):
            try:
                v = ast.literal_eval(x)
                return float(v[0]) if v else None
            except Exception:
                return None
        return x

    v = {"ok": True, "why": []}
    root = pathlib.Path(out_dir)
    design_dir = root / "design_outputs" / task_name
    v["design_dir"] = str(design_dir)

    samples = sorted(root.glob("global_run_*/*/seed_*/predictions/*_sample_*.cif"))
    v["n_generated_cif"] = len(samples)
    if len(samples) != expect_designs:
        v["ok"] = False
        v["why"].append("expected %d generated cif, found %d" % (expect_designs, len(samples)))

    bad, n_atoms = 0, []
    for p_ in samples:
        na = 0
        for line in p_.read_text().splitlines():
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

    summ = design_dir / "summary.csv"
    v["summary_csv"] = str(summ)
    if not summ.exists():
        v["ok"] = False
        v["why"].append("no design_outputs summary.csv - ranking never ran")
        return v
    sdf = pd.read_csv(summ)
    v["summary_rows"] = int(len(sdf))
    v["columns"] = list(sdf.columns)

    METRICS = ("af2_plddt", "af2_ptm", "af2_iptm", "af2_ipAE", "af2_monomer_plddt",
               "af2_bound_unbound_RMSD", "af2_binder_pred_design_rmsd",
               "af2_complex_pred_design_rmsd", "ptx_plddt", "ptx_iptm_binder",
               "ptx_ptm_binder", "ptx_iptm", "ptx_ptm", "ptx_pred_design_rmsd",
               "alpha", "beta", "loop", "Rg")
    v["metrics"] = {}
    for c in METRICS:
        if c not in sdf.columns:
            continue
        col = pd.to_numeric(sdf[c].map(unwrap), errors="coerce").dropna()
        if len(col):
            v["metrics"][c] = {"min": float(col.min()), "median": float(col.median()),
                               "max": float(col.max()), "n": int(len(col))}
    v["filters"] = {}
    for c in ("AF2-IG-easy-success", "AF2-IG-success", "Protenix-success",
              "Protenix-basic-success"):
        if c in sdf.columns:
            v["filters"][c] = int(sdf[c].astype(str).str.lower().eq("true").sum())
    v["sequences"] = [str(x) for x in sdf.get("sequence", pd.Series(dtype=str)).head(3)]
    v["seq_lengths"] = sorted({len(str(x)) for x in sdf.get("sequence", pd.Series(dtype=str))})

    ti = design_dir / "task_info.json"
    if ti.exists():
        v["task_info"] = json.loads(ti.read_text())

    # the run measured nothing real unless the confidence metrics exist and are in range
    if "af2_plddt" not in v["metrics"]:
        v["ok"] = False
        v["why"].append("no af2_plddt in summary.csv - AF2-IG did not score anything")
    else:
        m = v["metrics"]["af2_plddt"]
        if not (0.0 <= m["min"] <= m["max"] <= 1.0):
            v["ok"] = False
            v["why"].append("af2_plddt out of [0,1]: %s" % m)
        if m["median"] < 0.5:
            v["ok"] = False
            v["why"].append("median af2_plddt %.3f - designs are not folded" % m["median"])
    if v["seq_lengths"] and min(v["seq_lengths"]) < 10:
        v["ok"] = False
        v["why"].append("a design sequence is %d residues" % min(v["seq_lengths"]))
    return v



# ---------------------------------------------------------------------------------------------
# the generated artifact: parsed from the CIF's own _atom_site header, so the same reader works on
# upstream's writer and on tt_bio's. A benchmark that returns a plausible number from a run that
# folded nothing has happened on this fleet before, so the design is asserted, not the exit code.
# ---------------------------------------------------------------------------------------------
GEN_GLOB = "global_run_*/*/seed_*/predictions/*_sample_*.cif"


def parse_atom_site(text):
    """(field -> column index, rows) for the _atom_site loop. Column order is read, not assumed."""
    fields, rows, in_loop = [], [], False
    for line in text.splitlines():
        ls = line.strip()
        if ls.startswith("_atom_site."):
            fields.append(ls.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if not in_loop:
            continue
        if not ls or ls.startswith("#") or ls.startswith("loop_") or ls.startswith("_"):
            if rows:
                break
            continue
        parts = ls.split()
        if len(parts) == len(fields):
            rows.append(parts)
        elif rows:
            break
    return {f: i for i, f in enumerate(fields)}, rows


def read_design_cif(path):
    import math
    idx, rows = parse_atom_site(pathlib.Path(path).read_text())

    def col(*names):
        for n in names:
            if n in idx:
                return idx[n]
        return None

    cx, cy, cz = col("Cartn_x"), col("Cartn_y"), col("Cartn_z")
    ch = col("label_asym_id", "auth_asym_id")
    rs = col("label_seq_id", "auth_seq_id")
    xyz, chains, nonfinite = [], {}, 0
    for r in rows:
        if cx is None:
            break
        try:
            v = (float(r[cx]), float(r[cy]), float(r[cz]))
        except (ValueError, IndexError):
            nonfinite += 1
            continue
        if not all(math.isfinite(t) for t in v):
            nonfinite += 1
        xyz.append((r[cx], r[cy], r[cz]))
        if ch is not None:
            chains.setdefault(r[ch], set()).add(r[rs] if rs is not None else len(xyz))
    return {"path": str(path), "n_atom": len(xyz), "nonfinite": nonfinite,
            "chains": {k: len(v) for k, v in sorted(chains.items())},
            "digest": hashlib.sha256(
                " ".join(t for row in xyz for t in row).encode()).hexdigest()[:16]}


def gen_artifacts(out_dir, expect):
    """Every generated backbone under out_dir, digested and checked. Digest is over the CIF's own
    coordinate text, so it is exact, writer-independent and comparable across stacks."""
    cifs = sorted(pathlib.Path(out_dir).glob(GEN_GLOB))
    recs = [read_design_cif(p_) for p_ in cifs]
    v = {"ok": True, "why": [], "n_cif": len(recs), "designs": recs}
    if len(recs) != expect:
        v["ok"] = False
        v["why"].append("expected %d generated cif, found %d" % (expect, len(recs)))
    # At N_sample > 1 the backbones come out of one batched trajectory. They must all differ:
    # a repeat means the sample dim is not carrying independent noise, and "N designs" would be
    # a count of files rather than a count of designs.
    per = [r["digest"] for r in recs]
    v["designs_distinct"] = len(set(per)) == len(per)
    if not v["designs_distinct"]:
        v["ok"] = False
        v["why"].append("%d of %d generated backbones share a coordinate digest"
                        % (len(per) - len(set(per)), len(per)))
    for r in recs:
        if r["nonfinite"]:
            v["ok"] = False
            v["why"].append("%s: %d non-finite coordinates" % (r["path"], r["nonfinite"]))
        if r["n_atom"] < 100:
            v["ok"] = False
            v["why"].append("%s: only %d atoms" % (r["path"], r["n_atom"]))
    v["digest"] = hashlib.sha256(
        " ".join(r["digest"] for r in recs).encode()).hexdigest()[:16] if recs else None
    return v


def snap_stages():
    return {k: {"s": v["s"], "calls": v["calls"], "counts": dict(v.get("counts", {}))}
            for k, v in STAGES.items()}


def stage_delta(before, after):
    """What one round of the pipeline added to the cumulative stage table."""
    out = {}
    for k, v in after.items():
        b = before.get(k, {"s": 0.0, "calls": 0, "counts": {}})
        counts = {}
        for ck, cv in v["counts"].items():
            d = cv - b["counts"].get(ck, 0)
            if d:
                counts[ck] = d
        d = {"s": round(v["s"] - b["s"], 4), "calls": v["calls"] - b["calls"]}
        if counts:
            d["counts"] = counts
        if d["calls"] or d["s"]:
            out[k] = d
    # gen_feat is not a wrapped call: it is what gen_total has left over its two timed leaves.
    gt = out.get("gen_total", {}).get("s", 0.0)
    inner = sum(out.get(k, {}).get("s", 0.0) for k in ("gen_device", "gen_write"))
    if gt:
        out.setdefault("gen_feat", {"s": 0.0, "calls": 0})["s"] = round(max(0.0, gt - inner), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--preset", default="extended",
                    choices=["extended", "preview", "gen", "custom"],
                    help="gen = generator only: the four eval flags off, the artifact asserted from "
                         "the written CIFs instead of from AF2-IG's own metrics")
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--n-step", type=int, default=400)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--fast-ln", default="True")
    ap.add_argument("--ds-evo", default="True")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=1,
                    help="pipeline invocations in THIS process, round 0 dropped as cold. Seeds are "
                         "[0,1,2,3,0] so the last round repeats the first round's seed and its "
                         "digest has to match. Same protocol as perf/newmodelcells/pxd_pagecell.py")
    ap.add_argument("--label", default="run")
    ap.add_argument("--extra", default="", help="extra argv passed through to pxdesign pipeline")
    a = ap.parse_args()

    # `gen` is ours, not upstream's: the pipeline still gets a preset it knows, with every eval
    # stage switched off on top of it.
    pipeline_preset = "preview" if a.preset == "gen" else a.preset

    report = {"label": a.label, "preset": a.preset, "pipeline_preset": pipeline_preset,
              "n_sample": a.n_sample, "n_step": a.n_step, "dtype": a.dtype, "seed": a.seed,
              "rounds": a.rounds, "yaml": a.yaml, "yaml_sha256": None, "ok": False, "why": []}
    import hashlib as _h
    report["yaml_sha256"] = _h.sha256(pathlib.Path(a.yaml).read_bytes()).hexdigest()

    report["compute_apps_before"] = gpu_state()

    import torch
    torch.cuda.reset_peak_memory_stats()

    # importing pxdesign sets PROTENIX_DATA_ROOT_DIR / TOOL_WEIGHTS_ROOT defaults; the env vars
    # that pick the kernels are set by pxdesign itself from --use_fast_ln / --use_deepspeed_*.
    import pxdesign  # noqa: F401
    import pxdesign.runner.pipeline as PXD_PIPE

    counter_info = install_kernel_counters(report)
    install_jax_counters(report)
    found, subprocs = install_stage_timers()
    report["patch_targets_found"] = found
    report["counter_info"] = counter_info

    # pipeline.main() is the argparse layer BELOW click, so it wants the long names click's
    # build_argv emits (--dump_dir / --input_json_path), not click's -o / -i.
    def build_argv(out_dir, seed):
        argv = ["--preset", pipeline_preset, "--dump_dir", str(out_dir),
                "--input_json_path", a.yaml,
                "--N_sample", str(a.n_sample), "--N_step", str(a.n_step),
                "--dtype", a.dtype, "--use_fast_ln", a.fast_ln,
                "--use_deepspeed_evo_attention", a.ds_evo,
                "--eta_type", "const", "--eta_min", "2.5", "--eta_max", "2.5",
                "--seeds", str(seed), "--N_max_runs", "1"]
        on = ["true", "true"] if a.preset in ("preview", "extended") else ["false", "false"]
        ptx = "true" if a.preset == "extended" else "false"
        if a.preset in ("preview", "extended", "gen"):
            argv += ["--eval.binder.eval_complex", on[0],
                     "--eval.binder.eval_binder_monomer", on[1],
                     "--eval.binder.eval_protenix", ptx,
                     "--eval.binder.eval_protenix_mini", "false"]
        argv += ["--min_total_return", str(a.n_sample),
                 "--max_success_return", str(a.n_sample)]
        if a.extra:
            argv += a.extra.split()
        return argv

    seeds = ([0, 1, 2, 3, 0] if a.rounds <= 5
             else [0] + list(range(1, a.rounds - 1)) + [0])[:a.rounds]
    if a.rounds == 1:
        seeds = [a.seed]
    report["seeds"] = seeds
    task_name = pathlib.Path(a.yaml).stem

    t_all0 = time.time()
    rounds = []
    for i, seed in enumerate(seeds):
        out_dir = a.out_dir if a.rounds == 1 else os.path.join(a.out_dir, "r%d" % i)
        argv = build_argv(out_dir, seed)
        if i == 0:
            report["argv"] = argv
        before = snap_stages()
        t0 = time.time()
        rc, tb = 0, None
        try:
            PXD_PIPE.main(argv)
        except SystemExit as e:
            rc = int(e.code or 0)
        except Exception as e:
            rc = 1
            tb = traceback.format_exc()
            report["why"].append("round %d raised %s: %s" % (i, type(e).__name__, e))
        _sync()
        t1 = time.time()
        d = stage_delta(before, snap_stages())
        cell = sum(d.get(k, {}).get("s", 0.0) for k in ("gen_feat", "gen_device", "gen_write"))
        art = gen_artifacts(out_dir, a.n_sample)
        gen_counts = {}
        for k in ("gen_feat", "gen_device", "gen_write"):
            for ck, cv in (d.get(k, {}).get("counts") or {}).items():
                gen_counts[ck] = gen_counts.get(ck, 0) + cv
        rec = {"round": i, "seed": seed, "cold": i == 0, "out_dir": str(out_dir),
               "rc": rc, "t0": t0, "t1": t1, "round_wall_s": round(t1 - t0, 4),
               "gen_feat_s": d.get("gen_feat", {}).get("s", 0.0),
               "gen_device_s": d.get("gen_device", {}).get("s", 0.0),
               "gen_write_s": d.get("gen_write", {}).get("s", 0.0),
               "gen_cell_s": round(cell, 4),
               "counts_in_gen": gen_counts, "stages": d,
               "digest": art["digest"], "artifact": art}
        if tb:
            rec["traceback"] = tb
        rounds.append(rec)
        print("[pxd] %s r%d seed=%s cell=%.3fs (feat %.3f device %.3f write %.3f) "
              "digest=%s rc=%d%s"
              % (a.label, i, seed, rec["gen_cell_s"], rec["gen_feat_s"], rec["gen_device_s"],
                 rec["gen_write_s"], rec["digest"], rc, " COLD" if i == 0 else ""), flush=True)
        pathlib.Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        report["rounds"] = rounds
        pathlib.Path(a.report).write_text(json.dumps(report, indent=2, default=str))

    report["pipeline_rc"] = rounds[-1]["rc"] if rounds else 1
    report["total_s"] = round(time.time() - t_all0, 4)
    report["t_start"] = t_all0
    report["t_end"] = time.time()
    report["rounds"] = rounds

    warm = [r for r in rounds if not r["cold"]] or rounds
    w = sorted(r["gen_cell_s"] for r in warm)
    med = statistics.median(w)
    report["warm_n"] = len(w)
    report["warm_median_cell_s"] = round(med, 4)
    report["warm_min_cell_s"], report["warm_max_cell_s"] = w[0], w[-1]
    report["warm_spread_pct"] = round((w[-1] - w[0]) / med * 100, 3) if med else None
    for k in ("gen_feat_s", "gen_device_s", "gen_write_s"):
        report["warm_median_" + k] = round(statistics.median([r[k] for r in warm]), 4)

    # the determinism check: the last round repeats the first round's seed.
    if len(seeds) > 1 and seeds[0] == seeds[-1]:
        report["digest_repeat_ok"] = rounds[0]["digest"] == rounds[-1]["digest"]
        report["digest_repeat"] = [rounds[0]["digest"], rounds[-1]["digest"]]
    report["digests"] = {str(r["seed"]) + "@r" + str(r["round"]): r["digest"] for r in rounds}

    report["stages"] = {k: {"s": round(v["s"], 4), "calls": v["calls"],
                            **({"counts": v["counts"]} if v.get("counts") else {})}
                        for k, v in STAGES.items()}
    report["windows"] = WINDOWS
    report["subprocesses"] = subprocs
    report["subprocess_overlaps_gen"] = subprocess_overlap(report)
    jax_selftest(report)
    ds_selftest(report)
    report["counts"] = dict(COUNTS)
    report["module_census"] = CENSUS
    report["kernel_env_at_end"] = {k: os.environ.get(k) for k in
                                   ("DEEPSPEED_EVO", "LAYERNORM_TYPE", "CUTLASS_PATH")}
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
    report["s_per_design"] = round(report["total_s"] / max(1, a.n_sample * len(rounds)), 4)

    if a.preset == "gen":
        # AF2-IG did not run, so its metrics cannot be the gate. The written backbones are.
        bad = [r for r in rounds if not r["artifact"]["ok"]]
        report["validation"] = {"ok": not bad, "mode": "gen",
                               "why": [w_ for r in bad for w_ in r["artifact"]["why"]],
                               "rounds_checked": len(rounds)}
        # a raise AFTER gen_write is survivable here: the ranking half needs eval metrics that a
        # generator-only run never produces. A raise BEFORE it is not.
        report["validation"]["gen_write_reached"] = all(
            r["stages"].get("gen_write", {}).get("calls", 0) >= 1 for r in rounds)
        if not report["validation"]["gen_write_reached"]:
            report["validation"]["ok"] = False
            report["validation"]["why"].append("a round never reached gen_write")
    else:
        try:
            report["validation"] = validate_outputs(
                rounds[-1]["out_dir"], task_name, a.n_sample)
        except Exception as e:
            report["validation"] = {"ok": False, "why": ["validator raised %s: %s"
                                                         % (type(e).__name__, e)],
                                    "traceback": traceback.format_exc()}

    rc_ok = report["pipeline_rc"] == 0 or a.preset == "gen"
    report["ok"] = (rc_ok and report["gpu_exclusive"] and report["validation"].get("ok", False)
                    and report.get("digest_repeat_ok", True)
                    and not report["subprocess_overlaps_gen"])
    if not report["gpu_exclusive"]:
        report["why"].append("card was shared: before=%s after=%s"
                             % (report["compute_apps_before"], report["compute_apps_after"]))
    if report.get("digest_repeat_ok") is False:
        report["why"].append("the repeated seed did not reproduce its digest: %s"
                             % report["digest_repeat"])
    if report["subprocess_overlaps_gen"]:
        report["why"].append("a pxdbench subprocess ran inside a published stage: %s"
                             % report["subprocess_overlaps_gen"])
    report["why"] += report["validation"].get("why", [])

    pathlib.Path(a.report).write_text(json.dumps(report, indent=2, default=str))
    print("[pxd] %s warm_median_cell=%.3fs n=%d spread=%s%% | pxd-d=%.1f ptx=%.1f af2=%.1f "
          "mpnn=%.1f host=%.1f | jax_selftest=%s ds_in_gen=%s | ok=%s %s"
          % (a.label, report["warm_median_cell_s"], report["warm_n"],
             report["warm_spread_pct"], report["split"]["pxdesign_d_s"],
             report["split"]["protenix_s"], report["split"]["af2ig_s"],
             report["split"]["proteinmpnn_s"], report["split"]["host_data_s"],
             report.get("jax_counter_selftest"),
             sorted({k for r in rounds for k in r["counts_in_gen"]}),
             report["ok"], report["why"]), flush=True)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
