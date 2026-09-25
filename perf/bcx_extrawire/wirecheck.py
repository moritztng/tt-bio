"""Every attribute perf/bcx_extrawire/round_ab.py reaches into, checked without a device.

The A/B run is the expensive thing this row is waiting on, and the harness monkeypatches
BindCraft 2 and tt-bio internals by name. A name that moved dies in minute one on a card nobody
else can have. None of this opens a device: it is imports and introspection only.
"""
import os, sys, json, inspect
os.environ.setdefault("TT_VISIBLE_DEVICES", "")
sys.path.insert(0, "perf/bcx_extrawire")
sys.path.insert(0, "perf/bcx_round")
sys.path.insert(0, "perf/bcx_predictor")

fail, ok = [], []

def chk(name, cond, detail=""):
    (ok if cond else fail).append(f"{name}{(' -- ' + detail) if detail else ''}")

import round_ab as R
import meter as M
from tt_bio import bindcraft2 as bc2
import bindcraft.af2 as bc2_af2
import bindcraft.campaign as campaign
import bindcraft.trajectory as trajectory
import bindcraft.sequence_optimization as seqopt

# 1. the device seams round_ab retags
for cls_name in ("EvoformerOnDevice", "ExtraMsaOnDevice"):
    cls = getattr(bc2, cls_name, None)
    chk(f"bc2.{cls_name}", cls is not None)
    for m in ("_primal", "_taped", "_backward"):
        chk(f"bc2.{cls_name}.{m}", callable(getattr(cls, m, None)))

# 2. what meter.install patches
cls = bc2.design_model_class()
for m in ("sequence_gradients", "predict"):
    chk(f"design_model_class().{m}", callable(getattr(cls, m, None)))
chk("meter.install(splice_mod) needs .EvoformerOnDevice", hasattr(bc2, "EvoformerOnDevice"))
sig = inspect.signature(M.install)
chk("meter.install arity 5", len(sig.parameters) == 5, str(sig))

# 3. what per_arm_cache overrides -- the shared-trace guard, the part that must not silently break
chk("AlphaFoldDesignModel._compiled_sequence_gradients",
    callable(getattr(bc2_af2.AlphaFoldDesignModel, "_compiled_sequence_gradients", None)))
chk("bindcraft.af2.CompiledModelCache", hasattr(bc2_af2, "CompiledModelCache"))
cm = getattr(bc2_af2, "CompiledModelCache", None)
if cm is not None:
    chk("CompiledModelCache(max_size) accepts one positional",
        len([p for p in inspect.signature(cm).parameters.values()
             if p.default is inspect.Parameter.empty]) <= 1, str(inspect.signature(cm)))
    chk("CompiledModelCache instance exposes .max_size",
        "max_size" in (inspect.getsource(cm) if hasattr(cm, "__module__") else ""))
src = inspect.getsource(bc2_af2.AlphaFoldDesignModel)
for attr in ("gradient_compile_cache", "alphafold_runners"):
    chk(f"AlphaFoldDesignModel sets self.{attr}", f"self.{attr}" in src)

# 4. the rest of the campaign surface round_ab touches
chk("campaign.MULTIMER_POOL", hasattr(campaign, "MULTIMER_POOL"))
chk("campaign.AlphaFoldDesignModel", hasattr(campaign, "AlphaFoldDesignModel"))
chk("campaign.run_campaign", callable(getattr(campaign, "run_campaign", None)))
chk("trajectory stage fns present for meter",
    any(hasattr(trajectory, f) for f in ("run_gradient_design_stage",
                                         "run_sequence_mutation_stage")))
chk("seqopt has an update_sequence somewhere",
    any("update_sequence" in getattr(getattr(seqopt, n), "__dict__", {})
        for n in dir(seqopt) if isinstance(getattr(seqopt, n), type)))
from bindcraft.af.alphafold.model import layer_stack as LS
chk("layer_stack.layer_stack is patchable", callable(getattr(LS, "layer_stack", None)))

# 5. predictor really takes the keyword round_ab passes
chk("predictor(extra_msa=) exists",
    "extra_msa" in inspect.signature(bc2.predictor.__wrapped__).parameters)

print(json.dumps({"checked": len(ok) + len(fail), "passed": len(ok),
                  "FAILED": fail}, indent=1))
sys.exit(1 if fail else 0)
