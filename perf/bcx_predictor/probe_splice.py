"""Can the two Evoformer stacks be identified at the layer_stack call site?

modules.py uses layer_stack three times: :247 the template pair stack, :1528 the extra-MSA
stack, :1594 the Evoformer. Per-block interception is impossible because layer_stack is a
jax.lax.scan -- the body is traced once, not looped -- so the stack has to be replaced
whole, which means discriminating the three call sites. The closures are named
(`extra_msa_stack_fn`, `evoformer_fn`), so this checks whether that name survives to the
call site, including through hk.remat when gc.use_remat is on.
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bc2_state as B
from bindcraft.af.alphafold.model import modules, layer_stack as LS

seen = []
real = LS.layer_stack

def probe(num_layers, *a, **kw):
    made = real(num_layers, *a, **kw)
    def wrap(fn):
        seen.append({"num_layers": int(num_layers), "fn_name": getattr(fn, "__name__", None),
                     "wrapped": getattr(fn, "__wrapped__", None) is not None})
        return made(fn)
    return wrap

modules.layer_stack.layer_stack = probe
try:
    from ttbio_predictor import TTBioAlphaFoldDesignModel
    s = B.campaign_settings(overrides=["length_bucket_size=1"])
    _, states, _ = B.design_state(s)
    m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",),
                                  data_dir="/home/ttuser/bcx_e2e/af2_params",
                                  models=("model_1_ptm",), num_recycle=1,
                                  length_bucket_size=1, max_cache_size=2, trunk="jax")
    m.predict(states)
finally:
    modules.layer_stack.layer_stack = real

names = [e["fn_name"] for e in seen]
out = {"call_sites": seen,
       "distinct_names": sorted(set(n for n in names if n)),
       "evoformer_identifiable": names.count("evoformer_fn") == 1,
       "extra_msa_identifiable": names.count("extra_msa_stack_fn") == 1,
       "template_stack_present": any(e["fn_name"] not in ("evoformer_fn", "extra_msa_stack_fn")
                                     for e in seen),
       "discriminator_is_safe": (names.count("evoformer_fn") == 1
                                 and names.count("extra_msa_stack_fn") == 1)}
(HERE / "splice_probe.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
