"""What does BindCraft 2 actually hand the Evoformer stack?"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax, numpy as np
import bc2_state as B
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

seen = {}
real = LS.layer_stack
def factory(num_layers, *a, **kw):
    made = real(num_layers, *a, **kw)
    def choose(fn):
        if getattr(fn, "__name__", None) == "evoformer_fn":
            inner = made(fn)
            def spy(x):
                act, sk = x
                seen["msa_in"] = list(act["msa"].shape); seen["msa_dtype"] = str(act["msa"].dtype)
                seen["pair_in"] = list(act["pair"].shape)
                seen["act_keys"] = sorted(act)
                out, sk2 = inner(x)
                seen["msa_out"] = list(out["msa"].shape); seen["pair_out"] = list(out["pair"].shape)
                seen["out_keys"] = sorted(out)
                return out, sk2
            return spy
        return made(fn)
    return choose

modules.layer_stack.layer_stack = factory
try:
    s = B.campaign_settings(overrides=["length_bucket_size=1"])
    _, states, _ = B.design_state(s)
    m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                                  models=("model_1_ptm",), num_recycle=1, key=jax.random.PRNGKey(0),
                                  length_bucket_size=1, max_cache_size=2, trunk="jax")
    m.predict(states)
finally:
    modules.layer_stack.layer_stack = real
(HERE / "shape_probe.json").write_text(json.dumps(seen, indent=1))
print(json.dumps(seen, indent=1))
