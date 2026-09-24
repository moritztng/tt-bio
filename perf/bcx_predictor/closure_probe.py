"""Can the splice reach evoformer_masks from the closure of evoformer_fn?"""
import pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax
import bc2_state as B
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

seen = {}
real = LS.layer_stack
def factory(num_layers, *a, **kw):
    made = real(num_layers, *a, **kw)
    def choose(fn):
        if getattr(fn, "__name__", None) == "evoformer_fn":
            import types
            def walk(g, depth=0, seenids=None):
                seenids = seenids if seenids is not None else set()
                if depth > 6 or not isinstance(g, types.FunctionType) or id(g) in seenids:
                    return None
                seenids.add(id(g))
                names = g.__code__.co_freevars
                cells = g.__closure__ or ()
                for nm, cell in zip(names, cells):
                    try:
                        v = cell.cell_contents
                    except ValueError:
                        continue
                    if nm == "evoformer_masks" and isinstance(v, dict):
                        return v
                    r = walk(v, depth + 1, seenids)
                    if r is not None:
                        return r
                return None
            v = walk(fn)
            seen["found"] = v is not None
            if v is not None:
                seen["keys"] = sorted(v)
                seen["msa_shape"] = list(v["msa"].shape)
                seen["pair_shape"] = list(v["pair"].shape)
                seen["msa_is_tracer"] = type(v["msa"]).__name__
        return made(fn)
    return choose

modules.layer_stack.layer_stack = factory
try:
    s = B.campaign_settings(overrides=["length_bucket_size=1"])
    _, states, _ = B.design_state(s)
    TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                              models=("model_1_ptm",), num_recycle=1, key=jax.random.PRNGKey(0),
                              length_bucket_size=1, max_cache_size=2, trunk="jax").predict(states)
finally:
    modules.layer_stack.layer_stack = real
import json; print(json.dumps(seen, indent=1, default=str))
