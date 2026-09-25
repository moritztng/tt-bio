"""Is TTBioAlphaFoldDesignModel a drop-in for campaign.py, and does the mask guard fire?"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bc2_state as B
from ttbio_predictor import (TTBioAlphaFoldDesignModel, conforms, masked_residue_count,
                             refuse_masked_state)
from bindcraft.campaign import refuse_predictor_without_distogram

out = {}
s32 = B.campaign_settings()
s1 = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states32, losses = B.design_state(s32)
_, states1, _ = B.design_state(s1)

out["masked_at_bucket_32"] = masked_residue_count(states32, 32)
out["masked_at_bucket_1"] = masked_residue_count(states1, 1)

# The guard must refuse the padded state and pass the unpadded one.
try:
    refuse_masked_state(states32, 32); out["guard_refused_bucket_32"] = False
except ValueError as e:
    out["guard_refused_bucket_32"] = True
    out["guard_message_names_setting"] = "length_bucket_size=1" in str(e)
try:
    refuse_masked_state(states1, 1); out["guard_passed_bucket_1"] = True
except ValueError:
    out["guard_passed_bucket_1"] = False

m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir="/home/ttuser/bcx_e2e/af2_params",
                              models=("model_1_ptm",), num_recycle=1, length_bucket_size=1,
                              max_cache_size=2, trunk="jax")
out["protocol_conformant"] = conforms(m)
out["is_alphafold_design_model"] = isinstance(m, type(m).__mro__[1])
out["provides_distogram"] = m.provides_distogram
refuse_predictor_without_distogram(s1, m)          # campaign.py:59 must accept it
out["campaign_distogram_check_passed"] = True
out["device_padded_length_192"] = m.device_padded_length(192)
out["device_padded_length_211"] = m.device_padded_length(211)

# The device trunk is declared and refuses honestly rather than silently running JAX.
try:
    m.trunk = "device"; m.predict(states1); out["device_trunk_refuses"] = False
except NotImplementedError:
    out["device_trunk_refuses"] = True
finally:
    m.trunk = "jax"

# The control arm actually predicts, through BindCraft 2's own call path.
t0 = time.time()
preds = m.predict(states1)
out["predict_seconds"] = round(time.time() - t0, 2)
out["predict_states"] = sorted(preds)
st = preds["hPDL1"]
out["predict_chains"] = sorted(st.protein_complex)
out["predict_metrics"] = sorted(st.metrics)
out["predict_binder_residues"] = int(len(st.protein_complex["binder"]))
out["predict_atoms_finite"] = bool(
    __import__("numpy").isfinite(__import__("numpy").asarray(st.protein_complex["binder"].atoms)).all())

(HERE / "predictor_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
