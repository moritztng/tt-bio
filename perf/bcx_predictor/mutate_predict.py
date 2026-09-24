"""Is the untaped `predict` path broken when BindCraft 2 pads the TOTAL token axis?

`post_seed100` passed all four gradient stages at i_pTM 0.74-0.80 and then read 0.11 on
mutate round 1 and 0.11 flat for all fifteen rounds. Mutate is the first stage to call
`predict`, which pads the total token axis (`bindcraft/af2.py:298`), where
`sequence_gradients` pads the DESIGN chain (`:385`). Every validated `predict` reading this
row owns was taken at binder 77, where the complex is 192 exactly and the predict path pads
NOTHING, so a padded two-chain `predict` has never been graded here.

Three configurations of BindCraft 2's own `predict`, each run by both arms on the identical
state, at the mutate stage's own settings (design model, num_recycle 1, dropout False):

  A  seed 0's draw, binder 77, complex 192. Bucket 32 pads nothing. The control: this is the
     regime every earlier `predict` reading was taken in.
  B  seed 100's draw, binder 148, complex 263, bucket 32. BindCraft 2 pads 25 tokens onto
     the end of the complex. The test, and a heavier dose than the 6 tokens the failing
     trajectory had.
  C  the same draw at bucket 1. BindCraft 2 pads nothing; the splice's own `_pad_inputs`
     still pads 263 -> 288 with its own mask. Separates BindCraft 2's pad from ours.

If the device reads i_pTM far below JAX on B and agrees on A, the collapse is a port defect
on the padded predict path. If all three agree, the predict path is exonerated and the
mutate collapse is upstream of it.
"""
import json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import numpy as np, jax
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.af2 import padded_prediction_length

PARAMS = "/home/ttuser/bcx_e2e/af2_params"

CONFIGS = [
    ("A_binder77_b32", 0, 32),
    ("B_binder148_b32", 100, 32),
    ("C_binder148_b1", 100, 1),
]


def build(seed):
    ov = [] if seed == 0 else [f"campaign_seed={seed}"]
    s = B.campaign_settings(overrides=ov)
    _ds, states, _ = B.design_state(s)
    return states


STATES = {seed: build(seed) for seed in (0, 100)}


def model(bucket):
    # The mutate stage's own settings: the DESIGN model, design_recycles 1, and
    # run_mutation_polish sets dropout False (trajectory.py:263).
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                     max_cache_size=4, dropout=False, trunk="jax")


def summarise(pred):
    o = {}
    for st, sp in pred.items():
        plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
        chains, i = {}, 0
        for name in sorted(sp.protein_complex):
            n = len(sp.protein_complex[name])
            chains[name] = round(float(plddt[i:i + n].mean()), 4)
            i += n
        o[st] = {"n": int(plddt.shape[0]), "chains": chains,
                 "plddt": round(float(plddt.mean()), 4),
                 "ptm": round(float(np.asarray(sp.metrics["ptm"])), 4),
                 "iptm": round(float(np.asarray(sp.metrics["iptm"])), 4)}
    return o


out = {"why": "mutate reads i_pTM 0.11 flat where harden read 0.74; predict pads the total axis",
       "mutate_settings": {"num_recycle": 1, "dropout": False, "model": "model_1_ptm"},
       "configs": {}}
for tag, seed, bucket in CONFIGS:
    total = sum(len(v) for v in STATES[seed]["hPDL1"].values())
    out["configs"][tag] = {"seed": seed, "bucket": bucket, "complex_tokens": total,
                           "bc2_padded_to": padded_prediction_length(total, bucket),
                           "bc2_pad": padded_prediction_length(total, bucket) - total}
print(json.dumps(out["configs"], indent=1), flush=True)

res = {}
for tag, seed, bucket in CONFIGS:
    t0 = time.time()
    res.setdefault(tag, {})["jax"] = summarise(model(bucket).predict(STATES[seed]))
    res[tag]["jax_s"] = round(time.time() - t0, 1)
    print("JAX", tag, json.dumps(res[tag]["jax"]), res[tag]["jax_s"], "s", flush=True)
    out["result"] = res
    (HERE / "mutate_predict.json").write_text(json.dumps(out, indent=1, default=str))

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); w0 = time.time()
with evoformer_on_device(evo):
    for tag, seed, bucket in CONFIGS:
        t0 = time.time()
        res[tag]["device"] = summarise(model(bucket).predict(STATES[seed]))
        res[tag]["device_s"] = round(time.time() - t0, 1)
        print("DEV", tag, json.dumps(res[tag]["device"]), res[tag]["device_s"], "s", flush=True)
        out["result"] = res
        (HERE / "mutate_predict.json").write_text(json.dumps(out, indent=1, default=str))
w1 = time.time(); clock.stop()
out["aiclk"] = clock.window([(w0, w1)])
out["device_calls"] = dict(evo.calls)

for tag in res:
    j, d = res[tag]["jax"]["hPDL1"], res[tag]["device"]["hPDL1"]
    out.setdefault("delta", {})[tag] = {
        "iptm_jax": j["iptm"], "iptm_device": d["iptm"],
        "iptm_diff": round(d["iptm"] - j["iptm"], 4),
        "ptm_diff": round(d["ptm"] - j["ptm"], 4),
        "plddt_diff": round(d["plddt"] - j["plddt"], 4)}
out["mutate_reading_being_explained"] = {"harden_round_5_iptm": 0.74, "mutate_iptm": 0.11,
                                         "mutate_ptm": 0.59, "harden_ptm": 0.74}
out["stamp"] = A.stamp(3)
(HERE / "mutate_predict.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out["delta"], indent=1), flush=True)
print("AICLK", json.dumps(out["aiclk"], default=str), flush=True)
