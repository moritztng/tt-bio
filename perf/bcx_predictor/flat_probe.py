"""Does `predict` still measure the sequence after the first call in a process?

The mutate stage read i_pTM 0.11 on all fifteen rounds and pTM 0.59 on all fifteen.
`run_mutation_polish` proposes a different point mutation every round, so fifteen different
sequences returned the same two numbers to two decimals. A metric that does not move when
the sequence moves is not measuring the sequence, and that is a defect signature
independent of whether 0.11 is a plausible value.

Same shape, same process, three sequences that are not close to each other:

  v0  the seed-0 draw's binder, untouched.
  v1  the same binder with 20 of its 77 residues changed.
  v2  the binder replaced by poly-alanine.

Nothing about the fold is subtle here: poly-alanine cannot read the same i_pTM as a
designed binder. Both arms run all three in one process, device first, in the order the
mutate stage would. The reading is the SPREAD within each arm, not the absolute value.

Complex is 77 + 115 = 192, a multiple of 32, so BindCraft 2's predict path pads nothing --
this isolates staleness from the padding that `mutate_predict.py` tests.
"""
import json, pathlib, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    sys.path.insert(0, p)

import numpy as np, jax, jax.numpy as jnp
import bc2_state as B
import afgrad as A, stack as S
from splice import EvoformerOnDevice, evoformer_on_device
from ttbio_predictor import TTBioAlphaFoldDesignModel
from bindcraft.af.alphafold.common import residue_constants

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
ALA = residue_constants.restype_order["A"]

s = B.campaign_settings()
_ds, states, _ = B.design_state(s)
binder = states["hPDL1"]["binder"]
seq = np.asarray(binder.sequence)
n_res, n_aa = seq.shape
base = seq.argmax(-1)


def one_hot(idx):
    out = np.zeros((n_res, n_aa), dtype=seq.dtype)
    out[np.arange(n_res), idx] = 1.0
    return jnp.asarray(out)


variants = {"v0_original": jnp.asarray(seq)}
mut = base.copy()
pos = np.linspace(0, n_res - 1, 20).astype(int)
mut[pos] = (mut[pos] + 7) % n_aa
variants["v1_mut20"] = one_hot(mut)
variants["v2_polyA"] = one_hot(np.full(n_res, ALA))


def state_for(name):
    return {"hPDL1": {**states["hPDL1"],
                      "binder": binder.replace(sequence=variants[name])}}


def model():
    # The mutate stage's own settings: design model, design_recycles 1, dropout False.
    return TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                     models=("model_1_ptm",), num_recycle=1,
                                     key=jax.random.PRNGKey(0), length_bucket_size=32,
                                     max_cache_size=4, dropout=False, trunk="jax")


def summarise(pred):
    sp = pred["hPDL1"]
    plddt = np.asarray(sp.metrics["plddt"]).reshape(-1)
    return {"n": int(plddt.shape[0]),
            "binder_plddt": round(float(plddt[:n_res].mean()), 4),
            "plddt": round(float(plddt.mean()), 4),
            "ptm": round(float(np.asarray(sp.metrics["ptm"])), 4),
            "iptm": round(float(np.asarray(sp.metrics["iptm"])), 4)}


out = {"why": "mutate read i_pTM 0.11 and pTM 0.59 on all 15 rounds with a different mutation each round",
       "binder_len": n_res, "complex_tokens": 192, "bucket": 32,
       "variant_differences_from_v0": {
           k: int((np.asarray(v).argmax(-1) != base).sum()) for k, v in variants.items()}}
print(json.dumps(out, indent=1), flush=True)

res = {}

lv = S.Levers(); dm, _r = A.load_models(A.DEFAULT_PARAMS); dev = A.Dev(dm.to_device()); lv.arm("stack")
evo = EvoformerOnDevice(dev, k_evo=48)
clock = S.Clock(); w0 = time.time()
m_dev = model()          # ONE model object across all three, as the campaign has
with evoformer_on_device(evo):
    for name in variants:
        t0 = time.time()
        res.setdefault(name, {})["device"] = summarise(m_dev.predict(state_for(name)))
        res[name]["device_s"] = round(time.time() - t0, 1)
        print("DEV", name, json.dumps(res[name]["device"]), res[name]["device_s"], "s", flush=True)
        out["result"] = res
        (HERE / "flat_probe.json").write_text(json.dumps(out, indent=1, default=str))
w1 = time.time(); clock.stop()
out["aiclk"] = clock.window([(w0, w1)])
out["device_calls"] = dict(evo.calls)
(HERE / "flat_probe.json").write_text(json.dumps(out, indent=1, default=str))

m_jax = model()
for name in variants:
    t0 = time.time()
    res[name]["jax"] = summarise(m_jax.predict(state_for(name)))
    res[name]["jax_s"] = round(time.time() - t0, 1)
    print("JAX", name, json.dumps(res[name]["jax"]), res[name]["jax_s"], "s", flush=True)
    out["result"] = res
    (HERE / "flat_probe.json").write_text(json.dumps(out, indent=1, default=str))

for arm in ("device", "jax"):
    sp = {}
    for metric in ("iptm", "ptm", "plddt", "binder_plddt"):
        vals = [res[k][arm][metric] for k in variants if arm in res.get(k, {})]
        if vals:
            sp[metric] = {"values": vals, "spread": round(max(vals) - min(vals), 4)}
    out.setdefault("spread", {})[arm] = sp
out["mutate_reading_being_explained"] = {"iptm_all_15_rounds": 0.11, "ptm_all_15_rounds": 0.59}
out["stamp"] = A.stamp(3)
(HERE / "flat_probe.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out["spread"], indent=1), flush=True)
print("AICLK", json.dumps(out["aiclk"], default=str), flush=True)
