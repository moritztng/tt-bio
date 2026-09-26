"""How peaked is this IPA's attention, and how big are its logits?

The synthetic `single` and `pair` the reference draws are standard normal. Both go straight
into a LayerNorm so their scale is removed, but their CHANNEL CORRELATION is not what a real
trunk produces, and the IPA's logit is a dot product over 16 channels with no 1/sqrt(d) on the
result beyond sqrt(1/16) on q. If the logits come out large the softmax is an argmax, and then
the instrument is grading tie-breaks rather than arithmetic.
"""
import importlib.util
import math
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("pi", "scripts/bcx_structmod/probe_ipa.py")
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)

raw = np.load("/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
params = {k: raw[k] for k in raw.files if "structure_module" in k}
ref = np.load(sys.argv[1] if len(sys.argv) > 1 else "perf/bcx_structmod/out/ref_n64.npz")

single = np.asarray(ref["single"], np.float64)
pair = np.asarray(ref["pair"], np.float64)
mask = np.asarray(ref["seq_mask"], np.float64)
live = np.flatnonzero(mask > 0)
nr = live.size

act = pi._ln(single, params, pi.PREFIX + "single_layer_norm")
act = act @ np.asarray(params[pi.PREFIX + "initial_projection//weights"], np.float64) \
    + np.asarray(params[pi.PREFIX + "initial_projection//bias"], np.float64)
act_2d = pi._ln(pair, params, pi.PREFIX + "pair_layer_norm")

keep = {}
pi.numpy_ipa(params, act, act_2d, mask, keep)

a = keep["attn"][np.ix_(live, live)]
print(f"act rms {np.sqrt((act ** 2).mean()):.4f}")
print(f"attn row max mean {a.max(axis=1).mean():.4f} median {np.median(a.max(axis=1)):.4f}")
ent = -(a * np.log(a + 1e-300)).sum(axis=1)
print(f"attn row entropy {ent.mean():.4f} nats, uniform over {nr} is {math.log(nr):.4f}")
for name in ("scalar_logits", "point_logits", "attention_2d", "logits"):
    v = keep[name][np.ix_(live, live)]
    print(f"{name:16s} rms {np.sqrt((v ** 2).mean()):10.3f} max {np.abs(v).max():10.3f}")
