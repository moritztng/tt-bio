"""What a CORRECT implementation loses to bfloat16 matmuls, on one real IPA.

jax on CPU ignores `jax_default_matmul_precision`, so the obvious control is unavailable: the
float32 and the "bfloat16 matmul" reference arms come out bit-identical. This is the control
that does work. `probe_ipa.numpy_ipa` is a float64 transcription already validated against
haiku to rel 1.0e-15; running it twice, once exact and once with every matmul operand rounded
to bfloat16, brackets what the card's matmul engine can cost a port that has no bug in it.

Operand rounding is a LOWER bound on the card's error, not a model of it: a measured ttnn
float32 matmul reads rel 1.5e-3 against float64 where rounding both operands to bfloat16 and
accumulating in float64 reads about 8e-4, so the card is roughly 2x this arm.
"""
import importlib.util
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("pi", "scripts/bcx_structmod/probe_ipa.py")
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)

raw = np.load("/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
params = {k: raw[k] for k in raw.files if "structure_module" in k}
ref = np.load(sys.argv[1])

single = np.asarray(ref["single"], np.float64)
pair = np.asarray(ref["pair"], np.float64)
mask = np.asarray(ref["seq_mask"], np.float64)
live = mask > 0

act = pi._ln(single, params, pi.PREFIX + "single_layer_norm")
act = act @ np.asarray(params[pi.PREFIX + "initial_projection//weights"], np.float64) \
    + np.asarray(params[pi.PREFIX + "initial_projection//bias"], np.float64)
act_2d = pi._ln(pair, params, pi.PREFIX + "pair_layer_norm")

arms = {}
for label, mode in (("exact", None), ("bf16_operands", "bfloat16")):
    pi.ROUND[0] = mode
    keep = {}
    arms[label] = (pi.numpy_ipa(params, act, act_2d, mask, keep), keep)

exact, kexact = arms["exact"]
rounded, kround = arms["bf16_operands"]


def rel(a, b, sel=None):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if sel is not None:
        a, b = a[sel], b[sel]
    return np.linalg.norm(a - b) / np.linalg.norm(b)


print(f"n={single.shape[0]} live={int(live.sum())}")
print(f"  ipa_output      live rel={rel(rounded, exact, live):.4e}")
for name in ("scalar_logits", "point_logits", "attn", "result_scalar", "final_act"):
    print(f"  {name:16s} live rel={rel(kround[name], kexact[name], live):.4e}")
