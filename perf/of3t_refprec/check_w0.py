"""The arms and the reference must be differentiating at the SAME point.

The reference's w_0 is float64 and an arm's is float32, so the check is that the arm's weights
are the reference's rounded to single precision, bit for bit. Anything else means the arms are
at a different point and the comparison is not a precision measurement.
"""
import json, torch
ref = torch.load("/home/ttuser/of3t_refprec/bundle_ref/w0_043.pt", map_location="cpu",
                 weights_only=False)
arm = torch.load("/home/ttuser/of3t_refprec/run/arm2_f32_upstream/w0.pt", map_location="cpu",
                 weights_only=False)
n_id = n_diff = n_skip = 0
worst = (0.0, None)
assert set(ref) == set(arm), (len(ref), len(arm))
for k, v in ref.items():
    a = arm[k]
    if not torch.is_tensor(v) or not v.is_floating_point():
        n_skip += 1
        continue
    if torch.equal(v.to(torch.float32), a):
        n_id += 1
    else:
        n_diff += 1
        d = float((v.to(torch.float32) - a).abs().max())
        if d > worst[0]:
            worst = (d, k)
print(json.dumps({"n_tensors": len(ref), "n_bit_identical_after_fp32_cast": n_id,
                  "n_different": n_diff, "n_non_float_skipped": n_skip,
                  "worst_abs_diff": worst[0], "worst_tensor": worst[1]}, indent=1))
