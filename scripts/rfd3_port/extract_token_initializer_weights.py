"""Extract model.token_initializer.* real weights from an RFD3 checkpoint.

    python extract_token_initializer_weights.py /root/work/ckpt/rfd3_latest.ckpt /root/work/capture

Saves token_initializer.real_weights.pt (a state_dict of just the token_initializer
submodule, fp32) + token_initializer.real_weights.meta.json (key->shape).
"""
import json
import os
import sys

import torch

ckpt_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

ck = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
model_sd = ck["model"]
prefix = "token_initializer."
sub = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
# Some keys may have a leading "model." depending on ckpt structure; handle both.
if not sub:
    prefix2 = "model.token_initializer."
    sub = {k[len(prefix2):]: v for k, v in model_sd.items() if k.startswith(prefix2)}
out_path = os.path.join(out_dir, "token_initializer.real_weights.pt")
torch.save({k: v.detach().cpu() for k, v in sub.items()}, out_path)
meta = {k: [list(v.shape), str(v.dtype)] for k, v in sub.items()}
with open(os.path.join(out_dir, "token_initializer.real_weights.meta.json"), "w") as fh:
    json.dump({"n_keys": len(sub), "keys": list(sub.keys()), "shapes": meta}, fh, indent=2)
print(f"[extract] saved {len(sub)} token_initializer weight tensors to {out_path}", flush=True)
for k in sorted(sub.keys()):
    print(f"  {k}\t{list(sub[k].shape)}\t{sub[k].dtype}")
