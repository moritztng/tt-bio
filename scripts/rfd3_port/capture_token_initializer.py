"""Capture RFD3 TokenInitializer I/O goldens by monkeypatching its forward.

Run on the vast.ai reference instance (after foundry+rfd3 installed + ckpt present):

    RFD3_CAPTURE_DIR=/root/work/capture python capture_token_initializer.py \
        ckpt_path=/root/work/ckpt/rfd3_latest.ckpt \
        out_dir=/root/work/cap_out inputs=./demo.json \
        inference_sampler.num_timesteps=1 diffusion_batch_size=1 n_batches=1 \
        skip_existing=False prevalidate_inputs=True seed=42 \
        json_keys_subset='[dsDNA_basic]'

Saves, under RFD3_CAPTURE_DIR:
  token_initializer.in_f_<key>.pt   (one per feature key, ~43)
  token_initializer.out_<name>.pt   (Q_L_init, C_L, P_LL, S_I, Z_II)
  token_initializer.meta.json        (shapes/dtypes + the captured example id)
"""
import json
import os

import torch

SAVE = os.environ["RFD3_CAPTURE_DIR"]
os.makedirs(SAVE, exist_ok=True)

# Import the class to wrap (must come before the engine builds the model).
from rfd3.model.layers.encoders import TokenInitializer  # noqa: E402

_orig_forward = TokenInitializer.forward
_done = {"captured": False}


def _to_tensor(v):
    if torch.is_tensor(v):
        return v.detach().cpu()
    try:
        return torch.as_tensor(v).cpu()
    except Exception:
        return None


def _wrapped_forward(self, f):
    out = _orig_forward(self, f)
    if not _done["captured"]:
        meta = {"in_keys": [], "out_keys": list(out.keys()), "in_shapes": {}, "out_shapes": {}}
        for k, v in f.items():
            t = _to_tensor(v)
            if t is None:
                continue
            torch.save(t, os.path.join(SAVE, f"token_initializer.in_f_{k}.pt"))
            meta["in_keys"].append(k)
            try:
                meta["in_shapes"][k] = [list(t.shape), str(t.dtype)]
            except Exception:
                meta["in_shapes"][k] = [None, str(t.dtype)]
        for name, v in out.items():
            t = _to_tensor(v)
            if t is None:
                continue
            torch.save(t, os.path.join(SAVE, f"token_initializer.out_{name}.pt"))
            try:
                meta["out_shapes"][name] = [list(t.shape), str(t.dtype)]
            except Exception:
                meta["out_shapes"][name] = [None, str(t.dtype)]
        with open(os.path.join(SAVE, "token_initializer.meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[capture] saved TokenInitializer I/O to {SAVE}", flush=True)
        print(f"[capture] in_keys={meta['in_keys']}", flush=True)
        print(f"[capture] out_shapes={meta['out_shapes']}", flush=True)
        _done["captured"] = True
    return out


TokenInitializer.forward = _wrapped_forward

if __name__ == "__main__":
    import sys

    # Ensure the hydra app sees the overrides as argv.
    if len(sys.argv) == 1 or sys.argv[1].startswith("ckpt_path=") or "=" in sys.argv[1]:
        # Called with hydra overrides as positional args.
        sys.argv = ["capture_token_initializer.py"] + sys.argv[1:]
    from rfd3.run_inference import run_inference

    run_inference()
