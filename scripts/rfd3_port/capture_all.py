"""Capture RFD3 per-module I/O goldens by monkeypatching each target class's forward.

Captures (at the FIRST forward call of each class — step 0 / recycle 0):
  - TokenInitializer        : in `f` dict -> {Q_L_init, C_L, P_LL, S_I, Z_II}
  - LocalAtomTransformer    : in (Q_L, C_L, P_LL, indices) -> Q_L          (atom encoder)
  - CompactStreamingDecoder : in (A_I, S_I, Z_II, Q_L, C_L, P_LL, tok_idx, indices)
                              -> (A_I, Q_L, o)                            (atom decoder)
  - LinearSequenceHead      : in (A_I) -> (logits, indices)

Run on the vast.ai reference instance (after foundry+rfd3 installed + ckpt present):

    RFD3_CAPTURE_DIR=/root/work/capture python capture_all.py \
        ckpt_path=/root/work/ckpt/rfd3_latest.ckpt \
        out_dir=/root/work/cap_out inputs=./demo.json \
        inference_sampler.num_timesteps=1 diffusion_batch_size=1 n_batches=1 \
        skip_existing=False prevalidate_inputs=True seed=42 \
        json_keys_subset='[dsDNA_basic]' read_sequence_from_sequence_head=False
"""
import json
import os

import torch

SAVE = os.environ["RFD3_CAPTURE_DIR"]
os.makedirs(SAVE, exist_ok=True)

# Import the classes to wrap (must come before the engine builds the model).
from rfd3.model.layers.encoders import TokenInitializer  # noqa: E402
from rfd3.model.layers.blocks import (  # noqa: E402
    LocalAtomTransformer,
    CompactStreamingDecoder,
    LinearSequenceHead,
)

_done = {c.__name__: False for c in (TokenInitializer, LocalAtomTransformer,
                                    CompactStreamingDecoder, LinearSequenceHead)}


def _to_tensor(v):
    if torch.is_tensor(v):
        return v.detach().cpu()
    try:
        return torch.as_tensor(v).cpu()
    except Exception:
        return None


def _save(prefix, name, t):
    t = _to_tensor(t)
    if t is None:
        return
    torch.save(t, os.path.join(SAVE, f"{prefix}.{name}.pt"))


def _save_meta(prefix, meta):
    with open(os.path.join(SAVE, f"{prefix}.meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------- TokenInitializer ----------
_orig_ti = TokenInitializer.forward


def _wrap_ti(self, f):
    out = _orig_ti(self, f)
    if not _done["TokenInitializer"]:
        meta = {"in_keys": [], "out_keys": list(out.keys()),
                "in_shapes": {}, "out_shapes": {}}
        for k, v in f.items():
            t = _to_tensor(v)
            if t is None:
                continue
            _save("token_initializer", f"in_f_{k}", t)
            meta["in_keys"].append(k)
            try:
                meta["in_shapes"][k] = [list(t.shape), str(t.dtype)]
            except Exception:
                meta["in_shapes"][k] = [None, str(t.dtype)]
        for name, v in out.items():
            t = _to_tensor(v)
            if t is None:
                continue
            _save("token_initializer", f"out_{name}", t)
            try:
                meta["out_shapes"][name] = [list(t.shape), str(t.dtype)]
            except Exception:
                meta["out_shapes"][name] = [None, str(t.dtype)]
        _save_meta("token_initializer", meta)
        _done["TokenInitializer"] = True
        print(f"[capture] TokenInitializer -> {SAVE}", flush=True)
    return out


TokenInitializer.forward = _wrap_ti


# ---------- LocalAtomTransformer (encoder) ----------
_orig_enc = LocalAtomTransformer.forward


def _wrap_enc(self, Q_L, C_L, P_LL, **kwargs):
    out = _orig_enc(self, Q_L, C_L, P_LL, **kwargs)
    if not _done["LocalAtomTransformer"]:
        _save("encoder", "in_Q_L", Q_L)
        _save("encoder", "in_C_L", C_L)
        _save("encoder", "in_P_LL", P_LL)
        if "indices" in kwargs:
            _save("encoder", "in_indices", kwargs["indices"])
        _save("encoder", "out_Q_L", out)
        _save_meta("encoder", {
            "in_Q_L": [list(Q_L.shape), str(Q_L.dtype)],
            "in_C_L": [list(C_L.shape), str(C_L.dtype)],
            "in_P_LL": [list(P_LL.shape), str(P_LL.dtype)] if P_LL is not None else None,
            "in_indices": [list(kwargs["indices"].shape), str(kwargs["indices"].dtype)]
                if "indices" in kwargs else None,
            "out_Q_L": [list(out.shape), str(out.dtype)],
            "kwargs": {k: (str(v) if not torch.is_tensor(v) else [list(v.shape), str(v.dtype)])
                       for k, v in kwargs.items()},
        })
        _done["LocalAtomTransformer"] = True
        print(f"[capture] LocalAtomTransformer(encoder) -> {SAVE}", flush=True)
    return out


LocalAtomTransformer.forward = _wrap_enc


# ---------- CompactStreamingDecoder (decoder) ----------
_orig_dec = CompactStreamingDecoder.forward


def _wrap_dec(self, A_I, S_I, Z_II, Q_L, C_L, P_LL, tok_idx, indices, f=None, **kwargs):
    out = _orig_dec(self, A_I, S_I, Z_II, Q_L, C_L, P_LL, tok_idx, indices, f=f, **kwargs)
    if not _done["CompactStreamingDecoder"]:
        a_out, q_out, o = out
        _save("decoder", "in_A_I", A_I)
        _save("decoder", "in_S_I", S_I)
        _save("decoder", "in_Z_II", Z_II)
        _save("decoder", "in_Q_L", Q_L)
        _save("decoder", "in_C_L", C_L)
        _save("decoder", "in_P_LL", P_LL)
        _save("decoder", "in_tok_idx", tok_idx)
        _save("decoder", "in_indices", indices)
        _save("decoder", "out_A_I", a_out)
        _save("decoder", "out_Q_L", q_out)
        _save_meta("decoder", {
            "in_A_I": [list(A_I.shape), str(A_I.dtype)],
            "in_S_I": [list(S_I.shape), str(S_I.dtype)],
            "in_Z_II": [list(Z_II.shape), str(Z_II.dtype)],
            "in_Q_L": [list(Q_L.shape), str(Q_L.dtype)],
            "in_C_L": [list(C_L.shape), str(C_L.dtype)],
            "in_P_LL": [list(P_LL.shape), str(P_LL.dtype)] if P_LL is not None else None,
            "in_tok_idx": [list(tok_idx.shape), str(tok_idx.dtype)],
            "in_indices": [list(indices.shape), str(indices.dtype)],
            "out_A_I": [list(a_out.shape), str(a_out.dtype)],
            "out_Q_L": [list(q_out.shape), str(q_out.dtype)],
        })
        _done["CompactStreamingDecoder"] = True
        print(f"[capture] CompactStreamingDecoder(decoder) -> {SAVE}", flush=True)
    return out


CompactStreamingDecoder.forward = _wrap_dec


# ---------- LinearSequenceHead ----------
_orig_seq = LinearSequenceHead.forward


def _wrap_seq(self, A_I, **_):
    out = _orig_seq(self, A_I)
    if not _done["LinearSequenceHead"]:
        logits, indices = out
        _save("sequence_head", "in_A_I", A_I)
        _save("sequence_head", "out_logits", logits)
        _save("sequence_head", "out_indices", indices)
        _save_meta("sequence_head", {
            "in_A_I": [list(A_I.shape), str(A_I.dtype)],
            "out_logits": [list(logits.shape), str(logits.dtype)],
            "out_indices": [list(indices.shape), str(indices.dtype)],
        })
        _done["LinearSequenceHead"] = True
        print(f"[capture] LinearSequenceHead -> {SAVE}", flush=True)
    return out


LinearSequenceHead.forward = _wrap_seq


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1 or sys.argv[1].startswith("ckpt_path=") or "=" in sys.argv[1]:
        sys.argv = ["capture_all.py"] + sys.argv[1:]
    from rfd3.run_inference import run_inference

    run_inference()
