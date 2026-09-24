#!/usr/bin/env python3
"""of3t-confpfe: the confidence head alone on the training step's own inputs, per input mode.

    head_probe.py --dump conf_DIAG.pt --checkpoint CK --out-dir D

Reads the inputs `devstep.py --conf-dump` recorded inside the step (s_input, s_trunk, z_trunk,
the structure), rebuilds the head the way the training adapter does (same compute kernel config,
weights materialised), builds the step's masks, and calls `forward_device` four times:
s_trunk uploaded bf16 or fp32, with and without the tape, each with the step's own pad rows
and with them zeroed. Each call is written as a dump in
conf_DIAG's format (`D/conf_<mode>.pt`), so `conf_split.py` scores every mode against upstream
float64 on the same inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only-zero-pads", action="store_true")
    ap.add_argument("--modes", default=None, help="comma-separated mode names to run (default all)")
    ap.add_argument("--exact-off", action="store_true",
                    help="run inside autograd.exact_training(False), the public off switch")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.openfold3_confidence import OF3ConfidenceHead
    from tt_bio.tenstorrent import get_device

    d = torch.load(a.dump, weights_only=False)
    sd = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    head = OF3ConfidenceHead(aux, dev, ckc)
    if hasattr(head, "materialize_device_weights"):   # older trees upload lazily
        head.materialize_device_weights()

    ft = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(  # noqa: E731
        x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    inp = d["inputs"]
    n_tok = int(inp["z_trunk"].shape[-2])
    repr_x = d["repr_x"][:n_tok].float()
    # The dump's own token mask when it has one; the step's dumps park pad tokens at the origin.
    tok = (d["token_mask"].reshape(-1)[:n_tok].float() if "token_mask" in d
           else (repr_x.abs().sum(-1) > 0).float())
    n_real = int(tok.sum())
    pm = tok[:, None] * tok[None, :]
    host = lambda t: torch.Tensor(ttnn.to_torch(getattr(t, "value", t))).float().cpu()  # noqa: E731

    ap_modes = [(st, tp, zp) for zp in (False, True) for st, tp in
                (("bf16", False), ("fp32", False), ("bf16", True), ("fp32", True))]
    zp_inp = {k: v.clone() for k, v in inp.items()}
    for k, v in zp_inp.items():
        # Padded tokens' features set to zero, as upstream's own padding leaves them.
        if k == "z_trunk":
            v.reshape(-1, n_tok, n_tok, v.shape[-1])[:, n_real:] = 0
            v.reshape(-1, n_tok, n_tok, v.shape[-1])[:, :, n_real:] = 0
        else:
            v.reshape(-1, n_tok, v.shape[-1])[:, n_real:] = 0
    for st_dt, taped, zero_pads in ap_modes:
        if a.only_zero_pads and not zero_pads:
            continue
        mode = f"{st_dt}_{'tape' if taped else 'notape'}{'_zeropad' if zero_pads else ''}"
        if a.modes and mode not in a.modes.split(","):
            continue
        inp_m = zp_inp if zero_pads else inp
        si = ft(inp_m["s_input"].reshape(1, n_tok, -1))
        st = ft(inp_m["s_trunk"].reshape(1, n_tok, -1),
                ttnn.float32 if st_dt == "fp32" else ttnn.bfloat16)
        zt = ft(inp_m["z_trunk"].reshape(1, n_tok, n_tok, -1))
        oh = head.distance_onehot(repr_x)
        pmask, amask = ft(pm.unsqueeze(0)), ft(((1.0 - tok) * -1e9).reshape(1, 1, 1, n_tok))
        import contextlib
        off = ag.exact_training(False) if a.exact_off else contextlib.nullcontext()
        if taped:
            with off, ag.tape():
                out = head.forward_device(ag.Tensor(si), ag.Tensor(st), ag.Tensor(zt),
                                          ag.Tensor(oh), use_zij_trunk_embedding=True,
                                          pair_mask_d=pmask, attn_mask_d=amask)
        else:
            with off:
                out = head.forward_device(si, st, zt, oh, use_zij_trunk_embedding=True,
                                          pair_mask_d=pmask, attn_mask_d=amask)
        torch.save({"inputs": inp_m, "repr_x": d["repr_x"], "mode": mode, "real_tokens": n_real,
                    "dtypes": {"s_trunk": st_dt},
                    "outputs": {k: host(v) for k, v in out.items()}}, a.out_dir / f"conf_{mode}.pt")
        print(f"PROBE {mode} written, real tokens {n_real}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
