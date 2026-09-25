#!/usr/bin/env python3
"""git-bisect probe: does one taped Evoformer block's forward+backward survive at a sequence length?

Exit 0 = good (forward and backward complete), 1 = bad (the trimul circular-buffer/L1 clash
escapes as `program.cpp:1052`), 125 = skip (anything else: an import or API this commit does not
have, so the commit says nothing about the clash).

tt_bio comes from PYTHONPATH (the commit under test). `Dev` comes from a fixed copy of
`perf/bcx_afgrad/afgrad.py`, and the model is built directly rather than through
`afgrad.load_models`, because older tt_bio has no `multimer=` keyword to pass it.
"""
import argparse
import os
import sys
import traceback

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=288)
    ap.add_argument("--params", default=os.path.expanduser(
        "~/.boltz/af2/params/params_model_1_ptm.npz"))
    args = ap.parse_args()
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import afgrad as A
        from tt_bio.af2_weights import load_af2_state_dict
        from tt_bio.af2 import load_af2_device_model
        A._float64_layernorm()
        state = load_af2_state_dict(args.params)
        dm = load_af2_device_model(state, template=False, trunk_dtype=torch.bfloat16)
        dev = A.Dev(dm)
    except Exception:
        traceback.print_exc()
        print("PROBE: SKIP (setup)")
        return 125
    n = args.seq
    torch.manual_seed(0)
    m = torch.randn(1, n, 256) * 0.1
    z = torch.randn(n, n, 128) * 0.1
    try:
        mt = dev.ag.Tensor(dev.up(m), requires_grad=True)
        zt = dev.ag.Tensor(dev.up(z), requires_grad=True)
        with dev.tt.tape():
            mo, zo = dev.stack(mt, zt, 0, 1, ckpt=True)
            dev.ag.backward([mo, zo], [dev.seed(torch.ones_like(m), mo),
                                       dev.seed(torch.ones_like(z), zo)])
        dev.sync()
    except Exception as exc:
        text = str(exc)
        if "program.cpp:1052" in text or "clash with L1 buffers" in text:
            print("PROBE: BAD (trimul CB/L1 clash)")
            return 1
        traceback.print_exc()
        print("PROBE: SKIP (other failure)")
        return 125
    print(f"PROBE: GOOD (seq {n} forward+backward)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
