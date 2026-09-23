#!/usr/bin/env python3
"""Inference is unchanged: MSAModule built with z_fp32_residual=True and run WITHOUT a tape must
return the default module's z bit for bit, in bf16. 64-token crop boundary, shipped checkpoint."""
import json, os, socket, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.openfold3_msa_embedder import MSAModule
    B = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    (m0, z0), kw = B["inputs"]["args"], B["inputs"]["kwargs"]
    n, n_seq = int(z0.shape[-2]), int(m0.shape[-3])
    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                    weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    up = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    pm = kw["pair_mask"].reshape(1, n, n)
    m1 = torch.diagonal(pm.reshape(n, n)).reshape(1, n).clamp(0, 1)
    pm_d, attn_d = up(pm), up((1 - m1).reshape(1, 1, 1, n) * -1e9)
    res = {}
    for flag in (False, True):
        msa = MSAModule(sd, ckc, z_fp32_residual=flag)
        _, z = msa(up(m0.reshape(1, n_seq, n, -1)), up(z0.reshape(1, n, n, -1)), pm_d, attn_d)
        res[flag] = (str(z.dtype), torch.Tensor(ttnn.to_torch(z)).double())
        del msa
    same = bool(torch.equal(res[False][1], res[True][1]))
    out = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "board": open("/sys/class/tenstorrent/tenstorrent!%s/tt_card_type"
                         % os.environ.get("TT_VISIBLE_DEVICES", "0")).read().strip(),
           "boundary": sys.argv[1], "dtype_default": res[False][0], "dtype_flag_no_tape": res[True][0],
           "bit_identical": same, "max_abs_diff": float((res[False][1] - res[True][1]).abs().max())}
    print(out)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
    return 0 if same else 8


if __name__ == "__main__":
    sys.exit(main())
