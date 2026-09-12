#!/usr/bin/env python3
"""Does the key window's wrong gather reach anything, once the atom attention masks it?

Host only, float64, no device: the question is which slots carry weight, and a masked slot's
weight underflows to exactly zero in every precision this runs in.

The atom bias carries an additive mask built from the SAME cached matrix
(`tenstorrent.py`, `_populate_diffusion_cache`): `mask = atom_mask @ keys_indexing`, then
`(-mask + 1) * -1e9`. So a key slot whose matrix column is zero gets -1e9. The slots
`_atom_key_window` fills with real atoms are exactly those slots. This runs the composition
both ways and reports where the attention output differs, window by window.

Windows 0..windows-1 are the real ones. A difference there is a user-visible defect. A
difference only at windows >= `windows` is confined to the bucket's padded tail, which the
atom-to-token aggregation and the final slice discard.
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

W, H = 32, 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-real", type=int, default=129)
    ap.add_argument("--windows-pad", type=int, default=140)
    ap.add_argument("--dim", type=int, default=64)      # head dim per atom head
    ap.add_argument("--atoms-real", type=int, default=None,
                    help="real atom count, default windows_real*W (a full last window)")
    ap.add_argument("--bias", default="matrix", choices=("matrix", "none"),
                    help="matrix: the -1e9 mask the model builds from the same one-hot. "
                         "none: what the defect would cost if that mask were not there.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from tt_bio.boltz2 import get_indexing_matrix

    kr, kp, D = a.windows_real, a.windows_pad, a.dim
    n_atom = a.atoms_real if a.atoms_real is not None else kr * W
    ki = get_indexing_matrix(kr, W, H, torch.device("cpu"))
    ki = torch.nn.functional.pad(ki, (0, 8 * kp - ki.shape[1], 0, 2 * kp - ki.shape[0]))

    torch.manual_seed(0)
    s = torch.randn(1, kp, W, D).to(torch.bfloat16).float()          # live tail, as in the model
    atom_mask = torch.zeros(1, kp * W)
    atom_mask[:, :n_atom] = 1.0

    def gather(x):                                                   # the matrix's own gather
        b, k, w, d = x.shape
        y = torch.einsum("bjid,jk->bkid", x.double().view(b, 2 * k, w // 2, d), ki.double())
        return y.reshape(b, k, H, d)

    # The two s_kv candidates, on the host, in exact arithmetic. The device scoring of these two
    # is gather_probe.py; here the question is what the attention does with them.
    kv_true = gather(s)
    flat = s.reshape(1, kp * W, D)
    front = H // 2 - W // 2
    padded = torch.nn.functional.pad(flat, (0, 0, front, (kp + H // W) * W - front - kp * W))
    blocks = padded.view(1, kp + H // W, W, D)
    kv_window = torch.cat([blocks[:, c:c + kp] for c in range(H // W)], dim=2).double()

    # The bias the model builds: the same matrix gathers the atom mask, then -1e9 on every zero.
    m = gather(atom_mask.view(1, kp, W, 1)).reshape(1, kp, H)
    bias = ((-m + 1) * -1e9).reshape(1, kp, 1, H)                     # broadcast over the 32 queries
    if a.bias == "none":
        bias = torch.zeros_like(bias)

    q = torch.randn(1, kp, W, D).to(torch.bfloat16).double()
    scale = D ** -0.5

    def attend(kv):
        logits = torch.einsum("bkwd,bkhd->bkwh", q, kv) * scale + bias.double()
        p = torch.softmax(logits, dim=-1)
        return torch.einsum("bkwh,bkhd->bkwd", p, kv)

    o_true, o_window = attend(kv_true), attend(kv_window)
    d_kv = (kv_window - kv_true).abs().amax(dim=(0, 2, 3))
    d_o = (o_window - o_true).abs().amax(dim=(0, 2, 3))
    bad_kv = (d_kv > 0).nonzero().flatten().tolist()
    bad_o = (d_o > 0).nonzero().flatten().tolist()
    res = {
        "windows_real": kr, "windows_pad": kp, "atoms_real": n_atom, "dim": D, "bias": a.bias,
        "host": os.uname().nodename,
        "gather_bad_windows": bad_kv,
        "attention_bad_windows": bad_o,
        "attention_bad_real_windows": [w for w in bad_o if w < kr],
        "gather_max_abs": float(d_kv.max()), "attention_max_abs": float(d_o.max()),
        "per_window_attention_max_abs": [float(x) for x in d_o],
    }
    print(json.dumps({k: res[k] for k in ("windows_real", "atoms_real", "bias",
                                          "attention_bad_real_windows",
                                          "gather_max_abs", "attention_max_abs")}, indent=1),
          flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
