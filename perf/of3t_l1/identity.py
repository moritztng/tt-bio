#!/usr/bin/env python3
"""Do the two changed OF3 modules produce the same bytes as `origin/main`?

The tape changes in this row are inert without a tape -- `ops.checkpoint_segment` with no hook
installed is `fn(*inputs)` -- but one change is NOT: `MSAModuleBlock` stopped writing its own
`m`, so the shipped inference path now runs `add_(upd, m)` where it ran `add_(m, upd)`. bf16
addition is commutative and the module's own comment said so before this row touched it, which
is an argument. This is the measurement.

Run the SAME file against two checkouts on the SAME card and compare sha256:

    identity.py --repo /path/to/wk/of3t-l1     --out a.json
    identity.py --repo /path/to/detached/main  --out b.json

`--repo` is why this file lives outside both trees: a harness that only exists on one side
cannot run on the other, and copying it in makes the two sides two files.

THE NEGATIVE CONTROL IS NOT OPTIONAL. A digest comparison that cannot fail proves nothing, and
"which check fails if our module is replaced by zeros?" is this campaign's standing question.
`--perturb` scales one weight by 1 + 2^-8 -- one bf16 ulp at unit magnitude, the smallest change
the dtype can carry -- and the digest must MOVE. A run without it reports `control: ABSENT`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path


def digest(t, ttnn):
    import torch
    x = ttnn.to_torch(t)
    return hashlib.sha256(x.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--depth", type=int, default=16, help="MSA rows")
    ap.add_argument("--templates", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", action="store_true",
                    help="the negative control: move one weight by one bf16 ulp")
    ap.add_argument("--ckpt", type=Path,
                    default=Path.home() / ".boltz" / "of3-p2-155k.pt")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    repo = a.repo.resolve()
    sys.path.insert(0, str(repo))

    out = {"argv": sys.argv[1:], "repo": str(repo), "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "head": os.popen(f"git -C {repo} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {repo} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "control": "PRESENT (--perturb)" if a.perturb else "ABSENT"}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    try:
        import torch
        import ttnn
        import tt_bio
        out["env"]["tt_bio_file"] = tt_bio.__file__
        if not tt_bio.__file__.startswith(str(repo)):
            raise SystemExit(f"tt_bio came from {tt_bio.__file__}, not {repo} -- an editable "
                             f"install shadowed the checkout and the comparison would be "
                             f"between one tree and itself")
        from tt_bio.tenstorrent import get_device
        from tt_bio.openfold3_msa_embedder import MSAModule
        from tt_bio.openfold3_template import TemplatePairStack
        from tt_bio.openfold3_weights import _sub

        dev = get_device()
        out["env"]["arch"] = str(dev.arch())
        torch.set_grad_enabled(False)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
            packer_l1_acc=True)
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        if a.perturb:
            # One weight, one bf16 ulp at unit magnitude. Named in the artifact so the control
            # is auditable rather than asserted.
            key = next(k for k in sd if k.startswith("msa_module.blocks.0")
                       and k.endswith("weight"))
            sd[key] = sd[key] * (1.0 + 2.0 ** -8)
            out["perturbed_key"] = key

        n, d, nt = a.tokens, a.depth, a.templates
        msa = MSAModule(sd, ckc)
        # The stack lives under  in the checkpoint, so the
        # sub-dict is what its remap expects; handed the whole dict it finds no
        # blocks and raises on an empty max().
        tps = TemplatePairStack(_sub(sd, "template_embedder"), ckc)
        # Dims off the checkpoint, never guessed: a wrong c_z builds a module that runs and
        # digests a different function on each side.
        c_m = int(sd["msa_module_embedder.linear_m.weight"].shape[0])
        c_z = int(sd["msa_module.blocks.0.outer_product_mean.linear_out.bias"].shape[0])
        c_t = int(tps.ln_w.shape[-1])

        g = torch.Generator().manual_seed(a.seed)
        m_t = torch.randn(1, d, n, c_m, generator=g) * 0.5
        z_t = torch.randn(1, n, n, c_z, generator=g) * 0.5
        t_t = torch.randn(nt, n, n, c_t, generator=g) * 0.5
        mk = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                       device=dev)
        out["shapes"] = {"m": list(m_t.shape), "z": list(z_t.shape), "t": list(t_t.shape)}

        mo, zo = msa(mk(m_t), mk(z_t))
        out["msa_module"] = {"m_sha256": digest(mo, ttnn), "z_sha256": digest(zo, ttnn)}
        dump()
        to = tps(mk(t_t))
        out["template_pair_stack"] = {"sha256": digest(to, ttnn)}
        out["ok"] = True
    except Exception:                                                    # noqa: BLE001
        out["ok"] = False
        out["error"] = traceback.format_exc()[-4000:]
    dump()
    print(json.dumps({k: out.get(k) for k in
                      ("ok", "control", "msa_module", "template_pair_stack", "perturbed_key")},
                     indent=1), flush=True)
    print("head", out["env"].get("head"), "->", a.out, flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
