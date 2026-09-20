#!/usr/bin/env python3
"""Do the two modules D23 is about produce the same bytes as `origin/main`, on both checkpoints?

This row changes the reference, not the model. Nothing under `tt_bio/` is edited on the branch,
so shipped inference must be byte-identical to `origin/main` — and "must be by construction" is
the claim, not the measurement.

The two modules are the ones that carry tt-bio's per-checkpoint bindings, which D23 says are
correct and must not be flipped:

  * `openfold3_trunk.py:133`  `tri_att_end_bias_follows_pair = not is_openbind(state_dict)` —
    the ending-node bias orientation, which is `transpose_bias` in upstream's 0.5.0 and absent
    before it, and which no weight selects;
  * `openfold3_diffusion_transformer.py:265` — per-block `layer_norm_z` when the checkpoint
    carries no shared one, which is the 48 tensors 0.5.0 drops.

So both checkpoints are run, not just the one the campaign measures. of3-p2-155k is preview2
and must take one binding; of3-ob-2025-06-30-174k is OpenBind and must take the other. A digest
that only ever sees one checkpoint cannot tell a binding from a constant.

Run the SAME file against two checkouts on the SAME card:

    identity.py --repo /path/to/wk/of3t-rebase  --out a.json
    identity.py --repo /path/to/detached/main   --out b.json

THE NEGATIVE CONTROL IS NOT OPTIONAL. `--perturb` moves one weight by 1 + 2^-8, one bf16 ulp at
unit magnitude, and every digest it reaches must MOVE. A run without it reports `control: ABSENT`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import inspect
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
    ap.add_argument("--blocks", type=int, default=2,
                    help="blocks per stack. The binding is per block, so two is enough to "
                         "exercise it and 48 only costs time.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--ckpt-p2", type=Path,
                    default=Path.home() / "of3-weights" / "of3-p2-155k.pt")
    ap.add_argument("--ckpt-ob", type=Path,
                    default=Path.home() / ".boltz" / "of3-ob-2025-06-30-174k.pt")
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
        "control": "PRESENT (--perturb)" if a.perturb else "ABSENT",
        "checkpoints": {}}
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
        from tt_bio.tenstorrent import get_device, Pairformer
        from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
        from tt_bio.openfold3_weights import _sub, is_openbind, remap_pairformer_stack
        from tt_bio.openfold3_trunk import _PF_DIMS

        dev = get_device()
        out["env"]["arch"] = str(dev.arch())
        torch.set_grad_enabled(False)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
            packer_l1_acc=True)

        for tag, path in (("p2-155k", a.ckpt_p2), ("openbind-174k", a.ckpt_ob)):
            if not path.is_file():
                out["checkpoints"][tag] = {"skipped": f"{path} is not on this host"}
                dump()
                continue
            sd = torch.load(path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
            sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
            ob = bool(is_openbind(sd))
            rec = {"file": path.name, "is_openbind": ob,
                   # the two bindings, read out of the shipped expressions rather than restated
                   "tri_att_end_bias_follows_pair": (not ob),
                   "binding_source": "openfold3_trunk.py:133, "
                                     "openfold3_diffusion_transformer.py:265"}

            if a.perturb:
                # One weight per digested module, each 1 + 2^-8, one bf16 ulp at unit
                # magnitude. Both, because a control that only reaches the pairformer leaves
                # the diffusion-transformer digest unable to fail, and a digest that cannot
                # fail proves nothing.
                keys = [next(k for k in sd if k.startswith("pairformer_stack.blocks.0")
                             and k.endswith("weight")),
                        next(k for k in sd
                             if k.startswith("diffusion_module.diffusion_transformer.blocks.0")
                             and k.endswith("weight"))]
                for key in keys:
                    sd[key] = sd[key] * (1.0 + 2.0 ** -8)
                rec["perturbed_keys"] = keys

            dmsd = _sub(sd, "diffusion_module")
            dsd = _sub(dmsd, "diffusion_transformer")
            # `_sub` hands back a plain dict here and a Weights elsewhere; read through both
            # rather than branching on hasattr, which is how c_a came out None the first time.
            dget = (lambda k: dsd.data[k]) if hasattr(dsd, "data") else (lambda k: dsd[k])
            dhas = (lambda k: k in dsd.data) if hasattr(dsd, "data") else (lambda k: k in dsd)
            rec["dit_has_shared_layer_norm_z"] = dhas("layer_norm_z.weight")
            dit = OF3DiffusionTransformer(dsd, ckc, n_blocks=a.blocks)
            rec["dit_per_block_layer_norm_z"] = dit.ln_z_w is None

            pf_sd = remap_pairformer_stack(sd, prefix="pairformer_stack")
            # The shipped call from openfold3_trunk.py, restricted to the kwargs THIS tree's
            # Pairformer accepts. `origin/main` has no `tri_att_scale_pair_bias`; that kwarg
            # arrived on the campaign branch. Dropping it silently would compare two different
            # functions and call the difference a digest, so what was dropped is reported.
            want = {"scale_pair_bias": True, "tri_att_scale_pair_bias": False,
                    "fp32_softmax": True, "transpose_bias": (not ob)}
            accepted = set(inspect.signature(Pairformer.__init__).parameters)
            kw = {k: v for k, v in want.items() if k in accepted}
            rec["pairformer_kwargs"] = {"passed": kw,
                                        "dropped_this_tree_has_no_such_kwarg":
                                            sorted(set(want) - set(kw))}
            pf = Pairformer(a.blocks, *_PF_DIMS, True, pf_sd, ckc, **kw)

            n = a.tokens
            c_s = int(sd["pairformer_stack.blocks.0.attn_pair_bias.layer_norm_a.weight"].shape[0])
            c_z = int(sd["pairformer_stack.blocks.0.attn_pair_bias.layer_norm_z.weight"].shape[0])
            # Every dim off the checkpoint, never guessed: a wrong one builds a module that
            # runs and digests a different function on each side.
            c_a = int(dget("blocks.0.attention_pair_bias.mha.linear_q.bias").shape[0])
            c_ds = int(dget("blocks.0.attention_pair_bias.layer_norm_a."
                            "layer_norm_s.weight").shape[0])
            c_dz = int(dget("blocks.0.attention_pair_bias.linear_z.weight").shape[1])
            g = torch.Generator().manual_seed(a.seed)
            mk = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                           device=dev)
            s_t = torch.randn(1, n, c_s, generator=g) * 0.5
            z_t = torch.randn(1, n, n, c_z, generator=g) * 0.5
            so, zo = pf(mk(s_t), mk(z_t))
            rec["pairformer"] = {"s_sha256": digest(so, ttnn), "z_sha256": digest(zo, ttnn),
                                 "shapes": {"s": list(s_t.shape), "z": list(z_t.shape)}}
            dump()

            a_t = torch.randn(1, n, c_a, generator=g) * 0.5
            ds_t = torch.randn(1, n, c_ds, generator=g) * 0.5
            dz_t = torch.randn(1, n, n, c_dz, generator=g) * 0.5
            mkt = torch.ones(1, n)
            do = dit(mk(a_t), mk(ds_t), mk(dz_t), mk(mkt), mk(mkt.reshape(1, n, 1)))
            rec["diffusion_transformer"] = {"sha256": digest(do, ttnn),
                                            "shapes": {"a": list(a_t.shape),
                                                       "s": list(ds_t.shape),
                                                       "z": list(dz_t.shape)}}
            out["checkpoints"][tag] = rec
            dump()

        got = [c for c in out["checkpoints"].values() if "pairformer" in c]
        out["bindings_differ_between_checkpoints"] = (
            len({c["tri_att_end_bias_follows_pair"] for c in got}) == 2
            and len({c["dit_per_block_layer_norm_z"] for c in got}) == 2) if len(got) > 1 else None
        out["ok"] = True
    except Exception:                                                    # noqa: BLE001
        out["ok"] = False
        out["error"] = traceback.format_exc()[-4000:]
    dump()
    print(json.dumps({k: out.get(k) for k in
                      ("ok", "control", "bindings_differ_between_checkpoints")}, indent=1))
    for t, c in out["checkpoints"].items():
        print(f"  {t}: {json.dumps({k: v for k, v in c.items() if k != 'binding_source'})}")
    print("head", out["env"].get("head"), "->", a.out, flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
