#!/usr/bin/env python3
"""Where inside ONE DiT block does our diffusion transformer stop computing their function?

`device_gradient_maskbisect.json` already localises the diffusion gap to the DiT as a whole:
everything upstream of it agrees (`ai_into_dit` 2.626e-03, `ql_encoder` 2.460e-03,
`plm_encoder` 3.246e-03) and `ai_out_of_dit` is 7.350e-02. `dit_conditioning.json`'s own
perturbation control says the DiT DAMPS an input error by 0.438x, so 2.6e-03 in cannot make
7.4e-02 out. The defect is inside the stack.

This is pass 47's method one level down: run ONE block on their captured activations and read
the two internal boundaries.

    a1 = a  + attention_pair_bias(a, s, z, mask)
    a2 = a1 + conditioned_transition(a1, s, mask)

Both RESIDUAL DELTAS are reported beside the running activations, because the residual stream
carries most of the norm and a rel on `a1` divides a sub-module's error by an activation the
sub-module did not produce. The delta is the sub-module's own output; that is the number that
says which one is wrong.

Ours is instrumented without touching shipped code: `_DiTBlock` calls `self.adaln_t(a, s)` as
its first act after the attention residual, so wrapping that attribute captures `a1` exactly
where the block produces it.

One hypothesis is settled by reading and re-checked by the numbers below. The brief's named
lead was whether `DiffusionAttentionPairBias` inherited the trunk's `scale_pair_bias` defect
(D1: OF3's token-level attention folds the bias INSIDE the sqrt(d) scale, so a bias added
outside arrives at 1/sqrt(d) of its value). It did not. Upstream scales the query --
`primitives/attention.py:321`, `q /= math.sqrt(self.c_hidden)` -- and then adds the bias to the
already-scaled scores at line 161; our block does `scale_add(sc, HEAD_DIM ** -0.5, zb)`, which
is `x * scale + bias` (eltwise_fusion.py:55), the same order. If that convention were wrong the
attention delta below would be off by ~1/sqrt(48) = 0.144, so this run measures the reading
rather than resting on it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/ttuser/of3t_gradients/ref")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "of3t_gradients"))

CAP = "/home/ttuser/of3t_diffusion_cap"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT_DIR = "perf/of3t_rebase"


def rel(x, y):
    return float(torch.linalg.vector_norm(x.double() - y.double())
                 / (torch.linalg.vector_norm(y.double()) + 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--struct", type=int, default=0)
    ap.add_argument("--block", type=int, default=0, help="which DiT block to open up")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()

    import bundle_min as BM
    import openfold3
    import ttnn

    # D23/R126. This instrument is only as good as the revision the reference is built from, and
    # the 0.5.0 reading it replaces was taken on a tree upstream's own registry refuses to load
    # this checkpoint into. The revision is an input, so it is recorded beside the numbers.
    ref_tree = Path(openfold3.__file__).resolve().parent.parent
    ref_version = "unknown"
    for meta in sorted(ref_tree.glob("*.dist-info/METADATA")) + [ref_tree / "PKG-INFO"]:
        if meta.is_file():
            import re as _re
            m = _re.search(r"^Version:\s*(\S+)", meta.read_text(errors="replace"), _re.M)
            if m:
                ref_version = m.group(1)
                break
    bb = (ref_tree / "openfold3/core/model/latent/base_blocks.py").read_text()
    apb_src = (ref_tree / "openfold3/core/model/layers/attention_pair_bias.py").read_text()
    reference_revision = {
        "tree": str(ref_tree), "version": ref_version,
        "transpose_bias_true_in_base_blocks": bb.count("transpose_bias=True"),
        "has_DiffusionAttentionPairBias": "class DiffusionAttentionPairBias" in apb_src,
    }
    print(f"[{time.time()-t0:.0f}s] reference revision {ref_version} "
          f"({reference_revision['tree']})", flush=True)
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
    from tt_bio.openfold3_weights import _sub

    S = torch.load(f"{CAP}/sub_boundary.pt", map_location="cpu", weights_only=False)
    _, dit_kw = S["dit_in"]
    k = a.struct
    aa_full = dit_kw["a"][0, k].double()
    ss_full = dit_kw["s"][0, k].double() if dit_kw["s"].dim() >= 3 and \
        dit_kw["s"].shape[1] == dit_kw["a"].shape[1] else \
        dit_kw["s"].reshape(dit_kw["s"].shape[-2], -1).double()
    n_full = aa_full.shape[0]
    zz_full = dit_kw["z"].reshape(n_full, n_full, -1).double()
    mk_full = dit_kw["mask"].reshape(-1)[:n_full].double()
    n = a.n
    aa, ss, zz, mk = aa_full[:n], ss_full[:n], zz_full[:n, :n], mk_full[:n]

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(kk[6:] if kk.startswith("model.") else kk): v for kk, v in sd.items()}
    dmsd = _sub(sd, "diffusion_module")

    built = BM.build(torch.float64, 20260919, "cpu", num_recycles=0)
    model = built[1]
    _inc = model.load_state_dict({kk: (v.to(torch.float64)
                                       if torch.is_tensor(v) and v.is_floating_point() else v)
                                  for kk, v in sd.items()}, strict=False)
    reference_revision["load"] = {"missing": len(_inc.missing_keys),
                                  "unexpected": len(_inc.unexpected_keys),
                                  "missing_keys": sorted(_inc.missing_keys)[:8]}
    if _inc.unexpected_keys:
        raise SystemExit(
            f"KEY GATE FAILED: {len(_inc.unexpected_keys)} checkpoint tensors have nowhere to go "
            f"in this reference and are dropped silently -- the measurement below would be a "
            f"statement about the reference, not about our block (D23/R126). First four: "
            f"{sorted(_inc.unexpected_keys)[:4]}")
    ref_dit = model.diffusion_module.diffusion_transformer
    full_blocks = list(ref_dit.blocks)
    B = a.block

    # ---- theirs, with the two residual deltas hooked ----------------------------------------
    grab = {}
    blk = full_blocks[B]
    h1 = blk.attention_pair_bias.register_forward_hook(
        lambda m, i, o: grab.__setitem__("d_apb", o.detach().reshape(n, -1).double()))
    h2 = blk.conditioned_transition.register_forward_hook(
        lambda m, i, o: grab.__setitem__("d_ct", o.detach().reshape(n, -1).double()))
    # Both residual branches read `s` through an AdaLN and gate on it. If the block's two
    # branches miss the bar by the same margin, the shared conditioning path is the first
    # thing to exonerate or convict, so it gets its own boundary.
    h3 = blk.attention_pair_bias.layer_norm_a.register_forward_hook(
        lambda m, i, o: grab.__setitem__("adaln_a", o.detach().reshape(n, -1).double()))
    h4 = blk.conditioned_transition.layer_norm.register_forward_hook(
        lambda m, i, o: grab.__setitem__("adaln_t", o.detach().reshape(n, -1).double()))
    ref_dit.blocks = torch.nn.ModuleList([blk])
    with torch.no_grad(), BM.no_autocast():
        ref = ref_dit(a=aa.reshape(1, 1, n, -1), s=ss.reshape(1, 1, n, -1),
                      z=zz.reshape(1, 1, n, n, -1), mask=mk.reshape(1, 1, n),
                      _mask_trans=True, use_high_precision_attention=True)
    h1.remove(); h2.remove(); h3.remove(); h4.remove()
    their_out = ref.reshape(n, -1).double()
    their_d_apb = grab["d_apb"]
    their_a1 = aa + their_d_apb
    their_d_ct = grab["d_ct"]
    print(f"[{time.time()-t0:.0f}s] theirs: |a| {float(aa.norm()):.4g} "
          f"|d_apb| {float(their_d_apb.norm()):.4g} |d_ct| {float(their_d_ct.norm()):.4g}",
          flush=True)

    # ---- ours, with a1 captured where the block produces it ---------------------------------
    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)

    with device_dtype_override(act):
        our_dit = OF3DiffusionTransformer(_sub(dmsd, "diffusion_transformer"), cfg, n_blocks=1)
        # block B's weights, not block 0's, when B != 0
        if B != 0:
            from tt_bio.openfold3_diffusion_transformer import _DiTBlock
            our_dit.blocks = [_DiTBlock(_sub(_sub(dmsd, "diffusion_transformer"),
                                             f"blocks.{B}"), cfg,
                                        norm_z=our_dit.ln_z_w is None)]
        blk0 = our_dit.blocks[0]
        inner_t, inner_a = blk0.adaln_t, blk0.adaln_a
        box = {}

        class _TapT:
            def __call__(self, x, s):
                box["a1"] = ttnn.to_torch(x).reshape(n, -1).double()
                o = inner_t(x, s)
                box["adaln_t"] = ttnn.to_torch(o).reshape(n, -1).double()
                return o

        class _TapA:
            def __call__(self, x, s):
                o = inner_a(x, s)
                box["adaln_a"] = ttnn.to_torch(o).reshape(n, -1).double()
                return o

        blk0.adaln_t, blk0.adaln_a = _TapT(), _TapA()
        out = our_dit(ft(aa.reshape(1, n, -1)), ft(ss.reshape(1, n, -1)),
                      ft(zz.reshape(1, n, n, -1)), ft(mk.reshape(1, n)),
                      ft(mk.reshape(1, n, 1)))
        blk0.adaln_t, blk0.adaln_a = inner_t, inner_a
    our_out = ttnn.to_torch(out).reshape(n, -1).double()
    our_a1 = box["a1"]
    our_d_apb = our_a1 - aa
    our_d_ct = our_out - our_a1

    m = mk.bool()
    m0 = m
    rows = {
        "adaln_a_out": {"rel": rel(box["adaln_a"], grab["adaln_a"]),
                        "rel_real": rel(box["adaln_a"][m0], grab["adaln_a"][m0]),
                        "our_norm": float(box["adaln_a"].norm()),
                        "their_norm": float(grab["adaln_a"].norm())},
        "adaln_t_out": {"rel": rel(box["adaln_t"], grab["adaln_t"]),
                        "rel_real": rel(box["adaln_t"][m0], grab["adaln_t"][m0]),
                        "our_norm": float(box["adaln_t"].norm()),
                        "their_norm": float(grab["adaln_t"].norm())},
        "a1_running": {"rel": rel(our_a1, their_a1), "rel_real": rel(our_a1[m], their_a1[m])},
        "d_attention_pair_bias": {"rel": rel(our_d_apb, their_d_apb),
                                  "rel_real": rel(our_d_apb[m], their_d_apb[m]),
                                  "our_norm": float(our_d_apb.norm()),
                                  "their_norm": float(their_d_apb.norm())},
        "d_conditioned_transition": {"rel": rel(our_d_ct, their_d_ct),
                                     "rel_real": rel(our_d_ct[m], their_d_ct[m]),
                                     "our_norm": float(our_d_ct.norm()),
                                     "their_norm": float(their_d_ct.norm())},
        "block_out": {"rel": rel(our_out, their_out), "rel_real": rel(our_out[m], their_out[m])},
    }
    for nm, r in rows.items():
        extra = ""
        if "our_norm" in r:
            extra = f"   |ours| {r['our_norm']:.5g} |theirs| {r['their_norm']:.5g}"
        print(f"{nm:28s} rel {r['rel']:.6e}   real-rows {r['rel_real']:.6e}{extra}", flush=True)

    ref_dit.blocks = torch.nn.ModuleList(full_blocks)
    rep = {"instrument": "one DiT block opened at its two residual boundaries",
           "reference_revision": reference_revision,
           "block": B, "n": n, "struct": k,
           "inputs": "their captured dit_in activations, byte-identical to both sides",
           "dtype_device": str(act), "dtype_reference": "float64",
           "rows": rows,
           "scale_convention": {
               "upstream": "primitives/attention.py:321 q /= sqrt(c_hidden), then line 161 "
                           "scores += bias -- bias added OUTSIDE the scale",
               "ours": "scale_add(sc, HEAD_DIM ** -0.5, zb) == x * scale + bias "
                       "(eltwise_fusion.py:55) -- the same order",
               "D1_twin": "refuted; a wrong convention here would show as a ~0.144 = "
                          "1/sqrt(48) error on d_attention_pair_bias"},
           "tenancy": subprocess.run(
               ["bash", "-lc", "pgrep -af TT_VISIBLE_DEVICES | grep -v pgrep | wc -l"],
               capture_output=True, text=True).stdout.strip()}
    os.makedirs(OUT_DIR, exist_ok=True)
    dst = a.out or os.path.join(OUT_DIR, f"dit_block_bisect_b{B}_n{n}.json")
    json.dump(rep, open(dst, "w"), indent=1)
    print("wrote", dst, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
