#!/usr/bin/env python3
"""Is our diffusion transformer wrong, or is it being handed a wrong operand?

Every number so far fits both readings. The bisect validates the DiT's `a` input at 2.63e-03
and `si` inside that same number, but `zij` reaches the DiT as `z` by a different route than
the NPE uses it, so a layout error there would pass the NPE check and still break the DiT.

Two checks, and the predictions are written down BEFORE the run (PROTOCOL A18):

  A. HOST ONLY. The `s` and `z` this row hands our DiT, against their captured `dit_in`. Both
     come from the same captured run, so the prediction is they agree to ~0. A material
     disagreement here IS the mis-wired operand and ends the search.

  B. DEVICE. Our DiT standalone on THEIR exact captured (a, s, z, mask), against their
     `dit_out`. This removes our own `a` (which carries 2.63e-03) from the comparison, so it
     isolates the DiT itself. Prediction: if our DiT is a different function this reads ~7e-02
     like the in-module number; if it reads ~1e-03 the DiT is sound and the in-module gap
     comes from its inputs after all. It also reconciles the refuted size ladder: the ladder
     used synthetic inputs and read 8.2e-01 at a size where our fixture gate is bit-exact, so
     if THIS reads small the ladder's synthetic inputs were the fault and not the call.

Tenancy is stamped into the artifact (amendment item 11). No timing claim is made here.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import torch

sys.path.insert(0, os.getcwd())
CAP = "/home/ttuser/of3t_diffusion_cap"
CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")


def rel(x, y):
    return float(torch.linalg.vector_norm(x.double() - y.double())
                 / (torch.linalg.vector_norm(y.double()) + 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", type=int, default=0)
    ap.add_argument("--device", action="store_true", help="run check B as well")
    ap.add_argument("--out", default="perf/of3t_diffusion/operand_check.json")
    a = ap.parse_args()
    t0 = time.time()

    S = torch.load(f"{CAP}/sub_boundary.pt", map_location="cpu", weights_only=False)
    B = torch.load(f"{CAP}/diffusion_boundary.pt", map_location="cpu", weights_only=False)
    dit_args, dit_kw = S["dit_in"]
    si_all, zij_ref = S["cond_out"][0], S["cond_out"][1]
    k = a.struct
    n_token = int(B["kwargs"]["token_mask"].shape[-1])

    rep = {"struct": k, "n_token": n_token,
           "tenancy": subprocess.run(
               ["bash", "-lc", "pgrep -af TT_VISIBLE_DEVICES | grep -v pgrep | wc -l"],
               capture_output=True, text=True).stdout.strip(),
           "shapes": {"dit_in_a": list(dit_kw["a"].shape),
                      "dit_in_s": list(dit_kw["s"].shape),
                      "dit_in_z": list(dit_kw["z"].shape),
                      "cond_si": list(si_all.shape), "cond_zij": list(zij_ref.shape)}}
    print(json.dumps(rep["shapes"], indent=1), flush=True)

    # ---- A. the operands this row builds, against their captured DiT inputs ------------------
    my_s = si_all[0, k]                       # what we pass as `si`
    their_s = dit_kw["s"]
    their_s_k = their_s[0, k] if their_s.dim() >= 3 and their_s.shape[1] == si_all.shape[1] \
        else their_s.reshape(their_s.shape[-2], their_s.shape[-1])
    my_z = zij_ref.reshape(n_token, n_token, -1)
    their_z = dit_kw["z"].reshape(n_token, n_token, -1)
    rep["A_operands"] = {
        "s_rel": rel(my_s, their_s_k), "z_rel": rel(my_z, their_z),
        "s_shapes": [list(my_s.shape), list(their_s_k.shape)],
        "z_shapes": [list(my_z.shape), list(their_z.shape)],
        "verdict": None}
    rep["A_operands"]["verdict"] = (
        "operands agree; the DiT is handed the right s and z"
        if max(rep["A_operands"]["s_rel"], rep["A_operands"]["z_rel"]) < 1e-12
        else "OPERAND MISMATCH -- this is the mis-wired operand")
    print(f"[{time.time()-t0:.0f}s] A: s {rep['A_operands']['s_rel']:.3e}  "
          f"z {rep['A_operands']['z_rel']:.3e}  -> {rep['A_operands']['verdict']}", flush=True)

    # ---- B. our DiT on THEIR exact inputs -----------------------------------------------------
    if a.device:
        import ttnn
        from tt_bio.tenstorrent import get_device, device_dtype_override
        from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer
        from tt_bio.openfold3_weights import _sub
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {(kk[6:] if kk.startswith("model.") else kk): v for kk, v in sd.items()}
        dmsd = _sub(sd, "diffusion_module")
        dev = get_device()
        cfg = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        act = ttnn.float32
        ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
        with device_dtype_override(act):
            dit = OF3DiffusionTransformer(_sub(dmsd, "diffusion_transformer"), cfg)
        their_a = dit_kw["a"][0, k]
        mask = dit_kw["mask"].reshape(-1)[:n_token]
        with device_dtype_override(act):
            out = dit(ft(their_a.reshape(1, n_token, -1)),
                      ft(their_s_k.reshape(1, n_token, -1)),
                      ft(their_z.reshape(1, n_token, n_token, -1)),
                      ft(mask.reshape(1, n_token)), ft(mask.reshape(1, n_token, 1)))
        ours = ttnn.to_torch(out).reshape(n_token, -1)
        theirs = S["dit_out"][0, k]
        r = rel(ours, theirs.reshape(n_token, -1))
        m = mask.bool()
        o2, t2 = ours.double(), theirs.reshape(n_token, -1).double()
        rep["B_dit_on_their_inputs"] = {
            "rel": r, "rel_real": rel(o2[m], t2[m]), "rel_pad": rel(o2[~m], t2[~m]),
            "in_module_rel_with_our_a": 7.349989198116111e-02,
            "ladder_synthetic_rel_n384": 8.761760e-01,
            "verdict": ("our DiT computes a different function" if r > 1e-2
                        else "our DiT is sound on their inputs; the in-module gap is its inputs")}
        print(f"[{time.time()-t0:.0f}s] B: {r:.6e} (real {rep['B_dit_on_their_inputs']['rel_real']:.3e}, "
              f"pad {rep['B_dit_on_their_inputs']['rel_pad']:.3e}) -> "
              f"{rep['B_dit_on_their_inputs']['verdict']}", flush=True)

    json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True, default=str)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
