"""PXDesign-d generator: device cost of the diffusion sampler on one Blackhole p150a.

PXDesign-d is Protenix's own `DiffusionModule` under a design-conditioning wrapper, at
depths 16 DiT / 4 atom-encoder / 4 atom-decoder and c_z=128 (read off
`pxdesign_v0.1.0.pt`; see state/pxdesign-port.md section 2). tt-bio already runs that
module on device, and since commit bd2b067d it takes those depths from the weights, so
the generator's arithmetic is buildable here today even though the surrounding pipeline
(host featurizer, eval half) is not ported yet.

What this measures: real checkpoint weights, real depths, real shapes, synthetic
conditioning VALUES. The diffusion sampler has no data-dependent control flow -- step
count, tensor shapes, window blocking and the op sequence are fixed by (n_tokens,
n_atoms, n_step, multiplicity) -- so device time is a function of shapes and weights,
not of the numbers inside the conditioning. Parity is the port's gate, not this one's;
nothing here is an accuracy claim.

Cells come from the GPU reference's own fixtures: target atoms counted from
perf/pxdesign/targets/laczc_*.cif (~8.0 heavy atoms/residue), binder 80 residues of the
`xpb` placeholder, whose atom set is GLY's (N/CA/C/O) per pxdesign
data/constants.py RES_ATOMS_DICT, so 4 atoms/residue.

Timing: a short run measures the per-step cost with one sync at the end of the sampler
(isolated per-op timing over-syncs and inflates cost ~2x). --check-linearity runs two
step counts and reports the fit, so the extrapolation to N_step=400 is measured rather
than assumed.
"""
import argparse, json, os, time

import torch
import ttnn

# Cells: name -> (target_aa, target_atoms). Binder is 80 residues x 4 atoms.
LADDER = {
    "laczc128": (128, 1024),
    "laczc256": (256, 2083),
    "laczc512": (512, 4108),
    "laczc768": (768, 6135),
}
BINDER_AA = 80
BINDER_ATOMS_PER_RES = 4


def make_cond(NT, N, c_s=384, c_s_inputs=449, c_z=128, c_atom=128, c_atompair=16,
              nq=32, nk=128, seed=0):
    """Shape-faithful conditioning, exactly the dict DiffusionModule.denoise documents.

    s_trunk is zeros because that is what PXDesign-d actually passes (ProtenixDesign sets
    s_trunk = zeros(c_s); it has no trunk). Everything else is random at the right shape.
    """
    g = torch.Generator().manual_seed(seed)
    NP = ((N + nq - 1) // nq) * nq
    nb = NP // nq
    # atom -> token onehot, contiguous runs of atoms per token (the real mapping's shape)
    S = torch.zeros(N, NT)
    per = [BINDER_ATOMS_PER_RES] * NT
    # distribute atoms over tokens, keeping the binder's 4-atom tokens last
    n_binder_tok = BINDER_AA
    n_tgt_tok = NT - n_binder_tok
    tgt_atoms = N - n_binder_tok * BINDER_ATOMS_PER_RES
    base, extra = divmod(tgt_atoms, n_tgt_tok)
    per = [base + (1 if i < extra else 0) for i in range(n_tgt_tok)] + \
          [BINDER_ATOMS_PER_RES] * n_binder_tok
    a = 0
    for t, cnt in enumerate(per):
        S[a:a + cnt, t] = 1.0
        a += cnt
    assert a == N, (a, N)
    return {
        "s_trunk": torch.zeros(NT, c_s),
        "s_inputs": torch.randn(NT, c_s_inputs, generator=g) * 0.1,
        "pair_z": torch.randn(NT, NT, c_z, generator=g) * 0.1,
        "c_l": torch.randn(N, c_atom, generator=g) * 0.1,
        "p_lm": torch.randn(nb, nq, nk, c_atompair, generator=g) * 0.1,
        "S": S,
        "mask_trunked": torch.ones(nb, nq, nk),
    }


def census():
    """Every tt_bio lever counter currently loaded, read in-process.

    `scripts/lever_census.py` is the org instrument for this, but it drives `tt-bio
    predict` and PXDesign has no CLI path yet. Its counters are plain module-level
    `[served, declined]` lists though, and this bench folds in the process it runs in (no
    multiprocessing spawn), so they can be read directly. Reported per cell, so a gate that
    admits work at 128 aa and silently declines at 768 aa shows up as a decline count that
    moves with size -- the failure mode of
    `tt-bio-tuned-at-512-l1-gates-go-dark-above-640aa`.
    """
    import sys
    out = {}
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("tt_bio.") or mod is None:
            continue
        for attr in dir(mod):
            if not attr.endswith("_STATS"):
                continue
            try:
                v = getattr(mod, attr)
            except Exception:
                continue
            if isinstance(v, (list, tuple)) and v and all(isinstance(x, int) for x in v):
                out[mod_name + "." + attr] = list(v)
            elif isinstance(v, dict):
                out[mod_name + "." + attr] = {str(k): v[k] for k in v}
    return out


def build(ckpt, device, ckc, fp32=None):
    from tt_bio.protenix import DiffusionModule, n_blocks
    raw = torch.load(ckpt, map_location="cpu")
    sd = raw["model"] if "model" in raw else raw
    pref = "module.diffusion_module."
    dm = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
    assert dm, f"no {pref}* keys in {ckpt}"
    depths = dict(
        dit=n_blocks(dm, "diffusion_transformer"),
        atom_enc=n_blocks(dm, "atom_attention_encoder.atom_transformer.diffusion_transformer"),
        atom_dec=n_blocks(dm, "atom_attention_decoder.atom_transformer.diffusion_transformer"),
    )
    mod = DiffusionModule(dm, device, ckc, diffusion_fp32=fp32)
    return mod, depths


def run_sampler(mod, cond, N, n_step, mult, trace, mps):
    from tt_bio.protenix import edm_sample
    ttnn.synchronize_device(mod.dev)
    t0 = time.perf_counter()
    out = edm_sample(mod, cond, N, n_step=n_step, multiplicity=mult,
                     max_parallel_samples=mps, seed=42, trace=trace)
    ttnn.synchronize_device(mod.dev)
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/pxd_ckpt/pxdesign_v0.1.0.pt"))
    ap.add_argument("--cells", nargs="+", default=["laczc128"])
    ap.add_argument("--n-step", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=2,
                    help="throwaway sampler pass before timing (device program cache + "
                         "first-iteration cost land here, not in the fit)")
    ap.add_argument("--n-step-2", type=int, default=0,
                    help="second step count for the linearity check (0 = skip)")
    ap.add_argument("--mult", type=int, nargs="+", default=[1])
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--mps", type=int, default=None, help="max_parallel_samples")
    ap.add_argument("--fp32", type=int, default=None, help="1/0 override of the diffusion dtype")
    ap.add_argument("--target-n-step", type=int, default=400,
                    help="N_step the extrapolation reports (PXDesign's own default)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    from tt_bio.tenstorrent import get_device
    ensure_p300_mesh_descriptor()
    trs = (1 << 30) if args.trace else 0
    device = get_device(trace_region_size=trs)
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    fp32 = None if args.fp32 is None else bool(args.fp32)
    mod, depths = build(args.ckpt, device, ckc, fp32=fp32)
    rec = {"chip": "p150a (Blackhole)", "ckpt": os.path.basename(args.ckpt),
           "depths": depths, "diffusion_fp32": mod._diffusion_fp32,
           "dtype": str(mod.dtype), "trace": args.trace,
           "target_n_step": args.target_n_step, "cells": []}
    print(f"[pxd] depths={depths} fp32={mod._diffusion_fp32} dtype={mod.dtype}", flush=True)

    for name in args.cells:
        tgt_aa, tgt_atoms = LADDER[name]
        NT = tgt_aa + BINDER_AA
        N = tgt_atoms + BINDER_AA * BINDER_ATOMS_PER_RES
        for M in args.mult:
            cell = {"cell": name, "target_aa": tgt_aa, "n_tokens": NT, "n_atoms": N,
                    "multiplicity": M, "trace": args.trace,
                    "max_parallel_samples": args.mps}
            cond = make_cond(NT, N)
            print(f"[pxd] {name} NT={NT} N={N} M={M} n_step={args.n_step} ...", flush=True)
            try:
                # the t-independent conditioning is hoisted once per fold; time it separately
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                mod._atom_cond(cond)
                if mod.device_dit:
                    cond["dit_z"] = mod._dit_z_device(cond["pair_z"])
                ttnn.synchronize_device(device)
                cell["cond_hoist_s"] = time.perf_counter() - t0

                if args.warmup_steps:
                    tw, _ = run_sampler(mod, cond, N, args.warmup_steps, M, args.trace, args.mps)
                    cell["warmup_s"] = tw
                    cell["warmup_steps"] = args.warmup_steps
                t_a, _ = run_sampler(mod, cond, N, args.n_step, M, args.trace, args.mps)
                cell["n_step_a"] = args.n_step
                cell["sampler_s_a"] = t_a
                if args.n_step_2:
                    t_b, _ = run_sampler(mod, cond, N, args.n_step_2, M, args.trace, args.mps)
                    cell["n_step_b"] = args.n_step_2
                    cell["sampler_s_b"] = t_b
                    # measured slope + intercept, so the extrapolation is a fit not a guess
                    slope = (t_b - t_a) / (args.n_step_2 - args.n_step)
                    icept = t_a - slope * args.n_step
                    cell["per_step_s"] = slope
                    cell["fixed_s"] = icept
                    cell["extrap_s"] = icept + slope * args.target_n_step
                else:
                    cell["per_step_s"] = t_a / args.n_step
                    cell["extrap_s"] = cell["per_step_s"] * args.target_n_step
                cell["extrap_s_per_design"] = cell["extrap_s"] / M
                # Positive proof that trace actually engaged: DiffusionModule._trace is only
                # set by _capture_trace. denoise_traced falls back silently when the
                # device_dit bias path is missing, so a trace A/B that never captured would
                # otherwise read as "trace buys nothing".
                cell["lever_census"] = census()
                _tr = getattr(mod, "_trace", None)
                cell["trace_captured"] = _tr is not None
                if _tr is not None:
                    cell["trace_id"] = int(_tr["tid"]); cell["trace_n"] = int(_tr["N"])
                if args.trace and M == 1 and not cell["trace_captured"]:
                    raise AssertionError("trace=True but no trace was captured -- "
                                         "denoise_traced fell back to the untraced path")
                cell["ok"] = True
                print(f"[pxd]   {json.dumps({k: cell[k] for k in cell if k != 'ok'})}", flush=True)
            except Exception as e:
                cell["ok"] = False
                cell["error"] = f"{type(e).__name__}: {e}"
                print(f"[pxd]   FAILED {cell['error']}", flush=True)
            rec["cells"].append(cell)
            # free the resident conditioning before the next cell
            for k in list(cond):
                if isinstance(cond[k], ttnn.Tensor):
                    ttnn.deallocate(cond[k])
                cond.pop(k)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[pxd] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
