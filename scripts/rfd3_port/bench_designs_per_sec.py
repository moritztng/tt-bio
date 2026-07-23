"""p23: real designs/sec throughput measurement for the RFD3 port.

Runs the real end-to-end path (featurize a real PDB -> on-device
TokenInitializer -> RFD3Sampler EDM loop over the real ttnn DiffusionModule)
for the p12/p21-verified IAI_protein.pdb + "A1-10,20,A31-40" fixture, at a
few different `num_timesteps`, so per-step device-forward cost can be
isolated from the one-time TokenInitializer/weight-load cost (step-count
method: per_step = (t(n2) - t(n1)) / (n2 - n1); trunk = t(n1) - n1*per_step).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/bench_designs_per_sec.py
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification
from tt_bio.rfd3_sampler import RFD3Sampler

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
CONTIG = "A1-10,20,A31-40"
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def main():
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    print(f"[setup] I={f['restype'].shape[0]} L={L} ({int(is_motif.sum())} motif atoms)")

    t0 = time.time()
    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    print(f"[setup] weight load + device bring-up: {time.time() - t0:.2f}s")

    coord0 = f["motif_pos"].float().unsqueeze(0)

    def run(n_ts, label):
        sampler = RFD3Sampler(num_timesteps=n_ts)
        with torch.no_grad():
            t_init0 = time.time()
            init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
            t_init = time.time() - t_init0
            g = torch.Generator().manual_seed(42)
            t0 = time.time()
            X, _ = sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif, generator=g)
            t_sample = time.time() - t0
        print(f"[{label}] num_timesteps={n_ts} token_init={t_init:.3f}s sample={t_sample:.3f}s "
              f"total={t_init + t_sample:.3f}s ({(n_ts - 1)} device-forward steps)")
        return t_init, t_sample

    # cold (first shapes seen this process -> pays full kernel-compile cost)
    run(4, "cold")
    # warm, small N
    ti_w, n8 = run(8, "warm")
    # warm, larger N (same shapes already compiled -> isolates per-step cost)
    ti_w2, n40 = run(40, "warm")

    per_step = (n40 - n8) / (40 - 8)
    trunk = n8 - 7 * per_step  # token-init + 1st-step fixed overhead baked into "sample" itself is ~0 (init timed separately)
    print(f"\n[step-count] per_step={per_step * 1000:.1f} ms/step  "
          f"(from warm 8-step={n8:.3f}s vs 40-step={n40:.3f}s)")

    for n_full in (50, 200):
        est_sample = (n_full - 1) * per_step + max(0.0, n8 - 7 * per_step)
        est_total = ti_w2 + est_sample
        print(f"[estimate] num_timesteps={n_full}: sample~{est_sample:.2f}s total~{est_total:.2f}s "
              f"-> {1.0 / est_total:.4f} designs/sec ({est_total:.1f}s/design)")


if __name__ == "__main__":
    main()
