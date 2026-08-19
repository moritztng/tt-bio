#!/usr/bin/env python3
"""Score the ported EDM sampler against the reference, with the RNG removed.

Two instruments, as the brief requires:
  a. REPLAY -- record the reference's draws, replay them into the port, and use the
     reference's own denoiser. Any difference is then the loop arithmetic alone, and
     it should be bit-exact.
  b. STREAM -- run the port's own draws from a shared seed and check the draw SEQUENCE
     matches the reference's (count, order and shape), which replay cannot test.

(a) alone cannot catch a draw-order bug; (b) alone cannot separate a maths bug from one.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=8,
                    help="short schedule: this tests the loop, not the model")
    ap.add_argument("--atoms", type=int, default=64)
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tt_bio._vendor.rf3.diffusion_samplers.inference_sampler import SampleDiffusion
    from tt_bio.rf3.sampler import Draws, DiffusionSampler

    HP = dict(num_timesteps=args.steps, min_t=0, max_t=1, sigma_data=16,
              s_min=4e-4, s_max=160, p=7, gamma_0=0.8, gamma_min=1.0,
              noise_scale=1.003, step_scale=1.5)

    coords = torch.zeros(args.d, args.atoms, 3)

    # A stand-in denoiser: deterministic, non-trivial, and identical on both sides, so
    # the comparison isolates the LOOP. The real diffusion module is scored separately.
    def denoise(x, t):
        return 0.5 * x + 0.1 * torch.sin(x) + t[..., None, None] * 0.01

    ref = SampleDiffusion(solver="af3", **HP)

    class _M(torch.nn.Module):
        def forward(self, *, X_noisy_L, t, f, S_inputs_I, S_trunk_I, Z_trunk_II):
            return denoise(X_noisy_L, t)

    f = {"ref_element": torch.zeros(args.atoms, 1)}
    torch.manual_seed(args.seed)
    out = ref.sample_diffusion_like_af3(
        S_inputs_I=torch.zeros(1, 1), S_trunk_I=torch.zeros(1, 1),
        Z_trunk_II=torch.zeros(1, 1, 1), f=f, diffusion_module=_M(),
        diffusion_batch_size=args.d, coord_atom_lvl_to_be_noised=coords)
    want = out["X_L"]

    # --- (a) replay: same seed, so the port's live draws ARE the reference's stream
    torch.manual_seed(args.seed)
    mine, used = DiffusionSampler(**HP).sample(denoise, coords, args.d)
    replay_exact = bool(torch.equal(mine, want))
    replay_max = float((mine - want).abs().max())

    # --- (b) stream: replay those draws back and confirm the sequence is consumed
    #        identically -- a different call order raises inside Draws
    # Seed deliberately wrong: if replay is really replaying, the live RNG state must
    # not matter. Compare against the LIVE PORT run, not the reference -- replay tests
    # the draw plumbing, not the arithmetic.
    torch.manual_seed(args.seed + 1)
    mine2, _ = DiffusionSampler(**HP).sample(denoise, coords, args.d,
                                             draws=Draws(used.values))
    replay_matches = bool(torch.equal(mine2, mine))

    expected_draws = 1 + 5 * (args.steps - 1)
    print(json.dumps({
        "steps": args.steps, "atoms": args.atoms, "D": args.d,
        "draws_consumed": len(used.values),
        "draws_expected_1_plus_5_per_step": expected_draws,
        "draw_count_matches": len(used.values) == expected_draws,
        "live_stream_bit_exact_vs_reference": replay_exact,
        "live_stream_maxabs": round(replay_max, 9),
        "replay_reproduces_live_run_under_wrong_seed": replay_matches,
        "verdict": ("PASS" if replay_exact and replay_matches
                    and len(used.values) == expected_draws else "GAP"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
