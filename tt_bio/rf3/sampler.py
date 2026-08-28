"""RF3's EDM sampler (AF-3 Algorithm 18), host-side, around a device denoiser.

The loop is arithmetic on [D, L, 3] -- centring, a 3x3 rotation, a translation and two
scalar multiplies -- wrapped around one denoiser call per step. All the cost is the
denoiser, so none of this wants to be on device; the same call this repo already made
for Kabsch alignment.

The RNG order is the load-bearing part. Per step the reference consumes FIVE draws, in
this sequence, and one more before the loop starts:

    (before the loop)  normal (D, L, 3)          initial structure
    1-3.               rand (D) x3               rotation angles theta_x, theta_y, theta_z
    4.                 normal (D, 1, 3)          translation
    5.                 normal X.shape            epsilon

Re-implementing the rotation with a different parameterisation -- a quaternion draw is
the better way to sample a uniform rotation and an obvious "improvement" -- consumes a
different number of draws and desynchronises the stream. Every downstream comparison
then becomes a cross-RNG comparison, which produces plausible structures and an RMSD
that reads as a porting bug. Hence `Draws`: record once from the reference, replay into
the port, and the RNG leaves the comparison entirely.

The rotation and centring helpers are imported from the vendored reference rather than
reimplemented, so the only thing under test is the loop and the denoiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tt_bio._vendor.foundry.utils.rigid import rot_vec_mul
from tt_bio._vendor.foundry.utils.rotation_augmentation import centre


class Draws:
    """Records or replays the sampler's RNG stream.

    `record=True` draws live and remembers; otherwise it hands back what was recorded,
    in order, and asserts the shapes still line up -- a shape mismatch means the call
    sequence changed, which is exactly the failure this exists to catch.
    """

    def __init__(self, values: list[torch.Tensor] | None = None):
        self.record = values is None
        self.values: list[torch.Tensor] = values if values is not None else []
        self._i = 0

    def _take(self, shape, fn):
        if self.record:
            v = fn(shape)
            self.values.append(v.clone())
            return v
        if self._i >= len(self.values):
            raise AssertionError(
                f"draw {self._i} requested but only {len(self.values)} recorded: "
                "the call sequence consumes a different number of draws than the "
                "reference did")
        v = self.values[self._i]
        if tuple(v.shape) != tuple(shape):
            raise AssertionError(
                f"draw {self._i}: recorded {tuple(v.shape)} but asked for "
                f"{tuple(shape)} -- the call sequence has diverged from the reference")
        self._i += 1
        return v

    def rand(self, shape):
        return self._take(shape, lambda s: torch.rand(s))

    def normal(self, shape):
        return self._take(shape, lambda s: torch.normal(mean=0.0, std=1.0, size=s))

    def exhausted(self) -> bool:
        return self.record or self._i == len(self.values)


def uniform_random_rotation(d: int, draws: Draws) -> torch.Tensor:
    """The reference's rotation, draw-for-draw: three rand(D) in x, y, z order."""
    tx = draws.rand((d,)) * 2 * torch.pi
    ty = draws.rand((d,)) * 2 * torch.pi
    tz = draws.rand((d,)) * 2 * torch.pi
    def rots(c, s, axis):
        m = torch.zeros(len(c), 3, 3)
        if axis == 0:
            m[:, 0, 0] = 1; m[:, 1, 1] = c; m[:, 1, 2] = -s; m[:, 2, 1] = s; m[:, 2, 2] = c
        elif axis == 1:
            m[:, 1, 1] = 1; m[:, 0, 0] = c; m[:, 0, 2] = s; m[:, 2, 0] = -s; m[:, 2, 2] = c
        else:
            m[:, 2, 2] = 1; m[:, 0, 0] = c; m[:, 0, 1] = -s; m[:, 1, 0] = s; m[:, 1, 1] = c
        return m
    rx = rots(torch.cos(tx), torch.sin(tx), 0)
    ry = rots(torch.cos(ty), torch.sin(ty), 1)
    rz = rots(torch.cos(tz), torch.sin(tz), 2)
    # rz @ (ry @ rx), matching the reference's association. Left-associating instead
    # is the same rotation and a different fp32 rounding -- worth 3.8e-06 on the
    # trajectory, which is harmless but stops the loop being bit-exact, and bit-exact
    # is the only bar that leaves no room to hide.
    return torch.matmul(rz, torch.matmul(ry, rx))


@dataclass
class DiffusionSampler:
    num_timesteps: int = 200
    min_t: float = 0.0
    max_t: float = 1.0
    sigma_data: float = 16.0
    s_min: float = 4e-4
    s_max: float = 160.0
    p: float = 7.0
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5

    def noise_schedule(self) -> torch.Tensor:
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps)
        return self.sigma_data * (
            self.s_max ** (1 / self.p)
            + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
        ) ** self.p

    def sample(self, denoise, coord_to_be_noised: torch.Tensor, d: int,
               draws: Draws | None = None, s_trans: float = 1.0,
               partial_t: int = 0, progress_fn=None):
        """`denoise(x_noisy [D,L,3], t [D]) -> [D,L,3]` is the ported diffusion module.

        `partial_t` starts the rollout part-way down the schedule, which is upstream's
        `SamplePartialDiffusion` (`inference_sampler.py:206`, a one-line
        `t_hat_full[self.partial_t:]`). It is an INDEX, not a noise level: 0 is the full
        rollout from pure noise and `num_timesteps - 1` is one step away from the input
        structure. Everything else is unchanged, so the initial noising still reads
        `sched[0]` -- of the truncated schedule, which is the whole mechanism.
        """
        if not 0 <= partial_t < self.num_timesteps:
            raise ValueError(
                f"partial_t must be in [0, {self.num_timesteps}), got {partial_t}")
        draws = draws if draws is not None else Draws()
        sched = self.noise_schedule()[partial_t:]
        n_atom = coord_to_be_noised.shape[-2]

        x = sched[0] * draws.normal((d, n_atom, 3)) + coord_to_be_noised
        exists = torch.ones((d, n_atom)).bool()

        for k, (c_prev, c_t) in enumerate(zip(sched, sched[1:])):
            if progress_fn:
                progress_fn("diffusion", step=k, total=len(sched) - 1)
            x = centre(x, exists)
            r = uniform_random_rotation(d, draws)
            x = rot_vec_mul(r[:, None], x) + s_trans * draws.normal((d, 1, 3))

            gamma = self.gamma_0 if c_t > self.gamma_min else 0.0
            t_hat = c_prev * (gamma + 1)
            eps = (self.noise_scale
                   * torch.sqrt(t_hat ** 2 - c_prev ** 2)
                   * draws.normal(tuple(x.shape)))
            x_noisy = x + eps
            x_denoised = denoise(x_noisy, t_hat.tile(d))
            delta = (x_noisy - x_denoised) / t_hat
            x = x_noisy + self.step_scale * (c_t - t_hat) * delta

        return x, draws
