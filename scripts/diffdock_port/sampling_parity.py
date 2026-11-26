"""Grade ``tt_bio.diffdock.sampling`` against upstream's own ``utils.sampling.sampling``.

Run in the reference environment, which has torch_geometric and the upstream clone:

    cd ~/scratch/dfd/upstream && \
      PYTHONPATH=.:/home/moritz/.coworker/wt/dfd-sampling \
      ~/scratch/dfd/venv-ref/bin/python \
      /home/moritz/.coworker/wt/dfd-sampling/scripts/diffdock_port/sampling_parity.py

The reference is upstream's loop itself. The score network is replaced on both sides by the
same deterministic stub -- a smooth function of the pose and of t that draws no randomness --
so what is graded is exactly what this row owns: the t schedule, the dt at the last step, the
three g coefficients, the nine low-temperature constants, the draw order, the conformer
update and the Kabsch re-alignment. A stub that reads the pose and t means a schedule or a
dt that is off by one step shows up as a diverging trajectory, not as a wash.

**Upstream's DataLoader eats one draw from the global torch stream before the first step.**
``_BaseDataLoaderIter`` pulls a base seed (``torch.empty((), dtype=torch.int64).random_()``)
each time an iterator is made, so upstream's first ``tr_z`` is not the first draw after the
seed -- it is the second. The port has no loader, so its stream legitimately starts one draw
earlier; aligning the two is the harness's job, not the sampler's, and :func:`align_stream`
is where it happens. Measured: with that one draw consumed, a noise-driven rollout is
**bit-identical** to upstream; without it, the trajectories are unrelated from the first
noisy step and the deviation looks like a sampler bug.

Both sides are seeded identically before the rollout, so they consume the same draws in the
same order, and the comparison is over the whole trajectory rather than the final pose only.
Upstream has no trajectory hook, and a k-step run is NOT a prefix of a 19-step one -- the last
step of each run takes dt all the way to t=0 and skips its noise -- so the grade runs both
sides to k steps for every k from 1 to 19 and compares the 19 end states.
"""

import copy
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/moritz/.coworker/wt/dfd-sampling")

from torch_geometric.data import HeteroData                      # noqa: E402
from utils.sampling import sampling as ref_sampling              # noqa: E402
from utils.diffusion_utils import get_t_schedule, t_to_sigma as ref_t_to_sigma  # noqa: E402

from tt_bio.diffdock import sampling as ours                     # noqa: E402

N_ATOMS, N_RES, N_POSES = 14, 40, 4
SCORE_SCALE = 1e-3
SEED = 11


class _Args:
    """The slice of upstream's ``score_model_args`` that the sampler reads.

    ``__contains__`` because upstream writes ``'all_atoms' in model_args``: its own args
    object is a ``Namespace`` subclass that supports membership, and a plain class does not.
    """

    tr_sigma_min, tr_sigma_max = 0.1, 19.0
    rot_sigma_min, rot_sigma_max = 0.03, 1.55
    tor_sigma_min, tor_sigma_max = 0.0314, 3.14
    no_torsion = False
    all_atoms = False
    crop_beyond = None          # graded separately: it needs a real receptor graph

    def __contains__(self, key):
        return hasattr(self, key)


Args = _Args()


BOUNDS = ours.SigmaBounds(Args.tr_sigma_min, Args.tr_sigma_max, Args.rot_sigma_min,
                          Args.rot_sigma_max, Args.tor_sigma_min, Args.tor_sigma_max)


def align_stream():
    """Consume what upstream's ``DataLoader`` consumes before the rollout: one int64 draw."""
    torch.empty((), dtype=torch.int64).random_()


def ligand():
    """A small rigid-plus-rotatable test ligand: positions, bonds, rotatable mask."""
    g = torch.Generator().manual_seed(3)
    pos = torch.randn(N_ATOMS, 3, generator=g) * 1.5
    bonds = [(i, i + 1) for i in range(N_ATOMS - 1)]
    pairs = []
    for u, v in bonds:
        pairs += [[u, v], [v, u]]
    edge_index = torch.tensor(pairs, dtype=torch.long).T
    edge_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    rotatable = [3, 7, 9]                       # forward half-edges of three bonds
    mask_rotate = np.zeros((len(rotatable), N_ATOMS), dtype=bool)
    for k, e in enumerate(rotatable):
        edge_mask[2 * e] = True
        mask_rotate[k, edge_index[1, 2 * e] + 0:] = True     # everything past the bond moves
    return pos, edge_index, edge_mask, mask_rotate


def complex_graph(pos, edge_index, edge_mask, mask_rotate):
    data = HeteroData()
    data["ligand"].pos = pos.clone()
    data["ligand"].x = torch.zeros(N_ATOMS, 1)
    data["ligand"].edge_mask = edge_mask.clone()
    data["ligand"].mask_rotate = [mask_rotate]
    data["ligand", "ligand"].edge_index = edge_index.clone()
    data["receptor"].pos = torch.zeros(N_RES, 3)
    data["receptor"].x = torch.zeros(N_RES, 1)
    data.name = "stub"
    return data


class StubModel(torch.nn.Module):
    """A deterministic, pose- and time-dependent stand-in for the score network.

    Not equivariant and not physical; it only has to be the *same* function on both sides and
    to depend on everything the rollout feeds it, so that any disagreement in the state handed
    to the network shows up in the trajectory.

    It is scaled to ``SCORE_SCALE`` on purpose. At sigma=19 the translation coefficient
    ``g**2 * dt`` is ~190, so an O(1) stub score throws the pose thousands of Angstrom per
    step and the trajectory becomes chaotic in the pose -- two bit-identical implementations
    then diverge visibly, and the grade measures the stub, not the port. A real score network
    outputs ~1/sigma, which is what this scale stands in for.
    """

    def __init__(self, n_torsions):
        super().__init__()
        self.n_torsions = n_torsions

    @staticmethod
    def _scores(pos, t_tr, t_rot, t_tor, b, n_torsions):
        pos = pos.reshape(b, -1, 3)
        centre = pos.mean(dim=1)
        tr = torch.tanh(centre / 10.0) * (1.0 + t_tr) * SCORE_SCALE
        rot = torch.sin((pos[:, 0] - pos[:, -1]) / 10.0) * (1.0 + t_rot) * SCORE_SCALE
        tor = torch.cos(pos[:, :n_torsions, 0] / 10.0 + t_tor).reshape(-1) * SCORE_SCALE
        return tr, rot, tor

    def forward(self, data):
        b = data.num_graphs if hasattr(data, "num_graphs") else 1
        t = data.complex_t
        return self._scores(data["ligand"].pos, float(t["tr"][0]), float(t["rot"][0]),
                            float(t["tor"][0]), b, self.n_torsions)


def upstream_trajectory(pos, edge_index, edge_mask, mask_rotate, steps, inference_steps,
                        **kwargs):
    """Upstream's rollout, end state recorded after every k of ``steps`` steps."""
    schedule = get_t_schedule(inference_steps=inference_steps, sigma_schedule="expbeta")
    model = StubModel(int(mask_rotate.shape[0]))
    out = []
    for k in range(1, steps + 1):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        data_list = [complex_graph(pos, edge_index, edge_mask, mask_rotate)
                     for _ in range(N_POSES)]
        done, _ = ref_sampling(
            data_list=data_list, model=model, inference_steps=k, tr_schedule=schedule,
            rot_schedule=schedule, tor_schedule=schedule, device="cpu",
            t_to_sigma=lambda a, b_, c: ref_t_to_sigma(a, b_, c, Args), model_args=Args,
            batch_size=N_POSES, **kwargs)
        out.append(torch.stack([g["ligand"].pos for g in done]))
    return out


def our_trajectory(pos, edge_index, edge_mask, mask_rotate, cfg):
    """Our rollout, run to k steps for every k, matching how upstream is read out."""
    from dataclasses import replace

    start = pos.unsqueeze(0).repeat(N_POSES, 1, 1)
    n_torsions = int(mask_rotate.shape[0])
    rot_edges = edge_index.T[edge_mask]

    def score_fn(step):
        return StubModel._scores(step.lig_pos, step.t_tr, step.t_rot, step.t_tor,
                                 step.lig_pos.shape[0], n_torsions)

    out = []
    for k in range(1, cfg.steps + 1):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        align_stream()
        out.append(ours.sample(score_fn, start, cfg=replace(cfg, actual_steps=k),
                               bounds=BOUNDS, noise=ours.Noise(), edge_index=rot_edges,
                               mask_rotate=mask_rotate).poses)
    return out


def grade(label, cfg, **ref_kwargs):
    pos, edge_index, edge_mask, mask_rotate = ligand()
    ref = upstream_trajectory(pos, edge_index, edge_mask, mask_rotate, cfg.steps,
                              cfg.inference_steps, **ref_kwargs)
    mine = our_trajectory(pos, edge_index, edge_mask, mask_rotate, cfg)
    worst, worst_step, deltas = 0.0, -1, []
    for k, (a, b) in enumerate(zip(ref, mine)):
        d = float((a - b).abs().max())
        deltas.append(d)
        if d > worst:
            worst, worst_step = d, k + 1
    if worst >= 1e-5:
        print("  per-k |delta|:", " ".join(f"{d:.1e}" for d in deltas))
    scale = float(torch.stack(ref).abs().max())
    print(f"{label:28s} steps={cfg.steps:2d}  worst |delta| = {worst:.3e} A "
          f"at step {worst_step} (pose scale {scale:.1f} A)")
    return worst


GOLDEN = ("/home/moritz/.coworker/wt/dfd-sampling/tests/fixtures/diffdock/"
          "sampling_rollout_golden.npz")


def write_golden(path=GOLDEN):
    """Freeze upstream's trajectories so the CPU suite grades without this environment.

    Small on purpose (4 poses, 14 atoms, 19 steps, two cases): the regression signal that
    runs on every checkout, with the grading itself staying here against upstream live.
    """
    pos, edge_index, edge_mask, mask_rotate = ligand()
    recipe = ours.recipe()
    blobs = dict(pos=pos.numpy(), edge_index=edge_index.numpy(),
                 edge_mask=edge_mask.numpy(), mask_rotate=mask_rotate, seed=np.int64(SEED))
    for label, cfg in (("sde", ours.SamplingConfig.from_recipe(recipe)),
                       ("ode", ours.SamplingConfig.from_recipe(
                           recipe, ode=True, temp_sampling=(1.0, 1.0, 1.0),
                           temp_psi=(0.0, 0.0, 0.0)))):
        traj = upstream_trajectory(
            pos, edge_index, edge_mask, mask_rotate, cfg.steps, cfg.inference_steps,
            no_final_step_noise=cfg.no_final_step_noise, ode=cfg.ode, no_random=cfg.no_random,
            temp_sampling=list(cfg.temp_sampling), temp_psi=list(cfg.temp_psi),
            temp_sigma_data=list(cfg.temp_sigma_data))
        blobs[f"{label}_trajectory"] = torch.stack(traj).numpy()
    np.savez_compressed(path, **blobs)
    print(f"wrote {path}")


def main():
    if "--write-golden" in sys.argv:
        write_golden()
        return 0
    recipe = ours.recipe()
    cases = [
        ("default recipe (SDE)", ours.SamplingConfig.from_recipe(recipe), {}),
        ("no_final_step_noise off",
         ours.SamplingConfig.from_recipe(recipe, no_final_step_noise=False), {}),
        ("no temperature",
         ours.SamplingConfig.from_recipe(recipe, temp_sampling=(1.0, 1.0, 1.0),
                                         temp_psi=(0.0, 0.0, 0.0)), {}),
        ("ode",
         ours.SamplingConfig.from_recipe(recipe, ode=True, temp_sampling=(1.0, 1.0, 1.0),
                                         temp_psi=(0.0, 0.0, 0.0)), {}),
        ("no_random",
         ours.SamplingConfig.from_recipe(recipe, no_random=True, temp_sampling=(1.0, 1.0, 1.0),
                                         temp_psi=(0.0, 0.0, 0.0)), {}),
        ("actual_steps = inference_steps",
         ours.SamplingConfig.from_recipe(recipe, actual_steps=None), {}),
        ("5 steps",
         ours.SamplingConfig.from_recipe(recipe, inference_steps=5, actual_steps=4), {}),
    ]
    worst = 0.0
    for label, cfg, extra in cases:
        kw = dict(no_final_step_noise=cfg.no_final_step_noise, ode=cfg.ode,
                  no_random=cfg.no_random, temp_sampling=list(cfg.temp_sampling),
                  temp_psi=list(cfg.temp_psi), temp_sigma_data=list(cfg.temp_sigma_data),
                  **extra)
        worst = max(worst, grade(label, cfg, **kw))
    print(f"\nworst over all cases: {worst:.3e} A")
    return 0 if worst < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
