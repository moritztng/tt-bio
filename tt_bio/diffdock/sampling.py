"""DiffDock-L reverse diffusion: the rollout that turns a score network into a pose.

The sampler owns the pose, not the model. It knows the t schedule, the per-step update on
R^3 x SO(3) x SO(2)^m, the nine low-temperature constants and where every random draw comes
from; it knows nothing about how the three scores are produced. The score network arrives as
a callable taking a :class:`StepInput` and returning ``(tr_score, rot_score, tor_score)``, so
the host CPU reference and the ttnn path drop in unchanged.

Every constant is read from ``recipe.yaml`` (D-dfd-2), never retyped here. The sigma bounds
come from the checkpoint's own ``model_parameters.yml`` through :class:`DiffDockConfig`.

Three things are deliberately faithful to upstream rather than tidier:

* **The draw order.** ``randomize_position`` draws torsions (numpy), then a rotation (scipy's
  ``Rotation.random``, which consumes the numpy global stream), then a translation (torch);
  the loop draws ``tr_z``, ``rot_z``, ``tor_z`` per step in that order. Matching upstream's
  trajectory means matching that order exactly, so :class:`Noise` wraps the same generators
  in the same sequence rather than taking one tidy generator of its own. Seeding all three in
  one place is what makes ``--seed`` reproducible (upstream exposes no seed at all).
* **The last step's dt.** With ``actual_steps`` (19) below ``inference_steps`` (20) the final
  step's ``dt`` is ``t[18]`` itself, not ``t[18] - t[19]``: the rollout jumps to t=0 instead
  of stopping short. That is what the two flags mean together, and it is why ``--actual_steps``
  is not "run fewer steps of the same schedule".
* **The rotation composition.** ``axis_angle_to_matrix`` goes through the quaternion, as
  upstream's pytorch3d copy does, including the small-angle series below 1e-6. Rodrigues is
  the same map to ~1e-7, but the bar here is upstream's trajectory to fp32 rounding, so the
  arithmetic is the same arithmetic.

Torsion handling splits by where it happens, also upstream's split: the initial randomisation
runs in numpy float64 via scipy and is cast down, the per-step update runs batched in torch
fp32. Both are reproduced in their own precision.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor

from tt_bio.diffdock.job import RECIPE

def recipe() -> dict:
    """The shipped inference recipe, verbatim (D-dfd-2); a fresh copy of ``job.RECIPE``."""
    return dict(RECIPE)


# ---------------------------------------------------------------------------
# schedule and sigmas
# ---------------------------------------------------------------------------

def t_schedule(inference_steps: int, *, sigma_schedule: str = "expbeta",
               inf_sched_alpha: float = 1.0, inf_sched_beta: float = 1.0,
               t_max: float = 1.0) -> np.ndarray:
    """Upstream's ``get_t_schedule``: ``inference_steps`` values, descending from t_max.

    At alpha=beta=1 the beta CDF is the identity and this is a linear schedule; the recipe
    ships those values, and the campaign's gated recipe runs them.
    """
    if sigma_schedule != "expbeta":
        raise ValueError(f"unknown sigma_schedule {sigma_schedule!r}; upstream ships expbeta")
    from scipy.stats import beta as beta_dist

    lin_max = beta_dist.cdf(t_max, a=inf_sched_alpha, b=inf_sched_beta)
    c = np.linspace(lin_max, 0, inference_steps + 1)[:-1]
    return beta_dist.ppf(c, a=inf_sched_alpha, b=inf_sched_beta)


def pocket_t_max(pocket_tr_max: float, tr_sigma_min: float, tr_sigma_max: float) -> float:
    """``evaluate.py:318``: the t at which tr_sigma equals ``pocket_tr_max``.

    Pocket-conditioned docking starts the translation schedule at the pocket's width instead
    of the receptor's, which only bites with ``different_schedules``.
    """
    return float((np.log(pocket_tr_max) - np.log(tr_sigma_min))
                 / (np.log(tr_sigma_max) - np.log(tr_sigma_min)))


@dataclass(frozen=True)
class SigmaBounds:
    """The six sigma bounds, from the checkpoint's ``model_parameters.yml``."""

    tr_sigma_min: float
    tr_sigma_max: float
    rot_sigma_min: float
    rot_sigma_max: float
    tor_sigma_min: float
    tor_sigma_max: float

    @classmethod
    def from_config(cls, cfg) -> "SigmaBounds":
        return cls(cfg.tr_sigma_min, cfg.tr_sigma_max, cfg.rot_sigma_min,
                   cfg.rot_sigma_max, cfg.tor_sigma_min, cfg.tor_sigma_max)

    def sigmas(self, t_tr: float, t_rot: float, t_tor: float) -> tuple[float, float, float]:
        """Upstream's ``t_to_sigma``: geometric interpolation between min and max."""
        return (self.tr_sigma_min ** (1 - t_tr) * self.tr_sigma_max ** t_tr,
                self.rot_sigma_min ** (1 - t_rot) * self.rot_sigma_max ** t_rot,
                self.tor_sigma_min ** (1 - t_tor) * self.tor_sigma_max ** t_tor)

    def g(self, kind: str, sigma: float) -> float:
        """The diffusion coefficient ``sigma * sqrt(2 log(max/min))`` for one head."""
        lo, hi = getattr(self, f"{kind}_sigma_min"), getattr(self, f"{kind}_sigma_max")
        return float(sigma * np.sqrt(2 * np.log(hi / lo)))


@dataclass(frozen=True)
class SamplingConfig:
    """The recipe's sampling half, in upstream's names.

    ``steps`` is what the loop runs (``actual_steps`` when set), while the schedule is built
    with ``inference_steps``; see the module docstring on why that is not a truncation.
    """

    inference_steps: int = 20
    actual_steps: int | None = 19
    sigma_schedule: str = "expbeta"
    inf_sched_alpha: float = 1.0
    inf_sched_beta: float = 1.0
    different_schedules: bool = False
    no_final_step_noise: bool = True
    ode: bool = False
    no_random: bool = False
    initial_noise_std_proportion: float = 1.4601642460337794
    choose_residue: bool = False
    samples_per_complex: int = 10
    temp_sampling: tuple[float, float, float] = (1.0, 1.0, 1.0)
    temp_psi: tuple[float, float, float] = (0.0, 0.0, 0.0)
    temp_sigma_data: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pocket_knowledge: bool = False
    pocket_tr_max: float = 3.0
    pocket_cutoff: float = 5.0
    no_random_pocket: bool = False
    crop_beyond: float | None = None

    @classmethod
    def from_recipe(cls, data: dict | None = None, **overrides) -> "SamplingConfig":
        """Build from ``recipe.yaml``'s keys (or any dict carrying them), then override."""
        r = dict(recipe() if data is None else data)
        heads = ("tr", "rot", "tor")
        cfg = cls(
            inference_steps=int(r["inference_steps"]),
            actual_steps=None if r.get("actual_steps") is None else int(r["actual_steps"]),
            sigma_schedule=r.get("sigma_schedule", "expbeta"),
            inf_sched_alpha=float(r.get("inf_sched_alpha", 1)),
            inf_sched_beta=float(r.get("inf_sched_beta", 1)),
            different_schedules=bool(r.get("different_schedules", False)),
            no_final_step_noise=bool(r.get("no_final_step_noise", False)),
            ode=bool(r.get("ode", False)),
            no_random=bool(r.get("no_random", False)),
            initial_noise_std_proportion=float(r.get("initial_noise_std_proportion", -1.0)),
            choose_residue=bool(r.get("choose_residue", False)),
            samples_per_complex=int(r.get("samples_per_complex", 10)),
            temp_sampling=tuple(float(r[f"temp_sampling_{h}"]) for h in heads),
            temp_psi=tuple(float(r[f"temp_psi_{h}"]) for h in heads),
            temp_sigma_data=tuple(float(r[f"temp_sigma_data_{h}"]) for h in heads),
            pocket_knowledge=bool(r.get("pocket_knowledge", False)),
            pocket_tr_max=float(r.get("pocket_tr_max", 3.0)),
            pocket_cutoff=float(r.get("pocket_cutoff", 5.0)),
            no_random_pocket=bool(r.get("no_random_pocket", False)),
        )
        return replace(cfg, **overrides) if overrides else cfg

    @property
    def steps(self) -> int:
        return self.inference_steps if self.actual_steps is None else self.actual_steps

    def schedules(self, bounds: SigmaBounds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(tr, rot, tor)`` schedules; one shared schedule unless ``different_schedules``.

        ``different_schedules`` only changes the translation schedule, and only under pocket
        knowledge: it starts tr at the pocket's width (``evaluate.py:317-322``).
        """
        kw = dict(sigma_schedule=self.sigma_schedule, inf_sched_alpha=self.inf_sched_alpha,
                  inf_sched_beta=self.inf_sched_beta)
        shared = t_schedule(self.inference_steps, **kw)
        if not (self.different_schedules and self.pocket_knowledge):
            return shared, shared, shared
        t_max = pocket_t_max(self.pocket_tr_max, bounds.tr_sigma_min, bounds.tr_sigma_max)
        return t_schedule(self.inference_steps, t_max=t_max, **kw), shared, shared


# ---------------------------------------------------------------------------
# randomness
# ---------------------------------------------------------------------------

class Noise:
    """Every random draw the sampler makes, in upstream's order and from its generators.

    Upstream has no seed: it draws from the numpy global stream (torsions, and the random
    rotation via ``scipy.spatial.transform.Rotation.random``), the torch global stream (every
    translation and every per-step z) and the stdlib ``random`` module (``--choose_residue``).
    Seeding is a tt-bio extension, so it seeds exactly those three and changes nothing else:
    same process or another, same seed gives the same poses, and an unseeded run is upstream's
    behaviour bit for bit.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed & 0xFFFFFFFF)
            torch.manual_seed(seed)
            random.seed(seed)

    def torsions(self, n: int) -> np.ndarray:
        return np.random.uniform(low=-np.pi, high=np.pi, size=n)

    def rotation_matrix(self) -> Tensor:
        from scipy.spatial.transform import Rotation

        return torch.from_numpy(Rotation.random().as_matrix()).float()

    def normal(self, *, mean=0.0, std=1.0, size=None) -> Tensor:
        if isinstance(mean, Tensor):
            return torch.normal(mean=mean, std=std)
        return torch.normal(mean=mean, std=std, size=size)

    def residue(self, n: int) -> int:
        return random.randint(0, n - 1)


# ---------------------------------------------------------------------------
# rigid and torsional moves
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Upstream's pytorch3d copy: axis-angle ``[..., 3]`` to a rotation matrix ``[..., 3, 3]``.

    Through the quaternion, with the same ``1e-6`` small-angle series, because the grading bar
    is upstream's own trajectory.
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half = 0.5 * angles
    small = angles.abs() < 1e-6
    ratio = torch.where(small, 0.5 - (angles * angles) / 48, torch.sin(half) / angles.clamp(min=1e-30))
    q = torch.cat([torch.cos(half), axis_angle * ratio], dim=-1)
    r, i, j, k = torch.unbind(q, -1)
    two = 2.0 / (q * q).sum(-1)
    out = torch.stack([
        1 - two * (j * j + k * k), two * (i * j - k * r), two * (i * k + j * r),
        two * (i * j + k * r), 1 - two * (i * i + k * k), two * (j * k - i * r),
        two * (i * k - j * r), two * (j * k + i * r), 1 - two * (i * i + j * j),
    ], dim=-1)
    return out.reshape(q.shape[:-1] + (3, 3))


def kabsch_batch(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Upstream's batched Kabsch: the rigid map taking ``a`` onto ``b``, both ``[B, N, 3]``.

    Returns ``(R [B,3,3], t [B,3,1])`` with ``b ~ a @ R.T + t.T``. ``tt_bio.align`` is the
    shipped fp64 single-pair implementation and stays the one for scoring and output; this one
    is batched, fp32 and inside the step loop, where it must be upstream's arithmetic.
    """
    a, b = a.permute(0, 2, 1), b.permute(0, 2, 1)
    ca, cb = a.mean(dim=2, keepdim=True), b.mean(dim=2, keepdim=True)
    h = torch.bmm(a - ca, (b - cb).transpose(1, 2))
    u, _, vt = torch.linalg.svd(h)
    rot = torch.bmm(vt.transpose(1, 2), u.transpose(1, 2))
    flip = torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=a.dtype, device=a.device))
    mirrored = torch.bmm(vt.transpose(1, 2) @ flip, u.transpose(1, 2))
    rot = torch.where(torch.linalg.det(rot)[:, None, None] < 0, mirrored, rot)
    return rot, torch.bmm(-rot, ca) + cb


def torsion_update_batch(pos: Tensor, edge_index: Tensor, mask_rotate, updates: Tensor) -> Tensor:
    """Rotate about each rotatable bond, ``[B, N, 3]`` poses advanced together.

    One serial pass over the rotatable bonds (their count is fixed for a ligand), batched over
    poses inside it. ``edge_index`` is ``[m, 2]``: the rotatable bonds only, ``u`` on the fixed
    side, ``v`` on the moving side.
    """
    pos = pos + 0
    mask_rotate = _as_mask(mask_rotate)
    for idx, (u, v) in enumerate(edge_index):
        rot_vec = pos[:, u] - pos[:, v]
        rot_vec = rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True) * updates[:, idx:idx + 1]
        rot_mat = axis_angle_to_matrix(rot_vec)
        moving = mask_rotate[idx]
        pos[:, moving] = torch.bmm(pos[:, moving] - pos[:, v:v + 1],
                                   rot_mat.transpose(1, 2)) + pos[:, v:v + 1]
    return pos


def _as_mask(mask_rotate) -> Tensor:
    """``mask_rotate`` as a torch bool tensor; the builders hand it over as numpy."""
    if isinstance(mask_rotate, Tensor):
        return mask_rotate.bool()
    return torch.from_numpy(np.ascontiguousarray(mask_rotate)).bool()


def torsion_update_single(pos, edge_index, mask_rotate, updates) -> Tensor:
    """The initial randomisation's torsion move: numpy/scipy float64, cast down, one pose.

    Upstream runs the initial draw through ``scipy``'s ``Rotation.from_rotvec`` in float64 and
    the per-step update through torch in fp32. Keeping both means keeping both precisions.
    """
    from scipy.spatial.transform import Rotation

    out = np.array(pos.cpu().numpy() if isinstance(pos, Tensor) else pos, copy=True)
    edges = edge_index.cpu().numpy() if isinstance(edge_index, Tensor) else np.asarray(edge_index)
    for idx, (u, v) in enumerate(edges):
        if updates[idx] == 0:
            continue
        rot_vec = out[u] - out[v]
        rot_vec = rot_vec * updates[idx] / np.linalg.norm(rot_vec)
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()
        moving = mask_rotate[idx]
        out[moving] = (out[moving] - out[v]) @ rot_mat.T + out[v]
    return torch.from_numpy(out.astype(np.float32))


def modify_conformer(pos: Tensor, tr_update: Tensor, rot_update: Tensor,
                     torsion_updates: Tensor | None, *, edge_index: Tensor,
                     mask_rotate) -> Tensor:
    """Apply one step's update to ``[B, N, 3]`` poses: rotate, translate, twist, re-align.

    The torsion move is taken about the ligand's own bonds and then Kabsch-aligned back onto
    the rigid pose, so the rigid-body part of the update is exactly the tr/rot heads' and the
    torsion head only changes the internal geometry. That alignment is upstream's, and it is
    why the three heads do not fight each other.
    """
    centre = pos.mean(dim=1, keepdim=True)
    rot_mat = axis_angle_to_matrix(rot_update)
    rigid = torch.bmm(pos - centre, rot_mat.permute(0, 2, 1)) + tr_update.unsqueeze(1) + centre
    if torsion_updates is None:
        return rigid
    flexible = torsion_update_batch(rigid, edge_index, mask_rotate,
                                    torsion_updates.reshape(rigid.shape[0], -1))
    rot, t = kabsch_batch(flexible, rigid)
    return torch.bmm(flexible, rot.transpose(1, 2)) + t.transpose(1, 2)


# ---------------------------------------------------------------------------
# initial pose
# ---------------------------------------------------------------------------

def pocket_centre(receptor_pos: Tensor, reference_lig_pos: Tensor, cutoff: float) -> Tensor:
    """The centroid of the receptor residues within ``cutoff`` of a reference ligand.

    Upstream's ``pocket_knowledge`` path. With no residue inside the cutoff it falls back to
    the single closest residue, loudly, rather than to the receptor centre.
    """
    d = torch.cdist(receptor_pos, reference_lig_pos.float())
    label = torch.any(d < cutoff, dim=1)
    if bool(torch.any(label)):
        return receptor_pos[label].mean(dim=0)
    return receptor_pos[torch.argmin(torch.min(d, dim=1)[0])]


def randomize_position(lig_pos: Tensor, *, receptor_pos: Tensor, n_samples: int,
                       cfg: SamplingConfig, bounds: SigmaBounds, noise: Noise,
                       edge_index: Tensor | None = None, mask_rotate=None,
                       no_torsion: bool = False, centre: Tensor | None = None) -> Tensor:
    """``n_samples`` randomised starting poses from one conformer, as ``[B, N, 3]``.

    Upstream's ``randomize_position``, draw for draw: random torsions, then a uniformly random
    rotation about the ligand's own centre, then a translation. The translation prior is
    ``initial_noise_std_proportion`` times the receptor's RMS radius (the recipe's 1.46), or
    ``-proportion * tr_sigma_max`` when the proportion is negative, or a tight draw about one
    random residue under ``--choose_residue``.
    """
    centre = receptor_pos.mean(dim=0) if centre is None else centre
    tr_sigma_max = cfg.pocket_tr_max if cfg.pocket_knowledge else bounds.tr_sigma_max
    no_random = cfg.no_random or (cfg.pocket_knowledge and cfg.no_random_pocket)

    poses = []
    if not no_torsion and mask_rotate is not None and len(mask_rotate):
        for _ in range(n_samples):
            poses.append(torsion_update_single(lig_pos, edge_index, mask_rotate,
                                               noise.torsions(len(mask_rotate))))
    else:
        poses = [lig_pos.clone() for _ in range(n_samples)]

    out = []
    for pos in poses:
        pos = pos - pos.mean(dim=0, keepdim=True)
        pos = pos @ noise.rotation_matrix().T + centre
        if not no_random:
            if cfg.choose_residue:
                idx = noise.residue(len(receptor_pos))
                tr = noise.normal(mean=receptor_pos[idx:idx + 1], std=0.01)
            elif cfg.initial_noise_std_proportion >= 0.0:
                std_rec = torch.sqrt(torch.mean(torch.sum(receptor_pos ** 2, dim=1)))
                tr = noise.normal(std=std_rec * cfg.initial_noise_std_proportion / 1.73,
                                  size=(1, 3))
            else:
                tr = noise.normal(std=-cfg.initial_noise_std_proportion * tr_sigma_max,
                                  size=(1, 3))
            pos = pos + tr
        out.append(pos)
    return torch.stack(out)


# ---------------------------------------------------------------------------
# receptor cropping
# ---------------------------------------------------------------------------

def crop_beyond(receptor: dict, lig_pos: Tensor, cutoff: float) -> dict:
    """Upstream's ``crop_beyond``: drop receptor residues further than ``cutoff`` from the pose.

    Returns a new dict; the input is untouched. Node features, positions, side-chain vectors
    and the k-NN edge index are all re-indexed together, and the atom graph with them when the
    receptor carries one. The score model's cutoff moves with the schedule
    (``3 * tr_sigma + crop_beyond``); the confidence model's is a fixed 80 A.
    """
    pos = receptor["pos"]
    keep = torch.any(torch.cdist(pos, lig_pos.float()) < cutoff, dim=1)
    remap = torch.cumsum(keep.long(), dim=0) - 1

    out = dict(receptor)
    out["pos"] = pos[keep]
    for key in ("x", "side_chain_vecs", "node_t"):
        if isinstance(receptor.get(key), Tensor):
            out[key] = receptor[key][keep]
    out["edge_index"] = _subgraph(keep, remap, receptor["edge_index"])
    if isinstance(receptor.get("atom"), dict):
        atom = dict(receptor["atom"])
        atom_to_res = receptor["atom"]["res_index"]
        atoms_keep = keep[atom_to_res]
        atom["x"] = atom["x"][atoms_keep]
        atom["pos"] = atom["pos"][atoms_keep]
        atom["res_index"] = remap[atom_to_res][atoms_keep]
        if isinstance(atom.get("edge_index"), Tensor):
            atom_remap = torch.cumsum(atoms_keep.long(), dim=0) - 1
            atom["edge_index"] = _subgraph(atoms_keep, atom_remap, atom["edge_index"])
        out["atom"] = atom
    return out


def _subgraph(keep: Tensor, remap: Tensor, edge_index: Tensor) -> Tensor:
    both = keep[edge_index[0]] & keep[edge_index[1]]
    return remap[edge_index[:, both]]


# ---------------------------------------------------------------------------
# the rollout
# ---------------------------------------------------------------------------

@dataclass
class StepInput:
    """What the score network is asked for, once per step.

    ``lig_pos`` is ``[B, N, 3]`` in the receptor's centred frame. ``crop_cutoff`` is the
    receptor crop radius for this step (``3 * tr_sigma + crop_beyond``), ``None`` when the
    checkpoint sets no ``crop_beyond``; the score adapter applies it with :func:`crop_beyond`,
    because the receptor graph belongs to the model, not to the rollout.
    """

    lig_pos: Tensor
    step: int
    t_tr: float
    t_rot: float
    t_tor: float
    tr_sigma: float
    rot_sigma: float
    tor_sigma: float
    crop_cutoff: float | None = None


@dataclass
class Rollout:
    """The result: ``poses [B, N, 3]``, and the trajectory when it was asked for.

    ``failed`` is a ``[B]`` bool mask of poses that went non-finite during the rollout and
    were frozen at their last finite coordinates. Such a pose is a result the caller must
    not rank, not a reason to lose the complex: see :func:`sample`.
    """

    poses: Tensor
    trajectory: list[Tensor] = field(default_factory=list)
    failed: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.bool))


def _temperature_scale(kind: int, cfg: SamplingConfig, sigma: float, bounds: SigmaBounds,
                       name: str) -> tuple[float, float]:
    """``(score_factor, noise_factor)`` for one head under low-temperature sampling.

    The fitted constants reweight the score and the injected noise separately:
    ``lambda = (sigma_data + sigma) / (sigma_data + sigma / temp)`` tilts the score, and
    ``sqrt(1 + psi)`` widens the noise. At ``temp_sampling == 1`` the head is untouched, which
    is why the recipe's three values all differ from 1.
    """
    if cfg.temp_sampling[kind] == 1.0:
        return 1.0, 1.0
    lo, hi = getattr(bounds, f"{name}_sigma_min"), getattr(bounds, f"{name}_sigma_max")
    sd = np.exp(cfg.temp_sigma_data[kind] * np.log(hi)
                + (1 - cfg.temp_sigma_data[kind]) * np.log(lo))
    lam = (sd + sigma) / (sd + sigma / cfg.temp_sampling[kind])
    return float(lam + cfg.temp_sampling[kind] * cfg.temp_psi[kind] / 2), float(1 + cfg.temp_psi[kind])


def sample(score_fn, lig_pos: Tensor, *, cfg: SamplingConfig, bounds: SigmaBounds,
           noise: Noise | None = None, edge_index: Tensor | None = None, mask_rotate=None,
           no_torsion: bool = False, crop_beyond_a: float | None = None,
           trajectory: bool = False) -> Rollout:
    """Run the reverse diffusion. ``lig_pos`` is ``[B, N, 3]``, B poses advanced together.

    ``score_fn(StepInput) -> (tr_score [B,3], rot_score [B,3], tor_score [B*m])`` is the only
    thing the rollout knows about the model.

    One pose going bad does not take the complex down. The scores are repaired by
    :func:`guard_scores`, and if a pose still goes non-finite -- its update overflows, or the
    model hands back a number no repair can scale -- that pose is frozen where it last was
    and recorded in ``Rollout.failed`` for the caller to exclude from the ranking, while the
    other poses carry on. All of this is a no-op, value for value, on a rollout whose numbers
    stay finite: the freezing branch is not taken at all.
    """
    noise = Noise() if noise is None else noise
    if cfg.ode and any(t != 1.0 for t in cfg.temp_sampling):
        # Upstream's own combination of the two is unreachable: the low-temperature blocks
        # read tr_z/rot_z/tor_z, which the ODE branch never defines, so `--ode` with the
        # fitted constants raises NameError there. Say so instead of inventing a meaning.
        raise ValueError("--ode and low-temperature sampling cannot be combined: upstream's "
                         "ODE branch draws no noise for the temperature terms to reweight; "
                         "run --ode with temp_sampling_{tr,rot,tor} = 1.0")
    tr_sched, rot_sched, tor_sched = cfg.schedules(bounds)
    steps = cfg.steps
    batch = lig_pos.shape[0]
    torsion = not no_torsion and mask_rotate is not None and len(mask_rotate) > 0
    traj = [lig_pos.clone()] if trajectory else []
    # A start pose that is already non-finite cannot be rolled out, but it also cannot be
    # left in the batch: every step's arithmetic is batched. It takes a finite pose's
    # coordinates for the ride and is marked failed, so nothing downstream ranks or reads it.
    # Zeros would not do -- a ligand collapsed to a point has zero-length bond axes, and the
    # torsion update divides by their length.
    failed = _nonfinite_rows(lig_pos, batch)
    if bool(failed.all()):
        return Rollout(poses=torch.nan_to_num(lig_pos), trajectory=traj, failed=failed)
    if bool(failed.any()):
        donor = lig_pos[int((~failed).nonzero()[0])]
        lig_pos = torch.where(failed[:, None, None], donor[None], lig_pos)

    with torch.no_grad():
        for t_idx in range(steps):
            t_tr, t_rot, t_tor = (float(tr_sched[t_idx]), float(rot_sched[t_idx]),
                                  float(tor_sched[t_idx]))
            last = t_idx == steps - 1
            dt_tr = float(tr_sched[t_idx] - tr_sched[t_idx + 1]) if not last else t_tr
            dt_rot = float(rot_sched[t_idx] - rot_sched[t_idx + 1]) if not last else t_rot
            dt_tor = float(tor_sched[t_idx] - tor_sched[t_idx + 1]) if not last else t_tor
            tr_sigma, rot_sigma, tor_sigma = bounds.sigmas(t_tr, t_rot, t_tor)

            step = StepInput(
                lig_pos=lig_pos, step=t_idx, t_tr=t_tr, t_rot=t_rot, t_tor=t_tor,
                tr_sigma=tr_sigma, rot_sigma=rot_sigma, tor_sigma=tor_sigma,
                crop_cutoff=None if crop_beyond_a is None else tr_sigma * 3 + crop_beyond_a,
            )
            tr_score, rot_score, tor_score = score_fn(step)[:3]
            tr_score, rot_score, tor_score = _repair_nans(tr_score, rot_score, tor_score)

            tr_g, rot_g = bounds.g("tr", tr_sigma), bounds.g("rot", rot_sigma)
            tor_g = bounds.g("tor", tor_sigma) if torsion else None
            quiet = cfg.no_random or (cfg.no_final_step_noise and last)

            if cfg.ode:
                tr_perturb = 0.5 * tr_g ** 2 * dt_tr * tr_score
                rot_perturb = 0.5 * rot_g ** 2 * dt_rot * rot_score
                tor_perturb = 0.5 * tor_g ** 2 * dt_tor * tor_score if torsion else None
            else:
                tr_z = _z(batch, 3, quiet, noise)
                rot_z = _z(batch, 3, quiet, noise)
                tor_z = (torch.zeros(tor_score.shape) if quiet
                         else noise.normal(size=tuple(tor_score.shape))) if torsion else None
                a_tr, w_tr = _temperature_scale(0, cfg, tr_sigma, bounds, "tr")
                a_rot, w_rot = _temperature_scale(1, cfg, rot_sigma, bounds, "rot")
                tr_perturb = (tr_g ** 2 * dt_tr * a_tr * tr_score
                              + tr_g * np.sqrt(dt_tr * w_tr) * tr_z)
                rot_perturb = (rot_g ** 2 * dt_rot * a_rot * rot_score
                               + rot_g * np.sqrt(dt_rot * w_rot) * rot_z)
                if torsion:
                    a_tor, w_tor = _temperature_scale(2, cfg, tor_sigma, bounds, "tor")
                    tor_perturb = (tor_g ** 2 * dt_tor * a_tor * tor_score
                                   + tor_g * np.sqrt(dt_tor * w_tor) * tor_z)
                else:
                    tor_perturb = None

            lig_pos, failed = _move(lig_pos, tr_perturb, rot_perturb, tor_perturb, failed,
                                    edge_index=edge_index, mask_rotate=mask_rotate)
            if trajectory:
                traj.append(lig_pos.clone())
    return Rollout(poses=lig_pos, trajectory=traj, failed=failed)


def _move(pos: Tensor, tr_perturb: Tensor, rot_perturb: Tensor, tor_perturb: Tensor | None,
          failed: Tensor, *, edge_index, mask_rotate) -> tuple[Tensor, Tensor]:
    """One step's :func:`modify_conformer`, with a non-finite pose frozen instead of fatal.

    The Kabsch SVD inside ``modify_conformer`` is batched, so a single non-finite row raises
    for every pose at once ("input matrix contained non-finite values") and the complex is
    lost. Zeroing that row's update first keeps the batch finite, and reverting its position
    afterwards keeps it at the last coordinates it actually had.
    """
    batch = pos.shape[0]
    bad = _nonfinite_rows(pos, batch)
    for perturb in (tr_perturb, rot_perturb, tor_perturb):
        if perturb is not None:
            bad = bad | _nonfinite_rows(perturb, batch)
    if bool(bad.any()):
        tr_perturb, rot_perturb, tor_perturb = (
            _zero_rows(p, bad) for p in (tr_perturb, rot_perturb, tor_perturb))
    moved = modify_conformer(pos, tr_perturb, rot_perturb, tor_perturb,
                             edge_index=edge_index, mask_rotate=mask_rotate)
    bad = bad | _nonfinite_rows(moved, batch)
    if not bool(bad.any()):
        return moved, failed
    return torch.where(bad[:, None, None], pos, moved), failed | bad


def _z(batch: int, dim: int, quiet: bool, noise: Noise) -> Tensor:
    return torch.zeros((batch, dim)) if quiet else noise.normal(size=(batch, dim))


def guard_scores(score: Tensor) -> tuple[Tensor, Tensor]:
    """Repair one head's non-finite entries. Returns ``(repaired, n_bad_poses)``.

    This is the campaign's one score repair: the host rollout below and the traced device
    driver (:func:`tt_bio.diffdock.loop.diffusion_step`, which imports this) call it.

    A non-finite score would otherwise poison every later step of that pose, and then
    ``modify_conformer``'s Kabsch SVD raises on the whole batch. Upstream replaces it with
    1 % of the head's mean magnitude, which perturbs the pose slightly and usually recovers
    on the next step. Two differences from upstream's version, both forced by cases it does
    not survive:

    * It keys on ``isfinite``, not ``isnan``. A head that goes to +-inf with no NaN passes
      upstream's `isnan` test untouched and takes the complex down in the SVD. That is not
      hypothetical: it is what three of `tt-bio dock`'s output tests hit under torch 2.8.0,
      where the starting draw differs (D-dfd-sampling-9).
    * An infinity is excluded from the mean rather than poisoning it, so the replacement
      value stays finite. Upstream's ``nanmean`` keeps the infinity, and when a head is
      non-finite in *every* row the replacement is itself non-finite and the guard does
      nothing at all.

    The repair is computed unconditionally and selected with a ``where``: no host branch, so
    the device path keeps its shapes static, and on a run with finite numbers the result is
    the input tensor value for value.
    """
    finite = torch.isfinite(score)
    eps = 0.01 * torch.where(finite, score.abs(), torch.zeros_like(score)).sum() \
        / finite.sum().clamp_min(1)
    sign = torch.sign(torch.nan_to_num(score, nan=1.0, posinf=1.0, neginf=-1.0))
    repaired = torch.where(finite, score, eps * sign)
    n_bad = (~torch.isfinite(score.reshape(score.shape[0], -1).sum(dim=-1))).sum()
    return repaired, n_bad


def _repair_nans(tr_score: Tensor, rot_score: Tensor, tor_score: Tensor):
    """:func:`guard_scores` over the three heads, each scaled by its own magnitude."""
    return tuple(None if score is None else guard_scores(score)[0]
                 for score in (tr_score, rot_score, tor_score))


def _nonfinite_rows(x: Tensor, batch: int) -> Tensor:
    """``[batch]`` bool: which poses carry a non-finite number anywhere in *x*.

    *x* is anything with the pose axis first, including the torsion head's flat ``[B * m]``.
    """
    return ~torch.isfinite(x.reshape(batch, -1)).all(dim=1)


def _zero_rows(x: Tensor | None, rows: Tensor) -> Tensor | None:
    """*x* with the poses *rows* selects set to zero, in whatever shape *x* has."""
    if x is None:
        return None
    flat = x.reshape(rows.shape[0], -1)
    return torch.where(rows[:, None], torch.zeros_like(flat), flat).reshape(x.shape)
