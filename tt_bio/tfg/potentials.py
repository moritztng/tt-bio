"""TFG potentials: geometry/chemistry penalty energies on atom coordinates (host torch).

Port of OpenDDE v1.2.0 `opendde/tfg/potentials.py` (Apache-2.0, Copyright (c) 2026 Aureka AI
Research, Copyright 2024 ByteDance and/or its affiliates), modified: the Triton fast path of
VinaStericPotential (OPENDDE_VINA_FAST) is dropped and only its dense path is kept. Arithmetic is
op-for-op upstream's.

Conventions: `coords` is `[..., N, 3]`; `energy` returns `coords.shape[:-2]`; `energy_and_grad`
returns `(energy, dE/dcoords)`; `project` returns an additive `delta_x` shaped like `coords`.
"""

import math
from typing import Any, Optional

import torch

from tt_bio.data.const import vdw_radii as rdkit_vdws  # element-for-element opendde rdkit_vdws

CLASS_REGISTRY: dict[str, type] = {}

# fp32 [128] VDW radius table; rows 118..127 are 0.
_VDW_RADII_128_CPU = torch.zeros(128, dtype=torch.float32)
_VDW_RADII_128_CPU[:118] = torch.as_tensor(rdkit_vdws, dtype=torch.float32)
_VDW_RADII_128_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _get_vdw_radii_128(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    out = _VDW_RADII_128_CACHE.get(key)
    if out is None or out.device != device:
        out = _VDW_RADII_128_CPU.to(device=device)
        _VDW_RADII_128_CACHE[key] = out
    return out


def register(cls):
    """Register a potential class in CLASS_REGISTRY under its class name."""
    CLASS_REGISTRY[cls.__name__] = cls
    return cls


class Potential:
    """Base potential: subclasses implement `_eval` and optionally `_project`."""

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        self._default_params = default_params

    def _resolve_params(self, params: Optional[dict[str, Any]]) -> dict[str, Any]:
        res = {}
        if self._default_params is not None:
            res.update(self._default_params)
        if params is not None:
            res.update(params)
        return res

    def _eval(self, coords, feats, params, need_grad: bool):
        raise NotImplementedError

    def energy(self, coords, feats, params=None) -> torch.Tensor:
        return self._eval(coords, feats, self._resolve_params(params), False)

    def energy_and_grad(self, coords, feats, params=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self._eval(coords, feats, self._resolve_params(params), True)

    def project(self, coords, feats, params=None) -> torch.Tensor:
        return self._project(coords, feats, self._resolve_params(params))

    def _project(self, coords, feats, params) -> torch.Tensor:
        """No-op projection (zeros); TFGEngine skips terms that do not override this."""
        return torch.zeros_like(coords)


def _to_tensor(x, device, dtype):
    return torch.as_tensor(x, device=device, dtype=dtype)


def _distance_value_and_grad(coords, index, need_grad):
    """Distances `[..., M]` for pairs `index [2, M]`; Jacobian `[..., 2, M, 3]` if need_grad."""
    r = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    norm = torch.linalg.norm(r, dim=-1).clamp_min(1e-8)
    if not need_grad:
        return norm, None
    r_hat = r / norm.unsqueeze(-1)
    grad = torch.stack((r_hat, -r_hat), dim=-3)
    return norm, grad


def _angle_value_and_grad(coords, index, need_grad):
    """Angle at vertex j of triples (i, j, k) `index [3, M]`, radians; Jacobian `[..., 3, M, 3]`."""
    r_ji = coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
    r_jk = coords.index_select(-2, index[2]) - coords.index_select(-2, index[1])
    cross = torch.linalg.cross(r_ji, r_jk, dim=-1)
    dot = torch.linalg.vecdot(r_ji, r_jk) + 1e-8
    angle = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), dot)
    if not need_grad:
        return angle, None
    r2_ji = torch.linalg.vecdot(r_ji, r_ji) + 1e-8
    r2_jk = torch.linalg.vecdot(r_jk, r_jk) + 1e-8
    rp = torch.linalg.vector_norm(cross, dim=-1) + 1e-8
    grad_i = torch.linalg.cross(r_ji, cross, dim=-1) / (r2_ji * rp).unsqueeze(-1)
    grad_k = torch.linalg.cross(cross, r_jk, dim=-1) / (r2_jk * rp).unsqueeze(-1)
    grad_j = -grad_i - grad_k
    grad = torch.stack((grad_i, grad_j, grad_k), dim=-3)
    return angle, grad


def _dihedral_value_and_grad(coords, index, need_grad):
    """Signed dihedral of (i, j, k, l) `index [4, M]`, radians; Jacobian `[..., 4, M, 3]`."""
    r_ij = coords.index_select(-2, index[1]) - coords.index_select(-2, index[0])
    r_kj = coords.index_select(-2, index[1]) - coords.index_select(-2, index[2])
    r_kl = coords.index_select(-2, index[3]) - coords.index_select(-2, index[2])
    m = torch.linalg.cross(r_ij, r_kj, dim=-1)
    n = torch.linalg.cross(r_kj, r_kl, dim=-1)
    w = torch.linalg.cross(m, n, dim=-1)
    wlen = torch.linalg.vector_norm(w, dim=-1)
    s = torch.linalg.vecdot(m, n) + 1e-8
    phi = torch.atan2(wlen, s)
    ipr = torch.linalg.vecdot(r_ij, n)
    phi = -phi * torch.sign(ipr)
    if not need_grad:
        return phi, None
    iprm = torch.linalg.vecdot(m, m) + 1e-8
    iprn = torch.linalg.vecdot(n, n) + 1e-8
    nrkj2 = torch.linalg.vecdot(r_kj, r_kj)
    nrkj_1 = torch.rsqrt(nrkj2 + 1e-8)
    nrkj_2 = torch.square(nrkj_1)
    nrkj = nrkj2 * nrkj_1
    a = -nrkj / iprm
    f_i = -a.unsqueeze(-1) * m
    b = nrkj / iprn
    f_l = -b.unsqueeze(-1) * n
    p = torch.linalg.vecdot(r_ij, r_kj) * nrkj_2
    q = torch.linalg.vecdot(r_kl, r_kj) * nrkj_2
    uvec = p.unsqueeze(-1) * f_i
    vvec = q.unsqueeze(-1) * f_l
    svec = uvec - vvec
    f_j = f_i - svec
    f_k = f_l + svec
    grad = torch.stack((f_i, -f_j, -f_k, f_l), dim=-3)
    return phi, grad


def _abs_dihedral_value_and_grad(coords, index, need_grad):
    """|phi| and its Jacobian."""
    phi, grad = _dihedral_value_and_grad(coords, index, need_grad)
    if not need_grad:
        return torch.abs(phi), None
    sign = torch.where(phi < 0, -1.0, 1.0)
    grad = grad * sign[..., None, :, None]
    return torch.abs(phi), grad


def _planar_improper_value_and_grad(coords, index, need_grad, zero_tol: float = 1e-10):
    """Improper planarity energy `1 - sin(Y)` (RDKit UFF inversion form) and its coordinate grad.

    index rows: neighbour 1, neighbour 2, centre, neighbour 3.
    """
    p1 = coords.index_select(-2, index[0])
    p3 = coords.index_select(-2, index[1])
    p2 = coords.index_select(-2, index[2])
    p4 = coords.index_select(-2, index[3])
    r_ji = p1 - p2
    r_jk = p3 - p2
    r_jl = p4 - p2
    l2_ji = torch.linalg.vecdot(r_ji, r_ji)
    l2_jk = torch.linalg.vecdot(r_jk, r_jk)
    l2_jl = torch.linalg.vecdot(r_jl, r_jl)
    dist_mask = (l2_ji > zero_tol) & (l2_jk > zero_tol) & (l2_jl > zero_tol)
    d_ji = torch.sqrt(l2_ji).clamp_min(1e-8)
    d_jk = torch.sqrt(l2_jk).clamp_min(1e-8)
    d_jl = torch.sqrt(l2_jl).clamp_min(1e-8)
    r_ji_n = r_ji / d_ji.unsqueeze(-1)
    r_jk_n = r_jk / d_jk.unsqueeze(-1)
    r_jl_n = r_jl / d_jl.unsqueeze(-1)
    n = torch.linalg.cross(-r_ji_n, r_jk_n, dim=-1)
    l2_n = torch.linalg.vecdot(n, n)
    mask = dist_mask & (l2_n > zero_tol)
    n_normalized = n / torch.sqrt(l2_n).unsqueeze(-1).clamp_min(1e-8)
    cos_y = torch.where(mask, torch.linalg.vecdot(n_normalized, r_jl_n), 0.0).clamp(-1.0, 1.0)
    sin_y_sq = (1.0 - cos_y * cos_y).clamp_min(0.0)
    sin_y = torch.sqrt(sin_y_sq).clamp_min(1e-8)
    energy = 1.0 - sin_y
    if not need_grad:
        return energy, None
    cos_theta = torch.linalg.vecdot(r_ji_n, r_jk_n).clamp(-1.0, 1.0)
    sin_theta_sq = (1.0 - cos_theta * cos_theta).clamp_min(1e-12)
    sin_theta = torch.sqrt(sin_theta_sq).clamp_min(1e-8)
    t1 = torch.linalg.cross(r_jl_n, r_jk_n, dim=-1)
    t2 = torch.linalg.cross(r_ji_n, r_jl_n, dim=-1)
    t3 = torch.linalg.cross(r_jk_n, r_ji_n, dim=-1)
    term1 = sin_y * sin_theta
    term2 = cos_y / (sin_y * sin_theta_sq)
    tg1 = (
        t1 / term1.unsqueeze(-1) - (r_ji_n - r_jk_n * cos_theta.unsqueeze(-1)) * term2.unsqueeze(-1)
    ) / d_ji.unsqueeze(-1)
    tg3 = (
        t2 / term1.unsqueeze(-1) - (r_jk_n - r_ji_n * cos_theta.unsqueeze(-1)) * term2.unsqueeze(-1)
    ) / d_jk.unsqueeze(-1)
    tg4 = (t3 / term1.unsqueeze(-1) - r_jl_n * (cos_y / sin_y).unsqueeze(-1)) / d_jl.unsqueeze(-1)
    tg2 = -(tg1 + tg3 + tg4)
    # dE/dY = -cos_y and grad_Y = -tg, so grad_E = cos_y * tg.
    grad = torch.stack((tg1, tg3, tg2, tg4), dim=-3) * cos_y.unsqueeze(-2).unsqueeze(-1)
    grad = torch.where(mask.unsqueeze(-2).unsqueeze(-1), grad, 0.0)
    return energy, grad


def _flat_bottom_linear(value, k, lower, upper):
    """E = k * (max(0, lower - v) + max(0, v - upper)); returns (E, dE/dv)."""
    energy = torch.zeros_like(value)
    grad = torch.zeros_like(value)
    if lower is not None:
        diff_lb = lower - value
        mask_lb = diff_lb > 0
        energy += torch.where(mask_lb, k * diff_lb, 0.0)
        grad -= torch.where(mask_lb, k, 0.0)
    if upper is not None:
        diff_ub = value - upper
        mask_ub = diff_ub > 0
        energy += torch.where(mask_ub, k * diff_ub, 0.0)
        grad += torch.where(mask_ub, k, 0.0)
    return energy, grad


def _flat_bottom_parabolic(value, k, lower, upper):
    """E = 0.5 k (max(0, lower - v)^2 + max(0, v - upper)^2); returns (E, dE/dv)."""
    energy = torch.zeros_like(value)
    grad = torch.zeros_like(value)
    if lower is not None:
        diff_lb = lower - value
        mask_lb = diff_lb > 0
        energy += torch.where(mask_lb, 0.5 * k * (diff_lb**2), 0.0)
        grad -= torch.where(mask_lb, k * diff_lb, 0.0)
    if upper is not None:
        diff_ub = value - upper
        mask_ub = diff_ub > 0
        energy += torch.where(mask_ub, 0.5 * k * (diff_ub**2), 0.0)
        grad += torch.where(mask_ub, k * diff_ub, 0.0)
    return energy, grad


def _aggregate_atom_gradients(coords, indices, grad_values, dE_dvals):
    """Scatter-add `dE/dv * dv/dx` (`[..., K, M, 3]`) onto a dense `[..., N, 3]` gradient."""
    g_combined = dE_dvals.unsqueeze(-2).unsqueeze(-1) * grad_values
    g_flat = g_combined.flatten(start_dim=-3, end_dim=-2)
    idx_flat = indices.flatten()
    if g_flat.ndim > coords.ndim:
        extra_dims = g_flat.ndim - coords.ndim
        g_flat = g_flat.sum(dim=list(range(extra_dims)))
    batch_shape = coords.shape[:-2]
    target_idx = idx_flat.view(*([1] * len(batch_shape)), -1, 1).expand(*batch_shape, -1, 3)
    out = torch.zeros_like(coords)
    out.scatter_add_(dim=-2, index=target_idx, src=g_flat)
    return out


def _solve_constraint_projection(coords, index, value, grad_value, mask, eps: float = 1e-8):
    """Minimum-norm linearised projection `dx = -J^T (J J^T + eps I)^-1 v` over masked constraints.

    Shapes: coords `[*B, N, 3]`, index `[K, M]`, value/mask `[*B, M]`, grad_value `[*B, K, M, 3]`.
    Solved per batch element (each has its own active set); fp16/bf16 solve in fp32.
    """
    device, dtype = coords.device, coords.dtype
    batch_shape = coords.shape[:-2]
    n_atom = coords.shape[-2]
    k = int(index.shape[0])
    m_total = int(index.shape[1])
    if value.shape[-1] != m_total:
        raise ValueError(
            f"_solve_constraint_projection: value.shape[-1]={value.shape[-1]} != index.shape[1]={m_total}"
        )
    if mask.shape != value.shape:
        raise ValueError(
            f"_solve_constraint_projection: mask.shape={tuple(mask.shape)} != value.shape={tuple(value.shape)}"
        )
    if grad_value.shape[-3:] != (k, m_total, 3):
        raise ValueError(
            f"_solve_constraint_projection: grad_value.shape[-3:]={tuple(grad_value.shape[-3:])} "
            f"!= (K,M,3)=({k},{m_total},3)"
        )
    b = math.prod(batch_shape) if len(batch_shape) > 0 else 1
    value_b = value.reshape(b, m_total)
    mask_b = mask.reshape(b, m_total)
    grad_b = grad_value.reshape(b, k, m_total, 3)
    delta = torch.zeros_like(coords.reshape(b, n_atom, 3))
    for bi in range(b):
        active = mask_b[bi]
        n_active = int(active.sum().item())
        if n_active == 0:
            continue
        v = value_b[bi, active]
        g = grad_b[bi, :, active, :]
        idx = index[:, active]
        grad_atom = torch.zeros((n_active, n_atom, 3), device=device, dtype=dtype)
        row = torch.arange(n_active, device=device)
        for ki in range(k):
            grad_atom[row, idx[ki]] += g[ki]
        grad_flat = grad_atom.reshape(n_active, n_atom * 3)
        work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        gf = grad_flat.to(work_dtype)
        vv = v.to(work_dtype)
        denom = gf @ gf.transpose(-1, -2)
        denom = denom + (float(eps) * torch.eye(n_active, device=device, dtype=work_dtype))
        lam = torch.linalg.solve(denom, vv)
        dx_flat = -(gf.transpose(-1, -2) @ lam)
        delta[bi] = dx_flat.to(dtype).reshape(n_atom, 3)
    return delta.reshape(*coords.shape)


def _zeros_energy(coords):
    return torch.zeros(coords.shape[:-2], device=coords.device, dtype=coords.dtype)


def _zeros_energy_and_grad(coords):
    return _zeros_energy(coords), torch.zeros_like(coords)


def _zeros(coords, need_grad):
    return _zeros_energy_and_grad(coords) if need_grad else _zeros_energy(coords)


def _sum_energy(e):
    if e.ndim == 0:
        return e
    return e.sum(dim=-1)


@register
class InterchainBondPotential(Potential):
    """Linear upper bound `buffer` (default 2.0) on `interchain_bond_index [2, M]` distances."""

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"buffer": 2.0}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["interchain_bond_index"]
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        value, grad_value = _distance_value_and_grad(coords, idx, need_grad)
        k = torch.ones_like(value)
        upper = torch.full((idx.shape[1],), float(params["buffer"]), device=coords.device, dtype=value.dtype)
        e, dE = _flat_bottom_linear(value, k, None, upper)
        if not need_grad:
            return _sum_energy(e)
        return _sum_energy(e), _aggregate_atom_gradients(coords, idx, grad_value, dE)


@register
class UserDistanceRestraintPotential(Potential):
    """Parabolic flat-bottom on user pair distances, bounds used as given (no VDW clamp).

    feats: `user_distance_restraint_index [2, M]`, `user_distance_restraint_{lower,upper}_bound [M]` (A).
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults: dict[str, Any] = {}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)

    def _bounds(self, coords, feats):
        lower = feats["user_distance_restraint_lower_bound"].to(device=coords.device, dtype=coords.dtype)
        upper = feats["user_distance_restraint_upper_bound"].to(device=coords.device, dtype=coords.dtype)
        return lower, upper

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["user_distance_restraint_index"]
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        lower, upper = self._bounds(coords, feats)
        value, grad_value = _distance_value_and_grad(coords, idx, need_grad)
        k = torch.ones_like(value)
        e, dE = _flat_bottom_parabolic(value, k, lower, upper)
        if not need_grad:
            return _sum_energy(e)
        return _sum_energy(e), _aggregate_atom_gradients(coords, idx, grad_value, dE)

    def _project(self, coords, feats, params):
        idx = feats["user_distance_restraint_index"]
        if idx.numel() == 0:
            return torch.zeros_like(coords)
        lower, upper = self._bounds(coords, feats)
        value, grad_value = _distance_value_and_grad(coords, idx, True)
        mask_lb = value < lower
        mask_ub = value > upper
        mask = mask_lb | mask_ub
        if mask.sum() == 0:
            return torch.zeros_like(coords)
        v = value - torch.where(mask_lb, lower, upper)
        return _solve_constraint_projection(coords, idx, v, grad_value, mask)


@register
class PairwiseDistancePotential(Potential):
    """Parabolic flat-bottom on pair distance bounds with bond/angle/clash buffers and VDW clamps.

    feats: `pairwise_distance_index [2, M]`, `pairwise_distance_{lower,upper}_bound [M]`,
    `pairwise_distance_is_{bond,angle} [M]`, `ref_element [N, E]` one-hot (argmax -> element).
    Clash pairs (neither bond nor angle) get upper = inf and lower >= 0.35 + 0.5 (r_i + r_j);
    bond pairs get upper <= that same VDW limit.
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"bond_buffer": 0.05, "angle_buffer": 0.05, "clash_buffer": 0.05}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)
        # Static per-feats tensors, keyed on feature tensor identity.
        self._cache_key = None
        self._cached_state_code = None
        self._cached_vdw_limit_f32 = None
        self._scale_table_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @staticmethod
    def _cache_key_from_feats(feats):
        def _tid(x):
            return (x.data_ptr(), tuple(x.shape), str(x.dtype))

        idx = feats["pairwise_distance_index"]
        return (
            str(idx.device),
            _tid(idx),
            _tid(feats["pairwise_distance_lower_bound"]),
            _tid(feats["pairwise_distance_upper_bound"]),
            _tid(feats["pairwise_distance_is_bond"]),
            _tid(feats["pairwise_distance_is_angle"]),
            _tid(feats["ref_element"]),
        )

    def _get_scale_tables(self, *, device, lb_dtype, ub_dtype, b_buf, a_buf, c_buf):
        min_ba = min(b_buf, a_buf)
        key = (device.type, device.index, str(lb_dtype), str(ub_dtype), b_buf, a_buf, c_buf)
        out = self._scale_table_cache.get(key)
        if out is not None:
            return out
        l_table = torch.tensor([c_buf, b_buf, a_buf, min_ba], device=device, dtype=lb_dtype)
        u_table = torch.tensor([0.0, b_buf, a_buf, min_ba], device=device, dtype=ub_dtype)
        self._scale_table_cache[key] = (l_table, u_table)
        return l_table, u_table

    def _get_distance_bounds(self, feats, params):
        idx = feats["pairwise_distance_index"]
        if idx.numel() == 0:
            return idx, None, None
        lb_base = feats["pairwise_distance_lower_bound"]
        ub_base = feats["pairwise_distance_upper_bound"]
        cache_key = self._cache_key_from_feats(feats)
        if cache_key != self._cache_key:
            is_bond = feats["pairwise_distance_is_bond"].to(torch.int64)
            is_angle = feats["pairwise_distance_is_angle"].to(torch.int64)
            # 0 = clash, 1 = bond, 2 = angle, 3 = bond and angle.
            self._cached_state_code = is_bond + (is_angle << 1)
            vdw_radii = _get_vdw_radii_128(idx.device)
            element_idx = feats["ref_element"].argmax(dim=-1)
            atom_radii = vdw_radii.index_select(0, element_idx.to(torch.long))
            vdw_pair_sum = atom_radii[idx[0]] + atom_radii[idx[1]]
            self._cached_vdw_limit_f32 = 0.35 + 0.5 * vdw_pair_sum
            self._cache_key = cache_key
        state_code = self._cached_state_code
        vdw_limit = self._cached_vdw_limit_f32.to(dtype=lb_base.dtype)
        l_table, u_table = self._get_scale_tables(
            device=idx.device,
            lb_dtype=lb_base.dtype,
            ub_dtype=ub_base.dtype,
            b_buf=float(params["bond_buffer"]),
            a_buf=float(params["angle_buffer"]),
            c_buf=float(params["clash_buffer"]),
        )
        lower = lb_base * (1.0 - l_table[state_code])
        upper = ub_base * (1.0 + u_table[state_code])
        upper = torch.where(state_code == 0, float("inf"), upper)
        bond_mask = (state_code & 1).bool()
        lower = torch.where(~bond_mask, torch.maximum(lower, vdw_limit), lower)
        upper = torch.where(bond_mask, torch.minimum(upper, vdw_limit), upper)
        return idx, lower, upper

    def _eval(self, coords, feats, params, need_grad: bool):
        idx, lower, upper = self._get_distance_bounds(feats, params)
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        value, grad_value = _distance_value_and_grad(coords, idx, need_grad)
        k = torch.ones_like(value)
        e, dE = _flat_bottom_parabolic(value, k, lower, upper)
        if not need_grad:
            return _sum_energy(e)
        return _sum_energy(e), _aggregate_atom_gradients(coords, idx, grad_value, dE)

    def _project_masked(self, coords, idx, lower, upper, idx_mask):
        idx = idx[..., idx_mask]
        if idx.numel() == 0:
            return torch.zeros_like(coords)
        lower = lower[idx_mask]
        upper = upper[idx_mask]
        value, grad_value = _distance_value_and_grad(coords, idx, True)
        mask_lb = value < lower
        mask_ub = value > upper
        mask = mask_lb | mask_ub
        if mask.sum() == 0:
            return torch.zeros_like(coords)
        v = value - torch.where(mask_lb, lower, upper)
        return _solve_constraint_projection(coords, idx, v, grad_value, mask)

    def _project(self, coords, feats, params):
        """Project angle-derived pairs, then bond pairs (on the angle-projected coords)."""
        idx, lower, upper = self._get_distance_bounds(feats, params)
        bond_mask = feats["pairwise_distance_is_bond"].bool()
        angle_mask = feats["pairwise_distance_is_angle"].bool()
        delta_x_angle = self._project_masked(coords, idx, lower, upper, angle_mask)
        delta_x_bond = self._project_masked(coords + delta_x_angle, idx, lower, upper, bond_mask)
        return delta_x_bond + delta_x_angle


@register
class StereoBondPotential(Potential):
    """Cis/trans double-bond term on |dihedral| of `stereo_bond_index [4, M]`.

    `stereo_bond_orientation > 0.5` wants |phi| >= pi - buffer (trans), else |phi| <= buffer.
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"buffer": 0.52360}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["stereo_bond_index"]
        orient = feats["stereo_bond_orientation"].to(coords.dtype)
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        value, grad_value = _abs_dihedral_value_and_grad(coords, idx, need_grad)
        dev, dt = value.device, value.dtype
        lower = torch.where(
            orient > 0.5, _to_tensor(torch.pi - params["buffer"], dev, dt), _to_tensor(float("-inf"), dev, dt)
        )
        upper = torch.where(orient > 0.5, _to_tensor(float("inf"), dev, dt), _to_tensor(params["buffer"], dev, dt))
        k = torch.ones_like(value)
        e, dE = _flat_bottom_linear(value, k, lower, upper)
        if not need_grad:
            return _sum_energy(e)
        return _sum_energy(e), _aggregate_atom_gradients(coords, idx, grad_value, dE)


@register
class ChiralAtomPotential(Potential):
    """Chirality: dihedral of `chiral_index [4, M]` >= buffer if `chiral_orientation > 0`, else <= -buffer.

    `_project` repairs violated centres with the linearised solver and, with `scale_x` (default),
    rescales the touched atoms of each chain so their per-chain radius of gyration is unchanged.
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"buffer": 0.34906, "scale_x": True}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["chiral_index"]
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        orient = feats["chiral_orientation"]
        value, grad_value = _dihedral_value_and_grad(coords, idx, need_grad)
        dev, dt = value.device, value.dtype
        lower = torch.where(orient > 0, _to_tensor(params["buffer"], dev, dt), _to_tensor(float("-inf"), dev, dt))
        upper = torch.where(orient > 0, _to_tensor(float("inf"), dev, dt), -_to_tensor(params["buffer"], dev, dt))
        k = torch.ones_like(value)
        e, dE = _flat_bottom_linear(value, k, lower, upper)
        if not need_grad:
            return _sum_energy(e)
        return _sum_energy(e), _aggregate_atom_gradients(coords, idx, grad_value, dE)

    def _project(self, coords, feats, params):
        idx = feats["chiral_index"]
        if idx.numel() == 0:
            return torch.zeros_like(coords)
        orient = feats["chiral_orientation"]
        buffer = float(params["buffer"])
        value, grad_value = _dihedral_value_and_grad(coords, idx, True)
        value = value * orient - buffer
        grad_value = grad_value * orient.reshape(*([1] * len(coords.shape[:-2])), 1, -1, 1)
        mask = value < 0
        if mask.sum() == 0:
            return torch.zeros_like(coords)
        delta_x = _solve_constraint_projection(coords, idx, value, grad_value, mask)
        if not bool(params.get("scale_x", True)):
            return delta_x

        device, dtype = coords.device, coords.dtype
        batch_shape = coords.shape[:-2]
        n_atom = coords.shape[-2]
        mask_atom = torch.zeros(n_atom, dtype=bool, device=device)
        mask_atom[idx] = True
        atom_chain_id = feats["asym_id"][..., feats["atom_to_token_idx"]][mask_atom]
        n_chain = int(atom_chain_id.max()) + 1
        masked_delta_x = delta_x[..., mask_atom, :]
        masked_coords = coords[..., mask_atom, :]
        cid = atom_chain_id.reshape(*([1] * len(batch_shape)), -1, 1)

        def _chain_sum(src, width):
            out = torch.zeros(*batch_shape, n_chain, width, device=device, dtype=dtype)
            return out.scatter_add_(-2, cid.expand(*batch_shape, -1, width), src)

        atom_count = _chain_sum(torch.ones_like(masked_coords[..., :1]), 1)
        center = (_chain_sum(masked_coords, 3) / atom_count.clamp_min(1))[..., atom_chain_id, :]
        rg_sum = _chain_sum(((masked_coords - center) ** 2).sum(dim=-1, keepdim=True), 1)
        rg = (rg_sum / atom_count.clamp_min(1))[..., atom_chain_id, :]
        new_x = masked_coords + masked_delta_x
        new_center = (_chain_sum(new_x, 3) / atom_count.clamp_min(1))[..., atom_chain_id, :]
        new_rg_sum = _chain_sum(((new_x - new_center) ** 2).sum(dim=-1, keepdim=True), 1)
        new_rg = (new_rg_sum / atom_count.clamp_min(1))[..., atom_chain_id, :].clamp_min(1e-12)
        new_x = (new_x - new_center) * ((rg / new_rg) ** 0.5) + center
        delta_x[..., mask_atom, :] = new_x - masked_coords
        return delta_x


@register
class PlanarImproperPotential(Potential):
    """Planarity (sp2) term `1 - sin(Y)` per `planar_improper_index [4, M]` improper, unit weight.

    `planar_improper_is_carbonyl` is read only for its shape and dtype (k = ones_like).
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"buffer": 0.1309}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["planar_improper_index"]
        is_carbonyl = feats["planar_improper_is_carbonyl"].to(coords.dtype)
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        energy, grad = _planar_improper_value_and_grad(coords, idx, need_grad)
        k = torch.ones_like(is_carbonyl)
        energy = k * energy
        e_sum = _sum_energy(energy)
        if not need_grad:
            return e_sum
        return e_sum, _aggregate_atom_gradients(coords, idx, grad, k)


@register
class LinearBondPotential(Potential):
    """Triple-bond linearity: angle of `linear_triple_bond_index [3, M]` >= pi - buffer (linear penalty)."""

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"buffer": 0.08726646259}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["linear_triple_bond_index"]
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        value, grad_value = _angle_value_and_grad(coords, idx, need_grad)
        k = torch.ones_like(value)
        lower = torch.full(
            (idx.shape[1],), torch.pi - float(params["buffer"]), device=coords.device, dtype=value.dtype
        )
        e, dE = _flat_bottom_linear(value, k, lower, None)
        if not need_grad:
            return _sum_energy(e)
        return _sum_energy(e), _aggregate_atom_gradients(coords, idx, grad_value, dE)


@register
class ExperimentalTorsionPotential(Potential):
    """ETKDG-style torsion energy `sum_{n=1..6} k_n (1 + s_n cos(n phi))`.

    feats: `experimental_torsion_index [4, M]`, `experimental_torsion_{force_constant,sign} [M, 6]`.
    cos(n phi) is expanded as Chebyshev polynomials of cos(phi).
    """

    def _eval(self, coords, feats, params, need_grad: bool):
        idx = feats["experimental_torsion_index"]
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        fc = feats["experimental_torsion_force_constant"]
        sg = feats["experimental_torsion_sign"]
        phi, grad_phi = _dihedral_value_and_grad(coords, idx, need_grad)
        cosphi = torch.cos(phi)
        cosphi2 = cosphi * cosphi
        cosphi3 = cosphi * cosphi2
        cosphi4 = cosphi * cosphi3
        cosphi5 = cosphi * cosphi4
        cosphi6 = cosphi * cosphi5
        cos2 = 2.0 * cosphi2 - 1.0
        cos3 = 4.0 * cosphi3 - 3.0 * cosphi
        cos4 = 8.0 * cosphi4 - 8.0 * cosphi2 + 1.0
        cos5 = 16.0 * cosphi5 - 20.0 * cosphi3 + 5.0 * cosphi
        cos6 = 32.0 * cosphi6 - 48.0 * cosphi4 + 18.0 * cosphi2 - 1.0
        energy = (
            fc[..., 0] * (1.0 + sg[..., 0] * cosphi)
            + fc[..., 1] * (1.0 + sg[..., 1] * cos2)
            + fc[..., 2] * (1.0 + sg[..., 2] * cos3)
            + fc[..., 3] * (1.0 + sg[..., 3] * cos4)
            + fc[..., 4] * (1.0 + sg[..., 4] * cos5)
            + fc[..., 5] * (1.0 + sg[..., 5] * cos6)
        )
        e_sum = _sum_energy(energy)
        if not need_grad:
            return e_sum
        sinphi = torch.sin(phi)
        dcos2 = 4.0 * cosphi
        dcos3 = 12.0 * cosphi2 - 3.0
        dcos4 = 32.0 * cosphi3 - 16.0 * cosphi
        dcos5 = 80.0 * cosphi4 - 60.0 * cosphi2 + 5.0
        dcos6 = 192.0 * cosphi5 - 192.0 * cosphi3 + 36.0 * cosphi
        dE_dphi = -sinphi * (
            fc[..., 0] * sg[..., 0]
            + fc[..., 1] * sg[..., 1] * dcos2
            + fc[..., 2] * sg[..., 2] * dcos3
            + fc[..., 3] * sg[..., 3] * dcos4
            + fc[..., 4] * sg[..., 4] * dcos5
            + fc[..., 5] * sg[..., 5] * dcos6
        )
        return e_sum, _aggregate_atom_gradients(coords, idx, grad_phi, dE_dphi)


@register
class VinaStericPotential(Potential):
    """AutoDock-Vina-like steric term (two Gaussians + quadratic repulsion) on inter-chain pairs.

    Candidates are all atom pairs across chains with more than one atom, excluding chain pairs
    joined by an `interchain_bond_index` bond; r_eq = r_vdw(i) + r_vdw(j). Only pairs with
    `dist < r_eq * (1 - buffer)` contribute. Dense path only (upstream's Triton kernel is not ported).
    """

    def __init__(self, default_params: Optional[dict[str, Any]] = None):
        defaults = {"buffer": 0.225}
        if default_params is not None:
            defaults.update(default_params)
        super().__init__(defaults)
        self._cache_key = None
        self._cache_val = None

    @staticmethod
    def _cache_key_from_feats(feats):
        def _tensor_id(x):
            return (x.data_ptr(), tuple(x.shape), str(x.dtype))

        asym_id = feats["asym_id"]
        b_idx = feats.get("interchain_bond_index")
        return (
            str(asym_id.device),
            _tensor_id(asym_id),
            _tensor_id(feats["atom_to_token_idx"]),
            _tensor_id(feats["ref_element"]),
            None if b_idx is None else _tensor_id(b_idx),
        )

    def _get_collision_candidates(self, feats):
        """Return `(sel_idx [2, M] long, r_eq [M] fp32)`, cached on feature tensor identity."""
        key = self._cache_key_from_feats(feats)
        if key == self._cache_key and self._cache_val is not None:
            return self._cache_val
        device = feats["asym_id"].device
        empty = (
            torch.empty((2, 0), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.float32),
        )

        def _cache_and_return(out):
            self._cache_key = key
            self._cache_val = out
            return out

        c_map = feats["asym_id"][..., feats["atom_to_token_idx"]]
        # Upstream passes two positional args to _cache_and_return on these two exits (TypeError);
        # here they return the empty candidate set, which is what the later exits return.
        if c_map.size(-1) == 0 or torch.all(c_map == c_map[..., :1]):
            return _cache_and_return(empty)
        n_chains = int(c_map.max().item()) + 1
        r_atom = _get_vdw_radii_128(device)[feats["ref_element"].argmax(dim=-1)]
        prohibited = torch.eye(n_chains, dtype=torch.bool, device=device)
        b_idx = feats.get("interchain_bond_index")
        if b_idx is not None and b_idx.numel() > 0:
            ca, cb = c_map[b_idx[0]], c_map[b_idx[1]]
            prohibited[ca, cb] = True
            prohibited[cb, ca] = True
        order = torch.argsort(c_map)
        c_sorted = c_map.index_select(0, order)
        chain_ids, counts = torch.unique_consecutive(c_sorted, return_counts=True)
        if chain_ids.numel() <= 1:
            return _cache_and_return(empty)
        chain_ids_cpu = chain_ids.to("cpu").tolist()
        counts_cpu = counts.to("cpu").tolist()
        prohibited_cpu = prohibited.to("cpu")
        # Single-atom chains (e.g. ions) take no part.
        valid_chain = {int(cid) for cid, cnt in zip(chain_ids_cpu, counts_cpu) if int(cnt) > 1}
        if len(valid_chain) <= 1:
            return _cache_and_return(empty)
        chain_atoms: dict[int, torch.Tensor] = {}
        start = 0
        for cid, cnt in zip(chain_ids_cpu, counts_cpu):
            end = start + int(cnt)
            if int(cid) in valid_chain:
                chain_atoms[int(cid)] = order[start:end]
            start = end
        i_list: list[torch.Tensor] = []
        j_list: list[torch.Tensor] = []
        for ii, ca in enumerate(chain_ids_cpu):
            if int(ca) not in valid_chain:
                continue
            a_atoms = chain_atoms[int(ca)]
            na = int(a_atoms.numel())
            if na == 0:
                continue
            for cb in chain_ids_cpu[ii + 1 :]:
                if int(cb) not in valid_chain or bool(prohibited_cpu[int(ca), int(cb)]):
                    continue
                b_atoms = chain_atoms[int(cb)]
                nb = int(b_atoms.numel())
                if nb == 0:
                    continue
                i_list.append(a_atoms.repeat_interleave(nb))
                j_list.append(b_atoms.repeat(na))
        if len(i_list) == 0:
            return _cache_and_return(empty)
        sel_idx = torch.stack((torch.cat(i_list, dim=0), torch.cat(j_list, dim=0)), dim=0)
        r_eq = r_atom[sel_idx[0]] + r_atom[sel_idx[1]]
        return _cache_and_return((sel_idx, r_eq))

    def _eval(self, coords, feats, params, need_grad: bool):
        idx, eq = self._get_collision_candidates(feats)
        if idx.numel() == 0:
            return _zeros(coords, need_grad)
        buf = float(params["buffer"])
        device, dtype = coords.device, coords.dtype
        batch_shape = coords.shape[:-2]
        n_atom = int(coords.shape[-2])
        b = math.prod(batch_shape) if len(batch_shape) > 0 else 1
        coords_b = coords.reshape(b, n_atom, 3)
        # Pass 1: distances of all candidates -> active mask.
        value0, _ = _distance_value_and_grad(coords_b, idx, False)
        thr = eq.to(value0.dtype) * (1.0 - buf)
        active = value0 < thr.unsqueeze(0)
        b_idx, m_idx = active.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            return _zeros(coords, need_grad)
        i_atom = idx[0].index_select(0, m_idx)
        j_atom = idx[1].index_select(0, m_idx)
        eq_a = eq.index_select(0, m_idx).to(dtype)
        v = value0[b_idx, m_idx].to(dtype)
        dist_diff = v - eq_a
        norm_d = dist_diff / 0.5
        g1 = -0.0356 * torch.exp(-(norm_d**2))
        g2 = -0.00516 * torch.exp(-(((dist_diff - 3.0) / 2.0) ** 2))
        rep = 0.840 * torch.where(dist_diff < 0.0, dist_diff**2, 0.0)
        e_pair = g1 + g2 + rep
        e_out = torch.zeros((b,), device=device, dtype=dtype)
        e_out.scatter_add_(0, b_idx, e_pair)
        e_out = e_out[0] if len(batch_shape) == 0 else e_out.reshape(*batch_shape)
        if not need_grad:
            return e_out
        # Pass 2: gradients on active pairs only.
        r = coords_b[b_idx, i_atom] - coords_b[b_idx, j_atom]
        norm = torch.linalg.norm(r, dim=-1).clamp_min(1e-8)
        r_hat = r / norm.unsqueeze(-1)
        dg1 = -2.0 * g1 * norm_d * (1.0 / 0.5)
        dg2 = -0.5 * g2 * (dist_diff - 3.0)
        drep = 0.840 * torch.where(dist_diff < 0.0, 2.0 * dist_diff, 0.0)
        gi = (dg1 + dg2 + drep).unsqueeze(-1) * r_hat
        gj = -gi
        g_flat = torch.zeros((b * n_atom, 3), device=device, dtype=dtype)
        g_flat.scatter_add_(0, (b_idx * n_atom + i_atom).unsqueeze(-1).expand(-1, 3), gi)
        g_flat.scatter_add_(0, (b_idx * n_atom + j_atom).unsqueeze(-1).expand(-1, 3), gj)
        return e_out, g_flat.reshape(*coords.shape)
