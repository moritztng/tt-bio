"""Parity of tt_bio.tfg potentials and TFGEngine.update against OpenDDE v1.2.0's own classes.

Live tests import upstream from $TFG_UPSTREAM (default /tmp/tfgsrc/up, a v1.2.0 checkout) and skip
when it is absent. The fixture tests compare against tests/data/tfg/engine_parity.pt, upstream outputs
on the same inputs; regenerate it with `python tests/test_tfg_engine.py --regen` (needs the clone).
All comparisons are torch.equal: the port is op-for-op upstream's arithmetic.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from tt_bio.tfg import engine as tt_engine
from tt_bio.tfg import potentials as tt_pot

UPSTREAM = Path(os.environ.get("TFG_UPSTREAM", "/tmp/tfgsrc/up"))
FIXTURE = Path(__file__).parent / "data" / "tfg" / "engine_parity.pt"
N_STEPS = 200

# Each case: (name, config edits, step_i, mc std, hook)
ENGINE_CASES = [
    ("default_step0", {}, 0, 0.0, False),
    ("last_step_hook_interval2_sched", {"interval2": True, "sched": True}, N_STEPS - 1, 0.0, True),
    ("mc_eps", {"mc": {"std": 0.05, "batch": 2}}, 7, 0.05, False),
]


def make_inputs(seed: int = 0):
    """Synthetic 3-sample, 201-atom complex: chain 0 (30 tokens x 4 atoms), chain 1 (60 one-atom
    tokens), chain 2 (5 tokens x 4 atoms), chain 3 (one ion atom). Coordinates are bonded random walks
    with the chains interpenetrating, so every term has violated constraints."""
    g = torch.Generator().manual_seed(seed)
    n_tok = [30, 60, 5, 1]
    atoms_per_tok = [4, 1, 4, 1]
    asym_id = torch.cat([torch.full((n,), c, dtype=torch.long) for c, n in enumerate(n_tok)])
    atom_to_token_idx = torch.repeat_interleave(
        torch.arange(sum(n_tok)), torch.tensor([a for a, n in zip(atoms_per_tok, n_tok) for _ in range(n)])
    )
    chain_atoms = [n * a for n, a in zip(n_tok, atoms_per_tok)]
    n_atom = sum(chain_atoms)
    starts = [sum(chain_atoms[:i]) for i in range(len(chain_atoms))]

    def walk(n, origin):
        d = torch.randn(n, 3, generator=g)
        d = d / d.norm(dim=-1, keepdim=True) * 1.5
        return origin + torch.cumsum(d, 0)

    origins = [torch.zeros(3), torch.tensor([2.0, 1.0, 0.0]), torch.tensor([-1.0, 3.0, 1.0]), torch.tensor([1.0, 1.0, 1.0])]
    base = torch.cat([walk(n, o) for n, o in zip(chain_atoms, origins)])
    x_noisy = base + 4.0 * torch.randn(3, n_atom, 3, generator=g)
    x0 = base + 0.4 * torch.randn(3, n_atom, 3, generator=g)

    # Elements (Z - 1): C, N, O, S on polymer/ligand, Zn for the ion.
    elem = torch.tensor([5, 6, 7, 15])[torch.randint(0, 4, (n_atom,), generator=g)]
    elem[starts[3]] = 29
    ref_element = torch.nn.functional.one_hot(elem, 128).float()

    def consecutive(chain, k, stride=1):
        """[k, M] windows of k consecutive atoms of `chain`, every `stride`-th start."""
        s, n = starts[chain], chain_atoms[chain]
        return torch.stack([torch.arange(s + j, s + n - (k - 1) + j, stride) for j in range(k)])

    bonds = torch.cat([consecutive(c, 2) for c in range(3)], 1)
    angles = torch.cat([consecutive(c, 3)[[0, 2]] for c in range(3)], 1)
    clash_i = torch.randint(0, n_atom, (300,), generator=g)
    clash_j = torch.randint(0, n_atom, (300,), generator=g)
    keep = clash_i != clash_j
    clash = torch.stack([clash_i[keep], clash_j[keep]])
    pd_index = torch.cat([bonds, angles, clash], 1)
    nb, na, nc = bonds.shape[1], angles.shape[1], clash.shape[1]
    is_bond = torch.cat([torch.ones(nb), torch.zeros(na), torch.zeros(nc)])
    is_angle = torch.cat([torch.zeros(nb), torch.ones(na), torch.zeros(nc)])
    is_angle[:10] = 1.0  # bond-and-angle pairs (state code 3)
    lb = torch.cat([torch.full((nb,), 1.4), torch.full((na,), 2.3), torch.full((nc,), 3.0)])
    ub = torch.cat([torch.full((nb,), 1.6), torch.full((na,), 2.6), torch.full((nc,), 1e9)])

    quads0 = consecutive(0, 4, stride=3)
    chiral = quads0[:, :20]
    stereo = consecutive(1, 4, stride=4)[:, :12]
    planar = consecutive(0, 4, stride=5)[[0, 2, 1, 3], :15]
    linear = consecutive(2, 3, stride=2)
    torsion = quads0[:, 5:30]
    nt = torsion.shape[1]

    user_i = starts[0] + torch.randint(0, chain_atoms[0], (8,), generator=g)
    user_j = starts[1] + torch.randint(0, chain_atoms[1], (8,), generator=g)
    feats = {
        "asym_id": asym_id,
        "atom_to_token_idx": atom_to_token_idx,
        "ref_element": ref_element,
        "pairwise_distance_index": pd_index,
        "pairwise_distance_is_bond": is_bond,
        "pairwise_distance_is_angle": is_angle,
        "pairwise_distance_lower_bound": lb,
        "pairwise_distance_upper_bound": ub,
        "chiral_index": chiral,
        "chiral_orientation": torch.where(torch.rand(chiral.shape[1], generator=g) > 0.5, 1.0, -1.0),
        "stereo_bond_index": stereo,
        "stereo_bond_orientation": (torch.rand(stereo.shape[1], generator=g) > 0.5).float(),
        "planar_improper_index": planar,
        "planar_improper_is_carbonyl": (torch.rand(planar.shape[1], generator=g) > 0.5).float(),
        "linear_triple_bond_index": linear,
        "experimental_torsion_index": torsion,
        "experimental_torsion_force_constant": 2.0 * torch.rand(nt, 6, generator=g),
        "experimental_torsion_sign": torch.where(torch.rand(nt, 6, generator=g) > 0.5, 1.0, -1.0),
        # Chain 1 to chain 2: their pair is excluded from the steric term.
        "interchain_bond_index": torch.tensor([[starts[1] + 3], [starts[2] + 2]]),
        "user_distance_restraint_index": torch.stack([user_i, user_j]),
        "user_distance_restraint_lower_bound": torch.full((8,), 3.0),
        "user_distance_restraint_upper_bound": torch.full((8,), 8.0),
    }
    return feats, x_noisy, x0


def guidance_cfg(edits):
    cfg = tt_engine.default_guidance_config()
    cfg["enable"] = True
    if edits.get("interval2"):
        cfg["terms"]["StereoBondPotential"]["interval"] = 2
    if edits.get("sched"):
        cfg["terms"]["LinearBondPotential"]["weight"] = {"type": "exp_interpolation", "start": 0.5, "end": 0.1, "alpha": 2.0}
        cfg["terms"]["PairwiseDistancePotential"]["clash_buffer"] = {"type": "const", "value": 0.05}
    if "mc" in edits:
        cfg["mc"] = dict(edits["mc"])
    return cfg


def hook(x0, step_i):
    """Stand-in for the rigid x0 pass: a rigid shift of chain 1 that depends on step_i."""
    out = x0.clone()
    out[..., 120:180, :] += 0.01 * (step_i + 1)
    return out


def term_params():
    """(name, params) per registered potential, params from the default guidance terms."""
    terms = tt_engine.default_guidance_config()["terms"]
    out = []
    for name in sorted(tt_pot.CLASS_REGISTRY):
        p = {k: v for k, v in terms.get(name, {}).items() if k not in ("interval", "weight", "enable_projection")}
        out.append((name, p))
    return out


T_HAT, C_TAU, ETA = 14.25, 12.5, 1.5


def run_tt(feats, x_noisy, x0, case):
    name, edits, step_i, std, use_hook = case
    eng = tt_engine.TFGEngine(tt_engine.parse_tfg_config(guidance_cfg(edits)))
    gen = torch.Generator().manual_seed(1234)
    return eng.update(
        x_noisy,
        x0,
        t_hat=torch.tensor(T_HAT),
        c_tau=torch.tensor(C_TAU),
        step_scale_eta=ETA,
        step_i=step_i,
        num_diffusion_steps=N_STEPS,
        feats=feats,
        x0_hook=hook if use_hook else None,
        generator=gen,
    )


def tt_potential_outputs(feats, coords):
    out = {}
    for name, p in term_params():
        pot = tt_pot.CLASS_REGISTRY[name]()
        e, g = pot.energy_and_grad(coords, feats, p)
        out[name] = {"energy": pot.energy(coords, feats, p), "e": e, "grad": g, "project": pot.project(coords, feats, p)}
    return out


# ---------------------------------------------------------------- upstream

def _upstream():
    if not (UPSTREAM / "opendde" / "tfg" / "engine.py").exists():
        return None
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    os.environ["OPENDDE_VINA_FAST"] = "off"
    os.environ["OPENDDE_RIGID_CONTACT"] = "off"
    from opendde.tfg import config as up_config
    from opendde.tfg import engine as up_engine
    from opendde.tfg import epitope_guidance as up_epi
    from opendde.tfg import potentials as up_pot

    return up_config, up_engine, up_pot, up_epi


@pytest.fixture
def upstream(monkeypatch):
    mods = _upstream()
    if mods is None:
        pytest.skip(f"OpenDDE v1.2.0 clone not found at {UPSTREAM} (set TFG_UPSTREAM)")
    monkeypatch.setenv("OPENDDE_VINA_FAST", "off")
    monkeypatch.setenv("OPENDDE_RIGID_CONTACT", "off")
    return mods


def run_upstream(mods, feats, x_noisy, x0, case, patch_hook):
    up_config, up_engine, _, up_epi = mods
    name, edits, step_i, std, use_hook = case
    patch_hook(up_epi, "guide_x0", (lambda x, f, s: hook(x, s)) if use_hook else (lambda x, f, s: x))
    eng = up_engine.TFGEngine(up_config.parse_tfg_config(guidance_cfg(edits)), device=torch.device("cpu"), dtype=torch.float32)
    gen = torch.Generator().manual_seed(1234)
    return eng.step(
        lambda **kw: x0,
        x=x_noisy,
        t_hat=torch.tensor(T_HAT),
        c_tau=torch.tensor(C_TAU),
        step_scale_eta=ETA,
        step_i=step_i,
        num_diffusion_steps=N_STEPS,
        input_feature_dict=feats,
        s_inputs=None,
        s_trunk=None,
        z_trunk=None,
        pair_z=None,
        p_lm=None,
        c_l=None,
        chunk_size=None,
        inplace_safe=False,
        enable_efficient_fusion=False,
        torch_generator=gen,
    )


def upstream_potential_outputs(mods, feats, coords):
    up_pot = mods[2]
    out = {}
    for name, p in term_params():
        pot = up_pot.CLASS_REGISTRY[name]()
        e, g = pot.energy_and_grad(coords, feats, p)
        out[name] = {"energy": pot.energy(coords, feats, p), "e": e, "grad": g, "project": pot.project(coords, feats, p)}
    return out


def _assert_equal_tree(ours, ref, where=""):
    if isinstance(ref, dict):
        assert set(ours) == set(ref), where
        for k in ref:
            _assert_equal_tree(ours[k], ref[k], f"{where}/{k}")
        return
    assert ours.shape == ref.shape and ours.dtype == ref.dtype, where
    assert torch.equal(ours, ref), f"{where}: max abs diff {(ours - ref).abs().max().item():.3e}"


def _nonzero_check(pot_out):
    """Every term carries signal on these inputs (a vacuous parity is not a parity)."""
    for name, o in pot_out.items():
        assert o["grad"].abs().max() > 0, name


# ---------------------------------------------------------------- live upstream tests

def test_registry_matches_upstream(upstream):
    assert sorted(tt_pot.CLASS_REGISTRY) == sorted(upstream[2].CLASS_REGISTRY)


def test_vdw_table_matches_upstream(upstream):
    assert torch.equal(tt_pot._VDW_RADII_128_CPU, upstream[2]._VDW_RADII_128_CPU)


@pytest.mark.parametrize("batched", [True, False])
def test_potentials_live(upstream, batched):
    feats, _, x0 = make_inputs()
    coords = x0 if batched else x0[0]
    ours = tt_potential_outputs(feats, coords)
    ref = upstream_potential_outputs(upstream, feats, coords)
    _nonzero_check(ref)
    _assert_equal_tree(ours, ref)


@pytest.mark.parametrize("case", ENGINE_CASES, ids=[c[0] for c in ENGINE_CASES])
def test_engine_update_live(upstream, monkeypatch, case):
    feats, x_noisy, x0 = make_inputs()
    ref = run_upstream(upstream, feats, x_noisy, x0, case, monkeypatch.setattr)
    ours = run_tt(feats, x_noisy, x0, case)
    _assert_equal_tree(ours, ref)


def test_parse_config_matches_upstream(upstream):
    up_cfg = upstream[0].parse_tfg_config(guidance_cfg({"sched": True}))
    ours = tt_engine.parse_tfg_config(guidance_cfg({"sched": True}))
    for f in ("enable", "rho", "mu", "eps_std", "eps_batch", "outer_steps", "inner_steps",
              "projection_outer_steps", "projection_inner_steps", "log_last_step_energy"):
        assert getattr(ours, f) == getattr(up_cfg, f), f
    assert [t.name for t in ours.terms] == [t.name for t in up_cfg.terms]
    for a, b in zip(ours.terms, up_cfg.terms):
        assert (a.interval, a.enable_projection) == (b.interval, b.enable_projection)
        for t in (1.0, 0.37, 0.0):
            assert a.weight(t) == b.weight(t)
            assert {k: (v(t) if callable(v) else v) for k, v in a.param_templates.items()} == \
                   {k: (v(t) if callable(v) else v) for k, v in b.param_templates.items()}


def test_default_config_matches_upstream_model_base():
    path = UPSTREAM / "opendde" / "config" / "model_base.py"
    if not path.exists():
        pytest.skip("upstream clone absent")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from opendde.config.model_base import model_configs

    assert tt_engine.default_guidance_config() == model_configs["sample_diffusion"]["guidance"]


# ---------------------------------------------------------------- fixture tests (no clone needed)

@pytest.fixture(scope="module")
def fixture_data():
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE} missing; regenerate with `python {__file__} --regen`")
    d = torch.load(FIXTURE, weights_only=True)
    d["feats"]["ref_element"] = torch.nn.functional.one_hot(d["feats"]["ref_element"], 128).float()
    return d


def test_potentials_fixture(fixture_data):
    d = fixture_data
    _assert_equal_tree(tt_potential_outputs(d["feats"], d["x0"]), d["potentials"])


@pytest.mark.parametrize("case", ENGINE_CASES, ids=[c[0] for c in ENGINE_CASES])
def test_engine_update_fixture(fixture_data, case):
    d = fixture_data
    _assert_equal_tree(run_tt(d["feats"], d["x_noisy"], d["x0"], case), d["engine"][case[0]])


# ---------------------------------------------------------------- port-only behaviour

def test_rejects_rho_and_outer():
    for edit in ({"rho": 0.1}, {"steps": {"tfg_outer": 2}}):
        cfg = tt_engine.default_guidance_config()
        cfg.update(edit)
        with pytest.raises(ValueError):
            tt_engine.TFGEngine(tt_engine.parse_tfg_config(cfg))


def test_vina_single_chain_is_zero():
    """Upstream raises TypeError here (two-arg _cache_and_return); the port returns zero energy."""
    feats, _, x0 = make_inputs()
    feats = dict(feats, asym_id=torch.zeros_like(feats["asym_id"]))
    e, g = tt_pot.VinaStericPotential().energy_and_grad(x0, feats)
    assert not e.any() and not g.any()


def test_validate_features_names_missing_keys():
    feats, x_noisy, x0 = make_inputs()
    del feats["chiral_index"]
    eng = tt_engine.TFGEngine(tt_engine.parse_tfg_config(guidance_cfg({})))
    with pytest.raises(KeyError, match="chiral_index"):
        eng.update(x_noisy, x0, t_hat=T_HAT, c_tau=C_TAU, step_scale_eta=ETA, step_i=0,
                   num_diffusion_steps=N_STEPS, feats=feats)


def _regen():
    import contextlib

    mods = _upstream()
    assert mods is not None, f"no upstream clone at {UPSTREAM}"
    feats, x_noisy, x0 = make_inputs()
    engine_out = {}
    for case in ENGINE_CASES:
        patches = []

        def patch(obj, attr, val):
            patches.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, val)

        try:
            engine_out[case[0]] = run_upstream(mods, feats, x_noisy, x0, case, patch)
        finally:
            for obj, attr, val in patches:
                setattr(obj, attr, val)
    # ref_element stored as element indices (the one-hot is 100 KB); the loader re-expands it.
    data = {
        "feats": dict(feats, ref_element=feats["ref_element"].argmax(-1)),
        "x_noisy": x_noisy,
        "x0": x0,
        "potentials": upstream_potential_outputs(mods, feats, x0),
        "engine": engine_out,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        FIXTURE.unlink()
    torch.save(data, FIXTURE)
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")


if __name__ == "__main__" and "--regen" in sys.argv:
    _regen()
