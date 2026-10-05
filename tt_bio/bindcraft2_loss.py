"""Add, reweight or replace a BindCraft 2 loss term, without editing BindCraft 2.

BindCraft 2 already keeps its design objective as **named, weighted terms** rather than one
opaque scalar: ``@loss('<name>')`` in ``bindcraft/loss.py`` registers a function of
``(protein_states, predictions)`` into ``REGISTERED_LOSSES``, and
``build_design_losses(settings, ...)`` turns the ones with a non-zero ``weights_<name>`` setting
into a ``{name: DesignLoss}`` dict once per trajectory. What it does not have is a way in from
the outside, so this module is that way in::

    import jax.numpy as jnp
    from bindcraft.loss import chain_residue_slices
    from tt_bio import bindcraft2

    def interface_plddt(protein_states, predictions, prediction_state="complex",
                        binder="binder"):
        rows = chain_residue_slices(protein_states[prediction_state])[binder]
        return 1.0 - predictions[prediction_state].metrics["plddt"][rows].mean()

    with bindcraft2.predictor(card=0) as build, \\
         bindcraft2.loss_terms(add={"interface_plddt": (interface_plddt, 0.4)},
                               weight={"interface_contacts": 0.5}):
        campaign.run_campaign(settings, project, af2_weights=params, mpnn_weights=mpnn)

**One seam reaches both places a loss is used.** The dict
``build_design_losses`` returns is consumed by the gradient path
(``trajectory.py`` -> ``af2.py``, inside ``jax.jit(jax.value_and_grad(...))``) and by the
evaluation path (``weighted_design_loss`` in the mutation stage, and the switch metrics), and
each term's value is exported as a metric into the trajectory's ``losses.csv``. So a term added
here differentiates, scores, filters and is logged, with nothing further to wire.

**Your term is differentiated by JAX, not by tt-bio's tape.** tt-bio puts AlphaFold 2's Evoformer
on card as a ``jax.custom_vjp`` over a ``jax.pure_callback`` (``tt_bio/bindcraft2.py``), whose
backward is tt-bio's hand-written tape. Your loss sits *downstream* of that primitive, on the
heads' outputs, so the only rules it has to obey are ``jax.jit``'s and ``jax.grad``'s -- the tape
never sees it and there is no VJP of ours to be missing. What *can* go wrong is a term whose
gradient is silently **zero**: ``argmax``, ``round``, ``>``, an integer cast, ``stop_gradient``,
or indexing by a traced ``argsort`` all differentiate to 0.0 without complaint, and a term like
that runs for an hour changing nothing. ``loss_terms`` screens every term you add or replace for
exactly that before the campaign starts, and :func:`check_gradient` grades one against float64
central differences.
"""
from __future__ import annotations

import contextlib
import difflib
import functools
import inspect
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping

__all__ = ["Term", "NewTerm", "loss_term", "terms", "INTERMEDIATES", "loss_terms",
           "synthetic_design", "check_gradient", "LossTermError", "UnknownTerm",
           "DeadGradient", "TermSignature"]


class LossTermError(ValueError):
    """A custom loss term tt-bio refuses to install, with the reason."""


class UnknownTerm(LossTermError):
    """A name that is not one of BindCraft 2's registered terms."""


class DeadGradient(LossTermError):
    """A term whose gradient is identically zero or not finite, so it cannot move a design."""


class TermSignature(LossTermError):
    """A term whose signature BindCraft 2's builder cannot bind."""


#: Every intermediate a term may read, where it comes from, and what it means. ``metrics`` keys
#: are read off ``predictions[state].metrics``; the rest are fields of
#: ``protein_states[state][chain]``, a ``bindcraft.protein.Protein``. ``N`` is the padded token
#: count of that state, chains concatenated in ``sorted()`` order -- which is what
#: ``bindcraft.loss.chain_residue_slices`` gives you, and the only correct way to index a chain.
INTERMEDIATES: dict[str, tuple[str, str, str]] = {
    # key: (source, shape, meaning)
    "distogram": ("metrics", "[N, N, bins]",
                  "pair distance logits, 64 bins; bin edges from "
                  "bindcraft.loss.distogram_bin_distances(bins). Softmax it yourself."),
    "pae": ("metrics", "[N, N]",
            "predicted aligned error in Angstrom, 0 to 31; row i aligned on token j. "
            "BindCraft 2's own terms divide by 31.0 to work in 0-1."),
    "plddt": ("metrics", "[N]", "per-token confidence, 0 to 1 (not 0-100)"),
    "ptm": ("metrics", "scalar", "predicted TM-score of the whole state, 0 to 1"),
    "iptm": ("metrics", "scalar", "interface predicted TM-score, 0 to 1"),
    "experimentally_resolved_ca": ("metrics", "[N]",
                                   "probability the CA of this token is experimentally "
                                   "resolved, 0 to 1"),
    "sequence": ("Protein", "[L_chain, 20]",
                 "the design's own sequence logits, amino acids in bindcraft.protein.AMINO_ACIDS "
                 "order. This is what the optimiser updates, so a term may read it directly."),
    "atoms": ("Protein", "[L_chain, 37, 3]",
              "atom37 coordinates in Angstrom; index with bindcraft.protein.ATOM_INDEX"),
    "atom_mask": ("Protein", "[L_chain, 37]", "which atoms are present (bool, not differentiable)"),
    "flags": ("Protein", "[L_chain]",
              "bindcraft.protein.ResidueFlags bitfield: DESIGN, TEMPLATE, SEQUENCE, CONTACT, "
              "HOTSPOT, COLDSPOT, CYCLIC, PADDING. Read it with has_residue_flag; always mask "
              "PADDING out of a mean (real_residue_mask does it)."),
    "residue_index": ("Protein", "[L_chain]", "author residue numbering (int, not differentiable)"),
}

#: The float arrays :func:`synthetic_design` makes differentiable, so a screen and a gradcheck
#: cover everything a term can actually pull a gradient through.
_DIFFERENTIABLE = ("distogram", "pae", "plddt", "ptm", "iptm", "experimentally_resolved_ca",
                   "sequence", "atoms")


@dataclass(frozen=True)
class Term:
    """One of BindCraft 2's registered loss terms, as tt-bio reports it."""

    name: str
    #: The settings key that weights it. ``weights_<name>``.
    setting: str
    #: ``binder_only``, ``binds_target``, ``avoids_target`` or ``every_target``: how
    #: ``build_design_losses`` scales the term for each target state.
    target_weighting: str
    #: Keyword parameters beyond ``(protein_states, predictions)``, all with defaults.
    parameters: tuple[str, ...]
    #: Whether it takes ``prediction_state``, i.e. whether BindCraft 2 fans it out per target
    #: state (and so renames it ``<name>.<state>`` when there is more than one).
    per_target: bool
    #: First line of its docstring, or "" -- BindCraft 2's terms are mostly undocumented.
    doc: str

    @property
    def reads(self) -> tuple[str, ...]:
        """Which :data:`INTERMEDIATES` the term's own source mentions. A hint, not a contract."""
        return _reads(_registered()[self.name])


@dataclass(frozen=True)
class NewTerm:
    """A term to add, as :func:`loss_term` returns it."""

    function: Callable
    weight: float | tuple[float, ...]
    target_weighting: str


def loss_term(function: Callable, weight: float | tuple[float, ...] = 1.0, *,
              target_weighting: str = "binder_only") -> NewTerm:
    """Describe a term for ``loss_terms(add=...)``.

    ``weight`` may be a list or tuple, which BindCraft 2 then samples per trajectory seed exactly
    as it does for its own terms -- the same mechanism as a ``weights_*`` range in a settings
    file, so a custom term can be swept without new machinery.

    ``target_weighting`` is BindCraft 2's own vocabulary and decides what a multi-target campaign
    does with the term: ``binder_only`` (the default -- one copy, weight 1), ``binds_target``,
    ``avoids_target``, or ``every_target``. It only has an effect if your term takes a
    ``prediction_state`` parameter.
    """
    if target_weighting not in _TARGET_WEIGHTINGS:
        raise LossTermError(f"target_weighting={target_weighting!r} is not one of "
                            f"{', '.join(sorted(_TARGET_WEIGHTINGS))}")
    return NewTerm(function, weight, target_weighting)


_TARGET_WEIGHTINGS = frozenset({"binder_only", "binds_target", "avoids_target", "every_target"})


def _loss_module():
    """``bindcraft.loss``, or a refusal that says what to install."""
    try:
        import bindcraft.loss as module
    except ImportError as exc:
        raise LossTermError(
            "BindCraft 2 is not importable, so there are no loss terms to change. tt-bio does "
            "not ship or serve BindCraft 2: install it yourself from "
            "https://github.com/PacesaLab/BindCraft2 under your own licence, and tt-bio binds to "
            f"it ({exc})") from exc
    return module


def _registered() -> dict[str, Callable]:
    return _loss_module().REGISTERED_LOSSES


def _reads(function: Callable) -> tuple[str, ...]:
    """Which intermediates a term's source text mentions, for the report in :func:`terms`."""
    target = getattr(function, "func", function)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return ()
    return tuple(key for key in INTERMEDIATES if key in source)


def terms() -> dict[str, Term]:
    """Every loss term BindCraft 2 has registered, by name.

    Read off ``REGISTERED_LOSSES`` live rather than from a table here, so a BindCraft 2 release
    that adds a term shows up without a tt-bio change, and so this never claims a term that is
    not there. Terms you added with :func:`loss_terms` appear while that context is open.
    """
    module = _loss_module()
    out = {}
    for name, function in sorted(module.REGISTERED_LOSSES.items()):
        parameters = inspect.signature(function).parameters
        keywords = tuple(n for n in parameters if n not in ("protein_states", "predictions"))
        out[name] = Term(name=name, setting=f"weights_{name}",
                         target_weighting=module.LOSS_TARGET_WEIGHTING.get(name, "binder_only"),
                         parameters=keywords, per_target="prediction_state" in parameters,
                         doc=next(iter((inspect.getdoc(function) or "").splitlines()), "").strip())
    return out


def _check_signature(name: str, function: Callable, *, like: Callable | None = None) -> None:
    """Refuse a term BindCraft 2's builder would bind wrongly, saying which rule it broke."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError) as exc:
        raise TermSignature(f"loss term {name!r} has no inspectable signature ({exc}); it must be "
                            "a plain function of (protein_states, predictions, **keywords)"
                            ) from exc
    positional = [p for p in parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) < 2:
        raise TermSignature(
            f"loss term {name!r} takes {len(positional)} positional parameter(s); BindCraft 2 "
            "calls a term as fn(protein_states, predictions, **keywords), so the first two are "
            "mandatory")
    for parameter in list(parameters.values())[2:]:
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect.Parameter.empty:
            raise TermSignature(
                f"loss term {name!r} has parameter {parameter.name!r} with no default. Every "
                "parameter past the first two must default, because BindCraft 2 only passes the "
                "ones a settings file names under losses.<term>.params")
    if like is not None:
        original = inspect.signature(like).parameters
        for shared in ("prediction_state", "reference_state", "binder_shapes"):
            if (shared in original) != (shared in parameters):
                raise TermSignature(
                    f"loss term {name!r} replaces one whose signature "
                    f"{'has' if shared in original else 'does not have'} {shared!r} while the "
                    f"replacement {'does' if shared in parameters else 'does not'}. That changes "
                    "how BindCraft 2 fans the term out over states, so the weights would no "
                    "longer mean the same thing; match the original or add a new term instead")


def _closest(name: str, known) -> str:
    near = difflib.get_close_matches(name, sorted(known), n=3, cutoff=0.4)
    return f"; did you mean {', '.join(near)}?" if near else ""


@contextlib.contextmanager
def loss_terms(*, add: Mapping[str, object] | None = None,
               weight: Mapping[str, float | tuple[float, ...]] | None = None,
               replace: Mapping[str, Callable] | None = None,
               check: bool = True,
               design: tuple | None = None) -> Iterator[dict[str, Term]]:
    """Install loss-term changes for the duration, and put BindCraft 2 back exactly on exit.

    ``add`` maps a new name to ``(function, weight)``, to a bare ``function`` (weight 1.0), or to
    a :func:`loss_term` for the other options. ``weight`` reweights a term that already exists --
    including one of BindCraft 2's own, and including ``0`` to switch it off. ``replace`` swaps
    the function behind an existing name and keeps its weight and target weighting.

    Nothing about your settings file changes: the weights are injected where
    ``bindcraft.loss.build_losses`` reads them, so a term added here is picked up by the same
    ``build_design_losses`` call every trajectory already makes, and therefore by the gradient
    path, the evaluation path and ``losses.csv`` alike.

    With ``check=True`` (the default) every added or replaced term is run once on a small
    synthetic design and its gradient screened: a term that returns a non-scalar, a non-finite
    value, or a gradient that is identically zero with respect to every intermediate it could
    read is refused here, before a campaign spends hours on it. Pass ``design=`` a
    ``(protein_states, predictions)`` pair of your own if the synthetic one does not fit your
    term's chain layout, or ``check=False`` to skip it.

    Yields the term table as it now stands (:func:`terms`).
    """
    module = _loss_module()
    added = {name: _normalise(name, spec) for name, spec in (add or {}).items()}
    weights = dict(weight or {})
    replaced = dict(replace or {})

    for name, spec in added.items():
        if name in module.REGISTERED_LOSSES:
            raise LossTermError(
                f"loss term {name!r} already exists in BindCraft 2 (weight setting "
                f"weights_{name}). Use replace={{{name!r}: fn}} to change what it computes, or "
                "weight= to change how much it counts, or pick another name")
        _check_signature(name, spec.function)
    for name in replaced:
        if name not in module.REGISTERED_LOSSES:
            raise UnknownTerm(f"cannot replace loss term {name!r}: BindCraft 2 does not register "
                              f"it{_closest(name, module.REGISTERED_LOSSES)}")
    for name, function in replaced.items():
        _check_signature(name, function, like=module.REGISTERED_LOSSES[name])
    for name in weights:
        if name not in module.REGISTERED_LOSSES and name not in added:
            raise UnknownTerm(f"cannot reweight loss term {name!r}: BindCraft 2 does not register "
                              f"it and add= does not define it"
                              f"{_closest(name, set(module.REGISTERED_LOSSES) | set(added))}")

    if check and (added or replaced):
        screened = {**{n: s.function for n, s in added.items()}, **replaced}
        for name, function in screened.items():
            _screen(name, function, design=design)

    overlay = {**{name: spec.weight for name, spec in added.items()}, **weights}
    restore_registered = {name: module.REGISTERED_LOSSES.get(name, _ABSENT)
                          for name in (*added, *replaced)}
    restore_weighting = {name: module.LOSS_TARGET_WEIGHTING.get(name, _ABSENT)
                         for name in (*added, *replaced)}
    real_build_losses = module.build_losses

    @functools.wraps(real_build_losses)
    def build_losses(settings: dict, seed: int = 0):
        merged = dict(settings)
        for name, value in overlay.items():
            merged[f"weights_{name}"] = value
        return real_build_losses(merged, seed)

    aliases = [m for m in list(sys.modules.values())
               if getattr(m, "build_losses", None) is real_build_losses]
    try:
        for name, spec in added.items():
            module.REGISTERED_LOSSES[name] = spec.function
            module.LOSS_TARGET_WEIGHTING[name] = spec.target_weighting
        for name, function in replaced.items():
            module.REGISTERED_LOSSES[name] = function
        for alias in aliases:
            alias.build_losses = build_losses
        yield terms()
    finally:
        for alias in aliases:
            alias.build_losses = real_build_losses
        for table, saved in ((module.REGISTERED_LOSSES, restore_registered),
                             (module.LOSS_TARGET_WEIGHTING, restore_weighting)):
            for name, value in saved.items():
                if value is _ABSENT:
                    table.pop(name, None)
                else:
                    table[name] = value


_ABSENT = object()


def _normalise(name: str, spec) -> NewTerm:
    if isinstance(spec, NewTerm):
        return spec
    if callable(spec):
        return loss_term(spec)
    if isinstance(spec, tuple) and len(spec) == 2 and callable(spec[0]):
        return loss_term(spec[0], spec[1])
    raise LossTermError(
        f"add={{{name!r}: ...}} wants a function, a (function, weight) pair, or a "
        f"bindcraft2.loss_term(...); got {type(spec).__name__}")


# --------------------------------------------------------------------------------------------
# A synthetic design, for the screen and the gradcheck.
# --------------------------------------------------------------------------------------------


def synthetic_design(*, binder: int = 24, target: int = 40, bins: int = 64, seed: int = 0,
                     dtype=None, states: tuple[str, ...] = ("complex", "binder_alone")):
    """A small, non-degenerate ``(protein_states, predictions)`` pair, for grading a term.

    Not a prediction of anything: the coordinates are a smooth open helix, the confidences are a
    deterministic pseudo-random field, and the only thing that matters is that nothing is
    constant, zero or exactly symmetric, so a gradient that *should* be non-zero is. The
    ``complex`` state carries chains ``binder`` and ``target``; ``binder_alone`` carries only
    ``binder``, which is how BindCraft 2 names the monomer state.

    Two things here are deliberate rather than convenient, because without them a quarter of
    BindCraft 2's own terms are constant and a screen built on this would refuse them:

    * ``predictions[state].protein_complex`` is **not** ``protein_states[state]``. In a campaign
      the first is what AlphaFold 2 predicted and the second is the template and design input,
      and the terms that compare them (``target_rmsd``, ``target_rigidity``,
      ``induced_fit_*``, ``fold_switching``) are identically zero if they are the same object.
      So the predicted coordinates carry a smooth, residue-dependent displacement.
    * the target chain gets a ``HOTSPOT`` and a run of ``COLDSPOT`` residues, which is what the
      coldspot and hotspot terms select on; with no flags set their mask is empty.

    ``dtype=jnp.float64`` with x64 enabled is what :func:`check_gradient` uses.
    """
    import jax
    import jax.numpy as jnp
    from bindcraft.protein import ATOM_INDEX, AMINO_ACIDS, Protein, ResidueFlags

    dtype = dtype or jnp.float32
    n_atoms = max(ATOM_INDEX.values()) + 1

    def chain(length: int, offset: int, flag: ResidueFlags) -> "Protein":
        key = jax.random.PRNGKey(seed + offset)
        index = jnp.arange(length, dtype=dtype)
        # A smooth open helix: distinct distances, no exact symmetry, nothing superimposed.
        turn = 1.7 * (index + 0.3 * offset)
        backbone = jnp.stack([6.0 * jnp.cos(turn), 6.0 * jnp.sin(turn),
                              1.5 * index + 7.0 * offset], axis=-1)
        atoms = jnp.zeros((length, n_atoms, 3), dtype=dtype)
        atoms = atoms.at[:, ATOM_INDEX["CA"]].set(backbone)
        atoms = atoms.at[:, ATOM_INDEX["CB"]].set(backbone + 1.52)
        atoms = atoms.at[:, ATOM_INDEX["N"]].set(backbone - 1.33)
        atoms = atoms.at[:, ATOM_INDEX["C"]].set(backbone + jnp.asarray([1.0, 0.5, 0.2]))
        mask = jnp.zeros((length, n_atoms), dtype=bool)
        for atom in ("N", "CA", "C", "CB"):
            mask = mask.at[:, ATOM_INDEX[atom]].set(True)
        flags = jnp.full((length,), int(flag), dtype=jnp.uint8)
        if flag is ResidueFlags.DESIGN:
            sequence = 0.5 * jax.random.normal(key, (length, len(AMINO_ACIDS)), dtype=dtype)
        else:
            sequence = jax.nn.one_hot(jax.random.randint(key, (length,), 0, len(AMINO_ACIDS)),
                                      len(AMINO_ACIDS), dtype=dtype)
            flags = flags.at[length // 3].set(int(flag | ResidueFlags.HOTSPOT))
            flags = flags.at[2 * length // 3:2 * length // 3 + 3].set(
                int(flag | ResidueFlags.COLDSPOT))
        return Protein(sequence=sequence, atoms=atoms, atom_mask=mask, flags=flags,
                       residue_index=jnp.arange(1, length + 1, dtype=jnp.int32))

    def predicted(protein, offset: int) -> "Protein":
        """What the model returned, as opposed to the template that went in."""
        index = jnp.arange(len(protein), dtype=dtype)[:, None, None]
        shift = jnp.stack([0.9 * jnp.sin(0.4 * index + offset),
                           0.7 * jnp.cos(0.3 * index + offset),
                           0.5 + 0.02 * index], axis=-1).reshape(len(protein), 1, 3)
        return protein.replace(atoms=protein.atoms + shift.astype(dtype))

    templates = {
        "complex": {"binder": chain(binder, 0, ResidueFlags.DESIGN),
                    "target": chain(target, 1, ResidueFlags.TEMPLATE | ResidueFlags.SEQUENCE)},
        "binder_alone": {"binder": chain(binder, 0, ResidueFlags.DESIGN)},
    }
    protein_states, predictions = {}, {}
    for position, state in enumerate(states):
        if state not in templates:
            raise LossTermError(f"synthetic_design does not know state {state!r}; it builds "
                                f"{', '.join(templates)}")
        template = templates[state]
        tokens = sum(len(protein) for protein in template.values())
        key = jax.random.PRNGKey(seed + 100 + position)
        plddt_key, pae_key, dist_key, res_key = jax.random.split(key, 4)
        pae = 31.0 * jax.random.uniform(pae_key, (tokens, tokens), dtype=dtype)
        predictions[state] = _prediction(
            {name: predicted(protein, position + 1)
             for name, protein in template.items()},
            {"plddt": jax.random.uniform(plddt_key, (tokens,), dtype=dtype),
             "pae": 0.5 * (pae + pae.T),
             "distogram": jax.random.normal(dist_key, (tokens, tokens, bins), dtype=dtype),
             "experimentally_resolved_ca": jax.random.uniform(res_key, (tokens,), dtype=dtype),
             "ptm": jnp.asarray(0.71, dtype=dtype),
             "iptm": jnp.asarray(0.43, dtype=dtype)})
        protein_states[state] = template
    return protein_states, predictions


def _prediction(protein_complex, metrics):
    from bindcraft.protein import StructurePrediction

    return StructurePrediction(protein_complex=protein_complex, metrics=metrics)


def _leaves(protein_states, predictions) -> dict[str, object]:
    """Every float array a term can pull a gradient through, keyed by a readable path.

    Both structures are in here. ``design/`` is ``protein_states[state][chain]``, the template
    and the sequence the optimiser updates; ``folded/`` is
    ``predictions[state].protein_complex[chain]``, what AlphaFold 2 returned. A term that
    compares the two reads both, and a screen that only differentiated one of them would call
    such a term dead.
    """
    out = {}
    for state, prediction in predictions.items():
        for key, value in prediction.metrics.items():
            if key in _DIFFERENTIABLE:
                out[f"{state}/metrics/{key}"] = value
        for name, protein in prediction.protein_complex.items():
            out[f"{state}/folded/{name}/atoms"] = protein.atoms
    for state, protein_complex in protein_states.items():
        for name, protein in protein_complex.items():
            out[f"{state}/design/{name}/sequence"] = protein.sequence
            out[f"{state}/design/{name}/atoms"] = protein.atoms
    return out


def _rebuild(leaves, protein_states, predictions):
    """``(protein_states, predictions)`` with every leaf in :func:`_leaves` swapped for ``leaves``."""
    states = {
        state: {name: protein.replace(sequence=leaves[f"{state}/design/{name}/sequence"],
                                      atoms=leaves[f"{state}/design/{name}/atoms"])
                for name, protein in protein_complex.items()}
        for state, protein_complex in protein_states.items()}
    built = {}
    for state, prediction in predictions.items():
        built[state] = _prediction(
            {name: protein.replace(atoms=leaves[f"{state}/folded/{name}/atoms"])
             for name, protein in prediction.protein_complex.items()},
            {key: leaves.get(f"{state}/metrics/{key}", value)
             for key, value in prediction.metrics.items()})
    return states, built


def _call(function, leaves, protein_states, predictions, params):
    states, built = _rebuild(leaves, protein_states, predictions)
    return function(states, built, **(params or {}))


def _check_value(name: str, value) -> None:
    """Refuse what a term returned, before anything tries to differentiate it."""
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if array.ndim != 0:
        raise LossTermError(
            f"loss term {name!r} returned shape {tuple(array.shape)}; a term must return a "
            "scalar, because BindCraft 2 multiplies it by one weight and sums it with the "
            "others. Reduce it yourself -- and if it is a per-residue quantity, reduce it with a "
            "mask rather than .mean(), or padding and the target chain count towards your term "
            "(bindcraft.protein.real_residue_mask gives you the mask)")
    if not bool(jnp.isfinite(array)):
        raise LossTermError(
            f"loss term {name!r} returned {float(array)} on a small synthetic design. A "
            "non-finite term makes the whole design loss non-finite, and BindCraft 2 throws that "
            "trajectory away; guard your divisions, logs and sqrts with an eps the way "
            "BindCraft 2's own terms do")


def _screen(name: str, function: Callable, *, design=None) -> None:
    """Refuse a term that cannot move a design, before a campaign runs it for hours."""
    import jax
    import jax.numpy as jnp

    # The forward pass first and on its own: a term that does not return a scalar has to be
    # named for that, not reported as whatever jax.grad says when handed a vector.
    try:
        protein_states, predictions = design or synthetic_design()
        leaves = _leaves(protein_states, predictions)
        scalar = lambda current: _call(function, current, protein_states, predictions, None)
        value = scalar(leaves)
        _check_value(name, value)
        gradient = jax.grad(scalar)(leaves)
    except LossTermError:
        raise
    except Exception as exc:  # the term may need a chain layout the synthetic design lacks
        warnings.warn(
            f"loss term {name!r} could not be screened for a dead gradient: {type(exc).__name__}: "
            f"{exc}. tt-bio installed it anyway. If this is the synthetic design's chain layout "
            "rather than your term, pass loss_terms(design=(protein_states, predictions)) with a "
            "real pair, or check=False to silence this; but do grade it with "
            "bindcraft2.check_gradient before you spend a campaign on it.",
            RuntimeWarning, stacklevel=4)
        return
    magnitudes = {key: float(jnp.abs(value).max()) for key, value in gradient.items()
                  if jnp.size(value)}
    if any(not jnp.isfinite(value).all() for value in gradient.values()):
        dead = sorted(k for k, v in gradient.items() if not jnp.isfinite(v).all())
        raise DeadGradient(
            f"loss term {name!r} has a non-finite gradient with respect to {', '.join(dead)}. "
            "That is usually a sqrt, a log, a divide or a norm at zero: a value that is finite "
            "forward can still be NaN backward, so add an eps inside the sqrt rather than after "
            "it. A non-finite gradient would be fed straight to the optimiser")
    if any(value > 0 for value in magnitudes.values()):
        return
    # An all-zero gradient has two causes and they want opposite answers. If the term also
    # returned exactly zero it is INACTIVE on this design -- an empty flag mask, a hinge on its
    # flat side, a parameter it needs that nothing set -- which is how nine of BindCraft 2's own
    # terms read here and is not a defect. If it returned a real value and still cannot be
    # differentiated, it computes something it can never move, which is the defect this screen
    # exists for.
    if float(abs(value)) < 1e-12:
        warnings.warn(
            f"loss term {name!r} is inactive on the design it was screened against: it returned "
            "0.0 and its gradient is 0.0 everywhere, so the screen cannot tell a term with no "
            "gradient from one whose mask is empty here. That is normal for a term that selects "
            "on a residue flag, compares against a reference state, or is a hinge sitting on its "
            "flat side. tt-bio installed it. Grade it on a design where it IS active: "
            "bindcraft2.check_gradient(fn, design=(protein_states, predictions)).",
            RuntimeWarning, stacklevel=4)
        return
    raise DeadGradient(
        f"loss term {name!r} returns {float(value):.6g} but its gradient is exactly zero with "
        f"respect to every intermediate it could read ({len(magnitudes)} arrays). It would run, "
        "cost a full forward and backward, and change nothing. JAX differentiates argmax, round, "
        "floor, a comparison, an integer cast, an argsort index and stop_gradient to 0.0 without "
        "complaining -- a term built from those has no gradient. Use a soft form instead: "
        "softmax for argmax, a sigmoid for a threshold, a soft-ranked mean for a top-k (which is "
        "what BindCraft 2's own best_contact_mean does: it stop_gradients the ranking and keeps "
        "the gradient on the values). If the term really is only a diagnostic, add it with "
        "weight 0 and read it out of losses.csv, or pass check=False")


# --------------------------------------------------------------------------------------------
# The gradcheck.
# --------------------------------------------------------------------------------------------


#: Step sizes the grade sweeps, as a fraction of the input's own scale. One step size cannot
#: serve both kinds of term: BindCraft 2's helpers compute in float32, where a central difference
#: at 1e-6 of the scale is below the forward's own resolution and measures nothing but rounding
#: (plddt_loss grades 7.3e-02 at 1e-6 and 3.5e-05 at 1e-2), while a term written in plain
#: jax.numpy over float64 inputs wants the small step. So sweep and report the best, with the
#: step that achieved it -- the finite difference is the instrument here, not the answer.
_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


@dataclass(frozen=True)
class GradeRow:
    """One input array's grade: the analytic gradient against float64 central differences."""

    leaf: str
    #: How many entries were probed. 0 means the analytic gradient is zero everywhere in this
    #: array, so there was nothing to compare and the term does not read it.
    probes: int
    #: Worst absolute deviation over the probes, each probe taken at its best step size.
    max_abs: float
    #: ``max_abs`` over ``gradient``: the deviation against the scale of the gradient itself.
    #: Normalising per entry instead would read 1.0 wherever both numbers are near zero, which
    #: on a 64-bin distogram is most entries, and would say nothing about the gradient.
    max_rel: float
    #: The relative step at which the worst probe was measured, as a fraction of the input scale.
    step: float
    #: max |analytic| over the array, the scale ``max_rel`` is relative to.
    gradient: float

    def __str__(self) -> str:
        if not self.probes:
            return f"{self.leaf:<40s} no gradient"
        return (f"{self.leaf:<40s} n={self.probes:<3d} |g|max={self.gradient:.3e} "
                f"max|d|={self.max_abs:.2e} rel={self.max_rel:.2e} at h={self.step:.0e}")


def check_gradient(function: Callable, *, params: Mapping | None = None, design=None,
                   probes: int = 8, steps=_STEPS, seed: int = 0, bar: float = 1e-4):
    """Grade a term's analytic gradient against float64 central differences.

    Takes ``jax.grad`` of the term once on :func:`synthetic_design` with ``jax_enable_x64`` on
    (restored after), then re-evaluates the term at ``+-h`` on ``probes`` entries of each input
    the gradient is non-zero in, sweeping ``h`` over :data:`_STEPS` and keeping the best. Returns
    ``(rows, worst, dtype)``: a :class:`GradeRow` per input array, the worst relative deviation
    over all of them, and the dtype the term's own forward came out in.

    **Probes go where the gradient is**, largest ``|analytic|`` first and then at random. Drawing
    uniformly would be close to useless on ``atoms``, which is ``[L, 37, 3]`` with four atoms per
    residue set: 8 uniform draws out of 2664 entries essentially never land on the CA a term
    reads, and every one of them would agree at exactly zero.

    **The bar is not uniform, and that is a property of BindCraft 2 rather than of this grade.**
    Its helpers -- ``_masked_mean``, ``pairwise_atom_distances``, ``distogram_pair_loss``,
    ``kabsch`` -- compute in float32, so the *third* return value is ``float32`` for any term
    built on them and the best achievable relative agreement is around 1e-4 to 1e-5 however the
    step is chosen. A term written in plain ``jax.numpy`` over ``plddt``, ``pae`` or
    ``distogram`` stays in float64 and grades to around 1e-9. Read your number against the dtype
    this returns, not against a single figure. ``bar`` is only what it warns at; it never raises,
    because the number is the answer.
    """
    import jax
    import jax.numpy as jnp

    enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        protein_states, predictions = design or synthetic_design(dtype=jnp.float64)
        leaves = {key: jnp.asarray(value, dtype=jnp.float64)
                  for key, value in _leaves(protein_states, predictions).items()}

        def scalar(current):
            return _call(function, current, protein_states, predictions, params)

        value = scalar(leaves)
        _check_value("the term under grade", value)
        dtype = jnp.asarray(value).dtype
        analytic = jax.grad(scalar)(leaves)

        rows, worst, key = [], 0.0, jax.random.PRNGKey(seed)
        for leaf in sorted(leaves):
            base, exact = leaves[leaf], analytic[leaf]
            flat_exact = exact.reshape(-1)
            live = jnp.nonzero(flat_exact)[0]
            peak = float(jnp.abs(flat_exact).max()) if jnp.size(flat_exact) else 0.0
            if not jnp.size(live):
                rows.append(GradeRow(leaf, 0, 0.0, 0.0, 0.0, peak))
                continue
            order = jnp.argsort(-jnp.abs(flat_exact[live]))
            ranked = live[order]
            key, draw = jax.random.split(key)
            picks = ranked[:probes]
            if jnp.size(ranked) > probes:  # half by magnitude, half at random among the live
                spread = jax.random.choice(draw, ranked[probes // 2:],
                                           (min(probes - probes // 2, jnp.size(ranked) - probes // 2),),
                                           replace=False)
                picks = jnp.concatenate([ranked[:probes // 2], spread])
            scale = float(jnp.abs(base).max()) or 1.0
            flat = base.reshape(-1)
            max_abs = max_rel = 0.0
            best_step = float(steps[0])
            for position in picks.tolist():
                truth = float(flat_exact[position])
                attempts = []
                for relative in steps:
                    h = relative * scale
                    up = float(scalar({**leaves, leaf: flat.at[position].add(h).reshape(base.shape)}))
                    down = float(scalar({**leaves, leaf: flat.at[position].add(-h).reshape(base.shape)}))
                    numeric = (up - down) / (2 * h)
                    absolute = abs(numeric - truth)
                    attempts.append((absolute, float(relative)))
                absolute, relative = min(attempts)
                if absolute > max_abs:
                    max_abs, best_step = absolute, relative
            max_rel = max_abs / max(peak, 1e-30)
            rows.append(GradeRow(leaf, int(jnp.size(picks)), max_abs, max_rel, best_step, peak))
            worst = max(worst, max_rel)
        if worst > bar:
            warnings.warn(
                f"the worst relative deviation is {worst:.3e}, over the {bar:.0e} bar, with the "
                f"term's own forward in {dtype}. In float32 that can still be the finite "
                "difference rather than the gradient; in float64 it is the gradient and it is "
                "wrong somewhere", RuntimeWarning, stacklevel=2)
        return rows, worst, dtype
    finally:
        jax.config.update("jax_enable_x64", enabled)
