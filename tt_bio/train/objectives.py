"""The named objective rows. Tier 1's cut line is that the objective is a name, not a callable.

r3's loss abstraction, and it is small: an objective maps ``(batch, outputs) -> (scalar,
breakdown, seeds)`` plus a weight table that a caller may override per example. That shape
covers AF2, Protenix, OpenFold3, Boltz-1, Boltz-2 and BoltzGen. The only Protenix-specific
thing in the surface is the *list* of eight terms, not their shape.

``seeds`` is what makes it work on this stack rather than on torch: the loss is float64 numpy
on host, so what goes back to the tape is one gradient array per device output, and
``ag.backward(roots, seeds)`` replays the union of their ancestors once. Returning a scalar
and expecting the framework to differentiate it would need the loss on device, which is the
thing ``tt_bio/train/losses.py`` deliberately is not.

**Rollout length is the one thing that does not generalise**, and it is an argument here with
its own cost accounting rather than a hidden default. At Protenix's 20 diffusion steps it is a
rounding error beside 48 taped denoiser draws; at Boltz-1's 200 it dominates the step. So a
row carries its own ``rollout`` and a user who changes it is changing the cost of the step,
not a quality knob.

A row is registered, not hard-coded, so a model row can arrive with the model. ``af3`` is the
row that exists because Protenix-v2 is served today; nothing here anticipates a family we do
not run, per the plan's own R4 kill criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from . import losses

__all__ = ["Objective", "objective", "register", "names", "af3_loss",
           "MOL_TYPE_CONVENTIONS", "entity_flags"]


@dataclass(frozen=True)
class Objective:
    """One named row. ``fn`` is ``(batch, outputs, weights) -> (scalar, breakdown, seeds)``."""

    name: str
    fn: Callable
    weights: Dict[str, float]
    rollout: int
    terms: tuple = ()
    doc: str = ""

    def __call__(self, batch, outputs, *, weights: Optional[Dict[str, float]] = None):
        return self.fn(batch, outputs, weights or self.weights)

    def __str__(self) -> str:
        return (f"{self.name}: {len(self.terms)} terms, rollout {self.rollout} "
                f"-- {', '.join(self.terms)}")


_ROWS: Dict[str, Objective] = {}


def register(obj: Objective) -> Objective:
    """Add a row. Refuses to replace one silently.

    A redefined objective is the worst kind of silent change: every number in every log from
    before the redefinition is now incomparable and nothing says so.
    """
    if obj.name in _ROWS and _ROWS[obj.name] is not obj:
        raise ValueError(f"objective {obj.name!r} is already registered. Replacing it makes "
                         f"every result recorded against the old one incomparable -- register "
                         f"a new name instead")
    _ROWS[obj.name] = obj
    return obj


def objective(name: str) -> Objective:
    try:
        return _ROWS[name]
    except KeyError:
        raise KeyError(f"no objective {name!r}; rows are {sorted(_ROWS)}") from None


def names() -> list:
    return sorted(_ROWS)


#: What an integer ``mol_type`` column MEANS, per stack. Three of our five models emit the
#: entity fact under this one name instead of upstream's three flags, and **the two live
#: conventions disagree**: both put protein at 0 and ligand at 3, and they SWAP dna and rna.
#: Each entry carries its own source, because a provenance comment written above a table
#: outlives the module the table was read from and then nothing says the entry is stale.
#: ``tests/test_entity_flags.py`` re-derives every entry from the live source, so an upstream
#: reordering fails a test here rather than silently retraining nucleic acids at the wrong
#: weight.
MOL_TYPE_CONVENTIONS = {
    # OpenFold3's own `MoleculeType` (`core/data/resources/residues.py:24-28`, vendored) and
    # Protenix-v2's `MOL_TYPE_IDS` (`tt_bio/protenix_data.py:40`), which agree. Measured by
    # running `protenix_data.build_complex_features` on a protein/rna/dna/CCD_ATP complex:
    # 5 protein tokens -> 0, 4 rna -> 1, 4 dna -> 2, 31 ligand atom-tokens -> 3.
    "af3": {"protein": 0, "rna": 1, "dna": 2, "ligand": 3},
    # Boltz-2 and BoltzGen, both from `tt_bio.data.const.chain_type_ids`
    # (`tt_bio/data/const.py:8-14`), which their featurisers copy straight through
    # (`tt_bio/data/featurizer.py:651`, `tt_bio/boltzgen/data/featurizer.py:700`). Measured on
    # the same four-entity complex: Boltz-2 10/8/8/31 -> 0/1/2/3, BoltzGen 18/8/8/31 -> the
    # same. NONPOLYMER is upstream's name for the class upstream's loss calls ligand.
    "boltz": {"protein": 0, "dna": 1, "rna": 2, "ligand": 3},
}
#: The batch key a featuriser sets to name its own convention. Optional, and absent on every
#: featuriser we ship today, which is the case ``entity_flags`` has to be safe in.
CONVENTION_KEY = "mol_type_convention"
#: Where the two conventions AGREE, so a flag derived from these values is unambiguous no
#: matter which stack built the batch. ligand is the one that matters: it is the only class
#: upstream weights differently from the other two (10.0 against 5.0).
_AGREED = ("protein", "ligand")


def entity_flags(mol_type, convention=None) -> dict:
    """Upstream's ``is_dna`` / ``is_rna`` / ``is_ligand`` from one integer ``mol_type`` column.

    One derivation for every model, because three featuriser edits would be three places to
    drift and the fourth model would arrive uncovered.

    **The dna/rna split is genuinely ambiguous without a named convention**, and this does not
    pretend otherwise. An integer column alone cannot say whether 1 means dna (Boltz) or rna
    (Protenix, OpenFold3), so with ``convention=None`` the pair is assigned under ``af3`` and
    reported ``resolved: False``. That is safe only while upstream weights dna and rna the
    same, which is a fact about `losses.mse` rather than a hope: the guard below reads its
    signature and refuses rather than guessing if the two weights ever differ.

    ``is_ligand`` is never ambiguous -- both conventions put ligand at 3 -- and ligand is the
    class carrying the weight that differs (10.0 against 5.0).
    """
    mt = np.asarray(mol_type)
    if mt.dtype.kind not in "iub":
        raise TypeError(
            f"mol_type must be an integer class column, got dtype {mt.dtype}. A float or a "
            f"one-hot here would silently compare unequal to every class id and derive three "
            f"all-zero flags, which is bit-identical to deriving nothing")
    name = convention if convention is not None else "af3"
    try:
        table = MOL_TYPE_CONVENTIONS[name]
    except KeyError:
        raise KeyError(f"no mol_type convention {name!r}; conventions are "
                       f"{sorted(MOL_TYPE_CONVENTIONS)}") from None
    resolved = convention is not None
    if not resolved:
        _refuse_if_nucleic_weights_differ()
    unknown = sorted(set(np.unique(mt).tolist()) - set(table.values()))
    if unknown:
        raise ValueError(
            f"mol_type carries {unknown}, which convention {name!r} does not define "
            f"({table}). An undefined class would fall through as protein and train at "
            f"weight 1.0 with nothing saying so")
    flags = {f"is_{k}": (mt == table[k]).astype(np.float64)
             for k in ("dna", "rna", "ligand")}
    return {"flags": flags, "convention": name, "resolved": resolved,
            "unambiguous": list(_AGREED),
            "counts": {k: int(v.sum()) for k, v in flags.items()}}


def _refuse_if_nucleic_weights_differ() -> None:
    """The invariance an unresolved dna/rna split rests on, checked instead of assumed.

    Upstream weights dna and rna identically (5.0 and 5.0), so swapping the two leaves
    ``losses.mse`` and its gradient seed bit-identical and an unnamed convention cannot move a
    number. The moment a caller separates them that stops being true, and a silently wrong
    nucleic-acid weighting is exactly the class of defect D12 was.
    """
    import inspect
    d = inspect.signature(losses.mse).parameters
    w_dna, w_rna = d["w_dna"].default, d["w_rna"].default
    if w_dna != w_rna:
        raise ValueError(
            f"losses.mse now weights dna {w_dna} and rna {w_rna} differently, so which of "
            f"mol_type 1 and 2 is which has become load-bearing. Set batch["
            f"{CONVENTION_KEY!r}] to one of {sorted(MOL_TYPE_CONVENTIONS)} -- 'af3' for "
            f"OpenFold3 and Protenix-v2, 'boltz' for Boltz-2 and BoltzGen")


def _with_entity_flags(batch) -> tuple:
    """Add the three flags to a batch that carries ``mol_type`` instead, or leave it alone.

    Returns ``(batch, note)``. The batch is copied rather than mutated: a loss that edits the
    caller's batch makes the second call on the same batch a different computation from the
    first.
    """
    if "mol_type" not in batch or any(k in batch for k in _OPTIONAL["mse"]):
        return batch, None
    got = entity_flags(batch["mol_type"], batch.get(CONVENTION_KEY))
    return {**batch, **got["flags"]}, {k: got[k] for k in
                                       ("convention", "resolved", "counts")}


def af3_loss(batch, outputs, weights) -> tuple:
    """The AF3 objective: the eight terms Protenix trains, at Protenix's own weights.

    ``outputs`` is a dict of host float64 arrays keyed as the model produces them, and
    ``batch`` carries the labels under the same names. A term whose inputs are absent is
    SKIPPED and recorded as skipped -- not treated as zero. A zero for a missing term is a
    number that goes into the total and moves the weighted sum, and it is how a run reports a
    healthy loss while training on four of eight terms.
    """
    total = 0.0
    breakdown, seeds = {}, {}
    # The one place a `mol_type` column becomes upstream's three flags, on the path every
    # model's batch already takes (`recipes.py` calls the row with the dataset's own dict).
    # D16: OpenFold3's vendored featuriser emits the three flags, and Protenix-v2, Boltz-2
    # and BoltzGen emit the same fact as one integer column, so before this every
    # nucleic-acid and ligand token those three trained on was weighted as protein.
    batch, derived = _with_entity_flags(batch)

    def take(term, w, value_grad, seed_key):
        nonlocal total
        value, grad = value_grad
        total += w * value
        breakdown[term] = {"value": value, "weight": w, "contribution": w * value}
        if seed_key is not None and grad is not None:
            prev = seeds.get(seed_key)
            g = w * grad
            seeds[seed_key] = g if prev is None else prev + g

    have = lambda *ks: all(k in outputs or k in batch for k in ks)
    for term, w in weights.items():
        if w == 0.0:
            breakdown[term] = {"value": None, "weight": 0.0, "contribution": 0.0,
                               "skipped": "weight is zero in this stage"}
            continue
        fn = _TERMS.get(term)
        if fn is None:
            raise KeyError(f"no loss term {term!r} in tt_bio.train.losses; "
                           f"weights name {sorted(weights)}")
        need = _NEEDS[term]
        if not have(*need):
            absent = [k for k in need if k not in outputs and k not in batch]
            breakdown[term] = {"value": None, "weight": w, "contribution": 0.0,
                               "skipped": f"missing {absent}"}
            continue
        take(term, w, fn(batch, outputs), _SEED[term])
        absent = [k for k in _OPTIONAL.get(term, ()) if k not in batch]
        if absent:
            breakdown[term]["without"] = absent
        elif derived is not None and term == "mse":
            # Derived, not native. `without` cannot say this -- the keys ARE in the batch by
            # now -- and the difference matters: an unresolved dna/rna split is a claim this
            # report has to carry rather than bury.
            breakdown[term]["derived"] = derived
    return total, breakdown, seeds


# Each term's adapter from (batch, outputs) to `losses`' own signature, and what it needs.
# Kept as data next to the requirement list so a term cannot be added without declaring both.
_TERMS = {
    "mse": lambda b, o: losses.mse(o["pred_xyz"], b["true_xyz"], b["coord_mask"],
                                   is_dna=b.get("is_dna"), is_rna=b.get("is_rna"),
                                   is_ligand=b.get("is_ligand"),
                                   per_sample_scale=b.get("edm_scale")),
    "smooth_lddt": lambda b, o: losses.smooth_lddt(o["pred_dist"], b["true_dist"],
                                                   b["lddt_pair_mask"]),
    "bond": lambda b, o: losses.bond(o["pred_dist"], b["true_dist"], b["bond_mask"],
                                     b.get("coord_mask"),
                                     per_sample_scale=b.get("edm_scale")),
    "distogram": lambda b, o: losses.distogram(o["distogram_logits"], b["true_xyz"],
                                               b["coord_mask"]),
    "plddt": lambda b, o: losses.plddt(o["plddt_logits"], b["per_atom_lddt"],
                                       b["per_atom_weight"]),
    "pde": lambda b, o: losses.pde(o["pde_logits"], o["pred_xyz"], b["true_xyz"],
                                   b["coord_mask"]),
    "pae": lambda b, o: losses.pae(o["pae_logits"], o["pred_xyz"], b["true_xyz"],
                                   b["coord_mask"], b["frame_atom_index"]),
    "resolved": lambda b, o: losses.resolved(o["resolved_logits"], b["coord_mask"]),
}
_NEEDS = {
    "mse": ("pred_xyz", "true_xyz", "coord_mask"),
    "smooth_lddt": ("pred_dist", "true_dist", "lddt_pair_mask"),
    "bond": ("pred_dist", "true_dist", "bond_mask"),
    "distogram": ("distogram_logits", "true_xyz", "coord_mask"),
    "plddt": ("plddt_logits", "per_atom_lddt", "per_atom_weight"),
    "pde": ("pde_logits", "pred_xyz", "true_xyz", "coord_mask"),
    "pae": ("pae_logits", "pred_xyz", "true_xyz", "coord_mask", "frame_atom_index"),
    "resolved": ("resolved_logits", "coord_mask"),
}
# Labels a term uses when the batch carries them and computes a DIFFERENT loss without. An
# absent entry here is not a missing input -- the term still fires, at a weight nobody asked
# for -- so `af3_loss` names what was absent in the breakdown instead of leaving it silent.
# `mse` is why this table exists. Upstream's per-entity upweighting (dna 5.0, rna 5.0, ligand
# 10.0; `core/loss/diffusion.py:138-141` at `model_config.py:496-498`) lives in `losses.mse`
# and reaches it only through these three keys, so a featuriser that omits them trains every
# nucleic-acid and ligand token at protein weight and every metric still looks healthy.
# `edm_scale` is deliberately NOT here: absent, it means one sample and the loss is right.
# Absent entity flags mean "this batch has no DNA, RNA or ligand", which is a claim about the
# batch that a featuriser which simply never produced them does not get to make.
_OPTIONAL = {
    "mse": ("is_dna", "is_rna", "is_ligand"),
}
# Which device output each term's gradient seeds. The four bin-label terms reach only their
# logits, because upstream builds every true bin under no_grad -- including pLDDT's, whose
# label depends on the prediction. That is a property of the loss, not an approximation.
_SEED = {
    "mse": "pred_xyz", "smooth_lddt": "pred_dist", "bond": "pred_dist",
    "distogram": "distogram_logits", "plddt": "plddt_logits", "pde": "pde_logits",
    "pae": "pae_logits", "resolved": "resolved_logits",
}

register(Objective(
    name="af3", fn=af3_loss, weights=dict(losses.LOSS_WEIGHTS["pretrain"]),
    # Protenix samples 20 diffusion steps in training and takes 48 denoiser draws per step.
    # Both are upstream's, not ours, and both are here because they price the step.
    rollout=20, terms=tuple(_TERMS),
    doc="the eight AF3 terms at Protenix's pretraining weights; pass "
        "weights=losses.LOSS_WEIGHTS['finetune'] for stage 3, which turns bond and pae on "
        "and smooth_lddt off"))
