"""What each predict model honours from its input file, as one table.

Every model except Boltz-2 reads the same YAML/FASTA through
``tt_bio.main._read_bio_chains``, and that reader accepts more than any single model can
use: ligand and nucleic-acid chains, ``modifications:``, ``templates:``,
``constraints:``, ``cyclic: true``, ``properties: affinity``. What a model does with a
key it cannot use was decided in five separate validators, each wired per model, so a
model added to one path and not the others dropped the input and still reported success.
ESMFold2 folded a cocrystal YAML as bare protein that way, and ``cyclic: true`` reached
four models as a linear fold before its guard existed.

So the per-model facts are DATA here (``CAPABILITY``), the mechanism is one function
(``check_input``), and tests/test_input_capabilities.py fails when a shipped model has no
row or a predict path does not call the check. Three verdicts:

* ``HONOURED`` -- the input reaches the model and changes the answer.
* ``REFUSED``  -- dropping it would change the STRUCTURE, so the fold is refused. A clear
  error beats a confidently wrong structure.
* ``NOTED``    -- dropping it only omits an extra output, so the fold runs and says so.

A refusal names the models that do honour the key, derived from this table rather than
written by hand, so the hint cannot go stale when a port gains the feature.
"""
from __future__ import annotations

from pathlib import Path

HONOURED = "honoured"
REFUSED = "refused"
NOTED = "noted"

#: feature -> (how it appears in the input, what dropping it would do to the answer)
FEATURES: dict[str, tuple[str, str]] = {
    "ligand": ("a ligand chain", "the complex would fold without the ligand"),
    "rna": ("an RNA chain", "the complex would fold without the RNA"),
    "dna": ("a DNA chain", "the complex would fold without the DNA"),
    "cyclic": ("`cyclic: true`", "the fold would return a linear structure"),
    "modifications": ("`modifications:`", "the fold would return the unmodified residue"),
    "templates": ("`templates:`", "the fold would ignore the template you supplied"),
    "bond": ("a `bond` constraint", "the fold would ignore the covalent bond"),
    "pocket": ("a `pocket`/`contact` constraint",
               "the fold would ignore the binding constraint"),
    "affinity": ("`properties: affinity`", "no affinity value is written"),
}

_ALL_HONOURED = dict.fromkeys(FEATURES, HONOURED)


def _row(**overrides) -> dict[str, str]:
    return {**_ALL_HONOURED, **overrides}


#: --model -> feature -> verdict. Every id in ``main.PREDICT_MODELS`` needs a row; the
#: completeness test refuses a new port that does not declare one.
#:
#: Boltz-2 is the only model that honours the whole reader: its own upstream parser
#: (tt_bio/data/parse.py), a constraint embedder for pocket/contact, a template pipeline and
#: an affinity head.
#:
#: ``modifications`` is REFUSED rather than honoured for Protenix, OpenDDE and the OF3 family
#: because ``protenix_data.build_complex_features`` takes no modifications argument and the
#: OF3 query is built with ``non_canonical_residues=None`` -- upstream has the field, the port
#: hardcodes it empty. ``templates`` is REFUSED for Protenix/OpenDDE/RF3/ESMFold2 for the same
#: kind of reason: ``build_complex_features`` emits ``dummy_template_features`` and no path
#: leads from a YAML ``templates:`` key into it, even though the protenix-v2 and opendde
#: checkpoints do ship a template pairformer stack. Both are porting gaps, refused so they
#: cannot be mistaken for support.
CAPABILITY: dict[str, dict[str, str]] = {
    "boltz2": _row(),
    # ESMFold2 folds ligands, RNA and DNA and applies `modifications:` (one reader, one
    # fold_complex call). It has no constraint, template or affinity path.
    "esmfold2": _row(cyclic=REFUSED, templates=REFUSED, bond=REFUSED, pocket=REFUSED,
                     affinity=NOTED),
    "esmfold2-fast": _row(cyclic=REFUSED, templates=REFUSED, bond=REFUSED, pocket=REFUSED,
                          affinity=NOTED),
    # Protenix honours covalent bonds (token_bonds is the only constraint signal its trunk
    # reads); pocket/contact need a constraint embedder no Protenix checkpoint ships.
    "protenix-v1": _row(cyclic=REFUSED, modifications=REFUSED, templates=REFUSED,
                        pocket=REFUSED, affinity=NOTED),
    "protenix-v2": _row(cyclic=REFUSED, modifications=REFUSED, templates=REFUSED,
                        pocket=REFUSED, affinity=NOTED),
    # OpenDDE is protein/ligand: nucleic-acid structural tokens are not ported.
    "opendde": _row(rna=REFUSED, dna=REFUSED, cyclic=REFUSED, modifications=REFUSED,
                    templates=REFUSED, pocket=REFUSED, affinity=NOTED),
    "opendde-abag": _row(rna=REFUSED, dna=REFUSED, cyclic=REFUSED, modifications=REFUSED,
                         templates=REFUSED, pocket=REFUSED, affinity=NOTED),
    # Ligands stay refused for OF3-preview2 and honoured for OpenBind, the checkpoint
    # upstream trained for co-folding. Templates are opt-in per protein chain (a precomputed
    # alignment npz); there is no template search.
    "openfold3": _row(ligand=REFUSED, cyclic=REFUSED, modifications=REFUSED, bond=REFUSED,
                      pocket=REFUSED, affinity=NOTED),
    "openbind": _row(cyclic=REFUSED, modifications=REFUSED, bond=REFUSED, pocket=REFUSED,
                     affinity=NOTED),
    # RF3 the model carries bonds, modified residues and cyclic chains, but only through its
    # own JSON/CIF spec; the YAML door here builds a spec from the chain reader alone.
    "rf3": _row(cyclic=REFUSED, modifications=REFUSED, templates=REFUSED, bond=REFUSED,
                pocket=REFUSED, affinity=NOTED),
}

#: (model, feature) -> the reason this refusal exists, replacing the generic "what a silent
#: drop would do" line where that line would be wrong. OF3-preview2's featurizer WOULD build
#: a ligand, and that is exactly the problem.
WHY: dict[tuple[str, str], str] = {
    ("openfold3", "ligand"): (
        "preview2 was released as a polymer model and was never trained to place a ligand, "
        "so it is polymer-only here; the featurizer would build one and the sampler would "
        "return a status=ok structure anyway"),
    ("opendde", "rna"): "nucleic-acid structural tokens are not ported",
    ("opendde", "dna"): "nucleic-acid structural tokens are not ported",
    ("opendde-abag", "rna"): "nucleic-acid structural tokens are not ported",
    ("opendde-abag", "dna"): "nucleic-acid structural tokens are not ported",
}

#: A route the model offers outside the YAML front door, appended to its refusals.
ELSEWHERE: dict[str, str] = {
    "rf3": "RF3 also reads its own JSON/CIF spec, which does carry a bond graph, modified "
           "residues and cyclic chains.",
}


def honoured_by(feature: str) -> tuple[str, ...]:
    """The --model ids that honour ``feature``, read off CAPABILITY so a hint cannot go
    stale when a port gains the feature."""
    return tuple(sorted(m for m, caps in CAPABILITY.items() if caps[feature] == HONOURED))


def _yaml_doc(path) -> dict:
    if Path(path).suffix.lower() not in (".yml", ".yaml"):
        return {}
    import yaml

    try:
        doc = yaml.safe_load(Path(path).read_text())
    except Exception:
        return {}
    return doc if isinstance(doc, dict) else {}


def _ids(sub: dict) -> list[str]:
    ids = sub.get("id", "?")
    return ([str(x) for x in ids] if isinstance(ids, (list, tuple))
            else str(ids).split(","))


def detect(path, chains=None) -> dict[str, str]:
    """Features actually present in ``path``, as feature -> where it was found.

    ``chains`` is ``_read_bio_chains(path)`` output: molecule types come from it, because a
    FASTA expresses those too, and the keyed features come from the YAML document.
    """
    found: dict[str, str] = {}
    for mt in ("ligand", "rna", "dna"):
        hits = [cid for cid, _s, _sp, m, _mo in (chains or []) if m == mt]
        if hits:
            found[mt] = "chain(s) " + ", ".join(hits)
    doc = _yaml_doc(path)
    per_chain: dict[str, list[str]] = {"cyclic": [], "modifications": [], "templates": []}
    for entry in doc.get("sequences") or []:
        if not isinstance(entry, dict):
            continue
        for sub in entry.values():
            if not isinstance(sub, dict):
                continue
            for feature in per_chain:
                if sub.get(feature):
                    per_chain[feature] += _ids(sub)
    for feature, hits in per_chain.items():
        if hits:
            found[feature] = "chain(s) " + ", ".join(hits)
    for c in doc.get("constraints") or []:
        if not isinstance(c, dict):
            continue
        if "bond" in c:
            found.setdefault("bond", "constraints")
        elif "pocket" in c or "contact" in c:
            found.setdefault("pocket", "constraints")
    binders = [str(pr["affinity"].get("binder")) for pr in (doc.get("properties") or [])
               if isinstance(pr, dict) and isinstance(pr.get("affinity"), dict)]
    if binders:
        found["affinity"] = "binder " + ", ".join(binders)
    return found


def _click_note(msg: str) -> None:
    import click

    click.secho(msg, fg="yellow")


def check_input(path, chains, model: str, echo=_click_note) -> dict[str, str]:
    """Refuse the structure-changing input ``model`` cannot honour; note the rest.

    Raises RuntimeError naming every refused feature at once, so a user fixing one key does
    not have to run again to find the next, and calls ``echo`` once per NOTED feature.
    Returns the detected features. An unknown model raises KeyError, which is the point: a
    new port has to declare its row.
    """
    caps = CAPABILITY[model]
    found = detect(path, chains)
    refused = [f for f in FEATURES if f in found and caps[f] == REFUSED]
    if refused:
        lines = []
        for f in refused:
            label, generic = FEATURES[f]
            others = ", ".join(m for m in honoured_by(f) if m != model)
            lines.append(f"  - {label} ({found[f]}): {WHY.get((model, f), generic)}."
                         + (f" Honoured by: {others}." if others else ""))
        tail = ELSEWHERE.get(model)
        raise RuntimeError(
            f"--model {model} cannot honour {len(refused)} part(s) of "
            f"{Path(path).name}:\n" + "\n".join(lines) + (f"\n{tail}" if tail else ""))
    if echo is not None:
        for f in FEATURES:
            if f in found and caps[f] == NOTED:
                label, effect = FEATURES[f]
                others = ", ".join(m for m in honoured_by(f) if m != model)
                echo(f"Note: --model {model} ignores {label} ({found[f]} in "
                     f"{Path(path).name}): {effect}."
                     + (f" Use --model {others} for it." if others else ""))
    return found
