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
(``check_capabilities``), and tests/test_input_capabilities.py fails when a shipped model has no
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
    "protein_free": ("an input with no protein chain",
                     "there is no protein for the model to fold it with"),
    "cyclic": ("`cyclic: true`", "the fold would return a linear structure"),
    "modifications": ("`modifications:`", "the fold would return the unmodified residue"),
    "templates": ("a per-chain `templates:` alignment npz",
                  "the fold would ignore the template you supplied"),
    "template_structure": ("a top-level `templates:` structure file",
                           "the fold would ignore the template you supplied"),
    "bond": ("a `bond` constraint", "the fold would ignore the covalent bond"),
    "polymer_bond": ("a `bond` between two standard polymer residues (a disulfide, a "
                     "crosslink)", "the fold would ignore the covalent bond"),
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
#: Boltz-2 honours the rest of the reader: its own upstream parser (tt_bio/data/parse.py), a
#: constraint embedder for pocket/contact, a template pipeline and an affinity head.
#:
#: ``modifications`` reaches the featurizer on every model. Protenix and OpenDDE tokenize a
#: modified residue per atom from its CCD component (``protenix_data.polymer_chain_features``);
#: the OF3 family passes it as ``non_canonical_residues`` and RF3 as a ``(CCD)`` residue in
#: the sequence, each upstream's own field.
#:
#: ``templates`` takes the same alignment npz on the OF3 family and on protenix-v2 / OpenDDE,
#: whose template embedder used to run on ``dummy_template_features`` whatever the input said.
#: ``template_structure`` is Boltz-2's top-level structure-file form, and it reaches all of
#: them too: ``template_cif`` aligns it into that npz. Boltz-2 is the other way round: its
#: parser reads only the structure file, so a per-chain npz folded bit-identically to no
#: template (1a8q, CA-RMSD 9.19 A either way, 0.29 A with the same template as a cif).
#: Both stay REFUSED on protenix-v1, whose v0.5.0 checkpoint ships an EMPTY template pairformer
#: stack, so upstream returns literal 0 from the template embedder and a template could only
#: be dropped (pinned by tests/test_protenix_template_gate.py), and on ESMFold2, which has no
#: template stack at all. RF3 templates by coordinate: the same alignment puts the template's
#: CA positions on the aligned residues (``rf3.featurize.apply_template_ca``).
CAPABILITY: dict[str, dict[str, str]] = {
    "boltz2": _row(templates=REFUSED),
    # ESMFold2 folds ligands, RNA and DNA and applies `modifications:` (one reader, one
    # fold_complex call). A `bond` is upstream's covalent_bonds, which its pair init reads
    # through token_bonds; `cyclic: true` is the head-to-tail amide on the same route, as on
    # Protenix. It has no template, pocket or affinity path, and its trunk is conditioned on
    # the protein language model, so a complex needs at least one protein.
    "esmfold2": _row(protein_free=REFUSED, templates=REFUSED, template_structure=REFUSED,
                     pocket=REFUSED, affinity=NOTED),
    "esmfold2-fast": _row(protein_free=REFUSED, templates=REFUSED, template_structure=REFUSED,
                          pocket=REFUSED, affinity=NOTED),
    # Protenix honours covalent bonds (token_bonds is the only constraint signal its trunk
    # reads); pocket/contact need a constraint embedder no Protenix checkpoint ships.
    # `cyclic: true` reaches it as the head-to-tail amide bond (main._read_bio_bonds), the
    # route upstream documents; it has no relpos wrap. v1's checkpoint does not close it (WHY).
    "protenix-v1": _row(cyclic=REFUSED, templates=REFUSED, template_structure=REFUSED,
                        pocket=REFUSED, affinity=NOTED),
    "protenix-v2": _row(pocket=REFUSED, affinity=NOTED),
    "opendde": _row(pocket=REFUSED, affinity=NOTED),
    "opendde-abag": _row(pocket=REFUSED, affinity=NOTED),
    # Ligands stay refused for OF3-preview2 and honoured for OpenBind, the checkpoint
    # upstream trained for co-folding. Templates are opt-in, in either form; there is no
    # template search. `cyclic: true` is upstream's relpos wrap (cyclic_mask); a `bond`
    # joins the atoms before tokenization, where upstream reads them.
    "openfold3": _row(ligand=REFUSED, polymer_bond=REFUSED, pocket=REFUSED, affinity=NOTED),
    "openbind": _row(polymer_bond=REFUSED, pocket=REFUSED, affinity=NOTED),
    # RF3's YAML door writes upstream's own spec fields: `(CCD)` residues in the sequence and
    # top-level `bonds`. Upstream's cyclic_chains does not close the ring (WHY below).
    "rf3": _row(cyclic=REFUSED, polymer_bond=REFUSED, pocket=REFUSED, affinity=NOTED),
    # `tt-bio affinity --model nesso1`, not predict. It returns a scalar and no coordinates,
    # so nothing it drops can come back as a wrong structure, and the docs tell users to
    # reuse their Boltz-2 affinity yaml, which carries msa:, constraints: and properties:
    # blocks. Refusing those would break the documented path for no safety gain, so
    # everything it cannot read is NOTED. Its own vendored parser is the enforcement point
    # for molecule types and refuses a third entity type by name ("Unsupported entity type
    # 'rna' (only protein, ligand)"), which is why the chain columns here record a verdict
    # this module does not apply itself.
    "nesso1": _row(rna=REFUSED, dna=REFUSED, protein_free=REFUSED, cyclic=NOTED,
                   modifications=NOTED, templates=NOTED, template_structure=NOTED, bond=NOTED,
                   polymer_bond=NOTED, pocket=NOTED),
}

#: Molecule-type features: they come from the parsed chain list, not from a yaml key.
CHAIN_FEATURES = frozenset({"ligand", "rna", "dna", "protein_free"})

#: Models whose molecule types their own reader enforces, so check_capabilities is called
#: with no chain list and the chain columns are a record rather than an enforcement. Only
#: nesso1, which never goes through _read_bio_chains.
CHAINS_ELSEWHERE = frozenset({"nesso1"})

#: How a user reaches a model, for the "somewhere else to go" hint. predict is the default.
COMMAND: dict[str, str] = {"nesso1": "tt-bio affinity --model nesso1"}


def how(model: str) -> str:
    """The command form a hint should print for ``model``."""
    return COMMAND.get(model, f"--model {model}")

#: (model, feature) -> the reason this refusal exists, replacing the generic "what a silent
#: drop would do" line where that line would be wrong. OF3-preview2's featurizer WOULD build
#: a ligand, and that is exactly the problem.
WHY: dict[tuple[str, str], str] = {
    ("openfold3", "ligand"): (
        "preview2 was released as a polymer model and was never trained to place a ligand, "
        "so it is polymer-only here; the featurizer would build one and the sampler would "
        "return a status=ok structure anyway"),
    # Both upstream data pipelines drop these bonds before training: OF3's
    # cleanup.remove_intra/inter_chain_poly_links, and atomworks, which adds struct_conn
    # `covale` bonds only ("except for disulfides"). Measured: on RF3 a Cys-Cys bond leaves
    # every feature bit-identical; on OF3 it atomizes both residues into tokens it never saw.
    **{(m, "polymer_bond"): (
        f"{fam} was trained with bonds between two standard residues removed from its data, "
        "so it cannot read one; a bond to a ligand or a modified residue does reach it")
       for m, fam in (("openfold3", "OpenFold3"), ("openbind", "OpenFold3"), ("rf3", "RF3"))},
    # perf/mgx_constraints: cyclic QLEDSEVEAVAKG, 5 samples each, N1-C13 on TT and in the
    # upstream torch reference on CPU (rf3_upstream_ring.py; Protenix v0.5.0 folding the
    # same token_bonds its current inference builds, include_discont_poly_poly_bonds).
    ("rf3", "cyclic"): (
        "RF3's cyclic offset brings the chain ends together but does not form the closing "
        "peptide bond: on a cyclic 13-mer the N-C distance stays 2.4-2.9 A, in upstream RF3 "
        "as here, where a closed amide is 1.33 A"),
    ("protenix-v1", "cyclic"): (
        "the v0.5.0 checkpoint reads the closing bond but does not form it: on a cyclic "
        "13-mer the N-C distance stays 2.1-2.2 A, in upstream Protenix as here, where a "
        "closed amide is 1.33 A; protenix-v2 closes it"),
    ("nesso1", "protein_free"): "it scores a protein-ligand pair",
    ("boltz2", "templates"): "Boltz-2 reads a template as a structure file, a top-level "
        "`templates:` entry with `cif:` and `chain_id:`, not as a per-chain alignment npz",
    **{(m, f): "its v0.5.0 checkpoint ships an empty template stack, so upstream Protenix-v1 "
       "ignores a template too; protenix-v2 is the same model with one"
       for m in ("protenix-v1",) for f in ("templates", "template_structure")},
    **{(m, f): "ESMFold2 has no template input"
       for m in ("esmfold2", "esmfold2-fast") for f in ("templates", "template_structure")},
    **{(m, "protein_free"): "its trunk is conditioned on the ESM protein language model, so a "
       "complex needs at least one protein chain" for m in ("esmfold2", "esmfold2-fast")},
}

#: A route the model offers outside the YAML front door, appended to its refusals.
ELSEWHERE: dict[str, str] = {}


def honoured_by(feature: str) -> tuple[str, ...]:
    """The --model ids that honour ``feature``, read off CAPABILITY so a hint cannot go
    stale when a port gains the feature."""
    return tuple(sorted(m for m, caps in CAPABILITY.items() if caps[feature] == HONOURED))


def _alternatives(feature: str, found, model: str) -> list[str]:
    """Where to send an input ``model`` refuses for ``feature``: the models that honour it and
    refuse nothing else the input carries. An RNA-only input on esmfold2 should hear about the
    models that fold RNA with no protein, not about OpenDDE, which takes no RNA at all. Falls
    back to every model that honours the feature when none takes the whole input."""
    every = [m for m in honoured_by(feature) if m != model]
    whole = [m for m in every if all(CAPABILITY[m][f] != REFUSED for f in found)]
    return whole or every


#: Output/limit flags and the --model ids that actually read each one. A flag a model does
#: not read only omits a file or leaves a default in place, so this warns rather than refuses,
#: the same distinction CAPABILITY draws between REFUSED and NOTED. Kept here next to the
#: capability table because it is the same question -- does the output reflect what I asked
#: for -- and the same failure if it is answered per model in five places.
FLAG_READERS: dict[str, tuple[str, ...]] = {
    "--write_pae": ("boltz2", "protenix-v1", "protenix-v2", "opendde", "opendde-abag"),
    "--write_pde": ("boltz2",),
    "--write_embeddings": ("boltz2",),
    # Everything that folds from an alignment. Left at its default the flag does nothing to
    # protenix/opendde/rf3/openfold3/openbind: they fold the resolved alignment whole, and
    # taking boltz2's 8192 default to them would change every fold they already produced. Set
    # explicitly, it caps them -- the a3m at `main.cap_a3m_text` for protenix/opendde/rf3,
    # `make_openfold3_msa_features(max_sequences=)` for the OF3 family.
    "--max_msa_seqs": ("boltz2", "esmfold2", "esmfold2-fast", "openfold3", "openbind",
                       "protenix-v1", "protenix-v2", "opendde", "opendde-abag", "rf3"),
}

#: (flag, model) -> why that model does not read it, when the generic line is not the reason.
FLAG_WHY: dict[tuple[str, str], str] = {
    ("--write_pae", "openfold3"): "its confidence head computes PAE logits but the fold does "
                                  "not return the matrices",
    ("--write_pae", "openbind"): "its confidence head computes PAE logits but the fold does "
                                 "not return the matrices",
    ("--write_pae", "rf3"): "rf3 writes pTM, ipTM and chain-pair PAE/PDE into "
                            "<name>_summary_confidences.json next to each structure",
    ("--write_pae", "esmfold2"): "it has no PAE head",
    ("--write_pae", "esmfold2-fast"): "it has no PAE head",
    ("--write_pde", "protenix-v1"): "--write_pae already writes PAE and PDE in one npz",
    ("--write_pde", "protenix-v2"): "--write_pae already writes PAE and PDE in one npz",
    ("--write_pde", "opendde"): "it writes PAE only, under --write_pae",
    ("--write_pde", "opendde-abag"): "it writes PAE only, under --write_pae",
    ("--max_msa_seqs", "nesso1"): "it conditions on ESM-2 embeddings, not on an alignment, so "
                                  "there is no depth to cap",
}


def unread_flags(model: str, given: dict[str, bool]) -> list[str]:
    """Warning lines for the flags in ``given`` that ``model`` does not read.

    ``given`` maps a flag name to whether the user passed it, so a caller hands over its own
    parameters and gets back exactly the notes worth printing.
    """
    out = []
    for flag, readers in FLAG_READERS.items():
        if given.get(flag) and model not in readers:
            why = FLAG_WHY.get((flag, model), f"{model} does not emit it")
            out.append(f"Note: --model {model} ignores {flag}: {why}.")
    return out


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
    if chains and not any(m == "protein" for _c, _s, _sp, m, _mo in chains):
        found["protein_free"] = "chain(s) " + ", ".join(c for c, *_r in chains)
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
    # A top-level `templates:` block is the same template given as a structure file
    # (tt_bio/template_cif.py turns it into the per-chain alignment).
    tops = []
    for t in doc.get("templates") or []:
        if isinstance(t, dict):
            ids = t.get("chain_id")
            tops += ([str(x) for x in ids] if isinstance(ids, list)
                     else [str(ids) if ids is not None else "all"])
    per_chain["template_structure"] = tops
    for feature, hits in per_chain.items():
        if hits:
            found[feature] = "chain(s) " + ", ".join(hits)
    # A bond endpoint on a ligand or a modified residue makes that residue atomized upstream;
    # two standard residues (a disulfide) are the separate `polymer_bond` case.
    standard = {cid: (mt, {m["position"] for m in (mods or [])})
                for cid, _s, _sp, mt, mods in (chains or [])}

    def on_standard(atom):
        mt, modded = standard.get(str(atom[0]), ("ligand", ()))
        return mt != "ligand" and atom[1] not in modded

    for c in doc.get("constraints") or []:
        if not isinstance(c, dict):
            continue
        if "bond" in c:
            found.setdefault("bond", "constraints")
            b = c["bond"] if isinstance(c["bond"], dict) else {}
            ends = [b.get("atom1"), b.get("atom2")]
            if all(isinstance(a, (list, tuple)) and len(a) == 3 and on_standard(a) for a in ends):
                found.setdefault("polymer_bond", "constraints: " + " - ".join(
                    "/".join(str(x) for x in a) for a in ends))
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


def check_capabilities(path, chains, model: str, echo=_click_note) -> dict[str, str]:
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
            others = ", ".join(how(m) for m in _alternatives(f, found, model))
            lines.append(f"  - {label} ({found[f]}): {WHY.get((model, f), generic)}."
                         + (f" Honoured by: {others}." if others else ""))
        tail = ELSEWHERE.get(model)
        raise RuntimeError(
            f"{how(model)} cannot honour {len(refused)} part(s) of "
            f"{Path(path).name}:\n" + "\n".join(lines) + (f"\n{tail}" if tail else ""))
    if echo is not None:
        for f in FEATURES:
            if f in found and caps[f] == NOTED:
                label, effect = FEATURES[f]
                others = ", ".join(how(m) for m in _alternatives(f, found, model))
                echo(f"Note: {how(model)} ignores {label} ({found[f]} in "
                     f"{Path(path).name}): {effect}."
                     + (f" Use {others} for it." if others else ""))
    return found


#: Column order and heading for the published matrix, so docs/model-capabilities.md and this
#: table can never disagree about what a column means.
DOC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ligand", "ligand"),
    ("rna", "RNA"),
    ("dna", "DNA"),
    ("protein_free", "no protein chain"),
    ("cyclic", "cyclic"),
    ("modifications", "modifications"),
    ("templates", "template npz"),
    ("template_structure", "template cif"),
    ("bond", "bond constraint"),
    ("polymer_bond", "residue-residue bond"),
    ("pocket", "pocket/contact"),
    ("affinity", "affinity"),
)

_MARK = {HONOURED: "yes", REFUSED: "refused", NOTED: "ignored, warns"}


def markdown_table() -> str:
    """The capability matrix as the markdown block docs/model-capabilities.md carries.

    Rendered from CAPABILITY, checked against the committed doc by
    tests/test_capabilities_doc.py, and printed by ``python3 -m tt_bio.capabilities`` so
    updating the doc after a table change is one command.
    """
    from tt_bio.main import PREDICT_MODELS

    head = "| model | " + " | ".join(h for _f, h in DOC_COLUMNS) + " |"
    rule = "|" + "---|" * (len(DOC_COLUMNS) + 1)
    rows = ["| `" + m + "` | " + " | ".join(_MARK[CAPABILITY[m][f]] for f, _h in DOC_COLUMNS)
            + " |" for m in PREDICT_MODELS]
    return "\n".join([head, rule, *rows])


if __name__ == "__main__":
    print(markdown_table())
