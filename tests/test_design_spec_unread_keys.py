"""A design spec key the model will not read is an error, not a default.

RFD3 and PXDesign both used to take conditioning fields they never pass to the model:
the run cost the same and came back unconditioned, which is indistinguishable from a
design that ignored its own epitope. Both readers now refuse by name through the one
helper in `tt_bio.data.yaml_input`, the way BoltzGen's schema check already did.
"""
from pathlib import Path

import pytest
import yaml

import tt_bio.boltzgen
from tt_bio.boltzgen._config import (
    check_overrides,
    dotlist_to_dict,
    instantiate,
    load_yaml,
    target_kwargs,
)
from tt_bio.data.yaml_input import refuse_unread_keys, refuse_unresolved
from tt_bio.rfd3.input import InputSpecification

RFD3_BASE = {"input": "target.cif", "contig": "A1-50,70"}

# Every upstream field the featurizer never reads, with a value that asks for something.
RFD3_UNREAD = [
    ("select_hotspots", "A10-20"),
    ("select_hotspots", True),
    ("select_hotspots", {"A10-20": "ALL"}),
    ("select_partially_buried", "A10-20"),
    ("select_hbond_donor", "A10-20"),
    ("select_hbond_acceptor", "A10-20"),
    ("select_unfixed_sequence", False),
    ("redesign_motif_sidechains", True),
    ("ori_token", [1.0, 2.0, 3.0]),
    ("infer_ori_strategy", "com"),
    ("plddt_enhanced", False),
    ("dialect", 1),
    ("cif_parser_args", {"cache_dir": "/tmp/x"}),
    ("extra", {"anything": 1}),
    # ligand-only: on a spec with no `ligand` these reach no feature
    ("select_buried", "A10-20"),
    ("select_exposed", "A10-20"),
]


@pytest.mark.parametrize("key,value", RFD3_UNREAD)
def test_rfd3_refuses_a_field_it_never_reads(key, value):
    spec = InputSpecification.from_dict({**RFD3_BASE, key: value})
    with pytest.raises(ValueError, match=key):
        spec.validate()


def test_rfd3_refuses_an_unknown_key_by_name():
    spec = InputSpecification.from_dict({**RFD3_BASE, "select_hotspot": "A10"})
    with pytest.raises(ValueError, match=r"unknown key\(s\).*select_hotspot"):
        spec.validate()


@pytest.mark.parametrize("key,value", [
    ("select_unfixed_sequence", True),
    ("redesign_motif_sidechains", False),
    ("plddt_enhanced", True),
    ("dialect", 2),
])
def test_rfd3_allows_a_field_spelled_out_at_its_default(key, value):
    """Spelling out a default asks for nothing, so it is not an error."""
    InputSpecification.from_dict({**RFD3_BASE, key: value}).validate()


def test_rfd3_still_accepts_the_fields_it_honours():
    InputSpecification.from_dict({
        "input": "target.cif", "contig": "A1-50,70", "length": "70",
        "select_fixed_atoms": {"A5": "N,CA,C,O"}, "is_non_loopy": True, "partial_t": 0.5,
    }).validate()


def test_rfd3_ligand_spec_may_use_the_rasa_bins():
    """select_buried/select_exposed do reach the model for ligand atoms."""
    InputSpecification.from_dict({
        "input": "target.cif", "contig": "A1-50,70", "ligand": "IAI",
        "select_buried": {"IAI": "C1,C2"},
    }).validate()


def _px(tmp_path, chain_props, extra=None):
    cfg = {"target": {"file": "t.cif", "chains": {"A": chain_props}}, "binder_length": 80}
    cfg.update(extra or {})
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.mark.parametrize("props,needle", [
    ({"crop": ["1-10"], "hotspot": [4]}, "hotspot"),
    ({"crops": ["1-10"]}, "crops"),
    ({"crop": ["1-10"], "symmetry": "C3"}, "symmetry"),
    ({"crop": ["1-10"], "msa": "./msa/0"}, "msa"),
])
def test_pxdesign_refuses_a_chain_key_it_does_not_read(tmp_path, props, needle):
    from tt_bio.pxdesign.inputs import read_design_yaml

    with pytest.raises(ValueError, match=needle):
        read_design_yaml(_px(tmp_path, props))


def test_pxdesign_refuses_an_unknown_top_level_key(tmp_path):
    from tt_bio.pxdesign.inputs import read_design_yaml

    with pytest.raises(ValueError, match="num_designs"):
        read_design_yaml(_px(tmp_path, {"crop": ["1-10"]}, extra={"num_designs": 7}))


def test_refuse_unresolved_names_the_numbering_it_matched_against():
    with pytest.raises(ValueError) as e:
        refuse_unresolved("hotspots", [455, 486], list(range(1, 117)), what="chain A")
    msg = str(e.value)
    assert "455" in msg and "486" in msg and "label_seq 1..116" in msg


def test_refuse_unresolved_passes_when_every_residue_is_there():
    refuse_unresolved("hotspots", [40, 99], list(range(1, 117)), what="chain A")


def test_refuse_unread_keys_reports_both_kinds_in_one_message():
    with pytest.raises(ValueError) as e:
        refuse_unread_keys({"a": 1, "b": 2, "typo": 3},
                           honoured=("a",), unimplemented={"b": (None, "not built")},
                           what="spec")
    msg = str(e.value)
    assert "b=2" in msg and "not built" in msg and "typo" in msg


# ---------------------------------------------------------------------------
# BoltzGen `--config <step> <key>=<val>`
#
# Same defect, other front door: the flag deep-merged any key into the step's
# YAML and `Predict.__init__`'s **_ignored_legacy_kwargs swallowed it, so
# `--config design not_a_real_key=5` was a full-cost run at the default with no
# signal. A step config is a `_target_` tree, so the schema is the constructor
# signature at each node -- `check_overrides` validates against that, not
# against a key list somebody has to maintain.

CONFIG_DIR = Path(tt_bio.boltzgen.__file__).parent / "resources" / "config"


def _template(name):
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


@pytest.mark.parametrize("step,dotlist,needle", [
    ("design", ["not_a_real_key=5"], "not_a_real_key"),
    ("design", ["sampling_step=40"], "sampling_steps"),          # suggests the real key
    ("design", ["data.cfg.atom_14=true"], "atom14"),             # nested under a _target_
    ("design", ["data.cfg.tokenizer.atomize=1"], "atomize_modified_residues"),
    ("fold", ["recycling_step=3"], "recycling_steps"),
    ("analysis", ["compute_lddt=true"], "compute_lddts"),
    ("filtering", ["budgets=60"], "budget"),
    ("filtering", ["modalitiy=peptide"], "modality"),
    # `trainer` was a Lightning leftover nothing on Tenstorrent reads
    ("design", ["trainer.devices=4"], "trainer"),
    # the --config help text used to show both of these; neither ever landed
    ("fold", ["num_workers=4"], "num_workers"),
    ("fold", ["data.num_workers=4"], "cfg"),
])
def test_boltzgen_refuses_a_config_key_the_step_cannot_accept(step, dotlist, needle):
    with pytest.raises(ValueError) as e:
        check_overrides(_template(step), dotlist_to_dict(dotlist), step)
    msg = str(e.value)
    # the message names the first segment that is unknown, and what is accepted there
    assert dotlist[0].split("=")[0].split(".")[0] in msg and needle in msg


@pytest.mark.parametrize("step,dotlist", [
    ("design", ["sampling_steps=40"]),                  # in the template
    ("design", ["debug=true"]),                         # declared, absent from the template
    ("design", ["data.cfg.atom14=false"]),              # nested, in the template
    ("design", ["override.masker_args.mask=false"]),    # free-form runtime dict
    ("design", ["override.some_future_knob=1"]),        # same, nothing to check against
    ("fold", ["recycling_steps=1"]),                    # the --config help example
    ("fold", ["data.cfg.num_workers=4"]),               # the DataLoader knob, one level down
    ("filtering", ["budget=60", "alpha=0.05"]),         # filter.py's own docstring example
    ("analysis", ["num_processes=8"]),
])
def test_boltzgen_accepts_a_config_key_the_step_declares(step, dotlist):
    check_overrides(_template(step), dotlist_to_dict(dotlist), step)


@pytest.mark.parametrize("protocol", [
    "protein-anything", "peptide-anything", "nanobody-anything",
    "antibody-anything", "protein-redesign",
])
@pytest.mark.parametrize("flags", [
    [],
    ["--skip_inverse_folding"],
    ["--only_inverse_fold"],
    ["--step_scale", "1.8", "--noise_scale", "0.98"],
    ["--reuse", "--num_designs", "4", "--budget", "2"],
    ["--alpha", "0.05", "--metrics_override", "iptm=2"],
])
def test_boltzgen_pipeline_overrides_all_name_something_real(monkeypatch, protocol, flags):
    """The check has to hold for the pipeline's own overrides too.

    Every step gets a dozen internal `key=value` args (`data.cfg.multiplicity`,
    `writer.designfolding`, the filter's `outdir`). If one of them were misspelled
    it would be as silent as a user's typo, so it goes through the same check.
    """
    import tt_bio.boltzgen.cli.boltzgen as bg

    monkeypatch.setattr(bg, "get_artifact_path",
                        lambda args, spec, repo_type="model": Path("/stub/artifact"))
    args = bg.build_parser().parse_args(
        ["run", "spec.yaml", "--output", "/stub/out", "--protocol", protocol] + flags)
    pipeline = bg.BinderDesignPipeline(args, Path("/stub/artifact"))
    assert pipeline.steps
    for step in pipeline.steps:
        step.check()


def test_boltzgen_config_check_runs_before_anything_executes(monkeypatch, tmp_path):
    """A typo must cost nothing: refused at configure, not mid-pipeline."""
    import tt_bio.boltzgen.cli.boltzgen as bg

    monkeypatch.setattr(bg, "get_artifact_path",
                        lambda args, spec, repo_type="model": Path("/stub/artifact"))
    args = bg.build_parser().parse_args(
        ["run", "spec.yaml", "--output", str(tmp_path), "--config", "analysis", "num_processs=4"])
    pipeline = bg.BinderDesignPipeline(args, Path("/stub/artifact"))
    with pytest.raises(ValueError, match="num_processs"):
        for step in pipeline.steps:
            step.check()


def test_a_kwargs_catch_all_does_not_make_every_key_valid():
    """`Predict.__init__` keeps **kwargs so an old on-disk config still loads.

    That tolerance is for files, not for the CLI: `instantiate` still accepts a
    legacy key, `check_overrides` still refuses it as an override.
    """
    assert "trainer" not in target_kwargs("tt_bio.boltzgen.task.predict.predict.Predict")
    cfg = {**_template("design"), "trainer": {"devices": 4}, "matmul_precision": "high"}
    cfg["checkpoint"] = cfg["output"] = cfg["name"] = "x"
    cfg["data"] = cfg["writer"] = None
    instantiate(cfg)  # tolerated on load
    with pytest.raises(ValueError, match="matmul_precision"):
        check_overrides(_template("design"), {"matmul_precision": "high"}, "design")


# ---------------------------------------------------------------------------
# RFD3: the real reference specs have to keep parsing
#
# The unknown-key refusal above landed with `allow_ligand_on_existing_chain`
# missing from the schema, which is an upstream passthrough field every real
# enzyme and symmetric-with-ligand example sets. Those specs, verbatim in this
# repo as the port's own parity fixtures, were refused before their design ran.

PARITY_SPECS = sorted(
    (Path(__file__).parent.parent / "scripts" / "rfd3_port" / "parity_artifacts").glob(
        "*/spec*.json"))


def test_the_repo_ships_rfd3_reference_specs_to_check():
    assert len(PARITY_SPECS) >= 8, [p.name for p in PARITY_SPECS]


@pytest.mark.parametrize("spec_path", PARITY_SPECS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_no_reference_spec_hits_the_unknown_key_refusal(spec_path):
    """A reference spec may name a feature this port does not implement (one
    fixture is a `dialect: 1` case, deliberately refused). What it must never hit
    is "unknown key": that means the schema is missing a real upstream field."""
    import json

    try:
        InputSpecification.from_dict(json.loads(spec_path.read_text())).validate()
    except ValueError as exc:
        assert "unknown key" not in str(exc), str(exc)


def test_rfd3_accepts_allow_ligand_on_existing_chain_true():
    """True is what the port does: a ligand is found wherever it sits."""
    InputSpecification.from_dict(
        {"length": "60", "ligand": "IAI", "allow_ligand_on_existing_chain": True}).validate()


def test_rfd3_refuses_allow_ligand_on_existing_chain_false():
    spec = InputSpecification.from_dict(
        {"length": "60", "ligand": "IAI", "allow_ligand_on_existing_chain": False})
    with pytest.raises(ValueError, match="allow_ligand_on_existing_chain"):
        spec.validate()


def test_rfd3_refusals_do_not_quote_the_ports_own_pass_numbers():
    """A message a user reads names the condition, not the pass that scoped it."""
    import re

    src = (Path(__file__).parent.parent / "tt_bio" / "rfd3" / "featurize.py").read_text()
    raises = re.findall(r"raise (?:NotImplementedError|ValueError)\((.*?)\)\n", src, re.S)
    assert raises
    bad = [r for r in raises if re.search(r"\bp\d+\+|this pass|\bF\d/F\d\b", r)]
    assert not bad, bad


def test_every_shipped_boltzgen_template_only_names_keys_its_target_declares():
    """The template is a schema instance too, and nothing was checking it.

    Removing the dead `trainer` node left `logger: false` indented under
    `writer:` in two templates, so FoldingWriter got a kwarg it does not take
    and the pipeline died at the design_folding step. The override check could
    not see it: it validates overrides, not the file they merge into.
    """
    bad = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        def walk(node, where):
            if not isinstance(node, dict):
                return
            target = node.get("_target_")
            if isinstance(target, str):
                accepted = target_kwargs(target)
                for key in node:
                    if key != "_target_" and key not in accepted:
                        bad.append(f"{path.name}:{where}.{key} not a parameter of {target}")
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        walk(load_yaml(path), path.stem)
    assert not bad, bad
