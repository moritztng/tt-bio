#!/usr/bin/env python3
"""D16 arm 1: what `mol_type` actually encodes in each stack, by RUNNING their featurisers.

No card. The mapping is a claim about THEIR data, so a reading of their code is the weaker
evidence and is kept only as the cross-check: each stack's own parse/tokenise/featurise path
is executed on the same four-entity complex (protein + DNA + RNA + CCD_ATP) and the integer
`mol_type` column it emits is read back and grouped by the chain it came from.

The result is the finding: **the three stacks do NOT agree.** Protenix-v2 numbers rna 1 and
dna 2; Boltz-2 and BoltzGen number dna 1 and rna 2. Both put protein at 0 and ligand at 3.
So a single global table would mislabel every nucleic-acid token in one family or the other,
and the adapter has to take the disagreement as input.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

OUT = "perf/of3t_entity"
SCRATCH = "/tmp/of3t/of3t-entity"
MOL_DIR = pathlib.Path("/home/moritz/.boltz/mols")
# The same four entities in each stack's own schema, so the only thing that differs between
# the three answers is the stack. 10 protein / 8 DNA / 8 RNA residues and one CCD ATP.
PROTEIN, DNA, RNA, LIGAND = "GAVLIGAVLI", "ACGTACGT", "ACGUACGU", "ATP"

BOLTZ_YAML = f"""version: 1
sequences:
  - protein: {{id: A, sequence: {PROTEIN}}}
  - dna: {{id: B, sequence: {DNA}}}
  - rna: {{id: C, sequence: {RNA}}}
  - ligand: {{id: D, ccd: {LIGAND}}}
"""
# BoltzGen refuses a spec with nothing to design, so chain E is a design range. It is a
# protein chain and its tokens must come back as protein, which is itself a check.
BOLTZGEN_YAML = f"""entities:
  - protein: {{id: A, sequence: {PROTEIN}}}
  - dna: {{id: B, sequence: {DNA}}}
  - rna: {{id: C, sequence: {RNA}}}
  - ligand: {{id: D, ccd: {LIGAND}}}
  - protein: {{id: E, sequence: 8..8}}
"""


def _observed(mol_type, spans) -> dict:
    """Group the emitted integers by the chain class that produced them."""
    mt = np.asarray(mol_type).tolist()
    out, i = {}, 0
    for cls, n in spans:
        seen = sorted(set(mt[i:i + n]))
        if len(seen) != 1:
            raise AssertionError(f"{cls} span emitted {seen}, not one class")
        out.setdefault(cls, seen[0])
        if out[cls] != seen[0]:
            raise AssertionError(f"{cls} emitted {out[cls]} and {seen[0]}")
        i += n
    if i != len(mt):
        raise AssertionError(f"spans cover {i} tokens, the column has {len(mt)}")
    return out


def measure_protenix() -> dict:
    from tt_bio.protenix_data import MOL_TYPE_IDS, build_complex_features
    feats = build_complex_features([(PROTEIN, None, "protein"), (RNA, None, "rna"),
                                    (DNA, None, "dna"), (f"CCD_{LIGAND}", None, "ligand")])
    mt = feats["mol_type"]
    n_lig = int(len(mt)) - (len(PROTEIN) + len(RNA) + len(DNA))
    obs = _observed(mt, [("protein", len(PROTEIN)), ("rna", len(RNA)),
                         ("dna", len(DNA)), ("ligand", n_lig)])
    return {"stack": "protenix-v2", "observed": obs, "declared": dict(MOL_TYPE_IDS),
            "agrees_with_own_source": obs == dict(MOL_TYPE_IDS),
            "n_tokens": int(len(mt)), "n_ligand_tokens": n_lig,
            "ran": "tt_bio.protenix_data.build_complex_features",
            "source": "tt_bio/protenix_data.py:40 MOL_TYPE_IDS, written at "
                      "protenix_data.py:387 and emitted at protenix_data.py:519",
            "upstream": "upstream Protenix has NO integer mol_type feature -- its featuriser "
                        "emits is_protein/is_ligand/is_dna/is_rna directly "
                        "(protenix/data/core/featurizer.py:542-552) off a STRING mol_type "
                        "annotation, so the integer column is our port's own encoding and "
                        "its order follows OpenFold3's MoleculeType, not ByteDance's"}


def measure_boltz2() -> dict:
    from tt_bio.data import const
    from tt_bio.data.featurizer import Boltz2Featurizer
    from tt_bio.data.mol import load_canonicals
    from tt_bio.data.tokenize import Boltz2Tokenizer
    from tt_bio.main import prepare_features
    path = pathlib.Path(f"{SCRATCH}/boltz2_quad.yaml")
    path.write_text(BOLTZ_YAML)
    feats, _ = prepare_features(
        path, load_canonicals(MOL_DIR), MOL_DIR, pathlib.Path(f"{SCRATCH}/msa"),
        Boltz2Tokenizer(), Boltz2Featurizer(), False, None, None, None, None, None, 16,
        single_sequence=True)
    mt = feats["mol_type"]
    n_lig = int(len(mt)) - (len(PROTEIN) + len(DNA) + len(RNA))
    obs = _observed(mt, [("protein", len(PROTEIN)), ("dna", len(DNA)),
                         ("rna", len(RNA)), ("ligand", n_lig)])
    declared = {"protein": const.chain_type_ids["PROTEIN"],
                "dna": const.chain_type_ids["DNA"], "rna": const.chain_type_ids["RNA"],
                "ligand": const.chain_type_ids["NONPOLYMER"]}
    return {"stack": "boltz-2", "observed": obs, "declared": declared,
            "agrees_with_own_source": obs == declared,
            "n_tokens": int(len(mt)), "n_ligand_tokens": n_lig,
            "ran": "tt_bio.main.prepare_features (parse -> tokenize -> featurize)",
            "source": "tt_bio/data/const.py:8-14 chain_type_ids, assigned at "
                      "tt_bio/data/parse.py:1207-1211 and :1442, emitted at "
                      "tt_bio/data/featurizer.py:651 into featurizer.py:1093",
            "upstream": "NONPOLYMER is upstream Boltz's name for the class upstream's "
                        "diffusion loss calls ligand"}


def measure_boltzgen() -> dict:
    from tt_bio.boltzgen.data.parse.schema import YamlDesignParser
    from tt_bio.boltzgen.data.tokenizer import Tokenizer
    from tt_bio.data import const
    from tt_bio.data.mol import load_canonicals
    path = pathlib.Path(f"{SCRATCH}/boltzgen_quad.yaml")
    path.write_text(BOLTZGEN_YAML)
    target = YamlDesignParser(MOL_DIR).parse_yaml(path, load_canonicals(MOL_DIR), MOL_DIR)
    tok = Tokenizer().tokenize(target.structure)
    mt = tok.tokens["mol_type"]
    n_lig = int(len(mt)) - (2 * len(PROTEIN) - 2 + len(DNA) + len(RNA))
    obs = _observed(mt, [("protein", len(PROTEIN)), ("dna", len(DNA)), ("rna", len(RNA)),
                         ("ligand", n_lig), ("protein", len(mt) - len(PROTEIN)
                                             - len(DNA) - len(RNA) - n_lig)])
    declared = {"protein": const.chain_type_ids["PROTEIN"],
                "dna": const.chain_type_ids["DNA"], "rna": const.chain_type_ids["RNA"],
                "ligand": const.chain_type_ids["NONPOLYMER"]}
    return {"stack": "boltzgen", "observed": obs, "declared": declared,
            "agrees_with_own_source": obs == declared,
            "n_tokens": int(len(mt)), "n_ligand_tokens": n_lig,
            "ran": "tt_bio.boltzgen YamlDesignParser.parse_yaml -> Tokenizer.tokenize",
            "source": "tt_bio/data/const.py:8-14 chain_type_ids (BoltzGen ships no const "
                      "module of its own; tt_bio/boltzgen/data/featurizer.py:15 imports "
                      "this one), assigned at tt_bio/boltzgen/data/parse/schema.py:992 and "
                      ":1066, emitted at tt_bio/boltzgen/data/featurizer.py:700 into :906",
            "upstream": "same const as Boltz-2, and the design chain's tokens come back "
                        "protein, so the design path does not renumber"}


def measure_openfold3() -> dict:
    """OF3 is the model that already worked, and it is here as the fourth convention."""
    sys.path.insert(0, "/home/moritz/.coworker/scratch/of3t-reference/upstream050")
    from openfold3.core.data.resources.residues import MoleculeType
    enum = {"protein": int(MoleculeType.PROTEIN), "rna": int(MoleculeType.RNA),
            "dna": int(MoleculeType.DNA), "ligand": int(MoleculeType.LIGAND)}
    import torch
    batch = torch.load(f"/tmp/of3t/of3t-updaterule/batch_step003.pt", map_location="cpu",
                       weights_only=False)
    present = [k for k in ("is_dna", "is_rna", "is_ligand") if k in batch]
    counts = {k: int(np.asarray(batch[k], np.float64).sum()) for k in present}
    del batch
    return {"stack": "openfold3", "observed": None, "declared": enum,
            "agrees_with_own_source": True,
            "emits": "the three flags directly, so no mol_type adapter is needed",
            "entity_keys_in_the_frozen_5nw3_batch": present, "token_counts": counts,
            "ran": "of3t-reference's frozen batch_step003.pt, the real training artifact",
            "source": "MoleculeType at "
                      "tt_bio/_vendor/openfold3/core/data/resources/residues.py:24-28, "
                      "expanded into the four flags at "
                      "core/data/pipelines/featurization/structure.py:92-107",
            "upstream": "its integer enum is never exposed as a feature; it is expanded "
                        "into is_protein/is_rna/is_dna/is_ligand before the batch"}


def measure_af2() -> dict:
    """AF2 folds proteins. The weighting is vacuous and firing `without` there is correct."""
    import ast
    keys = set()
    for f in ("tt_bio/af2.py", "tt_bio/af2_data.py", "tt_bio/af2_confidence.py",
              "tt_bio/af2_weights.py"):
        tree = ast.parse(pathlib.Path(f).read_text(errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                keys.add(node.slice.value)
            elif isinstance(node, ast.Dict):
                keys |= {k.value for k in node.keys if isinstance(k, ast.Constant)}
    hits = sorted(k for k in keys if isinstance(k, str) and k in
                  ("mol_type", "is_dna", "is_rna", "is_ligand", "is_protein",
                   "entity_type", "molecule_type", "token_type", "atom_type"))
    return {"stack": "af2", "observed": None, "declared": None,
            "agrees_with_own_source": True,
            "molecule_type_keys": hits, "n_string_keys_scanned": len(keys),
            "ran": "AST census over AF2's four shipped modules",
            "source": "no molecule-type feature exists in tt_bio/af2*.py",
            "upstream": "AF2 folds proteins only, so every token is protein, every entity "
                        "flag would be identically zero and the weighting is VACUOUS -- "
                        "`without` firing is the correct report, not a miss"}


def main() -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    rows = [measure_protenix(), measure_boltz2(), measure_boltzgen(),
            measure_openfold3(), measure_af2()]
    by = {r["stack"]: r for r in rows}

    live = {s: by[s]["observed"] for s in ("protenix-v2", "boltz-2", "boltzgen")}
    distinct = {json.dumps(v, sort_keys=True) for v in live.values()}
    agree = len(distinct) == 1
    swapped = (by["protenix-v2"]["observed"]["dna"] == by["boltz-2"]["observed"]["rna"]
               and by["protenix-v2"]["observed"]["rna"] == by["boltz-2"]["observed"]["dna"])
    shared = sorted(k for k in ("protein", "ligand")
                    if len({v[k] for v in live.values()}) == 1)

    rep = {"defect": "D16 -- three of five models emit the entity fact as `mol_type`",
           "method": "each stack's own parse/tokenise/featurise path EXECUTED on the same "
                     "four-entity complex; the code reading is the cross-check, not the "
                     "evidence",
           "complex": {"protein": PROTEIN, "dna": DNA, "rna": RNA, "ligand": LIGAND},
           "stacks": by,
           "three_mol_type_stacks_agree": agree,
           "dna_and_rna_are_swapped_between_the_two_conventions": bool(swapped),
           "classes_all_three_agree_on": shared,
           "conventions": {"af3": by["protenix-v2"]["observed"],
                           "boltz": by["boltz-2"]["observed"]},
           "every_stack_matches_its_own_declared_source": all(
               r["agrees_with_own_source"] for r in rows)}

    for r in rows:
        print(f"[map] {r['stack']:12s} {r['observed'] or r['source'][:60]}"
              f"{'' if r['observed'] is None else '  (' + r['ran'][:42] + ')'}")
    print(f"[map] the three mol_type stacks agree: {agree}; dna/rna swapped between the two "
          f"conventions: {swapped}; all three agree on {shared}")

    ok = (not agree) and swapped and shared == ["ligand", "protein"] and \
        rep["every_stack_matches_its_own_declared_source"]
    rep["pass"] = bool(ok)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/mapping.json", "w") as f:
        json.dump(rep, f, indent=1, sort_keys=True)
    print(f"\n{'PASS' if ok else 'FAIL'} -- wrote {OUT}/mapping.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
