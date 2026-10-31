"""Protenix / Mol-1 input JSON -> tt-bio's bio-input document.

Upstream Protenix and Mol-1 take a JSON list of jobs (``docs/infer_json_format.md`` in either repo):
``[{name, sequences: [{proteinChain|dnaSequence|rnaSequence|ligand|ion: {...}}], covalent_bonds, modelSeeds}]``.
tt-bio's readers take the Boltz-style YAML document. ``read_jobs`` turns each upstream job into that document plus
the parts the YAML has no field for (two-file MSAs, template hits, seeds), so the whole Protenix family reads
upstream JSON through the readers it already has.

Chain ids follow upstream's ``SampleDictToFeatures.build_full_atom_array``: an entity's ``id`` list when given,
otherwise the next free letter in A, B, ..., Z, AA, AB order, skipping ids claimed anywhere in the job. Ligands and
ions take letters too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KINDS = {"proteinChain": "protein", "dnaSequence": "dna", "rnaSequence": "rna", "ligand": "ligand", "ion": "ligand"}
_MOD_KEYS = {"protein": ("ptmType", "ptmPosition"), "dna": ("modificationType", "basePosition"),
             "rna": ("modificationType", "basePosition")}
# deprecated left_/right_ bond keys upstream still accepts
_BOND_ALIASES = {"left_entity": "entity1", "left_copy": "copy1", "left_position": "position1", "left_atom": "atom1",
                 "right_entity": "entity2", "right_copy": "copy2", "right_position": "position2",
                 "right_atom": "atom2"}


@dataclass
class Job:
    name: str
    doc: dict                                         # Boltz-style YAML document
    seeds: list | None = None                         # modelSeeds
    paired_msa: dict = field(default_factory=dict)    # chain id -> a3m text
    unpaired_msa: dict = field(default_factory=dict)  # chain id -> a3m text (protein, and RNA)
    templates: dict = field(default_factory=dict)     # chain id -> templatesPath
    ions: list = field(default_factory=list)          # chain ids of `ion` entities (upstream's MSA skips them)


def is_job_list(path) -> bool:
    """True when ``path`` is an upstream JSON input (a list of jobs with ``sequences``)."""
    path = Path(path)
    if path.suffix.lower() != ".json":
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, list) and bool(data) and all(isinstance(j, dict) and "sequences" in j for j in data)


def int_to_letters(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA (upstream ``mol.data.utils.int_to_letters``)."""
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def _strip_ccd(code: str) -> str:
    code = str(code).strip()
    return code[4:] if code.upper().startswith("CCD_") else code


def _read_text(entity: dict, inline: str, path_key: str):
    if entity.get(inline) is not None:
        return entity[inline]
    p = entity.get(path_key)
    if p and Path(p).exists():       # a missing path counts as absent upstream, which then searches
        return Path(p).read_text()
    return None


def _legacy_msa(entity: dict):
    d = (entity.get("msa") or {}).get("precomputed_msa_dir") if isinstance(entity.get("msa"), dict) else None
    if not d or not Path(d).is_dir():
        return None, None
    rd = lambda f: (Path(d) / f).read_text() if (Path(d) / f).exists() else None
    return rd("pairing.a3m"), rd("non_pairing.a3m")


def _chain_ids(job: dict) -> list[list[str]]:
    """Per entity, the chain id of each copy, exactly as upstream assigns them."""
    used, explicit = set(), []
    for i, ent in enumerate(job["sequences"]):
        (body,) = ent.values()
        ids = body.get("id")
        count = int(body.get("count", 1))
        if ids:
            if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                raise ValueError(f'sequences[{i}]: "id" must be a list of strings')
            if len(ids) != count:
                raise ValueError(f'sequences[{i}]: len(id) ({len(ids)}) != count ({count})')
            if len(set(ids)) != len(ids) or set(ids) & used:
                raise ValueError(f'sequences[{i}]: duplicated chain ids {ids}')
            used.update(ids)
        explicit.append(ids)
    out, nxt = [], 0
    for ent, ids in zip(job["sequences"], explicit):
        (body,) = ent.values()
        if ids:
            out.append(list(ids))
            continue
        cur = []
        for _ in range(int(body.get("count", 1))):
            while True:
                nxt += 1
                cand = int_to_letters(nxt)
                if cand not in used:
                    used.add(cand)
                    cur.append(cand)
                    break
        out.append(cur)
    return out


def job_to_doc(job: dict) -> Job:
    """One upstream job -> ``Job`` (YAML document + the fields YAML has no slot for)."""
    if not isinstance(job.get("sequences"), list) or not job["sequences"]:
        raise ValueError(f"job {job.get('name')!r}: `sequences` must be a non-empty list")
    for i, ent in enumerate(job["sequences"]):
        if not isinstance(ent, dict) or len(ent) != 1 or next(iter(ent)) not in KINDS:
            raise ValueError(f"sequences[{i}]: expected one of {sorted(KINDS)}, got {list(ent)}")
    ids = _chain_ids(job)
    out = Job(name=str(job.get("name") or "job"), doc={"version": 1, "sequences": []},
              seeds=[int(s) for s in job["modelSeeds"]] if job.get("modelSeeds") else None)
    for ent, cids in zip(job["sequences"], ids):
        ((kind, body),) = ent.items()
        mt = KINDS[kind]
        if mt == "ligand":
            spec = body.get("ligand") if kind == "ligand" else body.get("ion")
            if not spec:
                raise ValueError(f"{kind} entity needs a `{kind}` value")
            spec = str(spec).strip()
            if kind == "ion":
                lig = {"ccd": [_strip_ccd(spec).upper()]}
            elif spec.upper().startswith("CCD_"):
                lig = {"ccd": [c.upper() for c in spec[4:].split("_") if c]}
            elif spec.startswith("FILE_"):
                lig = {"file": str(Path(spec[5:]).expanduser().resolve())}
            else:
                lig = {"smiles": spec}
            out.doc["sequences"].append({"ligand": {"id": cids, **lig}})
            if kind == "ion":
                out.ions += cids
            continue
        sub = {"id": cids, "sequence": str(body["sequence"])}
        tkey, pkey = _MOD_KEYS[mt]
        mods = [{"position": int(m[pkey]), "ccd": _strip_ccd(m[tkey]).upper()} for m in body.get("modifications") or []]
        if mods:
            sub["modifications"] = mods
        out.doc["sequences"].append({mt: sub})
        if mt == "protein":
            paired, unpaired = _read_text(body, "pairedMsa", "pairedMsaPath"), \
                _read_text(body, "unpairedMsa", "unpairedMsaPath")
            if paired is None and unpaired is None:
                paired, unpaired = _legacy_msa(body)
            for c in cids:
                if paired is not None:
                    out.paired_msa[c] = paired
                if unpaired is not None:
                    out.unpaired_msa[c] = unpaired
                if body.get("templatesPath"):
                    out.templates[c] = body["templatesPath"]
        elif mt == "rna":
            unpaired = _read_text(body, "unpairedMsa", "unpairedMsaPath")
            if unpaired is not None:
                out.unpaired_msa.update({c: unpaired for c in cids})
    bonds = [_bond(b, ids, job["sequences"]) for b in job.get("covalent_bonds") or []]
    cons = [{"bond": {"atom1": a1, "atom2": a2}} for pair in bonds for a1, a2 in pair]
    if cons:
        out.doc["constraints"] = cons
    return out


def _bond(b: dict, ids: list, seqs: list) -> list:
    """One upstream covalent_bonds record -> [(atom1, atom2), ...] as [chain, residue, atom] triples.

    Without copy1/copy2 the bond is made between corresponding copies, which needs equal copy counts
    (upstream's rule)."""
    b = {_BOND_ALIASES.get(k, k): v for k, v in b.items()}
    e1, e2 = int(b["entity1"]) - 1, int(b["entity2"]) - 1
    if not (0 <= e1 < len(ids) and 0 <= e2 < len(ids)):
        raise ValueError(f"covalent bond {b}: entity out of range 1..{len(ids)}")
    c1, c2 = b.get("copy1"), b.get("copy2")
    if (c1 is None) != (c2 is None):
        raise ValueError(f"covalent bond {b}: give both copy1 and copy2, or neither")
    if c1 is None:
        if len(ids[e1]) != len(ids[e2]):
            raise ValueError(f"covalent bond {b}: without copy1/copy2 both entities need the same count")
        pairs = list(zip(ids[e1], ids[e2]))
    else:
        pairs = [(ids[e1][int(c1) - 1], ids[e2][int(c2) - 1])]
    p1, p2 = int(b["position1"]), int(b["position2"])
    a1, a2 = _atom_name(b["atom1"], seqs[e1]), _atom_name(b["atom2"], seqs[e2])
    return [([x, p1, a1], [y, p2, a2]) for x, y in pairs]


def _atom_name(atom, entity: dict) -> str:
    """A bond atom as a name. Upstream also takes an integer (or a digit string) on a SMILES or
    FILE_ ligand: the SMILES atom-map number when the SMILES carries maps, otherwise the RDKit
    atom index, resolved to the element+ordinal name the featurizer gives that atom."""
    if not (isinstance(atom, int) or (isinstance(atom, str) and atom.isdigit())):
        return str(atom)
    (body,) = entity.values()
    spec = str(body.get("ligand") or "")
    if not spec or spec.upper().startswith("CCD_"):
        raise ValueError(f"bond atom {atom!r}: an integer atom names an atom only on a SMILES or FILE_ ligand")
    from rdkit import Chem
    from tt_bio.protenix_data import _file_to_mol, _name_atoms
    if spec.startswith("FILE_"):
        mol = _file_to_mol(str(Path(spec[5:]).expanduser().resolve()))
    else:
        mol = Chem.MolFromSmiles(spec)
        if mol is None:
            raise ValueError(f"could not parse SMILES ligand: {spec!r}")
        mol = _name_atoms(Chem.AddHs(mol))
    maps = {a.GetAtomMapNum(): a.GetProp("name") for a in mol.GetAtoms() if a.GetAtomMapNum()}
    table = maps or {a.GetIdx(): a.GetProp("name") for a in mol.GetAtoms()}
    if int(atom) not in table:
        raise ValueError(f"bond atom {atom!r} is not an atom {'map number' if maps else 'index'} of ligand {spec!r}")
    return table[int(atom)]


def read_jobs(path) -> list[Job]:
    """Every job in an upstream JSON input file, in file order."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"{path}: the top level must be a list of jobs")
    return [job_to_doc(j) for j in data]


SIDECAR = ".upstream.json"
SEED_SEP = "__seed_"
DEFAULT_SEEDS = (101,)          # `mol pred -s` default


def job_seeds(job: Job, seeds=None, use_seeds_in_json: bool = False) -> list[int]:
    """The seeds one job runs, as upstream picks them: the job's ``modelSeeds`` under ``--use_seeds_in_json``,
    otherwise the command-line seeds (``mol pred -s``, default 101)."""
    if use_seeds_in_json:
        if not job.seeds:
            raise ValueError(f"job {job.name!r}: --use_seeds_in_json needs `modelSeeds` in the job")
        return list(job.seeds)
    return [int(s) for s in (seeds or DEFAULT_SEEDS)]


def stage(path, dest, seeds=None, use_seeds_in_json: bool = False) -> Path:
    """Expand an upstream JSON input into one ``dest/<job>__seed_<s>.yaml`` per job and seed, plus one
    ``<job>.upstream.json`` sidecar per job carrying what YAML has no field for (two-file MSAs, template paths,
    seeds). Returns ``dest``, which the normal input discovery then walks like any directory of YAML inputs. Job
    names must be unique, as upstream's output directories are keyed by them."""
    import yaml

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = read_jobs(path)
    names = [j.name for j in jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"{Path(path).name}: job names {dup} repeat; every job needs its own `name`")
    bad = sorted(n for n in names if SEED_SEP in n or "/" in n)
    if bad:
        raise ValueError(f"{Path(path).name}: job names {bad} may not contain '/' or '{SEED_SEP}'")
    for j in jobs:
        text = yaml.safe_dump(j.doc, sort_keys=False)
        for s in job_seeds(j, seeds, use_seeds_in_json):
            (dest / f"{j.name}{SEED_SEP}{s}.yaml").write_text(text)
        (dest / f"{j.name}{SIDECAR}").write_text(json.dumps(
            {"seeds": j.seeds, "paired_msa": j.paired_msa, "unpaired_msa": j.unpaired_msa, "templates": j.templates,
             "ions": j.ions}))
    return dest


SEARCH_TOOLS = ("hmmsearch_binary_path", "hmmbuild_binary_path", "seqres_database_path", "nhmmer_binary_path",
                "hmmalign_binary_path", "hmmbuild_rna_binary_path", "ntrna_database_path", "rfam_database_path",
                "rna_central_database_path", "nhmmer_n_cpu")


def search(path, dest, use_template: bool = False, use_rna_msa: bool = False, **tools) -> Path:
    """Upstream's template and RNA MSA searches (``mol pred`` preprocess steps 2 and 3, Mol-1's own code under
    tt_bio/_vendor/mol1). Under ``use_template`` a protein chain without ``templatesPath`` whose MSA directory holds
    pairing.a3m/non_pairing.a3m gets ``hmmsearch.a3m`` there; under ``use_rna_msa`` an RNA chain without
    ``unpairedMsaPath`` is searched with nhmmer into ``dest/<job>/rna_msa/<k>/``. ``tools`` are upstream's
    ``SEARCH_TOOLS`` options (None: PATH and $MOL_ROOT_DIR/search_database, as upstream). Returns ``path`` when
    nothing changed, else ``dest/<stem>-final-updated.json``."""
    unknown = set(tools) - set(SEARCH_TOOLS)
    if unknown:
        raise TypeError(f"unknown search options {sorted(unknown)}")
    if not (use_template or use_rna_msa):
        return Path(path)
    data = json.loads(Path(path).read_text())
    t = {k: tools.get(k) for k in SEARCH_TOOLS}
    updated = False
    if use_template:
        from tt_bio._vendor.mol1.template_search import update_template_info
        updated |= update_template_info(data, hmmsearch_binary_path=t["hmmsearch_binary_path"],
                                        hmmbuild_binary_path=t["hmmbuild_binary_path"],
                                        seqres_database_path=t["seqres_database_path"])
    if use_rna_msa:
        from tt_bio._vendor.mol1.rna_msa_search import update_rna_msa_info
        updated |= update_rna_msa_info(
            data, out_dir=str(dest), nhmmer_binary_path=t["nhmmer_binary_path"],
            hmmalign_binary_path=t["hmmalign_binary_path"],
            hmmbuild_binary_path=t["hmmbuild_rna_binary_path"] or t["hmmbuild_binary_path"],
            ntrna_database_path=t["ntrna_database_path"], rfam_database_path=t["rfam_database_path"],
            rna_central_database_path=t["rna_central_database_path"], nhmmer_n_cpu=t["nhmmer_n_cpu"])
    if not updated:
        return Path(path)
    out = Path(dest) / f"{Path(path).stem.replace('-update-msa', '')}-final-updated.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=4))
    return out


def split_stem(stem: str):
    """``<job>__seed_<s>`` -> (job, s); anything else -> (stem, None)."""
    job, sep, s = str(stem).rpartition(SEED_SEP)
    return (job, int(s)) if sep and job and s.lstrip("-").isdigit() else (str(stem), None)


def sidecar(yaml_path):
    """The staged extras for an input written by ``stage`` (with ``job`` and ``seed`` added), or None for an
    ordinary YAML input."""
    job, seed = split_stem(Path(yaml_path).stem)
    p = Path(yaml_path).parent / (job + SIDECAR)
    if seed is None or not p.exists():
        return None
    return {**json.loads(p.read_text()), "job": job, "seed": seed}
