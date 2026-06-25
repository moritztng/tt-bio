"""Server-side enforcement of the free-demo input limits and safety rules.

A public demo must be impossible to overload or exploit no matter what a client
sends, so every limit is checked here on the *actual parsed input* — the UI
mirrors these rules for friendliness, but this module is the authority. The
numbers come from :data:`tt_bio.platform.catalog.LIMITS` (one source of truth).

Two independent guards back each other up:
  1. a raw size cap on every target/spec (parse-independent), and
  2. structural caps (residues / chains / ligands / constraints) on the parsed
     content, plus blocks on local-file references (custom MSA, templates,
     ``file:`` includes) that could read the server's filesystem.
"""

from __future__ import annotations

import re

import yaml

from .catalog import LIMITS, MODELS, PROTOCOLS

# Model id -> the set of capabilities it supports (the catalog is the authority).
_MODEL_CAPS = {m["id"]: set(m.get("caps", [])) for m in MODELS}
_DEFAULT_MODEL = "boltz2"  # mirrors jobs._build_cmd's "job.model or 'boltz2'"
MODEL_IDS = set(_MODEL_CAPS)                                   # valid predict model ids
PROTOCOL_IDS = {p["id"] for p in PROTOCOLS}                    # valid design protocol ids
_MODEL_NEEDS_MSA = {m["id"]: bool(m.get("needs_msa")) for m in MODELS}


def model_needs_msa(model: str | None) -> bool:
    """True if the model can't fold without an MSA (e.g. Boltz-2)."""
    return _MODEL_NEEDS_MSA.get(model or _DEFAULT_MODEL, False)


# A BoltzGen binder length range, e.g. "80..120" (counts the upper bound).
_RANGE = re.compile(r"^\s*(\d+)\s*\.\.\s*(\d+)\s*$")
_NON_LETTER = re.compile(r"[^A-Za-z]")
# Per-polymer sequence alphabets. Proteins allow any letter (the engine maps a
# non-standard residue to UNK rather than crashing); nucleic acids are strict.
# This rejects the real hazards — digits, punctuation, whitespace — that would
# otherwise reach the featurizer as a "residue" and crash or corrupt the fold.
_SEQ_ALPHABET = {
    "protein": re.compile(r"^[A-Za-z]+$"),
    "dna": re.compile(r"^[ACGTNacgtn]+$"),
    "rna": re.compile(r"^[ACGUNacgun]+$"),
}
_ALPHABET_DESC = {"protein": "amino-acid letters", "dna": "A/C/G/T/N", "rna": "A/C/G/U/N"}


class InputError(ValueError):
    """A user input we refuse — surfaced to the client as a clean 400."""


def _seq_residues(value) -> int:
    """Residue count of a sequence value: a plain sequence, or a ``A..B`` range."""
    s = str(value or "").strip()
    m = _RANGE.match(s)
    if m:
        return max(int(m.group(1)), int(m.group(2)))
    return len(_NON_LETTER.sub("", s))


def _copies(body: dict) -> int:
    """A chain entry's ``id`` may be a list (multiple copies of the sequence)."""
    idv = body.get("id")
    return max(1, len(idv)) if isinstance(idv, list) else 1


def _bad_range(value) -> bool:
    """A design binder length given as a range must be 'low..high' with
    1 <= low <= high. A plain sequence (no '..') is fine. Catches degenerate
    inputs like '120..80', '0..0' or 'a..z' that would otherwise queue a doomed
    design run."""
    s = str(value or "").strip()
    if ".." not in s:
        return False
    m = _RANGE.match(s)
    if not m:
        return True
    lo, hi = int(m.group(1)), int(m.group(2))
    return not (1 <= lo <= hi)


def inspect(content: str) -> dict:
    """Best-effort structural summary of one target/spec. Never raises — returns
    counts plus a ``blocked`` reason string if a forbidden reference is found."""
    info = {"chains": 0, "residues": 0, "ligands": 0, "nucleic": 0, "constraints": 0,
            "binding_constraints": 0, "bond_constraints": 0,
            "has_polymer": False, "blocked": None, "bad_seq": None,
            "ids": [], "ligand_ids": set(), "affinity_binders": [], "bad_range": False}
    text = content or ""

    # Security: refuse anything that could pull a file off the server, even if
    # the YAML doesn't otherwise parse. (Custom MSA must be the literal "empty".)
    if re.search(r"(^|\n)\s*-?\s*file\s*:", text, re.I):
        info["blocked"] = "file references aren't allowed in the demo"
        return info
    if re.search(r"(^|\n)\s*templates?\s*:", text, re.I):
        info["blocked"] = "structure templates aren't allowed in the demo"
        return info
    for m in re.finditer(r"(^|\n)\s*msa\s*:\s*(.+)", text, re.I):
        val = m.group(2).strip().strip("'\"")
        if val and val.lower() != "empty":
            info["blocked"] = "custom MSA files aren't allowed — use Generate MSA"
            return info

    try:
        data = yaml.safe_load(text)
    except Exception:
        data = None

    if isinstance(data, dict):
        entries = data.get("sequences") or data.get("entities") or []
        for e in entries if isinstance(entries, list) else []:
            if not isinstance(e, dict):
                continue
            for key, body in e.items():
                if not isinstance(body, dict):
                    continue
                k = str(key).lower()
                n = _copies(body)
                idv = body.get("id")
                ids = [str(x) for x in idv] if isinstance(idv, list) else ([str(idv)] if idv is not None else [])
                info["ids"].extend(ids)
                if k in ("protein", "dna", "rna"):
                    info["chains"] += n
                    info["residues"] += _seq_residues(body.get("sequence")) * n
                    info["has_polymer"] = True
                    if k in ("dna", "rna"):
                        info["nucleic"] += n
                    seqval = body.get("sequence")
                    if info["bad_seq"] is None:
                        if not isinstance(seqval, str):
                            info["bad_seq"] = f"the {k} chain's sequence must be text"
                        else:
                            s = seqval.strip()
                            # A "low..high" range is a design binder spec (validated by
                            # _bad_range), not a literal sequence — skip the alphabet check.
                            if ".." not in s:
                                if not s:
                                    info["bad_seq"] = f"the {k} chain has an empty sequence"
                                elif not _SEQ_ALPHABET[k].match(s):
                                    info["bad_seq"] = (f"the {k} sequence has invalid characters "
                                                       f"(expected {_ALPHABET_DESC[k]})")
                    if _bad_range(body.get("sequence")):
                        info["bad_range"] = True
                elif k == "ligand":
                    info["ligands"] += n
                    info["ligand_ids"].update(ids)
        cons = data.get("constraints")
        if isinstance(cons, list):
            info["constraints"] = len(cons)
            # Constraint kinds gate differently: pocket/contact "binding constraints"
            # need a constraint embedder (Boltz-2 only); covalent "bond" constraints
            # only need the token-bond graph (Boltz-2 and Protenix-v2 both honour it).
            info["binding_constraints"] = sum(
                1 for c in cons if isinstance(c, dict) and ("pocket" in c or "contact" in c))
            info["bond_constraints"] = sum(
                1 for c in cons if isinstance(c, dict) and "bond" in c)
        props = data.get("properties")
        if isinstance(props, list):
            for p in props:
                if isinstance(p, dict) and isinstance(p.get("affinity"), dict):
                    b = p["affinity"].get("binder")
                    if b is not None:
                        info["affinity_binders"].append(str(b))
    elif text.lstrip().startswith(">"):
        # FASTA: count records and residues.
        info["chains"] = text.count(">")
        info["residues"] = sum(
            len(_NON_LETTER.sub("", ln)) for ln in text.splitlines() if not ln.startswith(">")
        )
        info["has_polymer"] = info["chains"] > 0
    return info


def _too_many(what: str, got: int, cap: int) -> str:
    return (f"Too many {what} ({got}) — the free demo allows at most {cap}. "
            f"This limit exists only because this is a free public demo.")


def _check_model_caps(info: dict, model: str | None, *, where: str) -> None:
    """Reject inputs that ask for a capability the chosen model doesn't have.

    The catalog's per-model ``caps`` are the single source of truth (the UI
    mirrors them). Without this, a protein-only model (ESMFold-2 Fast, Protenix)
    would *silently drop* ligands / nucleic acids / extra chains and fold only
    part of the input — returning a confidently wrong structure. We block that
    here so the user gets a clear message instead of a misleading result.
    """
    caps = _MODEL_CAPS.get(model or _DEFAULT_MODEL)
    if caps is None:  # unknown model id — let the engine reject it, don't guess
        return
    name = model or _DEFAULT_MODEL
    if info["ligands"] and "ligands" not in caps:
        raise InputError(f"{where}: {name} can't fold ligands — use Boltz-2 for "
                         f"protein–ligand complexes (and binding affinity).")
    if info["nucleic"] and "nucleic" not in caps:
        raise InputError(f"{where}: {name} folds proteins only — use Boltz-2 for "
                         f"DNA / RNA.")
    if info["affinity_binders"] and "affinity" not in caps:
        raise InputError(f"{where}: {name} doesn't predict binding affinity — use Boltz-2.")
    if info["binding_constraints"] and "constraints" not in caps:
        raise InputError(f"{where}: {name} doesn't support pocket/contact binding "
                         f"constraints — use Boltz-2.")
    if info["bond_constraints"] and "bonds" not in caps:
        raise InputError(f"{where}: {name} doesn't support covalent bond constraints — "
                         f"use Boltz-2 or Protenix-v2.")
    if info["chains"] > 1 and "multichain" not in caps:
        raise InputError(f"{where}: {name} folds a single chain — use Boltz-2 or "
                         f"ESMFold-2 for multi-chain complexes.")


def _check_one(content: str, *, where: str, model: str | None = None) -> None:
    if len(content) > LIMITS["max_content_chars"]:
        raise InputError(
            f"{where} is too large for the free demo "
            f"(limit {LIMITS['max_content_chars']:,} characters).")
    info = inspect(content)
    if info["blocked"]:
        raise InputError(f"{where}: {info['blocked']}.")
    # Reject empty / unreadable inputs early so a garbage submission never
    # reaches a device. (Ligand-only inputs are caught separately upstream.)
    if not info["has_polymer"] and not info["ligands"]:
        raise InputError(f"{where}: no protein, DNA, or RNA sequence found — check the input.")
    # Reject malformed sequences (non-text, empty, or non-residue characters) so
    # garbage never reaches the featurizer (which would crash or fold nonsense).
    if info["bad_seq"]:
        raise InputError(f"{where}: {info['bad_seq']}.")
    # Reject inputs that exceed the chosen model's capabilities (e.g. a ligand or
    # a second chain sent to a protein-only model), so nothing is silently dropped.
    _check_model_caps(info, model, where=where)
    # Degenerate binder length range (e.g. 120..80, 0..0, a..z).
    if info["bad_range"]:
        raise InputError(f"{where}: binder length must be a range like '80..120' "
                         f"(low ≤ high, ≥ 1) or a sequence.")
    # Chain/entity ids must be unique within one structure.
    ids = [i for i in info["ids"] if i]
    if len(ids) != len(set(ids)):
        dup = next((i for i in ids if ids.count(i) > 1), "")
        raise InputError(f"{where}: duplicate chain id '{dup}' — every chain needs a unique id.")
    # An affinity 'binder' must name a ligand that's actually in the input.
    for b in info["affinity_binders"]:
        if b not in info["ligand_ids"]:
            raise InputError(f"{where}: affinity binder '{b}' must be a ligand id present in the input "
                             f"(affinity is predicted for a small-molecule ligand).")
    if info["chains"] > LIMITS["max_chains_per_complex"]:
        raise InputError(_too_many("chains in one complex", info["chains"],
                                   LIMITS["max_chains_per_complex"]))
    if info["ligands"] > LIMITS["max_ligands_per_complex"]:
        raise InputError(_too_many("ligands in one complex", info["ligands"],
                                   LIMITS["max_ligands_per_complex"]))
    if info["constraints"] > LIMITS["max_constraints_per_complex"]:
        raise InputError(_too_many("binding constraints", info["constraints"],
                                   LIMITS["max_constraints_per_complex"]))
    if info["residues"] > LIMITS["max_residues"]:
        raise InputError(
            f"{where} has ~{info['residues']} residues — the free demo is limited to "
            f"{LIMITS['max_residues']} per structure. Try a smaller construct or domain.")


def check_targets(targets: list, model: str | None = None) -> None:
    """Validate a predict submission's list of targets against the demo limits
    and the chosen model's capabilities."""
    if len(targets) > LIMITS["max_complexes"]:
        raise InputError(_too_many("structures in one run", len(targets),
                                   LIMITS["max_complexes"]))
    for i, t in enumerate(targets):
        content = t.get("content") if isinstance(t, dict) else None
        _check_one(str(content or ""), where=f"Structure {i + 1}", model=model)


def check_design(spec: str) -> None:
    """Validate a design submission's spec against the demo limits."""
    _check_one(spec or "", where="Design spec")


def clamp_params(params: dict, kind: str) -> dict:
    """Clamp every numeric knob into its allowed demo range — the client is never
    trusted. Returns a sanitised copy (the allow-listed command builder already
    drops unknown keys; this bounds the known ones)."""
    p = dict(params)

    def clamp(key: str, lo: int, hi: int) -> None:
        v = p.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return
        p[key] = int(max(lo, min(hi, v)))

    if kind == "predict":
        clamp("recycling_steps", 1, LIMITS["max_recycling_steps"])
        clamp("sampling_steps", 10, LIMITS["max_sampling_steps"])
        clamp("diffusion_samples", 1, LIMITS["max_diffusion_samples"])
    else:
        clamp("num_designs", 1, LIMITS["max_designs"])
        clamp("budget", 1, LIMITS["max_budget"])
    return p
