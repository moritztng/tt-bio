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

from .catalog import LIMITS

# A BoltzGen binder length range, e.g. "80..120" (counts the upper bound).
_RANGE = re.compile(r"^\s*(\d+)\s*\.\.\s*(\d+)\s*$")
_NON_LETTER = re.compile(r"[^A-Za-z]")


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


def inspect(content: str) -> dict:
    """Best-effort structural summary of one target/spec. Never raises — returns
    counts plus a ``blocked`` reason string if a forbidden reference is found."""
    info = {"chains": 0, "residues": 0, "ligands": 0, "constraints": 0,
            "has_polymer": False, "blocked": None}
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
                if k in ("protein", "dna", "rna"):
                    info["chains"] += n
                    info["residues"] += _seq_residues(body.get("sequence")) * n
                    info["has_polymer"] = True
                elif k == "ligand":
                    info["ligands"] += n
        cons = data.get("constraints")
        if isinstance(cons, list):
            info["constraints"] = len(cons)
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


def _check_one(content: str, *, where: str) -> None:
    if len(content) > LIMITS["max_content_chars"]:
        raise InputError(
            f"{where} is too large for the free demo "
            f"(limit {LIMITS['max_content_chars']:,} characters).")
    info = inspect(content)
    if info["blocked"]:
        raise InputError(f"{where}: {info['blocked']}.")
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


def check_targets(targets: list) -> None:
    """Validate a predict submission's list of targets against the demo limits."""
    if len(targets) > LIMITS["max_complexes"]:
        raise InputError(_too_many("structures in one run", len(targets),
                                   LIMITS["max_complexes"]))
    for i, t in enumerate(targets):
        content = t.get("content") if isinstance(t, dict) else None
        _check_one(str(content or ""), where=f"Structure {i + 1}")


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
