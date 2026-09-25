"""The af2ig input reader, on its own so a web process can run it without the model.

It needs PyYAML and the standard library only, so JapanFold's API host checks a submission with
the engine's own rules (the same file `tt-bio predict --model af2ig` reads) instead of a copy.
The structure is kept as the text that arrived, PDB or mmCIF; `af2ig.features` converts it.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TARGET_CHAIN = "A"
DEFAULT_BINDER_CHAIN = "B"

#: Every key a submission may carry. Anything else is refused rather than ignored, so no option
#: can be sent and silently dropped. af2ig has none: it always starts from the design's own
#: coordinates, and starting from zeros would be single-sequence AF2 under this name.
_KEYS = {"the document": frozenset({"target", "binder"}),
         "target": frozenset({"structure", "file", "chain"}),
         "binder": frozenset({"sequence", "chain"})}


@dataclass(frozen=True)
class AF2IGInput:
    """One af2ig submission: the designed complex, and the sequence to place on its binder."""

    structure: str
    binder_sequence: str
    target_chain: str = DEFAULT_TARGET_CHAIN
    binder_chain: str = DEFAULT_BINDER_CHAIN


def _as_text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _chain_id(value, where: str, default: str) -> str:
    if value is None:
        return default
    cid = str(value).strip()
    if len(cid) != 1 or cid not in string.ascii_letters + string.digits:
        raise ValueError(f"{where} must be a single-character chain id, got {cid!r}")
    return cid


def read_af2ig_input(path) -> AF2IGInput:
    """Parse an af2ig YAML. Raises ValueError naming what is missing, never a KeyError."""
    import yaml

    path = Path(path)
    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise ValueError(f"{path.name} is not valid YAML: {e}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name} must be a YAML mapping with `target:` and `binder:`")
    target, binder = doc.get("target"), doc.get("binder")
    if not isinstance(target, dict) or not isinstance(binder, dict):
        missing = [k for k in ("target", "binder") if not isinstance(doc.get(k), dict)]
        hint = ("--model af2ig scores a complex you already designed, so it takes a "
                "structure plus the binder's sequence, not a sequence list. "
                "To fold a sequence on its own, use one of the cofolders.")
        raise ValueError(f"{path.name} has no `{'`/`'.join(missing)}:` block. {hint}")
    for where, block in (("the document", doc), ("target", target), ("binder", binder)):
        extra = sorted(map(str, set(block) - _KEYS[where]))
        if extra:
            raise ValueError(f"{path.name}: {where} does not take {', '.join(extra)} "
                             f"(only {', '.join(sorted(_KEYS[where]))}); af2ig has no options")
    if target.get("structure") is not None and target.get("file") is not None:
        raise ValueError("target: give either `structure:` (inline text) or `file:`, not both")
    if target.get("structure") is not None:
        text = _as_text(target["structure"], "target.structure")
    elif target.get("file") is not None:
        ref = Path(_as_text(target["file"], "target.file")).expanduser()
        ref = ref if ref.is_absolute() else (path.parent / ref)
        if not ref.is_file():
            raise ValueError(f"target.file {ref} does not exist")
        text = ref.read_text()
    else:
        raise ValueError("target: needs `structure:` (inline PDB/mmCIF text) or `file:`")
    sequence = _as_text(binder.get("sequence"), "binder.sequence").strip().upper()
    unknown = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if unknown:
        raise ValueError(f"binder.sequence has non-standard residue(s) {''.join(unknown)}; "
                         "AF2-IG places one of the 20 standard types on each position")
    return AF2IGInput(structure=text, binder_sequence=sequence,
                      target_chain=_chain_id(target.get("chain"), "target.chain",
                                             DEFAULT_TARGET_CHAIN),
                      binder_chain=_chain_id(binder.get("chain"), "binder.chain",
                                             DEFAULT_BINDER_CHAIN))
