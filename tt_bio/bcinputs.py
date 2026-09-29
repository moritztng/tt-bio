"""What a BindCraft 2 campaign's inputs have to say before a card is opened.

BindCraft 2 reads hotspots in the target file's own residue numbering and flags the residues
whose number a span names. A span that names no residue of the target simply sets no flag, and
the interface loss reads that as "this campaign has no epitope"::

    return jnp.where(hotspot_mask.any(), hotspot_contacts, surface_contacts)
                                            -- bindcraft/loss.py, interface_contacts_loss

So a hotspot in an unresolved loop, a typo, or one written against a chain the campaign does not
design against costs no error and no warning: the campaign runs to the end and hands back
confident designs against an epitope nobody chose, and `Hotspot_Contact_Fraction` returns None
rather than failing them. Measured on this fleet 2026-09-29: hotspots `54,56,66,115` on a PD-L1
whose 60-70 loop is unresolved keep three of four, and `62,64,66` keep none.

This module is the check that turns that silence into a refusal. It runs on the host, before
`run_campaign` opens the card, and it names the residue, why it is not there, and what to do.
It refuses a span that resolves to nothing at all and only notes one that resolves in part: a
range across a gap is a region a user can legitimately ask for, a single number that is not
there is a mistake.

The same silence exists in two settings a researcher edits by hand, so they are checked here
too: a confidence threshold written as a percentage (`min_plddt_final: 80`) can never be met and
the campaign accepts nothing for as long as it is left running, and `binder_lengths: 80` raises
`TypeError: 'int' object is not iterable` out of the middle of BindCraft 2 rather than saying
that a length is written `[80]`.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping

#: Everything BindCraft 2 scores on a 0 to 1 confidence, as a campaign setting. A value above 1
#: is a percentage, and no design ever reaches it.
CONFIDENCE_SETTINGS = (
    "min_monomer_plddt_final", "min_ptm_final", "min_iptm_final", "max_ipae_final",
    "min_target_plddt_final",
    *(f"min_plddt_{stage}" for stage in ("screen", "refine", "anneal", "harden", "mutate",
                                         "final")),
    *(f"min_iptm_{stage}" for stage in ("screen", "refine", "anneal", "harden", "mutate",
                                        "final")),
    *(f"max_detarget_iptm_{stage}" for stage in ("screen", "refine", "anneal", "harden",
                                                 "mutate", "final")),
)

#: The same quantities as filter entries. The `*_Fraction` family is BindCraft 2's own check
#: (`settings.reject_percentage_thresholds`), so it is deliberately not repeated here.
CONFIDENCE_FILTERS = ("pTM", "i_pTM", "i_pTM_detarget", "i_pAE", "i_pAE_detarget", "i_pDAE",
                      "Unbound_Binder_pLDDT", "Target_pLDDT", "SS_pLDDT")

#: A hotspot span, in the form the user writes it: an optional chain, a residue, an optional end.
_SPAN = re.compile(r"(?P<chain>[A-Za-z]+)(?P<start>\d+)(?:-(?P<end>\d+))?$")


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def threshold_problems(settings: Mapping) -> list[str]:
    """Confidence thresholds written as percentages, which nothing can ever satisfy."""
    problems = []
    for name in CONFIDENCE_SETTINGS:
        value = _number(settings.get(name))
        if value is not None and value > 1:
            problems.append(f"{name} {value:g} reads a confidence of 0 to 1, not a percentage: "
                            f"write {value / 100:g}. Left as it is, no design is ever accepted.")
    filters = settings.get("filters")
    for name, entry in (filters or {}).items() if isinstance(filters, Mapping) else ():
        value = _number(entry.get("threshold")) if isinstance(entry, Mapping) else None
        if name in CONFIDENCE_FILTERS and value is not None and value > 1:
            problems.append(f"filters.{name}.threshold {value:g} reads a confidence of 0 to 1: "
                            f"{name} cannot exceed 1, so no design is ever accepted. Write "
                            f"{value / 100:g} if it was meant as a percentage.")
    return problems


def binder_length_problems(settings: Mapping) -> list[str]:
    """A binder length written as anything but a list, which BindCraft 2 raises TypeError on."""
    lengths = settings.get("binder_lengths")
    if lengths is None or isinstance(lengths, (list, tuple)) and lengths:
        return []
    written = f"{lengths!r}"
    if isinstance(lengths, str) and (match := re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", lengths)):
        wanted = f"[{match.group(1)}, {match.group(2)}]"
    elif _number(lengths) is not None:
        wanted = f"[{int(lengths)}]"
    else:
        wanted = "[80] for one length, [60, 90] for a range"
    return [f"binder_lengths {written} has to be a list: write {wanted}. BindCraft 2 iterates it, "
            f"so anything else stops the campaign from inside the length sampler."]


def file_problem(label: str, path: str, source: str) -> str | None:
    """A target file BindCraft 2's reader cannot open as text, named as what it is.

    The RCSB hands out `.pdb.gz` and `.cif.gz` by default, and BindCraft 2 reads a structure
    with `open(source).read()`, so a downloaded target arrives as
    `'utf-8' codec can't decode byte 0x8b in position 1`, which names neither the file nor gzip.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return None                             # BindCraft 2's preflight names an unreadable file
    if head[:2] == b"\x1f\x8b":
        return (f"{label}: {source} is gzipped and BindCraft 2 reads a target as text. Unpack it "
                f"first: gunzip -k {source}.")
    try:
        with open(path, encoding="utf-8") as fh:
            fh.read(4096)
    except (UnicodeDecodeError, OSError):
        return (f"{label}: {source} is not text, so it is neither a PDB nor an mmCIF file. "
                f"BindCraft 2 reads .pdb, .cif and a structure pasted in as text; convert a "
                f"binary format (BinaryCIF, an mmtf) first.")
    return None


def _chain_residues(path: str, chains: str) -> dict[str, set[int]]:
    """Every residue number BindCraft 2 will keep, per chain, read the way BindCraft 2 reads it.

    Its own readers, so a residue this drops is a residue the campaign drops: heteroatoms,
    waters and a partly trimmed residue are already gone here.
    """
    from bindcraft.protein import (polymer_atom_records, read_structure_atoms,
                                   selected_chain_names, structure_chain_labels,
                                   structure_chain_names)

    wanted = selected_chain_names(chains, structure_chain_labels(path)) or \
        structure_chain_names(path)
    residues: dict[str, set[int]] = {name: set() for name in wanted}
    for record in polymer_atom_records(read_structure_atoms(path), path):
        if record["chain_id"] in residues:
            residues[record["chain_id"]].add(int(record["res_id"]))
    return residues


def _fasta_residues(path: str, chains: str) -> dict[str, set[int]]:
    from bindcraft.protein import read_fasta_sequences

    records = read_fasta_sequences(path)
    name = (chains or "A").split(",")[0].strip() or "A"
    sequence = records.get(name) or (next(iter(records.values())) if len(records) == 1 else "")
    return {name: set(range(1, len(sequence) + 1))}


def _unresolved_stretch(residues: set[int], number: int) -> str:
    """The run of missing numbers a residue sits in, when the chain covers it and it is absent."""
    low = high = number
    while low - 1 not in residues and low - 1 > min(residues):
        low -= 1
    while high + 1 not in residues and high + 1 < max(residues):
        high += 1
    return f"{low}" if low == high else f"{low}-{high}"


def _span_problem(label: str, kind: str, raw: str, chain: str, start: int, end: int,
                  residues: dict[str, set[int]], source: str, available) -> str | None:
    if chain not in residues:
        designed = ", ".join(residues) or "none"
        where = (f'Add {chain} to the target\'s "chains", or move the {kind} to chain {designed}.'
                 if chain in available else
                 f"{source} holds chain {', '.join(available) or 'none'}, so check the letter.")
        return (f"{label} {kind} {raw}: this campaign designs against chain {designed} of "
                f"{source}, so chain {chain} is not in its target. {where}")
    present = sorted(residues[chain])
    if not present:
        return None     # BindCraft 2's own "has no chain" error says this one better
    if [number for number in range(start, end + 1) if number in residues[chain]]:
        return None
    span = f"residue {start}" if start == end else f"residues {start}-{end}"
    if start > present[-1] or end < present[0]:
        return (f"{label} {kind} {raw}: chain {chain} of {source} has no {span}, its residues run "
                f"{present[0]} to {present[-1]}. A {kind} is read in the file's own residue "
                f"numbering.")
    return (f"{label} {kind} {raw}: chain {chain} of {source} has no {span}, it falls in the "
            f"unresolved stretch {_unresolved_stretch(residues[chain], start)}. A {kind} is read "
            f"in the file's own residue numbering. Name a residue the structure resolves, or "
            f"model the missing one in first.")


def _span_note(label: str, kind: str, raw: str, chain: str, start: int, end: int,
               residues: dict[str, set[int]]) -> str | None:
    missing = [n for n in range(start, end + 1) if n not in residues.get(chain, ())]
    if not missing or len(missing) == end - start + 1:
        return None
    return (f"{label} {kind} {raw}: {len(missing)} of {end - start + 1} residues are not in the "
            f"structure ({missing[0]}-{missing[-1]} unresolved); the rest carry the {kind}.")


def _target_source(target) -> str:
    """What to call a target's file in a message: its name on disk, or that it was pasted in."""
    path = getattr(target, "path", "") or ""
    return "a pasted-in structure" if "\n" in path else os.path.basename(path) or "no file"


def target_problems(target, name: str = "") -> tuple[list[str], list[str]]:
    """One target's hotspot and coldspot spans against the residues its file actually holds."""
    from bindcraft.protein import structure_chain_names
    from bindcraft.protein_preparation import chain_qualified_residue_span
    from bindcraft.settings import is_fasta

    label = f"target {name or getattr(target, 'name', '')!r}"
    path = target.path
    if "\n" in path or not os.path.isfile(path):
        return [], []                                    # BindCraft 2's preflight names this one
    if problem := file_problem(label, path, os.path.basename(path)):
        return [problem], []
    try:
        residues = _fasta_residues(path, target.chains) if is_fasta(path) else \
            _chain_residues(path, target.chains)
    except Exception:
        return [], []                # an unreadable file is BindCraft 2's own error to report
    source = os.path.basename(path)
    available = [] if is_fasta(path) else structure_chain_names(path)
    if not is_fasta(path) and not available:
        # BindCraft 2 reads the polymer and drops every heteroatom, so a file whose every record
        # is one leaves `receptor_chain_names` empty, and the first hotspot span indexes it:
        # `chain_qualified_residue_span` raises IndexError before the campaign says anything.
        return [f"{label}: {source} holds no protein residue to design against. Every record in "
                f"it was read as a heteroatom, a ligand, a metal, a glycan or a water, so the "
                f"file has no chain. Point the target at the file that holds the polymer, and "
                f"check that its residues are ATOM records rather than HETATM."], []
    problems, notes = [], []
    for kind, spans in (("hotspot", target.hotspots), ("coldspot", target.coldspots)):
        for raw in (spans or "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            match = _SPAN.match(chain_qualified_residue_span(raw, list(residues) or ["A"]))
            if not match:
                continue                 # BindCraft 2 refuses a malformed span with its own error
            chain = match.group("chain")
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            start, end = min(start, end), max(start, end)
            problem = _span_problem(label, kind, raw, chain, start, end, residues, source,
                                    available)
            if problem:
                problems.append(problem)
            elif note := _span_note(label, kind, raw, chain, start, end, residues):
                notes.append(note)
    return problems, notes


def input_problems(settings: Mapping) -> tuple[list[str], list[str]]:
    """Everything wrong with a campaign's inputs that this module can name, and what to note."""
    problems = threshold_problems(settings) + binder_length_problems(settings)
    notes: list[str] = []
    if problems:
        return problems, notes                    # a settings error stops the targets resolving
    try:
        from bindcraft.settings import build_design_settings
        targets = build_design_settings(dict(settings)).targets
    except Exception:
        return problems, notes             # BindCraft 2's preflight reports a settings it cannot
    by_name: dict[str, list] = {}
    for target in targets:
        by_name.setdefault(getattr(target, "name", ""), []).append(target)
    for name, group in by_name.items():
        if len(group) > 1:
            # `prepare_targets` returns a dict keyed by target name, so a repeated name is not a
            # second target: it overwrites the first, and the campaign designs against the last
            # one only. Nothing downstream counts targets, so the loss never notices.
            files = ", ".join(_target_source(target) for target in group)
            problems.append(
                f"{len(group)} targets are named {name!r} ({files}). BindCraft 2 keys a "
                f"campaign's targets by name, so only the last of them is prepared and the rest "
                f'are dropped without a word. Give each target its own "name".')
    for target in targets:                 # resolve, with its own message
        target_problem, target_notes = target_problems(target)
        problems += target_problem
        notes += target_notes
    return problems, notes


def refuse_unusable_inputs(settings: Mapping) -> None:
    """Stop a campaign whose inputs would design against something the caller did not ask for."""
    problems, notes = input_problems(settings)
    for note in notes:
        print(f"[tt_bio.bcinputs] {note}", flush=True)
    if problems:
        raise ValueError("\n".join(problems))
