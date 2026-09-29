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
            text = fh.read(8192)
    except (UnicodeDecodeError, OSError):
        return (f"{label}: {source} is not text, so it is neither a PDB nor an mmCIF file. "
                f"BindCraft 2 reads .pdb, .cif and a structure pasted in as text; convert a "
                f"binary format (BinaryCIF, an mmtf) first.")
    return _extension_problem(label, path, source, text)


CIF_SUFFIXES = (".cif", ".mmcif", ".pdbx")
PDB_SUFFIXES = (".pdb", ".ent", ".pdb1")
PDB_RECORDS = ("HEADER", "REMARK", "CRYST1", "SEQRES", "TITLE ", "COMPND", "MODEL ",
               "ATOM  ", "HETATM", "EXPDTA")


def _extension_problem(label: str, path: str, source: str, text: str) -> str | None:
    """A file whose extension says one format and whose records say the other.

    BindCraft 2 dispatches on the suffix, so the reader that runs is the wrong one and the error
    comes from inside it: a PDB named .cif is `There are no blocks in the file` and an mmCIF
    named .pdb is `Illegal hybrid-36 string '8  .'`. Neither names the file or the format.
    """
    suffix = os.path.splitext(path)[1].lower()
    is_cif = "_atom_site." in text or text.lstrip().startswith("data_")
    is_pdb = any(line.startswith(PDB_RECORDS) for line in text.splitlines())
    if suffix in CIF_SUFFIXES and is_pdb and not is_cif:
        return (f"{label}: {source} is named {suffix} but holds PDB records, and BindCraft 2 "
                f"reads a {suffix} with an mmCIF reader, which fails with 'There are no blocks "
                f"in the file'. Rename it to .pdb, or convert it to mmCIF.")
    if suffix in PDB_SUFFIXES and is_cif and not is_pdb:
        return (f"{label}: {source} is named {suffix} but holds an mmCIF block, and BindCraft 2 "
                f"reads a {suffix} with a PDB reader, which fails with 'Illegal hybrid-36 "
                f"string'. Rename it to .cif.")
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


def _dropped_residue(path: str, chain: str, number: int) -> str | None:
    """The residue name a hotspot points at when the polymer reader has already dropped it.

    A researcher reads a residue number off a viewer, where the ligands, the metals, the glycans
    and the waters are all on screen with numbers of their own. `polymer_atom_records` drops
    every one of them, so "no residue 401" is true of the target and false of the file the
    caller is looking at: naming NAG 401 is what tells them which.
    """
    from bindcraft.protein import read_structure_atoms

    try:
        for record in read_structure_atoms(path):
            if str(record["chain_id"]) == chain and int(record["res_id"]) == number:
                return str(record["res_name"]).strip() or None
    except Exception:
        return None                 # the file reads or it does not; the caller has other errors
    return None


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
                  residues: dict[str, set[int]], source: str, available,
                  path: str = "") -> str | None:
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
    if start == end and path and (dropped := _dropped_residue(path, chain, start)):
        return (f"{label} {kind} {raw}: residue {start} of chain {chain} in {source} is "
                f"{dropped}, a heteroatom, and BindCraft 2 designs against the polymer only, so "
                f"it drops it with the ligands, the metals, the glycans and the waters. A {kind} "
                f"names a protein residue; chain {chain}'s run {present[0]} to {present[-1]}.")
    if start > present[-1] or end < present[0]:
        return (f"{label} {kind} {raw}: chain {chain} of {source} has no {span}, its residues run "
                f"{present[0]} to {present[-1]}. A {kind} is read in the file's own residue "
                f"numbering.")
    return (f"{label} {kind} {raw}: chain {chain} of {source} has no {span}, it falls in the "
            f"unresolved stretch {_unresolved_stretch(residues[chain], start)}. A {kind} is read "
            f"in the file's own residue numbering. Name a residue the structure resolves, or "
            f"model the missing one in first.")


_NEGATIVE = re.compile(r"[A-Za-z]*-\d")


def _negative_span_problem(label: str, kind: str, raw: str, residues: dict[str, set[int]],
                           source: str) -> str | None:
    """A residue named with a minus, which the span syntax reads as its range separator.

    A deposited structure that keeps an expression tag numbers residues from below 1, and those
    residues cannot be written as a span at all: BindCraft 2 answers `'A-2+HOTSPOT' outside any
    span`, which names the syntax but not the reason the file cannot use it.
    """
    if not _NEGATIVE.match(raw.strip()):
        return None
    lowest = min((number for numbers in residues.values() for number in numbers), default=1)
    numbering = (f"{source} numbers its residues from {lowest}, and a residue at or below zero "
                 f"cannot be named: ") if lowest < 1 else ""
    return (f"{label} {kind} {raw}: {numbering}a span is written A35 or A35-40, so the minus is "
            f"read as the range separator rather than as part of a residue number. Renumber the "
            f"file from 1 and name the residues in that numbering, which is the numbering "
            f"BindCraft 2 reads throughout.")


#: A span the user wrote with a chain letter in front, as opposed to a bare residue number.
_WRITTEN_CHAIN = re.compile(r"\s*([A-Za-z])\s*\d")


def _ambiguous_chain_note(label: str, kind: str, raw: str, start: int, end: int,
                          residues: dict[str, set[int]], source: str) -> str | None:
    """A bare residue number on a target whose chains share that number.

    `chain_qualified_residue_span` qualifies a bare span with the target's first chain, which is
    the right default and is silent about the alternative. Most deposited complexes number every
    chain from 1, so the number a researcher copies out of a paper about one chain exists in the
    others too, and the flag lands on whichever chain happens to be first.
    """
    if _WRITTEN_CHAIN.match(raw):
        return None                                   # the user named the chain; nothing to say
    wanted = set(range(start, end + 1))
    holders = [chain for chain, numbers in residues.items() if numbers & wanted]
    if len(holders) < 2:
        return None
    return (f"{label} {kind} {raw} names no chain, and chains {', '.join(holders)} of {source} "
            f"each hold those residues. BindCraft 2 takes the first, chain {holders[0]}, and "
            f"that is the one carrying the {kind}. Write {holders[0]}{raw.strip()} to say so, "
            f"or {holders[1]}{raw.strip()} for the other one.")


def _span_note(label: str, kind: str, raw: str, chain: str, start: int, end: int,
               residues: dict[str, set[int]]) -> str | None:
    missing = [n for n in range(start, end + 1) if n not in residues.get(chain, ())]
    if not missing or len(missing) == end - start + 1:
        return None
    return (f"{label} {kind} {raw}: {len(missing)} of {end - start + 1} residues are not in the "
            f"structure ({missing[0]}-{missing[-1]} unresolved); the rest carry the {kind}.")


SPAN_FIELDS = ("chains", "hotspots", "coldspots")


def shape_problems(label: str, target) -> list[str]:
    """A target field written as a JSON list where BindCraft 2 reads one comma-separated string.

    `target_binding_site_flags` and `selected_chain_names` both call `str.split` on these, so a
    list reaches them as a list: `AttributeError: 'list' object has no attribute 'split'`, from
    inside the campaign, naming neither the field nor the target.
    """
    problems = []
    for field in SPAN_FIELDS:
        value = getattr(target, field, None)
        if value is None or isinstance(value, str):
            continue
        written = ",".join(str(item) for item in value) \
            if isinstance(value, (list, tuple, set)) else str(value)
        problems.append(
            f'{label}: "{field}" is a {type(value).__name__}, and BindCraft 2 reads it as one '
            f'comma-separated string. Write it as "{field}": "{written}". A list reaches '
            f"str.split as a list and fails inside the campaign with AttributeError.")
    return problems


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
    if shaped := shape_problems(label, target):
        return shaped, []          # every reader below splits these fields on a comma
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
            if problem := _negative_span_problem(label, kind, raw, residues, source):
                problems.append(problem)
                continue
            match = _SPAN.match(chain_qualified_residue_span(raw, list(residues) or ["A"]))
            if not match:
                continue                 # BindCraft 2 refuses a malformed span with its own error
            chain = match.group("chain")
            start = int(match.group("start"))
            end = int(match.group("end") or start)
            start, end = min(start, end), max(start, end)
            problem = _span_problem(label, kind, raw, chain, start, end, residues, source,
                                    available, "" if is_fasta(path) else path)
            if problem:
                problems.append(problem)
            elif note := _span_note(label, kind, raw, chain, start, end, residues):
                notes.append(note)
            elif note := _ambiguous_chain_note(label, kind, raw, start, end, residues, source):
                notes.append(note)
    return problems, notes


RECYCLE_SETTINGS = ("design_recycles", "validation_recycles", "betasheet_reopt_recycles")


def count_problems(settings: Mapping) -> list[str]:
    """Counts that are accepted here and fail, or end the campaign, somewhere far from here."""
    problems = []
    for name in RECYCLE_SETTINGS:
        value = settings.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
            # `recycled_alphafold_outputs` splits a key into `num_recycle + 1`, so a negative
            # count is an invalid tensor dimension: MLIRError out of the first fold, with the
            # card already open and nothing said about the setting.
            problems.append(
                f"{name} {value} is not a recycle count. AlphaFold 2 folds {name} + 1 times, so "
                f"a negative count reaches JAX as an invalid tensor dimension and fails inside "
                f"the first fold with the card already open. Write 0 for a single pass with no "
                f"recycling, or a positive count.")
    designs = settings.get("number_of_final_designs")
    if isinstance(designs, (int, float)) and not isinstance(designs, bool) and designs < 1 \
            and not settings.get("trajectory_only"):
        # `claim_trajectory` hands out no trajectory once accepted >= requested, and that is
        # already true at zero: the campaign reports itself finished having designed nothing.
        problems.append(
            f"number_of_final_designs {designs} ends the campaign before it takes a single "
            f"trajectory, because BindCraft 2 stops as soon as the accepted count reaches it, "
            f"and it starts at zero. Write how many designs you want, or pass trajectory_only "
            f"with max_trajectories to run trajectories without accepting any.")
    return problems


#: A, C, G and T are nucleotides and they are also alanine, cysteine, glycine and threonine, so
#: a pasted-in DNA sequence is a valid protein sequence and BindCraft 2 folds it as one.
NUCLEOTIDE_LETTERS = frozenset("ACGT")
#: Below this length, and below three of the four letters, a real low-complexity peptide is
#: plausible: poly-alanine and the (GA)n elastin-like and silk-like designs are ordinary
#: research sequences, and refusing one of those would be worse than accepting a short DNA.
NUCLEOTIDE_MIN_LENGTH = 30
NUCLEOTIDE_MIN_DISTINCT = 3
NUCLEOTIDE_MIN_SHARE = 0.10


def _fasta_letters(path: str, chains: str) -> str:
    from bindcraft.protein import read_fasta_sequences

    records = read_fasta_sequences(path)
    name = (chains or "A").split(",")[0].strip() or "A"
    sequence = records.get(name) or (next(iter(records.values())) if len(records) == 1 else "")
    return "".join(sequence).upper()


def fasta_nucleotide_problems(settings: Mapping) -> list[str]:
    """A nucleotide sequence pasted in where a protein sequence goes.

    `from_fasta` refuses every non-standard letter it is given -- `X`, `*`, a stray digit, and
    the `U` that gives RNA away -- but a DNA sequence is spelled entirely in letters that are
    also amino acids, so it is accepted and folded as a poly-Ala/Cys/Gly/Thr peptide of the same
    length. Measured on qb2 2026-09-29: a 60-base CDS prepared as a 60-residue target, hotspots
    5 and 50 resolving cleanly onto residues that mean nothing. Nothing downstream can catch it,
    because there is nothing wrong with the peptide except that the researcher never meant it.

    The test is deliberately narrow, so that a real low-complexity peptide is not refused: every
    letter one of ACGT, at least three of the four present, each of them at least a tenth of the
    sequence, and at least `NUCLEOTIDE_MIN_LENGTH` residues.
    """
    from bindcraft.settings import build_design_settings, is_fasta

    try:
        targets = build_design_settings(dict(settings)).targets
    except Exception:
        return []
    problems = []
    for target in targets:
        path = getattr(target, "path", "") or ""
        if "\n" in path or not os.path.isfile(path) or not is_fasta(path):
            continue
        letters = _fasta_letters(path, target.chains)
        present = set(letters)
        if not letters or not present <= NUCLEOTIDE_LETTERS:
            continue
        if len(letters) < NUCLEOTIDE_MIN_LENGTH or len(present) < NUCLEOTIDE_MIN_DISTINCT:
            continue
        if min(letters.count(letter) for letter in present) < NUCLEOTIDE_MIN_SHARE * len(letters):
            continue
        counts = ", ".join(f"{letter} {letters.count(letter)}" for letter in sorted(present))
        problems.append(
            f"target {getattr(target, 'name', '')!r}: {os.path.basename(path)} reads as a "
            f"nucleotide sequence rather than a protein one. All {len(letters)} of its letters "
            f"are A, C, G or T ({counts}) -- which are also the codes for alanine, cysteine, "
            f"glycine and threonine, so BindCraft 2 accepts it and designs a binder against a "
            f"{len(letters)}-residue poly-Ala/Cys/Gly/Thr peptide instead of your target. "
            f"Translate the sequence to amino acids first. If it really is a protein of only "
            f"those four residues, hand it in as a structure file, which is not read this way.")
    return problems


def fasta_crop_problems(settings: Mapping) -> list[str]:
    """Hotspots on a FASTA target that BindCraft 2 crops: the crop takes the hotspots with it.

    A FASTA target is cropped to a window sampled at random from `crop_fasta_sequence`, which
    defaults to `(10, 40)` for every FASTA, and `prepare_targets` applies the crop by slicing
    `sequence, atoms, atom_mask, flags, residue_index` together -- so a hotspot outside the
    sampled window is dropped with the sequence around it, and the only line printed is
    `target=T crop=8-35/60`, which says nothing about hotspots. An empty hotspot mask is then
    read by `interface_contacts_loss` as a campaign with no epitope. Measured on qb2 2026-09-29:
    a 60-residue FASTA target with hotspots 5,50 kept NO hotspot in 5 of 6 campaign seeds, and
    the sixth kept one of two. It is the silent whole-surface case with a random draw in front
    of it.
    """
    from bindcraft.settings import build_design_settings, is_fasta, resolve_crop_length_bounds

    try:
        targets = build_design_settings(dict(settings)).targets
        bounds = resolve_crop_length_bounds(dict(settings))
    except Exception:
        return []                    # BindCraft 2 refuses a settings it cannot resolve, its way
    if not bounds:
        return []                                             # cropping is off: nothing is lost
    problems = []
    for target in targets:
        path = getattr(target, "path", "") or ""
        spans = f"{target.hotspots or ''},{target.coldspots or ''}".strip(",")
        if "\n" in path or not spans.strip(",") or not os.path.isfile(path) or not is_fasta(path):
            continue
        residues = _fasta_residues(path, target.chains)
        length = max((len(numbers) for numbers in residues.values()), default=0)
        if not length or max(bounds) >= length:
            continue                      # the crop cannot exclude anything: the target is kept
        label = f"target {getattr(target, 'name', '')!r}"
        problems.append(
            f"{label}: this is a FASTA target with {spans!r} asked for, and BindCraft 2 crops a "
            f"FASTA target to a window of {min(bounds)}-{max(bounds)} residues sampled at "
            f"random out of its {length}. The crop slices the hotspot flags with the sequence, "
            f"so a hotspot outside the window is dropped without a word and the campaign "
            f"designs against whatever surface is left. Write \"crop_fasta_sequence\": false to "
            f"keep the whole sequence, or a crop as long as the target, or hand in a structure.")
    return problems


def input_problems(settings: Mapping) -> tuple[list[str], list[str]]:
    """Everything wrong with a campaign's inputs that this module can name, and what to note."""
    problems = threshold_problems(settings) + binder_length_problems(settings) \
        + count_problems(settings)
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
    return problems + fasta_crop_problems(settings) + fasta_nucleotide_problems(settings), notes


def load_settings(request: Mapping, *args, **kwargs) -> dict:
    """`bindcraft.settings.load_settings`, with the spellings it cannot report checked first.

    A campaign is loaded before anything in tt-bio sees it, so the check above runs on settings
    `load_settings` has already returned -- and two spellings never get that far. A binder length
    written `80` rather than `[80]` stops inside BindCraft 2's length sampler as `TypeError:
    'int' object is not iterable`, and `"60-90"` as `ValueError: invalid literal for int() with
    base 10: '-'`; neither names `binder_lengths`, and the message that does is unreachable.
    Loading through here puts the message first and is otherwise `load_settings` exactly::

        settings = bcinputs.load_settings(request)

    It checks only what it can read off the request as written. Everything else waits for
    `refuse_unusable_inputs`, which needs the defaults filled in to know what it is looking at.
    """
    from bindcraft.settings import load_settings as bindcraft_load_settings

    if problems := binder_length_problems(request):
        raise ValueError("\n".join(problems))
    return bindcraft_load_settings(request, *args, **kwargs)


def refuse_unusable_inputs(settings: Mapping) -> None:
    """Stop a campaign whose inputs would design against something the caller did not ask for."""
    problems, notes = input_problems(settings)
    for note in notes:
        print(f"[tt_bio.bcinputs] {note}", flush=True)
    if problems:
        raise ValueError("\n".join(problems))
