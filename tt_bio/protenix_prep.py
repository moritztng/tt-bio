"""MSAs searched ahead of a fold, written where upstream Protenix / Mol-1 `msa` writes them.

`tt-bio prep jobs.json` searches every protein chain a job left without an alignment and writes
``<stem>-update-msa.json`` beside the input with ``pairedMsaPath`` / ``unpairedMsaPath`` filled in,
the files under ``<out_dir>/<job>/msa/<k>/{pairing,non_pairing}.a3m``. A FASTA writes one directory
per record plus ``msa_index.json``. The search is the one `tt-bio predict` runs (worker.search_msas
for the unpaired MSA, main._generate_paired_a3m for the species-paired one), so a prepared JSON folds
with the alignments predict would have searched itself.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tt_bio.cache import cached, publish_text, seq_hash

MSA_INDEX_NAME = "msa_index.json"


def _search(seqs: list[str], msa_dir: Path, name: str, cfg: dict) -> dict[str, tuple[str, str | None]]:
    """``sequence -> (unpaired a3m text, paired a3m text or None)`` for the unique ``seqs`` of one complex."""
    from tt_bio.main import _generate_paired_a3m
    from tt_bio.worker import search_msas

    cfg = {**cfg, "msa_dir": str(msa_dir)}
    msa_dir.mkdir(parents=True, exist_ok=True)
    search_msas(Path(name), [(str(i), s, None, "protein", []) for i, s in enumerate(seqs)], cfg)
    paired = None
    if len(seqs) > 1:
        paired = _generate_paired_a3m(
            {seq_hash(s): s for s in seqs}, name, msa_dir, cfg.get("msa_server_url"),
            cfg.get("msa_pairing_strategy"), cfg.get("msa_server_username"), cfg.get("msa_server_password"),
            cfg.get("api_key_value"), msa_db_path=cfg.get("msa_db_path"), use_envdb=cfg.get("use_envdb", False),
            search=bool(cfg.get("use_msa_server") or cfg.get("msa_db_path")))
    out = {}
    for s in seqs:
        a3m = msa_dir / f"{seq_hash(s)}.a3m"
        if not cached(a3m):
            raise RuntimeError(f"MSA search returned nothing for {name} ({s[:20]}...)")
        out[s] = (a3m.read_text(), (paired or {}).get(seq_hash(s)))
    return out


def _write(target: Path, unpaired: str, paired: str | None) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    publish_text(target / "non_pairing.a3m", unpaired)
    paths = {"unpairedMsaPath": str((target / "non_pairing.a3m").resolve())}
    if paired:
        publish_text(target / "pairing.a3m", paired)
        paths["pairedMsaPath"] = str((target / "pairing.a3m").resolve())
    return paths


def prep_json(path: Path, out_dir: Path, cfg: dict) -> Path | None:
    """Search the jobs in ``path`` that need it; returns the updated JSON, or None when no job needed a search."""
    jobs = json.loads(path.read_text())
    updated = False
    for i, job in enumerate(jobs):
        chains = [e["proteinChain"] for e in job.get("sequences", []) if "proteinChain" in e]
        if not chains or all("unpairedMsaPath" in c or "pairedMsaPath" in c or c.get("msa") for c in chains):
            continue
        name = job.get("name") or f"task_{i}"
        seqs = sorted({c["sequence"] for c in chains})
        found = _search(seqs, out_dir / name / "msa" / "cache", name, cfg)
        dirs = {s: _write(out_dir / name / "msa" / str(k), *found[s]) for k, s in enumerate(seqs)}
        for c in chains:
            c.update(dirs[c["sequence"]])
        updated = True
    if not updated:
        return None
    dest = path.resolve().parent / f"{path.stem}-update-msa.json"
    dest.write_text(json.dumps(jobs, indent=4))
    return dest


def msa_dir_names(record_ids: list[str]) -> list[str]:
    """A directory name for each FASTA record, as upstream names them: the id with characters other than letters,
    digits, ``.``, ``_`` and ``-`` made ``_``, a repeat suffixed ``_2``, ``_3``..."""
    names, taken = [], {MSA_INDEX_NAME}
    for i, rid in enumerate(record_ids):
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", rid).strip("._") or f"seq_{i}"
        name, n = base, 2
        while name in taken:
            name, n = f"{base}_{n}", n + 1
        taken.add(name)
        names.append(name)
    return names


def prep_fasta(path: Path, out_dir: Path, cfg: dict) -> Path:
    """One directory per FASTA record; the records are searched as one complex, so they pair when there are
    several. Returns ``<out_dir>/msa_index.json``."""
    from Bio import SeqIO

    records = list(SeqIO.parse(str(path), "fasta"))
    if not records:
        raise ValueError(f"no sequences in {path}")
    seqs = [str(r.seq) for r in records]
    found = _search(sorted(set(seqs)), out_dir / ".msa_cache", path.stem, cfg)
    index = []
    for r, name, s in zip(records, msa_dir_names([r.id for r in records]), seqs):
        _write(out_dir / name, *found[s])
        index.append({"id": r.id, "sequence": s, "dir": str((out_dir / name).resolve())})
    dest = out_dir / MSA_INDEX_NAME
    dest.write_text(json.dumps(index, indent=2))
    return dest
