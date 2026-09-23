"""Search every alignment this row folds with, once, into msa/, so both arms read the same bytes.
Only what is missing is searched, so a re-run never rewrites an alignment an arm has read.

  python pin_msas.py [msa_dir]

Writes the unpaired per-chain a3m every model reads (``{seq_hash}.a3m``, the worker's own
search), each heterodimer's paired a3ms in the new per-complex layout (plus the Boltz-2 CSVs built
from those same bytes, paired rows over the pinned unpaired ones), and the same paired
bytes in main's old ``paired/{seq_hash}.a3m`` layout so the BEFORE OpenDDE arm reads them from
cache too. The homodimer gets main's old one-sequence paired search, which is what main folds
it with. Prints the md5 of every file.
"""
import hashlib
import json
import sys
from pathlib import Path

from tt_bio.cache import paired_msa_dir, publish_text, seq_hash
from tt_bio.data.msa import run_mmseqs2
from tt_bio.main import _generate_esmfold2_a3m, _generate_paired_a3m, write_boltz_csvs

URL = "https://api.colabfold.com"
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "seqs.json").read_text())
HETERO = [("im9", "e9"), ("barnase", "barstar"), ("atfarel2", "farel"), ("isea", "cwlo")]
HOMO = "hivpr"


def main():
    msa = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "msa")
    msa.mkdir(parents=True, exist_ok=True)
    todo = {seq_hash(s): s for s in S.values() if not (msa / f"{seq_hash(s)}.a3m").exists()}
    if todo:
        _generate_esmfold2_a3m(todo, "pin", msa, None, False, URL, "greedy", None, None, None)
    old = msa / "paired"
    for pair in HETERO:
        if paired_msa_dir(msa, [S[k] for k in pair]).exists():
            continue
        seqs = {seq_hash(S[k]): S[k] for k in pair}
        got = _generate_paired_a3m(seqs, "pin", msa, URL, "greedy", None, None, None)
        for h, text in got.items():
            publish_text(old / f"{h}.a3m", text)
    for pair in HETERO:
        pdir = paired_msa_dir(msa, [S[k] for k in pair])
        hs = [seq_hash(S[k]) for k in pair]
        if not all((pdir / f"{h}.csv").exists() for h in hs):
            write_boltz_csvs(pdir, {h: (pdir / f"{h}.a3m").read_text() for h in hs},
                             {h: (msa / f"{h}.a3m").read_text() for h in hs})
    h = seq_hash(S[HOMO])
    if not (old / f"{h}.a3m").exists():
        res = run_mmseqs2([S[HOMO]], msa / "homo_paired_tmp", use_env=True, use_pairing=True,
                          host_url=URL, pairing_strategy="greedy")
        publish_text(old / f"{h}.a3m", res[0])
    for p in sorted(msa.rglob("*.a3m")):
        rows = p.read_text().count("\n>") + 1
        print(hashlib.md5(p.read_bytes()).hexdigest()[:12], rows, p.relative_to(msa))
    for pair in HETERO:
        print(" ".join(pair), paired_msa_dir(msa, [S[k] for k in pair]).name)


if __name__ == "__main__":
    main()
