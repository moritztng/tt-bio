"""Search every alignment this row folds with, once, into msa/, so both arms read the same bytes.

  python pin_msas.py [msa_dir]

Writes the unpaired per-chain a3m every model reads (``{seq_hash}.a3m``, the worker's own
search), each heterodimer's paired a3ms in the new per-complex layout, and the same paired
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
from tt_bio.main import _generate_esmfold2_a3m, _generate_paired_a3m

URL = "https://api.colabfold.com"
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "seqs.json").read_text())
HETERO = [("im9", "e9"), ("barnase", "barstar")]
HOMO = "hivpr"


def main():
    msa = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "msa")
    msa.mkdir(parents=True, exist_ok=True)
    _generate_esmfold2_a3m({seq_hash(s): s for s in S.values()}, "pin", msa, None, False,
                           URL, "greedy", None, None, None)
    old = msa / "paired"
    for pair in HETERO:
        seqs = {seq_hash(S[k]): S[k] for k in pair}
        got = _generate_paired_a3m(seqs, "pin", msa, URL, "greedy", None, None, None)
        for h, text in got.items():
            publish_text(old / f"{h}.a3m", text)
    h = seq_hash(S[HOMO])
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
