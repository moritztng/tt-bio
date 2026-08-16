#!/usr/bin/env python3
"""Correctness for lever 1: write_npz_many must produce the same bytes as the serial write_npz loop.

The .npz container is a zip and zipfile stamps each member with the local time, so a sha256 of the
file itself is not stable across two runs of the *same* writer and would be a meaningless check.
The digest here is over the decoded arrays: for each sequence, sha256 over id, sequence, and the
raw little-endian bytes of every array, in a fixed key order. Two arms agree only if every stored
value is bit-identical.

Host only, no device: the embeddings are synthesised at the served shapes so this runs anywhere.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.esmc import ESMCEmbedding, write_npz, write_npz_many  # noqa: E402


def digest_npz(path: Path) -> str:
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as z:
        for k in sorted(z.files):
            a = z[k]
            h.update(k.encode())
            h.update(str(a.dtype.str).encode())
            h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main() -> int:
    n_seqs, L, d = 8, 76, 960
    rng = np.random.default_rng(0)
    embs = []
    for i in range(n_seqs):
        per_res = rng.standard_normal((L, d), dtype=np.float32)
        embs.append(ESMCEmbedding(
            id=f"seq{i}", sequence="A" * L,
            per_residue=per_res, pooled=per_res.mean(axis=0), logits=None))

    work = Path(tempfile.mkdtemp(prefix="npz-parity-"))
    a_dir, b_dir = work / "serial", work / "threaded"
    a_dir.mkdir(); b_dir.mkdir()

    for e in embs:
        write_npz(e, a_dir / f"{e.id}.npz")
    write_npz_many(embs, b_dir)

    a_names = sorted(p.name for p in a_dir.glob("*.npz"))
    b_names = sorted(p.name for p in b_dir.glob("*.npz"))
    ok_names = a_names == b_names
    rows = []
    all_eq = True
    for name in a_names:
        da, db = digest_npz(a_dir / name), digest_npz(b_dir / name)
        # also a direct array-level equality, so the digest is corroborated not trusted
        with np.load(a_dir / name) as za, np.load(b_dir / name) as zb:
            arr_eq = (sorted(za.files) == sorted(zb.files)
                      and all(np.array_equal(za[k], zb[k]) for k in za.files))
        all_eq &= (da == db) and arr_eq
        rows.append(dict(file=name, serial_sha256=da, threaded_sha256=db,
                         digest_equal=da == db, arrays_equal=bool(arr_eq)))

    res = dict(n_seqs=n_seqs, residues=L, d_model=d,
               filenames_equal=ok_names, all_bit_identical=bool(all_eq and ok_names),
               per_file=rows)
    print(json.dumps(res, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    return 0 if res["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
