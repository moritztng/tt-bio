"""Card-free A/B of the batch CIF write: serial `write_design` per row (A) against `write_designs` (B).

    python perf/rfd4/gap224/wab.py <tt-bio root> <spec.yaml> <batch> <rounds>

Stand-in engine output (residue types legal per token, the decoder's atom37 table, input coordinates plus noise),
the same call perf/rfd4/fair/run.py times as `write_s`. Every B file is byte-compared with its A twin, minus
`_entry.time`-style timestamps. Prints one JSON line.
"""
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import torch  # noqa: E402

from tt_bio import rfd4  # noqa: E402
from tt_bio.rfd4 import ae_fold as AF, io as IO, network as N  # noqa: E402
from tt_bio.rfd4.engine import DesignOutput  # noqa: E402
from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_ae_masks  # noqa: E402


def stand_in(f, B, seed):
    tab = AF.tables()
    b = N._broadcast_batch(dict(f), B)
    _, emit = compute_ae_masks(b)
    g = torch.Generator().manual_seed(seed)
    cs = b[AF.C_SEQ].long()
    rt = torch.where((cs >= 1) & (cs <= 20), cs, torch.randint(1, 21, cs.shape, generator=g))
    am = tab["atom_mask"][rt] & emit[..., None]
    am[..., 1] |= b["padding_mask"].bool()
    xyz = (b["coords"].float() / 10 + 0.05 * torch.randn(*b["coords"].shape, generator=g)) * am[..., None]
    rt = torch.where(emit, rt, torch.full_like(rt, -1))
    n = f["padding_mask"].shape[1]
    return DesignOutput(coords_nm=xyz, residue_type=rt, atom_mask=am,
                        insertion_logits=torch.randn(B, n, n, generator=g))


def body(p):
    return [l for l in Path(p).read_text().splitlines() if "time" not in l.split(" ")[0]]


def main():
    spec, B, rounds = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    out = Path(tempfile.mkdtemp(prefix="wab-"))
    ex = rfd4.prepare_examples(spec, out / "inputs")[0]
    f, aa = rfd4.featurize(ex.cif, seed=1000)
    o = stand_in(f, B, 0)
    t0 = time.perf_counter()
    IO.start_writer(B)
    start = time.perf_counter() - t0
    time.sleep(float(sys.argv[5]) if len(sys.argv) > 5 else 0)  # the design the pool starts behind
    t0 = time.perf_counter()
    IO.write_designs(o, f, aa, [out / "first" / f"d{k}.cif" for k in range(B)])
    first = time.perf_counter() - t0
    IO.write_designs(o, f, aa, [out / "warm" / f"d{k}.cif" for k in range(B)])
    a, b, same = [], [], 0
    for r in range(rounds):
        for leg in ("A", "B"):
            d = out / f"{leg}{r}"
            t0 = time.perf_counter()
            if leg == "A":
                for k in range(B):
                    rfd4.write_design(o, f, aa, d / f"d{k}.cif", row=k)
            else:
                rfd4.write_designs(o, f, aa, [d / f"d{k}.cif" for k in range(B)])
            (a if leg == "A" else b).append(time.perf_counter() - t0)
        same += sum(body(out / f"A{r}" / f"d{k}.cif") == body(out / f"B{r}" / f"d{k}.cif") for k in range(B))
    print(json.dumps({"tokens": int(f["padding_mask"].shape[1]), "batch": B, "rounds": rounds,
                      "pool_start_s": round(start, 3), "first_batch_s": round(first, 3), "workers": IO.writer_count(B),
                      "serial_s": round(statistics.median(a), 3), "pool_s": round(statistics.median(b), 3),
                      "serial_all": [round(x, 3) for x in a], "pool_all": [round(x, 3) for x in b],
                      "speedup": round(statistics.median(a) / statistics.median(b), 2),
                      "byte_identical": f"{same}/{rounds * B}"}), flush=True)


if __name__ == "__main__":
    main()
