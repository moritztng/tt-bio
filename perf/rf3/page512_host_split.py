"""How much of RF3's 512 aa page fold is AtomWorks host featurisation, on the box that
produced the p150a cell.

The page's cells are whole warm folds, and on both GPU boxes about half of an RF3 fold at
512 aa turned out to be host featurisation rather than device work: 12.7 s of 22.0 s on the
H200, 15.0 s of 30.9 s on the A100. The same vendored AtomWorks pipeline runs inside the
81.051 s p150a number, so the fold ratio on the page carries a large platform-independent
constant on both sides. This measures the Tenstorrent side of it.

No accelerator and no run lock: `featurize()` is pure host work. It does need a quiet box,
so `/proc/loadavg` is recorded per call and a run taken under contention is an upper bound,
not the number.

`_predict_rf3_one` calls `featurize(...)` with no `pipeline=`, so every fold rebuilds the
pipeline. This harness does the same, or it would under-report by exactly the amount the
real fold pays.

    PYTHONPATH="$PWD:/home/ttuser/rf3_perf_deps" python3 perf/rf3/page512_host_split.py \
        --repeat 3 --out perf/rf3/page512/tt_host_split_qb2.json
"""

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
YAML = ROOT / "perf/size512/fixtures/cdk2x2_512.yaml"
A3M = ROOT / "perf/size512/fixtures/cdk2x2_512.a3m"


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def seq_from_yaml(p: pathlib.Path) -> str:
    """The one protein sequence in the page fixture, read without a yaml dependency."""
    seq, in_seq = [], False
    for line in p.read_text().splitlines():
        s = line.strip()
        if s.startswith("sequence:"):
            rest = s.split(":", 1)[1].strip()
            if rest:
                return rest
            in_seq = True
            continue
        if in_seq:
            if s and not s.startswith("-") and ":" in s and not s[0].isalpha():
                break
            if s and s[0].isalpha() and ":" not in s:
                seq.append(s)
            elif seq:
                break
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3, help="timed calls after one discarded warm-up")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-recycles", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize

    seq = seq_from_yaml(YAML)
    assert len(seq) == 512, "fixture sequence is %d aa, expected 512" % len(seq)
    a3m_rows = sum(1 for line in A3M.read_text().splitlines() if line.startswith(">"))
    assert a3m_rows == 35, "expected 35 MSA rows, read %d" % a3m_rows

    # Exactly the component _predict_rf3_one builds for this fixture: one protein chain with
    # an absolute msa_path, because upstream resolves it against the process cwd.
    component = {"seq": seq, "chain_id": "A", "msa_path": str(A3M.resolve())}

    rep = {"fixture_yaml_sha256_16": sha16(YAML), "fixture_a3m_sha256_16": sha16(A3M),
           "seq_sha256_16": hashlib.sha256(seq.encode()).hexdigest()[:16],
           "seq_len": len(seq), "a3m_rows": a3m_rows,
           "n_recycles": args.n_recycles, "seed": args.seed,
           "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
           "calls": [], "warmup_s": None}

    def one_call() -> tuple[float, dict]:
        with tempfile.TemporaryDirectory() as td:
            spec_path = pathlib.Path(td) / "cdk2x2_512.json"
            spec_path.write_text(json.dumps([{"name": "cdk2x2_512", "components": [component]}]))
            load = open("/proc/loadavg").read().split()[:3]
            t0 = time.perf_counter()
            out = featurize(spec_path, n_recycles=args.n_recycles,
                            diffusion_batch_size=1, seed=args.seed)[0]
            dt = time.perf_counter() - t0
        f = out["feats"]
        n_tokens = int(f["asym_id"].shape[-1])
        assert n_tokens == 512, "featurised %d tokens, expected 512" % n_tokens
        # The MSA has to have been read, or this is timing a single-sequence featurisation.
        # msa_stack is (n_recycles, depth, n_tokens, channels): the featuriser materialises
        # one MSA stack per recycle, so depth is axis 1 and the recycle count is axis 0.
        stack = f["msa_stack"]
        depth = int(stack.shape[1])
        assert depth == 35, "msa depth %d, expected the fixture's 35 rows" % depth
        assert int(stack.shape[0]) == args.n_recycles, \
            "msa_stack carries %d recycles, asked for %d" % (stack.shape[0], args.n_recycles)
        n_atoms = int(f["ref_pos"].shape[0])
        assert n_atoms == 4116, "featurised %d atoms, expected 4116" % n_atoms
        return dt, {"s": round(dt, 4), "n_tokens": n_tokens, "n_atoms": n_atoms,
                    "msa_depth": depth, "msa_stack_shape": [int(x) for x in stack.shape],
                    "loadavg": [float(x) for x in load]}

    warm, _ = one_call()                     # discarded: first call pays import-time caches
    rep["warmup_s"] = round(warm, 4)
    print("warm-up (discarded) %.4f s" % warm, flush=True)

    for i in range(args.repeat):
        dt, meta = one_call()
        rep["calls"].append(meta)
        print("call %d  %.4f s  loadavg %.2f  msa_depth %d"
              % (i, dt, meta["loadavg"][0], meta["msa_depth"]), flush=True)

    vals = [c["s"] for c in rep["calls"]]
    rep["median_s"] = round(statistics.median(vals), 4)
    rep["min_s"], rep["max_s"] = round(min(vals), 4), round(max(vals), 4)
    rep["spread_s"] = round(max(vals) - min(vals), 4)
    rep["spread_pct"] = round(100 * (max(vals) - min(vals)) / rep["median_s"], 2)
    rep["max_loadavg_1min"] = max(c["loadavg"][0] for c in rep["calls"])

    out_p = pathlib.Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rep, indent=1) + "\n")
    print("\nmedian %.4f s  spread %.4f s (%.2f %%)  max loadavg %.2f  ->  %s"
          % (rep["median_s"], rep["spread_s"], rep["spread_pct"],
             rep["max_loadavg_1min"], out_p))


if __name__ == "__main__":
    main()
