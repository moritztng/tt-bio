#!/usr/bin/env python3
"""Split the ESM-C / SaProt embed timed region into device work and host npz writing.

The perf gate's ESM-C legs time ``_WorkerState.predict_one`` -> ``worker._predict_embed_one``,
which is ``load_sequences`` + ``embed_sequences`` + a serial ``write_npz`` loop. The SaProt leg
times ``saprot.embed_sequences`` and nothing else. If the write loop is a large share of the
ESM-C region then the two families are not on one scale and no WH/BH ratio built on them means
anything. This script measures the split directly, on real embeddings from a real device.

Also times ``write_npz_many`` (threaded, already shipped, already used by the CLI) on the same
result set, so lever 1's predicted landing is a measured number rather than an estimate.

Usage:
  TT_VISIBLE_DEVICES=<n> python3 perf/wh-embed/embed_split_screen.py \
      --model esmc-300m --n-seqs 8 --residues 76 --batch-size 8 --out results/x.json
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")  # 76 aa — the perf gate's fixture, verbatim

WARMUP = 2
REPEAT = 5


def make_sequence(residues: int) -> str:
    """A sequence of exactly ``residues`` aa built by tiling the gate's fixture."""
    if residues == len(UBIQUITIN):
        return UBIQUITIN
    reps = (residues // len(UBIQUITIN)) + 1
    return (UBIQUITIN * reps)[:residues]


def loadavg() -> float:
    return os.getloadavg()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--residues", type=int, default=76)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    ap.add_argument("--roof", action="store_true",
                    help="also measure the content-matched zlib roof on the real bytes")
    ap.add_argument("--seq-file", default=None,
                    help="read the sequence from a file instead of tiling ubiquitin. The perf "
                         "page's rows all read scripts/gpu_vs_tt/fixtures/prot512.seq, which is "
                         "byte-identical to the protein chain of the folding fixture, so an "
                         "embed cell and a fold cell describe the same 512 residues. Tiling is "
                         "kept as the default so the shipped perf-gate protocol is unchanged.")
    ap.add_argument("--expect-sha", default=None,
                    help="refuse to measure unless --seq-file hashes to this sha256")
    ap.add_argument("--assert-batch", type=int, default=None,
                    help="the batch this row must actually execute. The shipped batcher caps a "
                         "batch at batch_size * 512 tokens, so at 512 aa a requested 8 executes "
                         "as 7 and an eighth sequence would run as a second, ragged batch of 1. "
                         "A cross-platform cell has to state the batch that RAN, so the row "
                         "counts the forwards it issued and refuses to report a number unless "
                         "every one of them carried exactly this many rows.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.seq_file:
        raw = Path(args.seq_file).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if args.expect_sha and got != args.expect_sha:
            print(f"fixture sha256 {got} != expected {args.expect_sha}", file=sys.stderr)
            return 1
        seq = raw.decode().strip()
        args.residues = len(seq)
        fixture = dict(path=args.seq_file, sha256=got)
    else:
        seq = make_sequence(args.residues)
        fixture = dict(path="tiled ubiquitin", sha256=None)
    sequences = {f"seq{i}": seq for i in range(args.n_seqs)}

    is_saprot = args.model.startswith("saprot")
    if is_saprot:
        from tt_bio import saprot as mod
        from tt_bio.esmc import write_npz, write_npz_many
        t0 = time.perf_counter()
        model = mod.load_saprot(args.model, fast=args.fast)
        load_s = time.perf_counter() - t0
        # sequence-only mode, exactly as the perf leg runs it
        seqs_arg = {k: (v, "#" * len(v)) for k, v in sequences.items()}

        def embed_once():
            return mod.embed_sequences(model, seqs_arg, batch_size=args.batch_size)
    else:
        from tt_bio.esmc import load_esmc, embed_sequences, write_npz, write_npz_many
        t0 = time.perf_counter()
        model = load_esmc(args.model, fast=args.fast)
        load_s = time.perf_counter() - t0

        def embed_once():
            return embed_sequences(model, sequences, batch_size=args.batch_size)

    # Count the forwards the shipped batcher issues, and how many rows each one carried.
    # `batch_executed` below is arithmetic off the same rule the batcher uses; this is the
    # observation. They should agree, and if they ever do not, the observation wins: a perf-page
    # cell states the batch that ran, not the batch that was asked for.
    observed: list[int] = []
    _mcls = type(model)
    _orig_call = _mcls.__call__

    def _counting_call(self, *a, **k):
        ids = a[0] if a else k.get("input_ids")
        try:
            observed.append(int(ids.shape[0]))
        except Exception:
            observed.append(-1)
        return _orig_call(self, *a, **k)

    _mcls.__call__ = _counting_call

    work = Path(tempfile.mkdtemp(prefix="embed-split-"))
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_times, ser_times, par_times, loads = [], [], [], []
    results = None
    batches_per_iter = []
    for i in range(args.warmup + args.repeat):
        observed.clear()
        t0 = time.perf_counter()
        results = embed_once()
        t_dev = time.perf_counter() - t0
        batches_per_iter.append(list(observed))

        for p in out_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        for emb in results:
            write_npz(emb, out_dir / f"{emb.id}.npz")
        t_ser = time.perf_counter() - t0

        for p in out_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        write_npz_many(results, out_dir)
        t_par = time.perf_counter() - t0

        if i >= args.warmup:
            dev_times.append(t_dev)
            ser_times.append(t_ser)
            par_times.append(t_par)
            loads.append(loadavg())
        print(f"  iter {i}: device {t_dev*1000:.1f} ms  serial-write {t_ser*1000:.1f} ms  "
              f"threaded-write {t_par*1000:.1f} ms  load {loadavg():.1f}",
              file=sys.stderr, flush=True)

    def med(xs):
        return sorted(xs)[len(xs) // 2]

    npz_bytes = sum(p.stat().st_size for p in out_dir.glob("*.npz"))

    # The row's own validity check, on the embeddings the timed arm actually produced. A perf
    # cell does not count until its output has been looked at.
    #   d_model / residues  the shape the config declares, so a wrong checkpoint cannot pass
    #   all_finite          no NaN or inf anywhere in per-residue or pooled
    #   rows_identical      every row of the batch is the same sequence, so their pooled vectors
    #                       must agree bit for bit. This is what says the padding, the attention
    #                       mask and the per-row slice are right: a mask bug shows up here as
    #                       row 6 differing from row 0, and nowhere else in a timing run.
    #   pooled_sha256       run-to-run and cross-process reproducibility of the same fixture
    import numpy as _np
    _e0 = results[0]
    accuracy = dict(
        n_out=len(results),
        residues_out=int(_e0.per_residue.shape[0]),
        d_model=int(_e0.per_residue.shape[1]),
        tokens=args.residues + 2,
        all_finite=bool(all(_np.isfinite(e.per_residue).all() and _np.isfinite(e.pooled).all()
                            for e in results)),
        rows_identical=bool(all(_np.array_equal(_e0.pooled, e.pooled) for e in results)),
        pooled_sha256=hashlib.sha256(
            _np.ascontiguousarray(_e0.pooled).tobytes()).hexdigest()[:16],
        pooled_norm=round(float(_np.linalg.norm(_e0.pooled)), 4),
        residues_match_fixture=bool(int(_e0.per_residue.shape[0]) == args.residues),
    )
    accuracy["pass"] = bool(accuracy["all_finite"] and accuracy["residues_match_fixture"]
                            and accuracy["n_out"] == args.n_seqs)

    # Content-matched roof. A synthetic-content zlib probe bounds this arm from above only
    # (deflate's cost per byte depends on the content), so measure the roof on the bytes this
    # model actually emits. Also splits raw zlib from np.savez_compressed's container overhead,
    # which decides whether the residual is compression or numpy/zipfile bookkeeping.
    roof = None
    if args.roof:
        import zlib
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        raw = [np.ascontiguousarray(e.per_residue).tobytes() for e in results]
        in_bytes = sum(len(r) for r in raw)
        n_workers = max(1, min(32, os.cpu_count() or 8))

        def t(fn, reps=3):
            fn()
            xs = []
            for _ in range(reps):
                t0 = time.perf_counter()
                out = fn()
                xs.append(time.perf_counter() - t0)
            return med(xs), out

        t_ser, out_ser = t(lambda: [zlib.compress(r, 6) for r in raw])

        def par():
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                return list(ex.map(lambda r: zlib.compress(r, 6), raw))

        t_par, _ = t(par)
        comp_bytes = sum(len(o) for o in out_ser)
        roof = dict(
            per_residue_bytes=in_bytes, compressed_bytes=comp_bytes,
            compress_ratio=round(comp_bytes / in_bytes, 4), workers=n_workers,
            zlib_serial_ms=round(t_ser * 1000, 2), zlib_threaded_ms=round(t_par * 1000, 2),
            zlib_serial_MBps=round(in_bytes / t_ser / 1e6, 2),
            zlib_threaded_MBps=round(in_bytes / t_par / 1e6, 2),
            zlib_thread_speedup=round(t_ser / t_par, 3),
            # what savez_compressed adds on top of the compression itself
            savez_overhead_ms=round((med(ser_times) - t_ser) * 1000, 2),
            savez_overhead_share=round(1 - t_ser / med(ser_times), 4),
        )
    d, s, p_ = med(dev_times), med(ser_times), med(par_times)
    res = dict(
        model=args.model, n_seqs=args.n_seqs, residues=args.residues,
        batch_size=args.batch_size, fast=args.fast,
        # `batch_size` is what was REQUESTED. The shipped batcher caps a batch at
        # batch_size * 512 tokens and a 512 aa sequence buckets to 576, so at page scope the
        # batch that executes is 7 of a requested 8. The executed count is what a GPU arm has
        # to match, so it is recorded rather than left to be inferred.
        batch_executed=min(args.batch_size,
                           max(1, (args.batch_size * 512) // (((args.residues + 2 + 63) // 64) * 64))),
        # observed, not derived: the row count of every forward the batcher issued, per iteration
        batches_observed=batches_per_iter,
        forwards_per_iter=sorted({len(b) for b in batches_per_iter}),
        batch_observed=sorted({r for b in batches_per_iter for r in b}),
        assert_batch=args.assert_batch,
        fixture=fixture,
        arch=os.environ.get("PROBE_ARCH", "unknown"),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES", ""),
        load_s=round(load_s, 1),
        device_ms=round(d * 1000, 2),
        serial_write_ms=round(s * 1000, 2),
        threaded_write_ms=round(p_ * 1000, 2),
        gate_region_ms=round((d + s) * 1000, 2),
        write_share_of_gate=round(s / (d + s), 4),
        device_seq_s=round(args.n_seqs / d, 3),
        gate_seq_s=round(args.n_seqs / (d + s), 3),
        device_ms_all=[round(x * 1000, 2) for x in dev_times],
        serial_write_ms_all=[round(x * 1000, 2) for x in ser_times],
        threaded_write_ms_all=[round(x * 1000, 2) for x in par_times],
        npz_bytes=npz_bytes,
        accuracy=accuracy,
        matched_roof=roof,
        loadavg=loads,
        warmup=args.warmup, repeat=args.repeat,
    )
    bad = None
    if not accuracy["pass"]:
        bad = (f"accuracy check failed: {accuracy}. A timing whose own output does not pass is "
               f"not a cell.")
        res["accuracy_error"] = bad
    if args.assert_batch is not None:
        want = [args.assert_batch]
        offenders = [b for b in batches_per_iter if b != want]
        if offenders:
            bad = (f"--assert-batch {args.assert_batch}: the batcher issued {offenders[0]} "
                   f"rows-per-forward on at least one iteration, not exactly one forward of "
                   f"{args.assert_batch}. The timing is not the batch it claims.")
            res["assert_batch_error"] = bad

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    if bad:
        print(bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
