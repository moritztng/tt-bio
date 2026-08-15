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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = make_sequence(args.residues)
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

    work = Path(tempfile.mkdtemp(prefix="embed-split-"))
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_times, ser_times, par_times, loads = [], [], [], []
    results = None
    for i in range(args.warmup + args.repeat):
        t0 = time.perf_counter()
        results = embed_once()
        t_dev = time.perf_counter() - t0

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
    d, s, p_ = med(dev_times), med(ser_times), med(par_times)
    res = dict(
        model=args.model, n_seqs=args.n_seqs, residues=args.residues,
        batch_size=args.batch_size, fast=args.fast,
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
        loadavg=loads,
        warmup=args.warmup, repeat=args.repeat,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
