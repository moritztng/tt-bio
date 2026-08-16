#!/usr/bin/env python3
"""Integrated A/B for lever 1, through the worker path the perf gate and JapanFold both use.

Arm A is the pre-change behaviour: ``_predict_embed_one``'s serial ``write_npz`` loop.
Arm B is the shipped behaviour after the change: ``write_npz_many``.
Both arms call ``_WorkerState.predict_one`` on one resident model in one process, so no model
load, no device open and no cross-process drift sits between them.

Order is A A B A B A B ... — the leading A A is the A/A control, and its spread is the noise
floor this box can resolve. Arms alternate after that so any drift over the run hits both.

Arm A is reconstructed by monkeypatching ``tt_bio.esmc.write_npz_many`` to a serial loop over
``write_npz``, which is byte-for-byte the code that was there before (verified separately by
perf/wh-embed/embed_write_parity.py). Nothing else differs between arms.

Usage:
  TT_VISIBLE_DEVICES=<n> python3 perf/wh-embed/embed_worker_ab.py --model esmc-300m --out x.json
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m")
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--residues", type=int, default=76)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import esmc as esmc_mod
    from tt_bio.worker import _WorkerState

    shipped_many = esmc_mod.write_npz_many

    def serial_many(embeddings, out_dir, max_workers=None):
        for e in embeddings:
            esmc_mod.write_npz(e, Path(out_dir) / f"{e.id}.npz")

    reps = (args.residues // len(UBIQUITIN)) + 1
    seq = (UBIQUITIN * reps)[:args.residues]

    work = Path(tempfile.mkdtemp(prefix="embed-ab-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    fasta = work / "embed.fasta"
    fasta.write_text("".join(f">seq{i}|protein\n{seq}\n" for i in range(args.n_seqs)))

    # Reuse the perf gate's own cfg builder so this A/B and the gate configure the
    # worker identically; only the fixture size is ours.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import perf_regression as pr

    (work / "msa").mkdir(exist_ok=True)
    cfg = pr._build_cfg(args.model, dict(pr.SPECS[args.model], batch_size=args.batch_size),
                        struct_dir, work / "msa")
    cfg["job_id"] = "ab"

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ab", cfg)
    state.pfn = lambda *a, **k: None

    def one(arm):
        esmc_mod.write_npz_many = serial_many if arm == "A" else shipped_many
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(fasta, dict(cfg))
        return time.perf_counter() - t0, metrics

    for _ in range(args.warmup):
        one("B")
        one("A")

    order = ["A", "A"] + [a for _ in range(args.repeat) for a in ("B", "A")]
    runs = []
    for i, arm in enumerate(order):
        wall, metrics = one(arm)
        runs.append(dict(i=i, arm=arm, wall_ms=round(wall * 1000, 2),
                         device_s=metrics.get("device_s"), write_s=metrics.get("write_s"),
                         load1=round(os.getloadavg()[0], 2)))
        print(f"  {i:2d} {arm}  {wall*1000:8.2f} ms  device {metrics.get('device_s')}  "
              f"write {metrics.get('write_s')}", file=sys.stderr, flush=True)

    aa = [r["wall_ms"] for r in runs[:2]]
    a = [r["wall_ms"] for r in runs[2:] if r["arm"] == "A"]
    b = [r["wall_ms"] for r in runs[2:] if r["arm"] == "B"]
    med = lambda x: statistics.median(x)
    res = dict(
        model=args.model, n_seqs=args.n_seqs, residues=args.residues,
        batch_size=args.batch_size,
        arch=os.environ.get("PROBE_ARCH", "unknown"),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES", ""),
        aa_control_ms=aa,
        aa_ratio=round(aa[0] / aa[1], 4) if aa[1] else None,
        aa_noise_pct=round(abs(aa[0] - aa[1]) / min(aa) * 100, 2) if aa[1] else None,
        arm_A_serial_write_ms=[round(x, 2) for x in a],
        arm_B_threaded_write_ms=[round(x, 2) for x in b],
        median_A_ms=round(med(a), 2), median_B_ms=round(med(b), 2),
        speedup_B_over_A=round(med(a) / med(b), 4),
        seq_s_A=round(args.n_seqs / (med(a) / 1000), 3),
        seq_s_B=round(args.n_seqs / (med(b) / 1000), 3),
        device_s_median=round(med([r["device_s"] for r in runs if r["device_s"]]), 4),
        runs=runs,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    state.reset()
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
