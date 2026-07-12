"""Warm stage-timing profile of the ttnn ESMFold2 fold, to locate the perf bottleneck.

Patches tt_bio.esmfold2.report_progress to timestamp stage transitions, folds one
protein twice (first fold pays kernel compilation; the second is the steady-state
number reported), and prints wall-clock seconds per stage: LM, trunk, diffusion,
confidence. Read-only w.r.t. the model — no code path changes.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import ttnn

SEQS = {
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "prot": "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", choices=SEQS, default="ubiquitin")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    import tt_bio.esmfold2 as E
    from tt_bio.esmfold2_runtime import load_ttnn_esmfold2, fold_complex

    marks = []
    _orig = E.report_progress

    def _rp(stage, step=0, total=0):
        marks.append((stage, step, total, time.perf_counter()))
        return _orig(stage, step, total)

    E.report_progress = _rp

    model = load_ttnn_esmfold2()
    seq = SEQS[args.protein]
    device = model.structure_head.m.tt_device
    exact = {"structure": [], "confidence": []}

    def timed(owner, name, label):
        original = getattr(owner, name)

        def wrapper(*call_args, **call_kwargs):
            ttnn.synchronize_device(device)
            started = time.perf_counter()
            result = original(*call_args, **call_kwargs)
            ttnn.synchronize_device(device)
            exact[label].append(time.perf_counter() - started)
            return result

        setattr(owner, name, wrapper)

    timed(model.structure_head, "sample", "structure")
    timed(model.confidence_head, "forward", "confidence")

    def one():
        marks.clear()
        for samples in exact.values():
            samples.clear()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fold_complex(
            model,
            [("A", seq)],
            num_loops=args.loops,
            num_sampling_steps=args.steps,
            num_diffusion_samples=1,
            seed=0,
        )
        ttnn.synchronize_device(device)
        return t0, time.perf_counter(), {k: list(v) for k, v in exact.items()}

    print(f"[warmup fold {args.protein} L={len(seq)}] ...", flush=True)
    t0, t1, _ = one()
    print(f"  warmup total {t1 - t0:.1f}s (includes kernel compilation)", flush=True)

    print(f"[timed fold] ...", flush=True)
    t0, t1, stage_exact = one()
    # stage boundaries from first-occurrence timestamps
    first = {}
    for stage, step, total, ts in marks:
        first.setdefault(stage, ts)
    order = sorted(first.items(), key=lambda kv: kv[1])
    print(
        f"  total {t1 - t0:.1f}s  "
        f"(L={len(seq)}, loops={args.loops}, steps={args.steps})",
        flush=True,
    )
    print("  stage first-seen offsets (s from fold start):", flush=True)
    for stage, ts in order:
        print(f"    {stage:12s} +{ts - t0:6.1f}s", flush=True)
    # rough per-stage durations from consecutive stage starts + fold end
    bounds = [ts for _, ts in order] + [t1]
    print("  approx stage durations:", flush=True)
    for i, (stage, ts) in enumerate(order):
        print(f"    {stage:12s} {bounds[i+1] - ts:6.1f}s", flush=True)
    structure = sum(stage_exact["structure"])
    confidence = sum(stage_exact["confidence"])
    print(
        json.dumps(
            {
                "model": "esmfold2",
                "tokens": len(seq),
                "loops": args.loops,
                "sampling_steps": args.steps,
                "timed_total_s": t1 - t0,
                "structure_s": structure,
                "structure_share": structure / (t1 - t0),
                "confidence_s": confidence,
                "confidence_share": confidence / (t1 - t0),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
