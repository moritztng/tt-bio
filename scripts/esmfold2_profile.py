"""Warm stage-timing profile of the ttnn ESMFold2 fold, to locate the perf bottleneck.

Patches tt_bio.esmfold2.report_progress to timestamp stage transitions, folds one
protein twice (first fold pays kernel compilation; the second is the steady-state
number reported), and prints wall-clock seconds per stage: LM, trunk, diffusion,
confidence. Read-only w.r.t. the model — no code path changes.
"""
from __future__ import annotations
import argparse, time
import torch

SEQS = {
    "gb1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein", default="ubiquitin")
    ap.add_argument("--loops", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    import tt_bio.esmfold2 as E
    from tt_bio.esmfold2_runtime import load_ttnn_esmfold2, fold_complex

    marks = []
    _orig = E.report_progress
    def _rp(stage, step=0, total=0):
        marks.append((stage, step, total, time.time()))
        return _orig(stage, step, total)
    E.report_progress = _rp

    model = load_ttnn_esmfold2()
    seq = SEQS[args.protein]

    def one():
        marks.clear()
        t0 = time.time()
        fold_complex(model, [("A", seq)], num_loops=args.loops,
                     num_sampling_steps=args.steps, num_diffusion_samples=1, seed=0)
        return t0, time.time()

    print(f"[warmup fold {args.protein} L={len(seq)}] ...", flush=True)
    t0, t1 = one()
    print(f"  warmup total {t1 - t0:.1f}s (includes kernel compilation)", flush=True)

    print(f"[timed fold] ...", flush=True)
    t0, t1 = one()
    # stage boundaries from first-occurrence timestamps
    first = {}
    for stage, step, total, ts in marks:
        first.setdefault(stage, ts)
    order = sorted(first.items(), key=lambda kv: kv[1])
    print(f"  total {t1 - t0:.1f}s  (L={len(seq)}, loops={args.loops}, steps={args.steps})", flush=True)
    print("  stage first-seen offsets (s from fold start):", flush=True)
    for stage, ts in order:
        print(f"    {stage:12s} +{ts - t0:6.1f}s", flush=True)
    # rough per-stage durations from consecutive stage starts + fold end
    bounds = [ts for _, ts in order] + [t1]
    print("  approx stage durations:", flush=True)
    for i, (stage, ts) in enumerate(order):
        print(f"    {stage:12s} {bounds[i+1] - ts:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
